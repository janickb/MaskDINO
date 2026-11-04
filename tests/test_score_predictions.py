"""Unit tests for tools/score_predictions.py and tools/export_predictions_jsonl.py.

Same style as tests/test_truncated_instance_filter.py: a real `import maskdino`,
throwaway DatasetCatalog entries (unique names) holding hand-drawn rectangle masks.
"""
import json
import math
import os
import sys
import uuid

import numpy as np
import pycocotools.mask as mask_util
import pytest
import torch
from detectron2.data import DatasetCatalog
from detectron2.structures import Instances

from maskdino.data.class_mapping import ClassMapping, apply_class_mapping_to_metadata
from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import export_predictions_jsonl as ep
import score_predictions as sp

H = W = 40
CLASSES = ("a", "b", "c")


# --------------------------------------------------------------------------- helpers


def rect(y0, x0, y1, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def rle(mask):
    r = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def gt_ann(mask, cid):
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return {
        "segmentation": rle(mask),
        "category_id": cid,
        "bbox": [x0, y0, int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0],
        "bbox_mode": 1,
        "iscrowd": 0,
    }


def register(frames, classes=CLASSES):
    """frames: one GT list [(mask, contiguous class id), ...] per frame -> a fresh
    dataset name, registered the way the surgical HDF5 sets are."""
    name = f"_test_score_{uuid.uuid4().hex[:8]}"
    dicts = [
        {
            "file_name": f"/data/{name}/f{i:03d}.hdf5",
            "image_id": i,
            "height": H,
            "width": W,
            "annotations": [gt_ann(m, c) for m, c in gt],
        }
        for i, gt in enumerate(frames)
    ]
    DatasetCatalog.register(name, lambda: dicts)
    cm = ClassMapping({i: {"category_id": i + 1, "name": n} for i, n in enumerate(classes)})
    apply_class_mapping_to_metadata(name, cm)
    return name


def det(mask, class_name, score):
    return {"class_name": class_name, "score": score, "segmentation": rle(mask)}


def line(i, dets):
    return {"file_name": f"f{i:03d}.hdf5", "height": H, "width": W, "instances": dets}


def write_jsonl(path, records):
    with open(path, "w") as fh:
        fh.writelines(json.dumps(rec) + "\n" for rec in records)
    return str(path)


# two frames: a correct hit, a mislabeled hit, a hallucination, a miss
GT = [
    [(rect(2, 2, 12, 12), 0), (rect(20, 20, 30, 30), 1)],
    [(rect(5, 5, 15, 15), 2)],
]
DETS = [
    [
        det(rect(2, 2, 12, 12), "a", 0.9),
        det(rect(20, 20, 30, 30), "c", 0.45),
        det(rect(14, 26, 18, 34), "b", 0.7),
    ],
    [],
]


def score(tmp_path, name, records, **settings):
    path = write_jsonl(tmp_path / "preds.jsonl", records)
    return sp.score(name, path, str(tmp_path / "out"), **settings)


# --------------------------------------------------------------------------- scoring


def test_perfect_predictions(tmp_path):
    name = register(GT)
    records = [
        line(0, [det(rect(2, 2, 12, 12), "a", 0.9), det(rect(20, 20, 30, 30), "b", 0.8)]),
        line(1, [det(rect(5, 5, 15, 15), "c", 0.7)]),
    ]
    res = score(tmp_path, name, records)
    assert res["num_images"] == 2
    assert res["sweep/f1_max"] == 1.0
    assert res["sweep/classaware_f1_max"] == 1.0
    assert res["classaware_f1"] == 1.0
    out = tmp_path / "out"
    assert json.loads((out / "metrics.json").read_text()) == res
    settings = json.loads((out / "score_settings.json").read_text())
    assert settings["num_frames_scored"] == 2
    assert settings["sweep_thresholds"] == sp.SWEEP_GRID
    assert (out / "instance_matching_threshold_sweep.csv").exists()
    assert (out / "instance_matching_per_image.csv").exists()


def test_matches_the_evaluator_fed_directly(tmp_path):
    """The JSON Lines path must not change a single number."""
    name = register(GT)
    res = score(tmp_path, name, [line(i, d) for i, d in enumerate(DETS)])

    ev = HungarianInstanceEvaluator(name, sp.build_cfg(name), distributed=False)
    ev.reset()
    for i, dets in enumerate(DETS):
        inst = Instances((H, W))
        inst.pred_masks = torch.from_numpy(
            np.stack([mask_util.decode({**d["segmentation"], "counts": d["segmentation"]["counts"].encode()}) for d in dets])
            if dets
            else np.zeros((0, H, W), dtype=np.uint8)
        ).bool()
        inst.scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
        inst.pred_classes = torch.tensor(
            [CLASSES.index(d["class_name"]) for d in dets], dtype=torch.int64
        )
        ev.process([{"image_id": i, "file_name": f"/data/{name}/f{i:03d}.hdf5"}], [{"instances": inst}])
    assert dict(ev.evaluate()["instance_matching"]) == res
    # and the mixed case really exercises every count
    assert res["misclassified_total"] == 0  # the mislabeled hit scores below 0.5
    assert res["false_positives_total"] == 1
    assert res["false_negatives_total"] == 2


def test_classaware_sweep_reported(tmp_path):
    name = register(GT)
    res = score(tmp_path, name, [line(i, d) for i, d in enumerate(DETS)])
    # Up to 0.45 the mislabeled hit counts: classagnostic F1 2/3 (ties -> lowest
    # threshold), classaware 1/3. Above 0.7 only the correct hit is left: both 1/2.
    assert res["sweep/f1_max"] == pytest.approx(2 / 3)
    assert res["sweep/thresh_at_f1_max"] == pytest.approx(0.1)
    assert res["sweep/classaware_f1_max"] == pytest.approx(0.5)
    assert res["sweep/classaware_thresh_at_f1_max"] == pytest.approx(0.75)


def test_singleclass_dataset(tmp_path):
    name = register([[(rect(2, 2, 12, 12), 0)]], classes=("surgical-instruments",))
    res = score(tmp_path, name, [line(0, [det(rect(2, 2, 12, 12), "surgical-instruments", 0.9)])])
    assert res["sweep/f1_max"] == 1.0
    assert not any(k.startswith("sweep/classaware") for k in res)
    assert json.loads((tmp_path / "out" / "score_settings.json").read_text())["num_classes"] == 1


def test_truncated_gt_is_an_ignore_region(tmp_path):
    name = register([[(rect(0, 0, 10, 10), 0)]])  # touches the frame edge
    res = score(tmp_path, name, [line(0, [])])
    assert res["num_gt_total"] == 0
    assert res["false_negatives_total"] == 0
    res = score(tmp_path, name, [line(0, [])], exclude_truncated=False)
    assert res["false_negatives_total"] == 1


def test_sample_stride(tmp_path):
    name = register([[(rect(2, 2, 12, 12), 0)]] * 4)
    hit = [det(rect(2, 2, 12, 12), "a", 0.9)]
    res = score(tmp_path, name, [line(i, hit) for i in range(4)], sample_stride=2)
    assert res["num_images"] == 2
    settings = json.loads((tmp_path / "out" / "score_settings.json").read_text())
    assert settings["num_lines_skipped"] == 2
    # the kept frames are enough, and scoring again doesn't stride the strided list
    res = score(tmp_path, name, [line(0, hit), line(2, hit)], sample_stride=2)
    assert res["num_images"] == 2
    assert len(DatasetCatalog.get(name)) == 4
    with pytest.raises(ValueError, match="has no line"):
        score(tmp_path, name, [line(0, hit), line(1, hit)], sample_stride=2)


# --------------------------------------------------------------------------- rejections


def _rejects(tmp_path, records, match, gt=GT):
    with pytest.raises(ValueError, match=match):
        score(tmp_path, register(gt), records)


def test_rejects_a_missing_frame(tmp_path):
    _rejects(tmp_path, [line(0, [])], "f001.hdf5' of .* has no line")


def test_rejects_an_unknown_frame(tmp_path):
    _rejects(tmp_path, [line(0, []), line(1, []), line(7, [])], "f007.hdf5' is not in")


def test_rejects_a_duplicate_frame(tmp_path):
    _rejects(tmp_path, [line(0, []), line(1, []), line(0, [])], "already on line 1")


def test_rejects_an_unknown_class(tmp_path):
    _rejects(tmp_path, [line(0, [det(rect(2, 2, 12, 12), "forcep01_mm", 0.9)]), line(1, [])], "class_name 'forcep01_mm'")


@pytest.mark.parametrize("bad", [1.5, -0.1, math.nan, "0.9", None, True])
def test_rejects_a_bad_score(tmp_path, bad):
    _rejects(tmp_path, [line(0, [det(rect(2, 2, 12, 12), "a", bad)]), line(1, [])], "score")


def test_rejects_a_wrong_mask_size(tmp_path):
    small = rle(np.ones((20, 20), dtype=bool))
    d = {"class_name": "a", "score": 0.9, "segmentation": small}
    _rejects(tmp_path, [line(0, [d]), line(1, [])], r"mask size is \[20, 20\]")


def test_rejects_a_wrong_frame_size(tmp_path):
    _rejects(tmp_path, [{**line(0, []), "height": 1024}, line(1, [])], "height x width")


def test_rejects_an_unregistered_dataset(tmp_path):
    path = write_jsonl(tmp_path / "p.jsonl", [line(0, [])])
    with pytest.raises(ValueError, match="not a registered surgical"):
        sp.score("_no_such_dataset", path, str(tmp_path / "out"))


def test_collects_all_problems_in_one_error(tmp_path):
    # unknown frame, missing frame, unknown class, bad score
    with pytest.raises(ValueError, match="4 problem"):
        score(
            tmp_path,
            register(GT),
            [line(0, [det(rect(2, 2, 12, 12), "zz", 2.0)]), line(9, [])],
        )


# --------------------------------------------------------------------------- export


def _write_cache(run_dir, dets_per_frame):
    """A COCOEvaluator-style instances_predictions.pth (contiguous category ids)."""
    cache = [
        {
            "image_id": i,
            "instances": [
                {
                    "image_id": i,
                    "category_id": CLASSES.index(d["class_name"]),
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "score": d["score"],
                    "segmentation": d["segmentation"],
                }
                for d in dets
            ],
        }
        for i, dets in enumerate(dets_per_frame)
    ]
    path = run_dir / "inference" / "instances_predictions.pth"
    path.parent.mkdir(parents=True)
    torch.save(cache, path)
    return str(path)


def test_export_round_trip(tmp_path):
    name = register(GT)
    pth = _write_cache(tmp_path / "run", DETS)
    out = tmp_path / "exported.jsonl"
    assert ep.export(pth, name, str(out)) == 2
    exported = [json.loads(s) for s in out.read_text().splitlines()]
    assert exported == [line(i, d) for i, d in enumerate(DETS)]
    res_exported = sp.score(name, str(out), str(tmp_path / "a"))
    res_written = score(tmp_path, name, [line(i, d) for i, d in enumerate(DETS)])
    assert res_exported == res_written


def test_export_checks_the_run_class_mapping(tmp_path):
    name = register(GT)
    pth = _write_cache(tmp_path / "run", DETS)
    other = ClassMapping({0: {"category_id": -1, "name": "surgical-instruments"}})
    (tmp_path / "run" / "class_mapping.json").write_text(json.dumps(other.to_json_dict()))
    with pytest.raises(ValueError, match="surgical-instruments"):
        ep.export(pth, name, str(tmp_path / "x.jsonl"))


def test_export_rejects_an_unknown_image_id(tmp_path):
    name = register(GT)
    pth = _write_cache(tmp_path / "run", DETS + [[]])  # image_id 2 doesn't exist
    with pytest.raises(ValueError, match="image_id 2 is not in"):
        ep.export(pth, name, str(tmp_path / "x.jsonl"))
