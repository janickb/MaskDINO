"""Fast synthetic unit tests for maskdino.evaluation.hungarian_instance_evaluation.

No GPU, no DatasetCatalog, no real data. The evaluator is built via object.__new__ +
direct attribute assignment (mirrors the dev-time smoke checks) so we don't need a full
cfg or a registered dataset. Masks are hand-drawn rectangles; GT annotations are real
pycocotools RLE (counts kept as an ascii str, matching list_hdf5_dicts output).
"""

import csv
import logging

import numpy as np
import pycocotools.mask as mask_util
import pytest
import torch
from detectron2.structures import Instances

from maskdino.evaluation.hungarian_instance_evaluation import (
    HungarianInstanceEvaluator,
    hungarian_match,
    mask_iou_matrix,
)

H = W = 40


# --------------------------------------------------------------------------- helpers


def rect(y0, x0, y1, x1, h=H, w=W):
    """Bool (h, w) mask with [y0:y1, x0:x1] set."""
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def rle_ann(mask_bool, category_id, h=H, w=W):
    """GT annotation dict; segmentation is real RLE with counts as an ascii str."""
    rle = mask_util.encode(np.asfortranarray(mask_bool.numpy().astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"segmentation": rle, "category_id": int(category_id)}


def make_instances(masks, scores, classes, h=H, w=W):
    if masks:
        m = torch.stack(masks).bool()
    else:
        m = torch.zeros((0, h, w), dtype=torch.bool)
    inst = Instances((h, w))
    inst.pred_masks = m
    inst.scores = torch.tensor(scores, dtype=torch.float32)
    inst.pred_classes = torch.tensor(classes, dtype=torch.int64)
    return inst


def make_evaluator(
    *,
    score_thresh=0.5,
    iou_thresh=0.5,
    num_classes=6,
    thing_classes=None,
    output_dir=None,
    box_prefilter=True,
    gt_by_image_id=None,
    hw_by_image_id=None,
):
    ev = object.__new__(HungarianInstanceEvaluator)
    ev._dataset_name = "test"
    ev._distributed = False
    ev._output_dir = str(output_dir) if output_dir is not None else None
    ev._logger = logging.getLogger("test_hungarian")
    ev._cpu_device = torch.device("cpu")
    ev._score_thresh = float(score_thresh)
    ev._iou_thresh = float(iou_thresh)
    ev._min_visibility = 0.0
    ev._box_prefilter = bool(box_prefilter)
    ev._num_classes = int(num_classes)
    ev._check_classes = num_classes > 1
    ev._thing_classes = (
        thing_classes
        if thing_classes is not None
        else [f"c{i}" for i in range(num_classes)]
    )
    ev._gt_by_image_id = gt_by_image_id or {}
    ev._hw_by_image_id = hw_by_image_id or {}
    ev._records = []
    return ev


def run_one(ev, image_id, gt_masks_classes, pred):
    """gt_masks_classes: list[(mask_bool, cid)]; pred: (masks, scores, classes)."""
    ev._gt_by_image_id[image_id] = [rle_ann(m, c) for m, c in gt_masks_classes]
    ev._hw_by_image_id[image_id] = (H, W)
    if not ev._records:
        ev.reset()
    ev.process(
        [{"image_id": image_id, "file_name": f"{image_id}.hdf5"}],
        [{"instances": make_instances(*pred)}],
    )
    return ev._records[-1]


# --------------------------------------------------------------------------- mask_iou_matrix


def test_iou_known_value():
    a = rect(0, 0, 10, 10)  # 100 px
    b = rect(0, 5, 10, 15)  # 100 px, overlap = 10x5 = 50, union = 150
    a = a[None]  # introduce leading 1 (40,40) -> (1,40,40)
    b = b[None]
    iou = mask_iou_matrix(a, b, box_prefilter=False)
    assert iou.shape == (1, 1)
    assert iou[0, 0].item() == pytest.approx(1 / 3, abs=1e-5)


def test_iou_identical_and_disjoint():
    a = rect(0, 0, 10, 10)
    far = rect(20, 20, 30, 30)
    iou = mask_iou_matrix(torch.stack([a, far]), a[None], box_prefilter=False)
    assert iou[0, 0].item() == pytest.approx(1.0)
    assert iou[1, 0].item() == pytest.approx(0.0)


def test_iou_empty_inputs():
    a = rect(0, 0, 10, 10)
    empty = torch.zeros((0, H, W), dtype=torch.bool)
    assert mask_iou_matrix(empty, a[None]).shape == (0, 1)
    assert mask_iou_matrix(a[None], empty).shape == (1, 0)


def test_iou_prefilter_agrees():
    preds = torch.stack([rect(0, 0, 12, 12), rect(0, 18, 12, 30), rect(25, 25, 38, 38)])
    gts = torch.stack([rect(0, 3, 12, 15), rect(22, 22, 35, 35), rect(2, 2, 6, 6)])
    a = mask_iou_matrix(preds, gts, box_prefilter=True)
    b = mask_iou_matrix(preds, gts, box_prefilter=False)
    assert torch.allclose(a, b)


def test_iou_dtype_agnostic():
    a = rect(0, 0, 10, 10)
    b = rect(0, 5, 10, 15)
    ref = mask_iou_matrix(a[None].bool(), b[None].bool(), box_prefilter=False)
    u = mask_iou_matrix(
        a[None].to(torch.uint8), b[None].to(torch.uint8), box_prefilter=False
    )
    f = mask_iou_matrix(a[None].float(), b[None].float(), box_prefilter=False)
    assert torch.allclose(ref, u) and torch.allclose(ref, f)


# --------------------------------------------------------------------------- hungarian_match


def test_hungarian_lexicographic_beats_maxsum():
    # review counterexample: max-sum keeps (0,0)=0.99 + (1,1)=0.49 -> 1 accepted.
    # lexicographic prefers (0,1)=0.80 + (1,0)=0.60 -> 2 accepted.
    iou = torch.tensor([[0.99, 0.80], [0.60, 0.49]])
    _, _, matched_iou = hungarian_match(iou, 0.5)
    assert int((matched_iou >= 0.5).sum()) == 2
    assert sorted(matched_iou[matched_iou >= 0.5].tolist()) == pytest.approx(
        [0.60, 0.80]
    )


def test_hungarian_tiebreak_prefers_higher_iou():
    # column-0 GT has two qualifying preds (0.7 and 0.9); the tie-break must pick 0.9.
    iou = torch.tensor([[0.7, 0.0], [0.9, 0.0]])
    row, col, matched_iou = hungarian_match(iou, 0.5)
    acc = np.where(matched_iou >= 0.5)[0]
    assert acc.size == 1
    assert int(col[acc[0]]) == 0
    assert int(row[acc[0]]) == 1
    assert float(matched_iou[acc[0]]) == pytest.approx(0.9)


def test_hungarian_empty():
    row, col, matched_iou = hungarian_match(torch.zeros(0, 0), 0.5)
    assert row.shape == (0,) and col.shape == (0,) and matched_iou.shape == (0,)


def test_hungarian_all_below_threshold():
    iou = torch.tensor([[0.30, 0.20], [0.10, 0.40]])
    _, _, matched_iou = hungarian_match(iou, 0.5)
    assert int((matched_iou >= 0.5).sum()) == 0


def test_hungarian_k_bound():
    iou = torch.eye(3) * 0.999
    _, _, matched_iou = hungarian_match(iou, 0.5)
    assert int((matched_iou >= 0.5).sum()) == 3
    assert matched_iou.tolist() == pytest.approx([0.999, 0.999, 0.999])


# --------------------------------------------------------------------------- process: 3 phenomena


def test_process_hallucination():
    ev = make_evaluator()
    r = run_one(
        ev,
        1,
        [(rect(0, 0, 12, 12), 3)],
        ([rect(0, 0, 12, 12), rect(20, 20, 32, 32)], [0.9, 0.8], [3, 5]),
    )
    assert r["num_gt"] == 1 and r["num_pred"] == 2
    assert r["num_mask_matched"] == 1
    assert r["num_false_pos"] == 1
    assert r["num_false_neg"] == 0
    assert r["num_misclassified"] == 0
    assert r["num_correct_class"] == 1
    assert r["false_pos_per_class"] == {5: 1}
    assert r["confusion_pairs"] == [[3, 3]]


def test_process_false_negative():
    ev = make_evaluator()
    r = run_one(
        ev,
        1,
        [(rect(0, 0, 10, 10), 3), (rect(0, 20, 10, 30), 3), (rect(25, 0, 35, 10), 5)],
        ([rect(0, 0, 10, 10), rect(0, 20, 10, 30)], [0.9, 0.9], [3, 3]),
    )
    assert r["num_gt"] == 3 and r["num_pred"] == 2
    assert r["num_mask_matched"] == 2
    assert r["num_false_neg"] == 1
    assert r["num_false_pos"] == 0
    assert r["num_misclassified"] == 0
    assert r["false_neg_per_class"] == {5: 1}


def test_process_misclassification():
    ev = make_evaluator()
    r = run_one(
        ev,
        1,
        [(rect(0, 0, 12, 12), 3)],
        ([rect(0, 0, 12, 12)], [0.9], [4]),
    )
    assert r["num_mask_matched"] == 1
    assert r["num_correct_class"] == 0
    assert r["num_misclassified"] == 1
    assert r["num_false_neg"] == 0
    assert r["num_false_pos"] == 0
    assert r["confusion_pairs"] == [[3, 4]]
    assert r["misclassified_per_class"] == {3: 1}


def test_process_score_gate():
    ev = make_evaluator(score_thresh=0.5)
    # third pred would match the third GT but is below threshold -> dropped
    r = run_one(
        ev,
        1,
        [(rect(0, 0, 10, 10), 3), (rect(0, 20, 10, 30), 3), (rect(25, 0, 35, 10), 5)],
        (
            [rect(0, 0, 10, 10), rect(0, 20, 10, 30), rect(25, 0, 35, 10)],
            [0.9, 0.9, 0.3],
            [3, 3, 5],
        ),
    )
    assert r["num_pred"] == 2
    assert r["num_false_neg"] == 1


def test_process_count_identities():
    ev = make_evaluator()
    r = run_one(
        ev,
        1,
        [(rect(0, 0, 12, 12), 3), (rect(0, 20, 12, 32), 4)],
        ([rect(0, 0, 12, 12), rect(25, 25, 38, 38)], [0.9, 0.8], [3, 4]),
    )
    assert r["num_mask_matched"] + r["num_false_neg"] == r["num_gt"]
    assert r["num_mask_matched"] + r["num_false_pos"] == r["num_pred"]


def test_process_resolution_mismatch_raises():
    ev = make_evaluator()
    ev._gt_by_image_id[1] = [rle_ann(rect(0, 0, 10, 10), 3)]
    ev._hw_by_image_id[1] = (H, W)
    ev.reset()
    big = make_instances(
        [torch.zeros((64, 64), dtype=torch.bool)], [0.9], [3], h=64, w=64
    )
    with pytest.raises(RuntimeError):
        ev.process([{"image_id": 1, "file_name": "x"}], [{"instances": big}])


# --------------------------------------------------------------------------- evaluate


def _two_image_scene(ev):
    # img1: GT [c1 @rectA, c2 @rectB]; preds match both, rectB mislabeled c3
    run_one(
        ev,
        1,
        [(rect(0, 0, 12, 12), 1), (rect(0, 20, 12, 32), 2)],
        ([rect(0, 0, 12, 12), rect(0, 20, 12, 32)], [0.9, 0.9], [1, 3]),
    )
    # img2: GT [c1 @rectC]; pred is a disjoint c2 blob -> miss + hallucination
    run_one(ev, 2, [(rect(0, 0, 12, 12), 1)], ([rect(25, 25, 38, 38)], [0.9], [2]))


def test_evaluate_headline_recall_identities():
    ev = make_evaluator(num_classes=4, thing_classes=["bg", "c1", "c2", "c3"])
    _two_image_scene(ev)
    res = ev.evaluate()["instance_matching"]
    assert res["classagnostic_recall"] == pytest.approx(
        res["num_mask_matched_total"] / res["num_gt_total"]
    )
    assert res["classaware_recall"] == pytest.approx(
        res["num_correct_class_total"] / res["num_gt_total"]
    )
    assert res["num_gt_total"] == 3
    assert res["num_mask_matched_total"] == 2
    assert res["num_correct_class_total"] == 1
    assert res["false_negatives_total"] == 1
    assert res["false_positives_total"] == 1
    assert res["misclassified_total"] == 1


def test_evaluate_classaware_le_classagnostic():
    ev = make_evaluator(num_classes=4, thing_classes=["bg", "c1", "c2", "c3"])
    _two_image_scene(ev)
    res = ev.evaluate()["instance_matching"]
    assert res["classaware_recall"] < res["classagnostic_recall"]
    assert res["classaware_precision"] < res["classagnostic_precision"]


def test_evaluate_rates():
    ev = make_evaluator(num_classes=4, thing_classes=["bg", "c1", "c2", "c3"])
    _two_image_scene(ev)
    res = ev.evaluate()["instance_matching"]
    assert res["false_negatives_per_gt"] == pytest.approx(
        res["false_negatives_total"] / res["num_gt_total"]
    )
    assert res["false_positives_per_pred"] == pytest.approx(
        res["false_positives_total"] / res["num_pred_total"]
    )
    assert res["misclassified_per_match"] == pytest.approx(
        res["misclassified_total"] / res["num_mask_matched_total"]
    )


def test_evaluate_per_class_microaverage():
    ev = make_evaluator(num_classes=4, thing_classes=["bg", "c1", "c2", "c3"])
    _two_image_scene(ev)
    res = ev.evaluate()["instance_matching"]
    # weighted mean of per-class recall == headline recall
    gt = {"c1": 2, "c2": 1}
    agn = sum(res[f"classagnostic_recall_per_class/{k}"] * v for k, v in gt.items())
    awe = sum(res[f"classaware_recall_per_class/{k}"] * v for k, v in gt.items())
    total = sum(gt.values())
    assert agn / total == pytest.approx(res["classagnostic_recall"])
    assert awe / total == pytest.approx(res["classaware_recall"])


def test_evaluate_result_all_float_and_csv_safe():
    from collections import OrderedDict

    from detectron2.evaluation.testing import flatten_results_dict, print_csv_format

    ev = make_evaluator(num_classes=4, thing_classes=["bg", "c1", "c2", "c3"])
    _two_image_scene(ev)
    result = ev.evaluate()
    flat = flatten_results_dict(result)
    for k, v in flat.items():
        assert isinstance(float(v), float), k
    print_csv_format(OrderedDict(result))  # must not raise (nested-dict regression)


def test_evaluate_empty_records():
    ev = make_evaluator()
    ev.reset()
    assert ev.evaluate() == {"instance_matching": {}}


# --------------------------------------------------------------------------- singleclass


def test_singleclass_disables_misclassification():
    ev = make_evaluator(num_classes=1, thing_classes=["c0", "c1"])
    r = run_one(ev, 1, [(rect(0, 0, 12, 12), 1)], ([rect(0, 0, 12, 12)], [0.9], [0]))
    assert r["num_misclassified"] == 0
    assert r["num_correct_class"] == r["num_mask_matched"] == 1
    res = ev.evaluate()["instance_matching"]
    assert not any(k.startswith("misclassified_per_class/") for k in res)
    assert res["classaware_recall"] == pytest.approx(res["classagnostic_recall"])
    assert res["classaware_precision"] == pytest.approx(res["classagnostic_precision"])


# --------------------------------------------------------------------------- artifacts


def _read_csv(path):
    with open(path) as fh:
        return list(csv.reader(fh))


def test_artifacts_confusion_csv_structure(tmp_path):
    ev = make_evaluator(
        num_classes=4, thing_classes=["bg", "c1", "c2", "c3"], output_dir=tmp_path
    )
    _two_image_scene(ev)
    ev.evaluate()

    rows = _read_csv(tmp_path / "instance_matching_confusion.csv")
    header, data = rows[0], rows[1:]
    assert header[-1] == "(false negative)"
    assert data[-1][0] == "(false positive)"
    n = len(header) - 1  # numeric columns
    assert len(data) == n  # square (C+1) x (C+1)

    # raw GT-class row sums == that class's GT count
    gt_count = {"c1": 2, "c2": 1, "c3": 0}
    by_label = {row[0]: [int(x) for x in row[1:]] for row in data}
    for cls, cnt in gt_count.items():
        assert sum(by_label[cls]) == cnt


def test_artifacts_rownorm_rows_sum_to_one(tmp_path):
    ev = make_evaluator(
        num_classes=4, thing_classes=["bg", "c1", "c2", "c3"], output_dir=tmp_path
    )
    _two_image_scene(ev)
    ev.evaluate()

    rows = _read_csv(tmp_path / "instance_matching_confusion_rownorm.csv")[1:]
    for row in rows:
        vals = [float(x) for x in row[1:]]
        if sum(vals) > 0:  # skip all-zero rows (classes with no GT and no preds)
            assert sum(vals) == pytest.approx(1.0, abs=1e-6)


def test_artifacts_png_written(tmp_path):
    ev = make_evaluator(
        num_classes=4, thing_classes=["bg", "c1", "c2", "c3"], output_dir=tmp_path
    )
    _two_image_scene(ev)
    ev.evaluate()
    png = tmp_path / "instance_matching_confusion.png"
    assert png.exists() and png.stat().st_size > 0
