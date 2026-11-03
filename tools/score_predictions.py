#!/usr/bin/env python
"""
Score instance-segmentation predictions from any framework (MaskDINO, Mask2Former,
Ultralytics YOLO, ...) with HungarianInstanceEvaluator, against a registered surgical
HDF5 dataset. Every model in a comparison goes through this one path, so all of them
are graded by exactly the same rules.

Input is a JSON Lines file, one line per frame of the dataset:

    {"file_name": "<frame>.hdf5", "height": 1024, "width": 1024,
     "instances": [{"class_name": "forcep02_mm", "score": 0.91,
                    "segmentation": {"size": [1024, 1024], "counts": "<COCO RLE>"}}]}

- file_name: the HDF5 frame's basename.
- class_name: one of the dataset's thing_classes; the one "surgical-instruments" class
  for a *_singleclass dataset.
- segmentation: the binary mask at the frame's full resolution, COCO RLE with counts
  as a string (pycocotools.mask.encode, counts decoded to ascii).
- Every frame of the dataset appears exactly once; a frame with no predictions has
  "instances": []. With --sample-stride N only the frames TEST_SAMPLE_STRIDE N keeps
  are scored and required; lines for the other frames are skipped.

tools/export_predictions_jsonl.py writes this format from a detectron2 run's
inference/instances_predictions.pth.

GT is prepared the way train_net.setup() prepares DATASETS.TEST: the sample stride,
then frame-edge-truncated instances as ignore regions (INPUT.EXCLUDE_TRUNCATED_INSTANCES),
then the evaluator's own MIN_VISIBILITY filter. Defaults come from maskdino/config.py;
the score-threshold sweep defaults to the grid tools/tuning/tune_single_class.py uses.

Writes to --output-dir: metrics.json (the flat instance_matching dict - the numbers to
report), score_settings.json (what was scored, and how), score_log.txt and the
instance_matching_* artifacts (see maskdino/evaluation/hungarian_evaluator_metrics.md).

Usage:
    ./.venv/bin/python tools/score_predictions.py \
        --dataset val_set_a_mm \
        --predictions runs/<run>/predictions/val_set_a_mm.jsonl \
        --output-dir runs/<run>/score/val_set_a_mm
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances
from detectron2.utils.logger import setup_logger

# repo root, so `maskdino` imports when this runs by path from any directory
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maskdino import (
    add_maskdino_config,
    apply_test_sample_stride,
    apply_truncated_instance_filter,
    set_num_classes_from_metadata,
)
from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator

# Same grid as tools/tuning/tune_single_class.py's SWEEP_GRID. It contains the default
# SCORE_THRESH 0.5, so the fixed-threshold numbers are one point of the sweep.
SWEEP_GRID = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90
MAX_ERRORS_SHOWN = 20


def build_cfg(
    dataset,
    *,
    score_thresh=None,
    iou_thresh=None,
    min_visibility=None,
    exclude_truncated=None,
    sweep_thresholds=None,
    sample_stride=1,
):
    """Default MaskDINO cfg with only the evaluation knobs set. None keeps the
    maskdino/config.py default; the class count comes from the dataset."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
    he.ENABLED = True
    if score_thresh is not None:
        he.SCORE_THRESH = score_thresh
    if iou_thresh is not None:
        he.IOU_THRESH = iou_thresh
    he.SWEEP_THRESHOLDS = list(SWEEP_GRID if sweep_thresholds is None else sweep_thresholds)
    if min_visibility is not None:
        cfg.INPUT.MIN_VISIBILITY = min_visibility
    if exclude_truncated is not None:
        cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES = exclude_truncated
    cfg.DATASETS.TEST = (dataset,)
    cfg.DATASETS.TEST_SAMPLE_STRIDE = sample_stride
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = -1
    set_num_classes_from_metadata(cfg, dataset)
    return cfg


def _check_dataset(dataset):
    def is_surgical(name):
        return bool(MetadataCatalog.get(name).get("class_mapping_entries", None))

    if dataset in DatasetCatalog.list() and is_surgical(dataset):
        return
    known = sorted(n for n in DatasetCatalog.list() if is_surgical(n))
    raise ValueError(
        f"{dataset!r} is not a registered surgical HDF5 dataset; registered: {known}"
    )


@contextlib.contextmanager
def _restored_registration(dataset):
    """apply_test_sample_stride / apply_truncated_instance_filter re-register the
    dataset in place, as train_net.setup() does once per process. Put the original back
    afterwards, so a second score() of the same dataset doesn't stride an already
    strided list."""
    original = DatasetCatalog[dataset]
    try:
        yield
    finally:
        DatasetCatalog.remove(dataset)
        DatasetCatalog.register(dataset, original)


def load_predictions(path):
    """JSON Lines -> {frame basename: (line number, record)}. Fails on a malformed
    line and on a frame listed twice."""
    by_frame = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from None
            if not isinstance(rec, dict) or "file_name" not in rec or "instances" not in rec:
                raise ValueError(f"{path}:{lineno}: needs 'file_name' and 'instances'")
            name = os.path.basename(rec["file_name"])
            if name in by_frame:
                raise ValueError(
                    f"{path}:{lineno}: frame {name!r} is already on line {by_frame[name][0]}"
                )
            by_frame[name] = (lineno, rec)
    return by_frame


def _frame_errors(where, rec, h, w, class_index):
    """Format problems in one frame's record -> list of messages (empty when fine)."""
    errors = []
    if (rec.get("height"), rec.get("width")) != (h, w):
        errors.append(
            f"{where}: height x width is {rec.get('height')} x {rec.get('width')}, "
            f"the frame is {h} x {w}"
        )
    if not isinstance(rec["instances"], list):
        return errors + [f"{where}: 'instances' must be a list"]
    for k, det in enumerate(rec["instances"]):
        at = f"{where}, instance {k}"
        if not isinstance(det, dict):
            errors.append(f"{at}: must be an object")
            continue
        if det.get("class_name") not in class_index:
            errors.append(
                f"{at}: class_name {det.get('class_name')!r} is not one of {list(class_index)}"
            )
        s = det.get("score")
        # NaN fails the range check too
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not 0.0 <= s <= 1.0:
            errors.append(f"{at}: score {s!r} is not a number in [0, 1]")
        seg = det.get("segmentation")
        if not isinstance(seg, dict) or not isinstance(seg.get("counts"), str):
            errors.append(f"{at}: segmentation must be COCO RLE with 'counts' as a string")
        elif list(seg.get("size", [])) != [h, w]:
            errors.append(f"{at}: mask size is {seg.get('size')}, the frame is [{h}, {w}]")
    return errors


def to_instances(rec, h, w, class_index, floor):
    """One frame's predictions -> detectron2 Instances, keeping scores >= floor. The
    evaluator drops the rest itself; skipping them here only saves decoding. The gate
    runs on float32 scores, as the evaluator's does, so both keep the same set."""
    dets = rec["instances"]
    scores = np.asarray([d["score"] for d in dets], dtype=np.float32)
    keep = [d for d, s in zip(dets, scores) if s >= floor]
    inst = Instances((h, w))
    if keep:
        masks = np.stack(
            [
                mask_util.decode(
                    {"size": [h, w], "counts": d["segmentation"]["counts"].encode("ascii")}
                )
                for d in keep
            ]
        )
        inst.pred_masks = torch.from_numpy(masks).bool()
    else:
        inst.pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
    inst.scores = torch.from_numpy(scores[scores >= floor])
    inst.pred_classes = torch.tensor(
        [class_index[d["class_name"]] for d in keep], dtype=torch.int64
    )
    return inst


def score(dataset, predictions, output_dir, **settings):
    """Score `predictions` (JSON Lines, see the module docstring) on `dataset` -> the
    flat instance_matching dict. Writes metrics.json, score_settings.json and the
    evaluator's artifacts to output_dir. `settings` are build_cfg's keyword arguments.
    Raises ValueError listing the problems when the file doesn't cover the dataset or
    breaks the format."""
    _check_dataset(dataset)
    cfg = build_cfg(dataset, **settings)
    by_frame = load_predictions(predictions)
    with _restored_registration(dataset):
        full = DatasetCatalog.get(dataset)
        apply_test_sample_stride(cfg)
        apply_truncated_instance_filter(cfg)
        frames = DatasetCatalog.get(dataset)
        evaluator = HungarianInstanceEvaluator(
            dataset, cfg, distributed=False, output_dir=output_dir
        )

    known = {os.path.basename(d["file_name"]) for d in full}
    if len(known) != len(full):
        raise ValueError(f"{dataset} has frames sharing a basename; can't key them by name")
    # dataset order, which is also the order of an in-training eval's records
    scored = {os.path.basename(d["file_name"]): d for d in frames}
    thing_classes = list(MetadataCatalog.get(dataset).thing_classes)
    class_index = {name: i for i, name in enumerate(thing_classes)}

    errors = [
        f"line {by_frame[n][0]}: frame {n!r} is not in {dataset}"
        for n in sorted(set(by_frame) - known)
    ]
    errors += [f"frame {n!r} of {dataset} has no line" for n in sorted(set(scored) - set(by_frame))]
    for name, d in scored.items():
        if name in by_frame:
            lineno, rec = by_frame[name]
            errors += _frame_errors(
                f"line {lineno} ({name})", rec, d["height"], d["width"], class_index
            )
    if errors:
        shown = "\n  ".join(errors[:MAX_ERRORS_SHOWN])
        if len(errors) > MAX_ERRORS_SHOWN:
            shown += f"\n  ... and {len(errors) - MAX_ERRORS_SHOWN} more"
        raise ValueError(f"{predictions}: {len(errors)} problem(s):\n  {shown}")

    he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
    floor = min([float(he.SCORE_THRESH), *(float(t) for t in he.SWEEP_THRESHOLDS)])
    evaluator.reset()
    for name, d in scored.items():
        inst = to_instances(by_frame[name][1], d["height"], d["width"], class_index, floor)
        evaluator.process(
            [{"image_id": d["image_id"], "file_name": d["file_name"]}],
            [{"instances": inst}],
        )
    metrics = dict(evaluator.evaluate()["instance_matching"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(output_dir, "score_settings.json"), "w") as fh:
        json.dump(
            {
                "dataset": dataset,
                "predictions": os.path.abspath(predictions),
                "num_frames_scored": len(scored),
                "num_lines_skipped": len(by_frame) - len(scored),  # outside the stride
                "sample_stride": int(cfg.DATASETS.TEST_SAMPLE_STRIDE),
                "score_thresh": float(he.SCORE_THRESH),
                "iou_thresh": float(he.IOU_THRESH),
                "sweep_thresholds": [float(t) for t in he.SWEEP_THRESHOLDS],
                "min_visibility": float(cfg.INPUT.MIN_VISIBILITY),
                "exclude_truncated_instances": bool(cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES),
                "num_classes": int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES),
                "thing_classes": thing_classes,
            },
            fh,
            indent=2,
        )
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="registered dataset, e.g. val_set_a_mm or val_set_a_mm_singleclass",
    )
    ap.add_argument("--predictions", required=True, help="JSON Lines file (format above)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--score-thresh", type=float, default=None,
        help="fixed operating point (default: config.py's HUNGARIAN_EVAL.SCORE_THRESH)",
    )
    ap.add_argument(
        "--iou-thresh", type=float, default=None,
        help="minimum mask IoU for a match (default: config.py's HUNGARIAN_EVAL.IOU_THRESH)",
    )
    ap.add_argument(
        "--min-visibility", type=float, default=None,
        help="GT visibility filter; match the training recipe being compared "
        "(default: config.py's INPUT.MIN_VISIBILITY)",
    )
    ap.add_argument(
        "--exclude-truncated", action=argparse.BooleanOptionalAction, default=None,
        help="score frame-edge-truncated GT as ignore regions "
        "(default: config.py's INPUT.EXCLUDE_TRUNCATED_INSTANCES)",
    )
    ap.add_argument(
        "--sweep-thresholds", type=float, nargs="+", default=None,
        help="score thresholds to sweep (default: 0.10 to 0.90 in steps of 0.05)",
    )
    ap.add_argument(
        "--sample-stride", type=int, default=1,
        help="score every N-th frame only, like DATASETS.TEST_SAMPLE_STRIDE (default: all)",
    )
    ap.add_argument(
        "--num-threads", type=int, default=4,
        help="torch CPU threads (default: 4; see below)",
    )
    args = ap.parse_args()

    # torch's default (one thread per core) makes every mask op fight the other users'
    # jobs on a shared machine: measured at load ~380 on 256 cores, 64 frames took 73 s
    # and 1872 CPU-s with the default, 8 s and 20 CPU-s with 4 threads.
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(output=os.path.join(args.output_dir, "score_log.txt"), name="maskdino")
    try:
        metrics = score(
            args.dataset,
            args.predictions,
            args.output_dir,
            score_thresh=args.score_thresh,
            iou_thresh=args.iou_thresh,
            min_visibility=args.min_visibility,
            exclude_truncated=args.exclude_truncated,
            sweep_thresholds=args.sweep_thresholds,
            sample_stride=args.sample_stride,
        )
    except ValueError as exc:
        sys.exit(f"score_predictions: {exc}")

    print(f"\n=== {args.dataset}: {int(metrics['num_images'])} frames ===")
    for k, v in metrics.items():
        # skip the per-class "<group>/<class>" breakdowns; keep sweep/*
        if "/" not in k or k.startswith("sweep/"):
            print(f"  {k:36s} {v:.4f}")
    print(f"\nwrote {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
