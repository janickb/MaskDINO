#!/usr/bin/env python
"""
Re-run HungarianInstanceEvaluator offline from a finished run's cached predictions
(inference/instances_predictions.pth), no GPU / no re-inference.

Lets you re-tune SCORE_THRESH / IOU_THRESH / MIN_VISIBILITY and regenerate the
instance_matching_* artifacts (confusion CSV/PNG, per-image CSV) in seconds.

Usage:
    ./.venv/bin/python tools/rerun_hungarian_from_cache.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune_multiclass.yaml \
        --predictions runs/hungarian_eval_multiclass/inference/instances_predictions.pth \
        --output-dir  runs/hungarian_eval_multiclass/inference \
        [--score-thresh 0.5] [--iou-thresh 0.5] [--min-visibility 0.0] [--dataset val]

Threshold sweep (--sweep): re-score at several SCORE_THRESH values in one pass and
write a summary CSV of precision/recall/F1 vs threshold, plus one artifact
subdirectory per threshold.

This matters for the MaskDINO-vs-Mask2Former comparison. MaskDINO scores with
mask_cls.sigmoid() (focal loss, no background class) while Mask2Former uses
F.softmax(mask_cls, -1)[:, :-1] (softmax CE over num_classes + 1); both then multiply
by the same mask score. The final scores therefore live on different scales, so a
single shared SCORE_THRESH=0.5 compares the two arms at DIFFERENT operating points.
Report threshold-free COCO segm AP as the headline number, and read these Hungarian
metrics at each arm's own best-F1 threshold as well as at a fixed 0.5:

    ./.venv/bin/python tools/rerun_hungarian_from_cache.py \
        --config-file runs/<run>/config.yaml \
        --predictions runs/<run>/inference/instances_predictions.pth \
        --output-dir  runs/<run>/inference/hungarian_sweep \
        --sweep 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
"""
import argparse
from collections import defaultdict

# Run as a bare script (`python tools/<name>.py`) puts tools/ on sys.path, not the
# repo root, so `maskdino` would not resolve. Same idiom as the other tools here.
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))


import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog
from detectron2.structures import Boxes, Instances

from maskdino import arch_ns, build_base_cfg, set_num_classes_from_metadata
from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator


def _process_one(ev, image_id, dets, fname_by_id=None):
    ev.process(
        [{"image_id": image_id, "file_name": (fname_by_id or {}).get(image_id, "")}],
        [{"instances": _build_instances(dets)}],
    )


def _build_instances(dets):
    """Cached detections -> Instances, in the shape HungarianInstanceEvaluator reads.

    pred_boxes is reconstructed because Instances slicing needs every field, but the
    evaluator never reads it: mask_iou_matrix derives its own box-prefilter boxes from
    the masks. That is what makes these metrics architecture-agnostic - Mask2Former's
    boxes are mask-derived, MaskDINO's come from a trained box head.
    """
    if dets:
        masks = np.stack([mask_util.decode(d["segmentation"]) for d in dets])
        h, w = masks.shape[1:]
        inst = Instances((h, w))
        inst.pred_masks = torch.from_numpy(masks).bool()
        inst.scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
        inst.pred_classes = torch.tensor([d["category_id"] for d in dets], dtype=torch.int64)
        xywh = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32).reshape(-1, 4)
        xyxy = xywh.clone()
        xyxy[:, 2:] += xyxy[:, :2]
        inst.pred_boxes = Boxes(xyxy)
    else:
        inst = Instances((1024, 1024))
        inst.pred_masks = torch.zeros((0, 1024, 1024), dtype=torch.bool)
        inst.scores = torch.zeros((0,), dtype=torch.float32)
        inst.pred_classes = torch.zeros((0,), dtype=torch.int64)
        inst.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--predictions", required=True, help="inference/instances_predictions.pth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default=None, help="defaults to cfg.DATASETS.TEST[0]")
    ap.add_argument("--score-thresh", type=float, default=None)
    ap.add_argument("--iou-thresh", type=float, default=None)
    ap.add_argument("--min-visibility", type=float, default=None)
    ap.add_argument(
        "--sweep",
        default=None,
        help="comma-separated SCORE_THRESH values; writes <output-dir>/sweep_summary.csv "
        "plus one t<value>/ subdirectory of artifacts per threshold. Overrides "
        "--score-thresh.",
    )
    args = ap.parse_args()

    cfg = build_base_cfg()
    cfg.merge_from_file(args.config_file)
    set_num_classes_from_metadata(cfg, args.dataset or cfg.DATASETS.TEST[0])
    he = arch_ns(cfg).TEST.HUNGARIAN_EVAL
    he.ENABLED = True
    if args.score_thresh is not None:
        he.SCORE_THRESH = args.score_thresh
    if args.iou_thresh is not None:
        he.IOU_THRESH = args.iou_thresh
    if args.min_visibility is not None:
        # the evaluator reads the GT visibility filter from INPUT.MIN_VISIBILITY
        cfg.INPUT.MIN_VISIBILITY = args.min_visibility
    cfg.freeze()

    dataset = args.dataset or cfg.DATASETS.TEST[0]
    fname_by_id = {d["image_id"]: d["file_name"] for d in DatasetCatalog.get(dataset)}

    preds = torch.load(args.predictions, weights_only=False)
    # pre-filter by score before decoding masks (the evaluator applies the same
    # >= SCORE_THRESH gate itself, so this is equivalent and ~10x less RLE decoding)
    # With --sweep, gate the pre-filter at the LOWEST threshold being swept, not at the
    # config's SCORE_THRESH: otherwise sweeping below the config value silently yields
    # zero predictions at every point (the detections were already dropped here).
    if args.sweep:
        st = min(float(t) for t in args.sweep.split(",") if t.strip())
    else:
        st = float(he.SCORE_THRESH)
    by_image = defaultdict(list)
    for p in preds:
        for det in p["instances"]:
            if det["score"] >= st:
                by_image[p["image_id"]].append(det)
    for image_id in fname_by_id:
        by_image.setdefault(image_id, [])

    def score_at(score_thresh, output_dir):
        """Run the evaluator over the cached predictions at one SCORE_THRESH.

        cfg is frozen by now, so clone and re-point the threshold rather than mutating
        it - the evaluator reads SCORE_THRESH from the cfg it is handed.
        """
        local_cfg = cfg.clone()
        local_cfg.defrost()
        arch_ns(local_cfg).TEST.HUNGARIAN_EVAL.SCORE_THRESH = float(score_thresh)
        local_cfg.freeze()
        ev = HungarianInstanceEvaluator(
            dataset, local_cfg, distributed=False, output_dir=output_dir
        )
        ev.reset()
        for image_id, dets in by_image.items():
            dets = [d for d in dets if d["score"] >= float(score_thresh)]
            _process_one(ev, image_id, dets, fname_by_id)
        return ev.evaluate()["instance_matching"]

    ev = HungarianInstanceEvaluator(
        dataset, cfg, distributed=False, output_dir=args.output_dir
    )
    ev.reset()
    for image_id, dets in by_image.items():
        _process_one(ev, image_id, dets, fname_by_id)

    if args.sweep:
        import csv
        import os

        thresholds = [float(t) for t in args.sweep.split(",") if t.strip()]
        os.makedirs(args.output_dir, exist_ok=True)
        rows = []
        for t in thresholds:
            sub_dir = os.path.join(args.output_dir, f"t{t}")
            os.makedirs(sub_dir, exist_ok=True)
            res = score_at(t, sub_dir)
            rows.append({"score_thresh": t, **{k: v for k, v in res.items() if "/" not in k}})
            print(
                f"  score>={t:<5} "
                f"P={res['classagnostic_precision']:.4f} "
                f"R={res['classagnostic_recall']:.4f} "
                f"F1={res['classagnostic_f1']:.4f} "
                f"classaware_F1={res['classaware_f1']:.4f} "
                f"n_pred={res['num_pred_total']:.0f}"
            )
        summary = os.path.join(args.output_dir, "sweep_summary.csv")
        with open(summary, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        best = max(rows, key=lambda r: r["classagnostic_f1"])
        print(f"\nwrote {summary}")
        print(
            f"best class-agnostic F1 {best['classagnostic_f1']:.4f} "
            f"at SCORE_THRESH={best['score_thresh']}"
        )
        print(
            "NOTE: compare the two architectures at their OWN best-F1 thresholds as well "
            "as at a shared 0.5 - their score scales differ (sigmoid-focal vs softmax-CE)."
        )
        return

    res = ev.evaluate()["instance_matching"]
    print(f"\n=== instance_matching  (score>={he.SCORE_THRESH}  iou>={he.IOU_THRESH}) ===")
    for k, v in res.items():
        if "/" not in k:
            print(f"  {k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
