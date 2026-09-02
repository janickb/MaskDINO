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
"""
import argparse
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Boxes, Instances

from maskdino import add_maskdino_config
from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--predictions", required=True, help="inference/instances_predictions.pth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default=None, help="defaults to cfg.DATASETS.TEST[0]")
    ap.add_argument("--score-thresh", type=float, default=None)
    ap.add_argument("--iou-thresh", type=float, default=None)
    ap.add_argument("--min-visibility", type=float, default=None)
    args = ap.parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
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
    st = float(he.SCORE_THRESH)
    by_image = defaultdict(list)
    for p in preds:
        for det in p["instances"]:
            if det["score"] >= st:
                by_image[p["image_id"]].append(det)
    for image_id in fname_by_id:
        by_image.setdefault(image_id, [])

    ev = HungarianInstanceEvaluator(
        dataset, cfg, distributed=False, output_dir=args.output_dir
    )
    ev.reset()
    for image_id, dets in by_image.items():
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
        ev.process(
            [{"image_id": image_id, "file_name": fname_by_id.get(image_id, "")}],
            [{"instances": inst}],
        )

    res = ev.evaluate()["instance_matching"]
    print(f"\n=== instance_matching  (score>={he.SCORE_THRESH}  iou>={he.IOU_THRESH}) ===")
    for k, v in res.items():
        if "/" not in k:
            print(f"  {k:28s} {v:.4f}")


if __name__ == "__main__":
    main()
