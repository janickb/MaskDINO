#!/usr/bin/env python
"""
Plot per-class precision-recall curves from a COCOEvaluator run.

COCOEvaluator (see train_net.py Trainer.build_evaluator) already writes the
two files this script needs into its output_dir/inference folder:

  - coco_instances_results.json          predictions, COCO result format
  - {dataset_name}_coco_format.json      ground truth, auto-cached because
                                          our HDF5 dataset has no native
                                          COCO json (register_hdf5_instance.py)

pycocotools' COCOeval.accumulate() computes a full precision array (not just
the single mAP number _derive_coco_results reports), with shape:

    eval["precision"]: [T, R, K, A, M]
      T = 10 IoU thresholds       0.50 : 0.05 : 0.95
      R = 101 recall thresholds   0.00 : 0.01 : 1.00
      K = one entry per category
      A = 4 area ranges           all, small, medium, large
      M = 3 maxDets settings      1, 10, 100

A per-class PR curve is just that array sliced at fixed T/A/M, plotted against
the R recall grid.

Usage:
    ./.venv/bin/python tools/plot_pr_curves.py \
        --gt-json-file output_v2/inference/val_coco_format.json \
        --dt-json-file output_v2/inference/coco_instances_results.json \
        --iou-type segm \
        --output output_v2/inference/pr_curves.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json-file", required=True)
    parser.add_argument("--dt-json-file", required=True)
    parser.add_argument("--iou-type", default="segm", choices=["bbox", "segm"])
    parser.add_argument("--iou-thr", type=float, default=0.5, help="which of the 10 IoU thresholds to plot")
    parser.add_argument("--area", default="all", choices=["all", "small", "medium", "large"])
    parser.add_argument("--max-dets", type=int, default=100, choices=[1, 10, 100])
    parser.add_argument("--output", default="pr_curves.png")
    args = parser.parse_args()

    coco_gt = COCO(args.gt_json_file)
    coco_dt = coco_gt.loadRes(args.dt_json_file)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=args.iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    params = coco_eval.params
    t_idx = int(np.argmin(np.abs(np.array(params.iouThrs) - args.iou_thr)))
    a_idx = params.areaRngLbl.index(args.area)
    m_idx = params.maxDets.index(args.max_dets)
    recall_grid = params.recThrs  # x-axis, shape [R]

    # precision: [T, R, K, A, M] -> fix T, A, M, keep the [R, K] slice
    precision = coco_eval.eval["precision"][t_idx, :, :, a_idx, m_idx]

    cat_ids = params.catIds
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    n = len(cat_ids)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for k, cat_id in enumerate(cat_ids):
        ax = axes[k // ncols][k % ncols]
        pr = precision[:, k]
        valid = pr > -1  # -1 marks recall levels with no GT/predictions for this class
        ap = pr[valid].mean() if valid.any() else float("nan")

        ax.plot(recall_grid, np.where(valid, pr, 0.0))
        ax.set_title(f"{cat_names[cat_id]}\nAP@{args.iou_thr:.2f} = {ap:.3f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle(f"Per-class PR curves ({args.iou_type}, IoU={args.iou_thr:.2f}, area={args.area}, maxDets={args.max_dets})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
