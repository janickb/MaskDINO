#!/usr/bin/env python
"""Render the crops the classifier actually sees, grouped by assigned class.

    python tools/viz_class_crops.py --classes forcep03_mm forcep04_mm --n 12

Built to answer one question: are two classes distinguishable in the crop representation
at all? If a person cannot tell them apart here, no amount of model work will.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import maskdino  # noqa: F401
from cropcls.assign import assign_gt_classes
from cropcls.config import load_config
from cropcls.crops import letterbox_crop
from cropcls.data import CropPipeline
from cropcls.detector import FrozenSingleClassDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cropcls/phase1_seta.yaml")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12, help="crops per class")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="runs/class_crops.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    det = FrozenSingleClassDetector(cfg, device=args.device)
    ds = args.dataset or cfg.val_dataset
    pipe = CropPipeline(cfg, det, ds, is_train=False)
    want = {pipe.class_names.index(c): c for c in args.classes}

    found: dict[int, list] = {k: [] for k in want}
    frame = 0
    while any(len(v) < args.n for v in found.values()) and frame < len(pipe):
        mapped = pipe.mapper(copy.deepcopy(pipe.dicts[frame])); frame += 1
        inst = mapped["instances"]
        gt = inst.gt_masks if torch.is_tensor(inst.gt_masks) else inst.gt_masks.tensor
        if gt.shape[0] == 0:
            continue
        d = det.detect([mapped["image"]])[0]
        if len(d) == 0:
            continue
        pm = d.masks.cpu()
        a = assign_gt_classes(pm, gt, inst.gt_classes,
                              iou_floor=cfg.iou_floor, runner_up_ceiling=cfg.runner_up_ceiling)
        img = mapped["image"].float()
        for k in torch.where(a.keep)[0].tolist():
            lab = int(a.labels[k])
            if lab in found and len(found[lab]) < args.n:
                out = letterbox_crop(img, pm[k], cfg.crop_size, pad_frac=cfg.pad_frac)
                if out is not None:
                    found[lab].append((out[0], float(a.best_iou[k])))

    rows = len(want) * ((args.n + 5) // 6)
    fig, axes = plt.subplots(rows, 6, figsize=(13, 2.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    r = 0
    for lab, name in want.items():
        for i, (crop, iou) in enumerate(found[lab]):
            ax = axes[r + i // 6][i % 6]
            ax.imshow(crop.clamp(0, 255).byte().permute(1, 2, 0).numpy())
            ax.set_title(f"{name}  IoU {iou:.2f}", fontsize=7)
        r += (args.n + 5) // 6
    fig.suptitle("crops as the classifier sees them (bbox-cropped, blacked, letterboxed 224)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=105)
    print("wrote", args.out, {want[k]: len(v) for k, v in found.items()}, f"from {frame} frames")


if __name__ == "__main__":
    main()
