"""Step-4 gate: render the crops the classifier will actually be trained on.

Writes one contact sheet per frame batch with three views per instance - the raw crop,
the blacked-out crop, and the predicted mask overlaid on the original frame - plus the
assignment verdict, so the IoU floor and runner-up ceiling can be calibrated by eye
before any GPU time is spent on training. The *dropped* instances are rendered too:
if that pile does not look genuinely ambiguous, the thresholds are wrong.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .assign import ASSIGNED, assign_gt_classes
from .crops import letterbox_crop
from .data import CropPipeline

__all__ = ["write_contact_sheets"]

_VERDICT_COLOR = {
    "assigned": "#2e7d32",
    "dropped_low_iou": "#e65100",
    "dropped_contaminated": "#c62828",
    "dropped_no_gt": "#6a1b9a",
}


def _to_uint8(chw: torch.Tensor) -> np.ndarray:
    return chw.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def _overlay(frame: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Predicted mask tinted onto the full original frame, so mask errors are visible."""
    img = frame.clamp(0, 255).float().clone()
    tint = torch.tensor([0.0, 255.0, 255.0], device=img.device).view(3, 1, 1)
    m = mask.unsqueeze(0).float()
    img = img * (1 - alpha * m) + tint * (alpha * m)
    return _to_uint8(img)


def write_contact_sheets(
    pipeline: CropPipeline,
    out_dir: str | Path,
    frame_indices: list[int],
    max_instances_per_sheet: int = 8,
) -> list[Path]:
    """One PNG per frame. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cfg = pipeline.cfg

    for fi in frame_indices:
        mapped = pipeline.mapper(copy.deepcopy(pipeline.dicts[fi]))
        image = mapped["image"].float()
        inst = mapped["instances"]
        gt_masks = inst.gt_masks
        if not torch.is_tensor(gt_masks):
            gt_masks = gt_masks.tensor

        det = pipeline.detector.detect([mapped["image"]])[0]
        pred = det.masks.cpu()
        a = assign_gt_classes(
            pred,
            gt_masks,
            inst.gt_classes,
            iou_floor=cfg.iou_floor,
            runner_up_ceiling=cfg.runner_up_ceiling,
        )

        n = min(len(a), max_instances_per_sheet)
        if n == 0:
            continue
        fig, axes = plt.subplots(n, 3, figsize=(9, 3.1 * n), squeeze=False)
        for row in range(n):
            raw = letterbox_crop(image, pred[row], cfg.crop_size, blackout=False)
            black = letterbox_crop(image, pred[row], cfg.crop_size, blackout=True)
            verdict = a.verdict[row]
            colour = _VERDICT_COLOR.get(verdict, "#424242")
            label = (
                pipeline.class_names[int(a.labels[row])]
                if a.keep[row]
                else "-"
            )

            for col, (title, arr) in enumerate(
                (
                    ("raw crop", _to_uint8(raw[0]) if raw else None),
                    ("blacked (model input)", _to_uint8(black[0]) if black else None),
                    ("mask on frame", _overlay(image, pred[row])),
                )
            ):
                ax = axes[row][col]
                if arr is None:
                    ax.text(0.5, 0.5, "empty mask\n(dropped)", ha="center", va="center")
                else:
                    ax.imshow(arr)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(title, fontsize=9)
            axes[row][0].set_ylabel(
                f"{label}\nIoU {a.best_iou[row]:.2f}\n2nd {a.runner_up_iou[row]:.2f}\n"
                f"score {det.scores[row]:.2f}",
                fontsize=8,
                color=colour,
            )
            for s in axes[row][0].spines.values():
                s.set_color(colour)
                s.set_linewidth(3)

        counts = a.counts()
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        handles = [
            mpatches.Patch(color=c, label=k)
            for k, c in _VERDICT_COLOR.items()
            if k in counts
        ]
        fig.suptitle(
            f"frame {fi}  {Path(str(mapped.get('file_name',''))).name}\n"
            f"gt={len(inst)}  pred={len(a)}  {summary}   "
            f"(floor {cfg.iou_floor}, runner-up ceiling {cfg.runner_up_ceiling})",
            fontsize=10,
        )
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
        fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
        path = out / f"frame_{fi:05d}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)
    return written
