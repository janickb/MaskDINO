"""Attach a ground-truth class to each predicted (class-agnostic) mask.

Rule, per spec_cropcls.md section 3:

    best_iou >= iou_floor (0.5)  AND  runner_up_iou < runner_up_ceiling (0.2)
        -> assign that GT instance's class
    otherwise -> drop the sample

`runner_up_iou` is the second-highest *overlap* in the prediction's row - not a spatial
neighbour. It is 0 for the common case of a mask touching exactly one GT instance, and
only rises when one predicted mask physically covers pixels of two different instances
(the forcep03/forcep04 pile case, which plain IoU would wave through).

The predicted mask is upsampled to GT resolution; the GT is never downsampled. At median
mask fill 0.103 - and 0.07 for forcep03/04 - downsampling the GT would thin exactly the
filigree shapes the classifier has to tell apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from maskdino.evaluation.hungarian_instance_evaluation import mask_iou_matrix

__all__ = ["Assignment", "assign_gt_classes", "ASSIGNED", "LOW_IOU", "CONTAMINATED", "NO_GT"]

ASSIGNED = "assigned"
LOW_IOU = "dropped_low_iou"
CONTAMINATED = "dropped_contaminated"
NO_GT = "dropped_no_gt"


@dataclass
class Assignment:
    """Per-prediction assignment outcome. All tensors are length K (one row per prediction)."""

    keep: torch.Tensor          # bool
    labels: torch.Tensor        # long, -1 where not kept
    gt_index: torch.Tensor      # long, -1 where there is no best GT
    best_iou: torch.Tensor      # float
    runner_up_iou: torch.Tensor # float
    verdict: list[str]

    def __len__(self) -> int:
        return int(self.keep.numel())

    @property
    def n_kept(self) -> int:
        return int(self.keep.sum())

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.verdict:
            out[v] = out.get(v, 0) + 1
        return out


def _empty(k: int, verdict: str) -> Assignment:
    return Assignment(
        keep=torch.zeros(k, dtype=torch.bool),
        labels=torch.full((k,), -1, dtype=torch.long),
        gt_index=torch.full((k,), -1, dtype=torch.long),
        best_iou=torch.zeros(k),
        runner_up_iou=torch.zeros(k),
        verdict=[verdict] * k,
    )


def assign_gt_classes(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_floor: float = 0.5,
    runner_up_ceiling: float = 0.2,
) -> Assignment:
    """Assign a GT class to each predicted mask.

    Args:
        pred_masks: (K, h, w) bool/float. Upsampled to the GT resolution if it differs.
        gt_masks:   (M, H, W) bool/float, post-augmentation (same frame as pred_masks).
        gt_classes: (M,) long.
    """
    if pred_masks.ndim != 3 or gt_masks.ndim != 3:
        raise ValueError(
            f"expected (N, H, W) masks, got pred {tuple(pred_masks.shape)} "
            f"gt {tuple(gt_masks.shape)}"
        )
    k = int(pred_masks.shape[0])
    m = int(gt_masks.shape[0])
    if k == 0:
        return _empty(0, ASSIGNED)
    # Frames with zero GT instances exist in this data (e.g. 012676FD8_01-008.hdf5), and
    # a (K, 0) IoU matrix makes .max(1) throw - so short-circuit rather than crash.
    if m == 0:
        return _empty(k, NO_GT)

    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        pred_masks = (
            F.interpolate(
                pred_masks.unsqueeze(1).float(),
                size=tuple(gt_masks.shape[-2:]),
                mode="nearest",
            )
            .squeeze(1)
            .bool()
        )

    # Exact, box-prefiltered, returns float32 on CPU.
    iou = mask_iou_matrix(pred_masks, gt_masks)

    n_take = min(2, m)
    top = iou.topk(n_take, dim=1)
    best = top.values[:, 0]
    gt_index = top.indices[:, 0]
    runner_up = top.values[:, 1] if n_take == 2 else torch.zeros_like(best)

    high_enough = best >= iou_floor
    distinct = runner_up < runner_up_ceiling
    keep = high_enough & distinct

    labels = torch.full((k,), -1, dtype=torch.long)
    labels[keep] = gt_classes.cpu().long()[gt_index[keep]]

    verdict: list[str] = []
    for i in range(k):
        if keep[i]:
            verdict.append(ASSIGNED)
        elif not high_enough[i]:
            verdict.append(LOW_IOU)
        else:
            verdict.append(CONTAMINATED)

    gt_index = gt_index.clone()
    gt_index[best <= 0] = -1

    return Assignment(
        keep=keep,
        labels=labels,
        gt_index=gt_index,
        best_iou=best,
        runner_up_iou=runner_up,
        verdict=verdict,
    )
