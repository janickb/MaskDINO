# Copyright (c) Facebook, Inc. and its affiliates.
"""Inference-time postprocessing shared by both meta-architectures.

Lives here rather than in maskdino/maskdino.py so mask2former/maskformer_model.py can
import it without depending on the MaskDINO meta-architecture module. One definition,
both arms - which is what makes the two architectures' postprocess comparable rather
than merely similar.
"""
import torch

__all__ = ["class_aware_mask_nms"]


def class_aware_mask_nms(masks, scores, labels, iou_thresh):
    """Greedy NMS restricted to same-label pairs: among predictions sharing a label,
    suppress the lower-scoring one whenever mask IoU exceeds iou_thresh. Predictions
    with different labels are never compared against each other, so two genuinely
    distinct overlapping instruments (e.g. two different tools in a pile) can't
    suppress one another - only near-duplicate detections of the SAME object under
    the SAME label are removed. Gated by MODEL.<ARCH>.TEST.NMS_IOU (see maskdino.config.add_surgical_arch_config);
    0 disables it, which is upstream Mask2Former's behaviour - it ships no dedup step
    of its own. Both architectures must run the SAME value or their precision is not
    comparable: duplicates count as false positives in the Hungarian evaluator.

    Args:
        masks: (K, H, W) bool.
        scores: (K,) float.
        labels: (K,) int.
    Returns:
        (K,) bool keep mask.
    """
    k = masks.shape[0]
    keep = torch.ones(k, dtype=torch.bool, device=masks.device)
    if k <= 1:
        return keep

    flat = masks.reshape(k, -1).float()
    inter = flat @ flat.t()
    area = flat.sum(dim=1)
    union = area[:, None] + area[None, :] - inter
    iou = inter / union.clamp_min(1e-6)

    order = torch.argsort(scores, descending=True).tolist()
    for a in range(k):
        i = order[a]
        if not keep[i]:
            continue
        for b in range(a + 1, k):
            j = order[b]
            if keep[j] and labels[i] == labels[j] and iou[i, j] > iou_thresh:
                keep[j] = False
    return keep
