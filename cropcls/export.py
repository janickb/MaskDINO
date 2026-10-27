"""Score the cropcls pipeline with the SAME evaluator the baseline was scored with.

`train_cropcls.evaluate()` reports top-1 over crops that passed *our* assignment rule
(IoU >= 0.5 and runner-up < 0.2). That is the right primary metric - it conditions on an
instance having been localized, so the two detectors' differing recall cannot flatter
either side - but it is NOT comparable to the baseline's `classaware_f1` of 0.9256, which
comes from `HungarianInstanceEvaluator`'s one-to-one matching.

This module feeds the classifier's predictions straight into that evaluator, so every number
is produced by identical code. No JSON round-trip, so the contiguous-vs-canonical
category id trap never arises.

Two traps this handles explicitly:

* **The evaluator silently disables misclassification accounting** when
  `cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES <= 1` (`_check_classes` in
  `hungarian_instance_evaluation.py`). The detector's own cfg says 1, so it is cloned with
  NUM_CLASSES set to the classifier's count - otherwise every `misclassified_*` comes back
  0.0 and the comparison looks perfect for the wrong reason.
* **`apply_test_sample_stride` is not idempotent** - it removes the catalog entry and
  re-registers a plain lambda, so a second call strides the already-strided list (4 -> 16).
  It is called at most once per process here.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from detectron2.structures import Instances

from maskdino.data.datasets.register_hdf5_instance import apply_test_sample_stride
from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator

from .config import CropClsConfig
from .crops import letterbox_crop, normalize_crop
from .data import CropPipeline

logger = logging.getLogger("cropcls.export")

__all__ = ["evaluate_hungarian", "STRIDE_APPLIED"]

# Module-level guard for the non-idempotent stride helper.
STRIDE_APPLIED: set[str] = set()


def _eval_cfg(detector_cfg, dataset_name: str, num_classes: int, stride: int):
    d2 = detector_cfg.clone()
    d2.defrost()
    d2.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes      # or _check_classes is False
    d2.DATASETS.TEST = (dataset_name,)
    d2.DATASETS.TEST_SAMPLE_STRIDE = stride
    d2.freeze()
    return d2


@torch.no_grad()
def evaluate_hungarian(
    cfg: CropClsConfig,
    detector,
    model,
    dataset_name: str,
    device,
    stride: int = 4,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
) -> dict:
    """Run detector -> crops -> classifier over the val set, score with the Hungarian evaluator.

    Returns its metric dict: classagnostic_/classaware_ precision, recall and f1, the
    per-class recalls, and the misclassification counts - the same keys the baseline run
    logged under `instance_matching/`.
    """
    # The head's own width, not a configured number: this is what the model predicts,
    # so the evaluator's NUM_CLASSES cannot disagree with it.
    n_classes = int(model.head.out_features)
    d2 = _eval_cfg(detector.d2_cfg, dataset_name, n_classes, stride)
    if stride > 1 and dataset_name not in STRIDE_APPLIED:
        apply_test_sample_stride(d2)          # mutates DatasetCatalog in place
        STRIDE_APPLIED.add(dataset_name)

    # Built AFTER the stride, so its GT index covers exactly the evaluated frames.
    evaluator = HungarianInstanceEvaluator(
        dataset_name, d2, distributed=False,
        output_dir=str(output_dir) if output_dir else None,
    )
    evaluator.reset()

    # is_train=False -> no crop-level augmentation; the mapper still yields GT.
    pipe = CropPipeline(cfg, detector, dataset_name, is_train=False)
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    model.eval()
    logger.info(
        "hungarian eval: %s stride=%d -> %d frames, NUM_CLASSES=%d (check_classes=%s)",
        dataset_name, stride, n, n_classes, n_classes > 1,
    )

    for i in range(n):
        mapped = pipe.mapper(copy.deepcopy(pipe.dicts[i]))
        image = mapped["image"]
        det = detector.detect([image], amp=cfg.amp)[0]
        h, w = image.shape[-2:]

        if len(det) == 0:
            inst = Instances((h, w))
            inst.pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
            inst.pred_classes = torch.zeros((0,), dtype=torch.long)
            inst.scores = torch.zeros((0,))
            evaluator.process([{"image_id": mapped["image_id"]}], [{"instances": inst}])
            continue

        masks = det.masks.cpu()
        img_f = image.float()
        crops, keep_idx = [], []
        for k in range(len(det)):
            out = letterbox_crop(
                img_f, masks[k], cfg.crop_size,
                mask_only=cfg.mask_only, blackout=cfg.blackout,
                dilate=cfg.mask_dilate, close=cfg.mask_close,
            )
            if out is None:
                continue
            crops.append(normalize_crop(out[0], cfg.pixel_mean, cfg.pixel_std))
            keep_idx.append(k)

        if crops:
            x = torch.stack(crops).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            prob = F.softmax(logits.float(), dim=1)
            cls_prob, pred_classes = prob.max(1)
            obj = det.scores.cpu()[keep_idx]
            if cfg.score_mode == "detector":
                scores = obj
            else:
                # MaskDINO parity: score = sigmoid(class_logit) * mask_score.
                scores = obj * cls_prob.cpu()
            inst = Instances((h, w))
            inst.pred_masks = masks[keep_idx]
            inst.pred_classes = pred_classes.cpu().long()
            inst.scores = scores
        else:
            inst = Instances((h, w))
            inst.pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
            inst.pred_classes = torch.zeros((0,), dtype=torch.long)
            inst.scores = torch.zeros((0,))

        evaluator.process([{"image_id": mapped["image_id"]}], [{"instances": inst}])

    results = evaluator.evaluate() or {}
    return dict(results.get("instance_matching", results))
