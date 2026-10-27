"""Frozen single-class MaskDINO: the class-agnostic mask source.

Bootstrapped exactly the way `tools/benchmark_inference.py` does it - `get_cfg()` +
`add_deeplab_config` + `add_maskdino_config` + merge the run's own `config.yaml`, then
`build_model` -> `.eval()` -> `DetectionCheckpointer(...).load(...)`. No Trainer, no
`launch()`, and no detectron2 META_ARCHITECTURE registration anywhere.

The detector is a plain object rather than a child module, so it never lands in the
classifier's `state_dict`, never needs a `create_ddp_model` broadcast dodge, and cannot be
switched into train mode by a stray `model.train()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config

from .config import CropClsConfig

__all__ = ["FrozenSingleClassDetector", "Detections"]


@dataclass
class Detections:
    """One frame's class-agnostic instance predictions, in the augmented image frame.

    No boxes: every crop is taken from its mask's tight bbox (see `crops.py`), so the
    detector's learned `pred_boxes` are never read. They would also not be in this frame -
    with no "height"/"width" in the batched dict, MaskDINO denormalizes boxes against the
    *padded* size (`maskdino/maskdino.py:411-413`) while postprocessing masks to the
    un-padded one, so re-adding them means re-framing them and extending the guard below.
    """

    masks: torch.Tensor   # (K, H, W) bool
    scores: torch.Tensor  # (K,) float

    def __len__(self) -> int:
        return int(self.masks.shape[0])


class FrozenSingleClassDetector:
    def __init__(self, cfg: CropClsConfig, device: str = "cuda"):
        d2 = get_cfg()
        add_deeplab_config(d2)
        add_maskdino_config(d2)
        # The run's dumped config.yaml is the fully-resolved state of whatever program
        # wrote it, so it can carry keys this branch does not register - the single-class
        # checkpoint was produced by code that also had Mask2Former's MODEL.MASK_FORMER
        # block (branch 41's mask2former/config.py). Those keys are inert for building a
        # MaskDINO, so accept them as data rather than failing the merge.
        d2.set_new_allowed(True)
        d2.merge_from_file(cfg.detector_config)
        d2.set_new_allowed(False)
        d2.MODEL.WEIGHTS = cfg.detector_checkpoint
        d2.MODEL.DEVICE = device
        d2.freeze()
        self.d2_cfg = d2
        self.device = torch.device(device)
        self.score_thresh = float(cfg.detector_score_thresh)

        self.model = build_model(d2)
        self.model.eval()
        DetectionCheckpointer(self.model).load(d2.MODEL.WEIGHTS)

        self._assert_single_class()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _assert_single_class(self) -> None:
        """Guard against pointing the config at a multiclass checkpoint.
        """
        w = self.model.sem_seg_head.predictor.class_embed.weight
        n_classes, hidden = tuple(w.shape)
        if n_classes != 1:
            raise ValueError(
                f"{self.d2_cfg.MODEL.WEIGHTS} has class_embed {(n_classes, hidden)}: that is a "
                f"{n_classes}-class model, not the class-agnostic frontend this expects. "
                "A single-class checkpoint has class_embed (1, HIDDEN_DIM)."
            )

    @property
    def num_classes(self) -> int:
        return int(self.model.sem_seg_head.predictor.class_embed.weight.shape[0])

    @torch.no_grad()
    def detect(self, images: list[torch.Tensor], amp: bool = True) -> list[Detections]:
        """Run the frozen detector on a batch of frames.

        Args:
            images: list of (3, H, W) uint8/float tensors in `INPUT.FORMAT` order (RGB),
                **unnormalized** - the detector applies its own pixel_mean/std.

        Returns one `Detections` per input, with masks at the *input* resolution so they
        share a coordinate frame with the augmented `instances.gt_masks`. "height"/"width"
        are deliberately not passed, which makes MaskDINO postprocess to the un-padded
        augmented size rather than to some original on-disk size.
        """
        self.model.eval()  # cheap insurance against a stray train() upstream
        batched = [{"image": img.to(self.device)} for img in images]
        use_amp = amp and self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = self.model(batched)

        results: list[Detections] = []
        for img, out in zip(images, outputs):
            inst = out["instances"]
            keep = inst.scores >= self.score_thresh
            masks = inst.pred_masks[keep].bool()
            if masks.shape[-2:] != img.shape[-2:]:
                raise RuntimeError(
                    f"detector returned masks at {tuple(masks.shape[-2:])} but the input frame "
                    f"is {tuple(img.shape[-2:])}; they must share a frame with the GT masks"
                )
            results.append(
                Detections(
                    masks=masks,
                    scores=inst.scores[keep].float(),
                )
            )
        return results
