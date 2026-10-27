"""Plain-dataclass config for the crop classifier.

No yacs, no detectron2 CfgNode, no registry: the detector side is driven by its own
run `config.yaml` (merged in `detector.py`), and everything here is classifier-only.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# MaskDINO's own pixel statistics, which are ImageNet's scaled to 0-255 (RGB).
# Reused verbatim so the classifier's preprocessing cannot drift from the frontend's.
PIXEL_MEAN = [123.675, 116.28, 103.53]
PIXEL_STD = [58.395, 57.12, 57.375]


@dataclass
class CropClsConfig:
    # --- frozen detector -----------------------------------------------------
    detector_checkpoint: str
    # Defaults to <checkpoint dir>/config.yaml - the run's own resolved config.
    detector_config: str = ""
    # Predictions below this score are not turned into crops. 0.5 matches
    # MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.SCORE_THRESH, so the crop population
    # matches the population the baseline f1 was measured on.
    detector_score_thresh: float = 0.5

    # --- data ----------------------------------------------------------------
    # No num_classes: it is len(thing_classes) of train_dataset, read by
    # cropcls.data.dataset_class_names.
    train_dataset: str = "train_seta_mm"
    val_dataset: str = "val_seta_mm"

    # --- augmentation --------------------------------------------------------
    # The detector is FROZEN, so augmenting its input regularizes nothing - it only
    # perturbs the masks, and the letterbox discards the scale/position variation
    # anyway. Keeping it off makes the detector's output deterministic per frame
    # (hence cacheable) and matches its inference condition. Augmentation happens at
    # crop level instead. Set True only to A/B against the original recipe.
    augment_frames: bool = False

    # Crop-level augmentation, standing in for what the frame-level pipeline did.
    photometric: bool = True     # brightness/contrast/saturation/lighting, pre-blackout
    mask_only: bool = False      # shape-only ablation: silhouette instead of RGB
    # Grow the mask by N pixels before cropping and blacking out. 0 = tight silhouette.
    # Sits between blackout (no context at all) and blackout=False (all of it, which acts
    # as a distractor): it recovers the boundary transition and any real instrument pixels
    # the detector's mask clipped, without admitting the surroundings.
    mask_dilate: int = 0
    # Use closing (dilate+erode) instead of plain dilation at that radius: repairs
    # fragmented masks without admitting any surrounding pixels.
    mask_close: bool = False

    blackout: bool = True        # zero everything outside the mask. False keeps the real
                                 # background - tests whether surrounding context helps,
                                 # which the spec's blackout throws away by design.

    # --- crops ---------------------------------------------------------------
    crop_size: int = 224
    # Lossless pixel permutations on a square canvas; applied after cropping so one
    # detector forward yields 4x the classifier samples (spec section 6, step 5).
    rotations: list[int] = field(default_factory=lambda: [0, 90, 180, 270])
    pixel_mean: list[float] = field(default_factory=lambda: list(PIXEL_MEAN))
    pixel_std: list[float] = field(default_factory=lambda: list(PIXEL_STD))

    # How the score fed to the evaluator is formed.
    #   "detector_x_class" - objectness * classifier softmax max. Exact analogue of
    #       MaskDINO's `sigmoid(class_logit) * mask_score`, so it is the comparable
    #       setting - but it couples localization to classifier confidence: a weak
    #       classifier pushes good masks below SCORE_THRESH and depresses even the
    #       class-AGNOSTIC metric.
    #   "detector" - objectness only. Diagnostic: classagnostic_f1 then reads out pure
    #       localization (should match the frontend's own 0.9763).
    score_mode: str = "detector_x_class"

    # --- GT class assignment -------------------------------------------------
    iou_floor: float = 0.5
    runner_up_ceiling: float = 0.2

    # --- model ---------------------------------------------------------------
    # The classifier's own trunk: any timm image-classification backbone, pluggable -
    # the segmentation map comes from MaskDINO, so there is no decoder of any kind here.
    # Named "trunk", not "encoder", because in this repo "encoder"/"decoder" already mean
    # MaskDINO's pixel decoder and transformer decoder (cf.
    # RECLASSIFY_FINETUNE.UNFREEZE_ENCODER) - and *those* are frozen forever.
    trunk: str = "resnet50"
    trunk_pretrained: bool = True
    freeze_trunk: bool = False
    # Classifier checkpoint to start from: an init/resume for training, the thing being
    # scored under --eval-only. Lives here and not on the command line for the same
    # reason detector_checkpoint does - a run's config.json is its own record, so the
    # checkpoint it was built on is reproducible from the file rather than from shell
    # history. Empty = train from the trunk's pretrained weights.
    weights: str = ""

    # --- solver --------------------------------------------------------------
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    max_iter: int = 20000
    ims_per_batch: int = 4          # FRAMES per detector forward
    warmup_iters: int = 250
    checkpoint_period: int = 2000
    eval_period: int = 1000
    amp: bool = True
    seed: int = 0

    # --- io ------------------------------------------------------------------
    output_dir: str = "runs/cropcls_phase1_seta"

    def __post_init__(self) -> None:
        if not self.detector_config:
            self.detector_config = str(
                Path(self.detector_checkpoint).parent / "config.yaml"
            )

    def validate(self) -> None:
        """Fail at startup rather than mid-training on a mistyped path."""
        paths = [
            ("detector_checkpoint", self.detector_checkpoint),
            ("detector_config", self.detector_config),
        ]
        if self.weights:  # empty is legal: training from the trunk's pretrained weights
            paths.append(("weights", self.weights))
        for label, p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"{label} does not exist: {p}")
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}")
        if not 0.0 <= self.runner_up_ceiling <= self.iou_floor <= 1.0:
            raise ValueError(
                "need 0 <= runner_up_ceiling <= iou_floor <= 1, got "
                f"{self.runner_up_ceiling} / {self.iou_floor}"
            )
        bad = [r for r in self.rotations if r % 90 != 0]
        if bad:
            raise ValueError(
                f"rotations must be multiples of 90 (lossless on a square crop), got {bad}"
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> CropClsConfig:
    """Read a YAML file into CropClsConfig. Unknown keys are an error, not ignored."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw.update(overrides)
    known = {f.name for f in dataclasses.fields(CropClsConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    cfg = CropClsConfig(**raw)
    cfg.validate()
    return cfg
