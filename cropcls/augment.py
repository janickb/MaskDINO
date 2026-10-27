"""Crop-level augmentation.

The detector is frozen and runs on clean frames, so nothing is augmented before it
(see spec_cropcls.md section 3). Everything MaskDINO's phase-1 pipeline did to the frame
is reproduced here on the crop instead, at the same ranges as
`build_transform_gen` in `coco_instance_new_baseline_dataset_mapper.py`:

    RandomBrightness(0.7, 1.3)   RandomContrast(0.7, 1.3)
    RandomSaturation(0.8, 1.2)   RandomLighting(0.9)

Photometric ops run on the RAW crop, before blackout. Contrast in particular blends
toward the image mean, and on a ~95%-black crop that mean is near zero, so applying it
after blackout would behave nothing like applying it to a frame.

No flipping: surgical instruments have fixed handedness, so a mirrored scalpel is a
shape that does not exist (the reason RandomFlip is absent from the MaskDINO pipeline).
"""

from __future__ import annotations

import torch

__all__ = ["PhotometricJitter", "RGB_WEIGHTS"]

# detectron2's grayscale weights (RGB order, matching INPUT.FORMAT: RGB)
RGB_WEIGHTS = (0.299, 0.587, 0.114)

# detectron2 RandomLighting's ImageNet PCA constants
_EIGEN_VECS = torch.tensor(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)
_EIGEN_VALS = torch.tensor([0.2175, 0.0188, 0.0045])


def _blend(src, dst, w: float):
    """detectron2's blend_transform: src * (1 - w) + dst * w."""
    return src * (1 - w) + dst * w


class PhotometricJitter:
    """Brightness / contrast / saturation / lighting on a 0-255 RGB crop."""

    def __init__(
        self,
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        saturation=(0.8, 1.2),
        lighting_scale=0.9,
        generator: torch.Generator | None = None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.lighting_scale = lighting_scale
        self.g = generator

    def _u(self, lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=self.g))

    def __call__(self, crop: torch.Tensor) -> torch.Tensor:
        """crop: (3, H, W) float in 0-255, background NOT yet removed."""
        x = crop.float()
        # brightness: blend toward black
        x = _blend(torch.zeros_like(x), x, self._u(*self.brightness))
        # contrast: blend toward the crop's grey mean
        w = torch.tensor(RGB_WEIGHTS, device=x.device).view(3, 1, 1)
        x = _blend((x * w).sum(0, keepdim=True).mean(), x, self._u(*self.contrast))
        # saturation: blend toward greyscale
        grey = (x * w).sum(0, keepdim=True).expand_as(x)
        x = _blend(grey, x, self._u(*self.saturation))
        # lighting: AlexNet-style PCA colour shift
        if self.lighting_scale:
            weights = torch.empty(3).normal_(0.0, self.lighting_scale, generator=self.g)
            shift = (_EIGEN_VECS @ (weights * _EIGEN_VALS)).to(x.device).view(3, 1, 1)
            x = x + shift
        return x.clamp_(0, 255)

