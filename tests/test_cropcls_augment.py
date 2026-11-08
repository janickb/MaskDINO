"""Crop-level augmentation: it replaces the frame-level pipeline, so it must match it."""

import pytest
import torch

from cropcls.augment import PhotometricJitter
from cropcls.crops import letterbox_crop, mask_tight_bbox

SIZE = 224


def _frame(h=256, w=256, v=120.0):
    return torch.full((3, h, w), v)


def _mask(h, w, y0, y1, x0, x1):
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_photometric_stays_in_range_and_changes_the_image():
    g = torch.Generator().manual_seed(0)
    jit = PhotometricJitter(generator=g)
    x = torch.rand(3, 32, 32) * 255
    y = jit(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 255.0
    assert not torch.equal(x, y)


def test_photometric_is_stochastic():
    g = torch.Generator().manual_seed(0)
    jit = PhotometricJitter(generator=g)
    x = torch.rand(3, 16, 16) * 255
    assert not torch.equal(jit(x), jit(x))


def test_photometric_runs_before_blackout_so_background_is_still_zero():
    """The op must not leave non-zero pixels outside the mask.

    Lighting adds a constant colour shift, so applying it *after* blackout would lift
    the background off zero. Running it pre-blackout, then masking, cannot.
    """
    img, m = _frame(), _mask(256, 256, 50, 150, 60, 160)
    crop, crop_mask, _ = letterbox_crop(
        img, m, SIZE, photometric=PhotometricJitter(generator=torch.Generator().manual_seed(1))
    )
    assert torch.all(crop[:, ~crop_mask] == 0.0)
    assert float(crop[:, crop_mask].abs().mean()) > 0


def test_mask_only_is_a_silhouette_and_carries_no_rgb():
    img, m = _frame(v=200.0), _mask(256, 256, 50, 150, 60, 160)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE, mask_only=True)
    vals = crop[:, crop_mask].unique()
    assert vals.numel() == 1 and float(vals[0]) == 255.0, "inside the mask must be constant"
    assert torch.all(crop[:, ~crop_mask] == 0.0)
    # all three channels identical -> no colour information survives
    assert torch.equal(crop[0], crop[1]) and torch.equal(crop[1], crop[2])


def test_tight_box_is_used_without_jitter():
    img, m = _frame(), _mask(256, 256, 50, 150, 60, 160)
    box = mask_tight_bbox(m)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE, box=box)
    assert int(crop_mask.sum()) > 0
    assert torch.all(crop[:, ~crop_mask] == 0.0)
