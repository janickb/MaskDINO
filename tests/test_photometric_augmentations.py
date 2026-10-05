"""Unit tests for maskdino/data/augmentations/photometric.py.

Two properties are worth pinning hard:

1. **Default-equivalence.** INPUT.PHOTOMETRIC's defaults must reproduce the four
   augs that used to be hardcoded in build_transform_gen, or every pre-existing
   config silently changes meaning the moment this module lands.
2. **Mask invariance.** These are photometric augs; if any of them ever touched
   coordinates or segmentation, the image would desync from its masks - the same
   class of bug as the FixedSizeCrop seg_pad_value one (see
   tests/test_transform_gen_padding.py).

Loaded by file path, like tests/test_copy_paste.py, so the suite does not trigger
`import maskdino` (which registers datasets from absolute data paths). The module
has no maskdino-internal imports of its own.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
from detectron2.config import get_cfg
from fvcore.transforms.transform import NoOpTransform

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "data", "augmentations", "photometric.py")
_spec = importlib.util.spec_from_file_location("photometric", _MOD_PATH)
ph = importlib.util.module_from_spec(_spec)
sys.modules["photometric"] = ph
_spec.loader.exec_module(ph)

_CONFIG_PATH = os.path.join(_HERE, "..", "maskdino", "config.py")
_config_spec = importlib.util.spec_from_file_location("maskdino_config", _CONFIG_PATH)
config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["maskdino_config"] = config_mod
_config_spec.loader.exec_module(config_mod)
add_maskdino_config = config_mod.add_maskdino_config


@pytest.fixture
def cfg():
    cfg = get_cfg()
    add_maskdino_config(cfg)
    return cfg


def _image(h=16, w=20):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_defaults_reproduce_the_previously_hardcoded_augs(cfg):
    augs = ph.build_photometric_augmentations(cfg)
    assert [type(a).__name__ for a in augs] == [
        "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomLighting",
    ]
    brightness, contrast, saturation, lighting = augs
    assert (brightness.intensity_min, brightness.intensity_max) == (0.7, 1.3)
    assert (contrast.intensity_min, contrast.intensity_max) == (0.7, 1.3)
    assert (saturation.intensity_min, saturation.intensity_max) == (0.8, 1.2)
    assert lighting.scale == 0.9


def test_new_axes_are_off_by_default(cfg):
    names = [type(a).__name__ for a in ph.build_photometric_augmentations(cfg)]
    assert "RandomGaussianBlur" not in names
    assert "RandomGaussianNoise" not in names
    assert "RandomGrayscale" not in names


def test_new_axes_are_appended_when_enabled(cfg):
    cfg.INPUT.PHOTOMETRIC.BLUR_PROB = 0.3
    cfg.INPUT.PHOTOMETRIC.BLUR_SIGMA = [0.5, 1.2]
    cfg.INPUT.PHOTOMETRIC.NOISE_PROB = 0.3
    cfg.INPUT.PHOTOMETRIC.NOISE_STD = [2.0, 6.0]
    cfg.INPUT.PHOTOMETRIC.GRAYSCALE_PROB = 0.05
    names = [type(a).__name__ for a in ph.build_photometric_augmentations(cfg)]
    assert names[4:] == ["RandomGaussianBlur", "RandomGaussianNoise", "RandomGrayscale"]


@pytest.mark.parametrize(
    "aug",
    [
        ph.RandomGaussianBlur((1.0, 1.0)),
        ph.RandomGaussianNoise((5.0, 5.0)),
        ph.RandomGrayscale("RGB"),
    ],
)
def test_transforms_never_touch_geometry(aug):
    img = _image()
    tfm = aug.get_transform(img)
    seg = np.zeros(img.shape[:2], np.uint8)
    seg[2:6, 3:9] = 1
    coords = np.array([[1.0, 2.0], [7.0, 11.0]])

    assert np.array_equal(tfm.apply_segmentation(seg), seg)
    assert np.array_equal(tfm.apply_coords(coords), coords)


@pytest.mark.parametrize(
    "aug",
    [
        ph.RandomGaussianBlur((1.0, 1.0)),
        ph.RandomGaussianNoise((5.0, 5.0)),
        ph.RandomGrayscale("RGB"),
    ],
)
def test_transforms_change_pixels_and_preserve_shape_and_dtype(aug):
    img = _image()
    out = aug.get_transform(img).apply_image(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_zero_probability_is_a_no_op():
    img = _image()
    for aug in (
        ph.RandomGaussianBlur((1.0, 2.0), prob=0.0),
        ph.RandomGaussianNoise((5.0, 9.0), prob=0.0),
        ph.RandomGrayscale("RGB", prob=0.0),
    ):
        assert isinstance(aug.get_transform(img), NoOpTransform)


def test_zero_magnitude_is_a_no_op():
    img = _image()
    assert isinstance(ph.RandomGaussianBlur((0.0, 0.0)).get_transform(img), NoOpTransform)
    assert isinstance(ph.RandomGaussianNoise((0.0, 0.0)).get_transform(img), NoOpTransform)


def test_noise_field_is_fixed_once_the_transform_exists():
    # A TransformList gets re-applied (image, then each annotation), so a transform
    # that redrew its randomness per call would desync its own outputs.
    img = _image()
    tfm = ph.RandomGaussianNoise((6.0, 6.0)).get_transform(img)
    assert np.array_equal(tfm.apply_image(img), tfm.apply_image(img))


def test_grayscale_equalizes_channels():
    img = _image()
    out = ph.RandomGrayscale("RGB").get_transform(img).apply_image(img)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_grayscale_weights_follow_channel_order():
    # Pure green under Rec.601 is much brighter than pure red; reading the image as
    # BGR when it is RGB would swap those, so the weights must follow the format.
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255  # red in RGB order, blue in BGR order
    as_rgb = ph.RandomGrayscale("RGB").get_transform(red).apply_image(red)[0, 0, 0]
    as_bgr = ph.RandomGrayscale("BGR").get_transform(red).apply_image(red)[0, 0, 0]
    assert as_rgb == pytest.approx(255 * 0.299, abs=1)
    assert as_bgr == pytest.approx(255 * 0.114, abs=1)
