"""Unit tests for maskdino/data/augmentations/perspective.py.

This is the pipeline's only *geometric* augmentation beyond 90-degree rotation and
resize/crop, so the properties worth pinning are the ones a geometric transform can
silently get wrong:

  - the mask must land exactly where the image content landed (the FixedSizeCrop
    seg_pad_value bug was this failure mode in a different disguise);
  - apply_coords must agree with apply_image, or boxes drift off their objects;
  - no empty/filled corners may ever appear, or the model gets a free shortcut
    feature marking "this frame was augmented";
  - off by default, and a true no-op when disabled.

Loaded by file path, like tests/test_copy_paste.py, so the suite does not trigger
`import maskdino`.
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import pytest
from fvcore.transforms.transform import NoOpTransform

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "data", "augmentations", "perspective.py")
_spec = importlib.util.spec_from_file_location("perspective", _MOD_PATH)
pe = importlib.util.module_from_spec(_spec)
sys.modules["perspective"] = pe
_spec.loader.exec_module(pe)

_CONFIG_PATH = os.path.join(_HERE, "..", "maskdino", "config.py")
_config_spec = importlib.util.spec_from_file_location("maskdino_config", _CONFIG_PATH)
config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["maskdino_config"] = config_mod
_config_spec.loader.exec_module(config_mod)

_SIZE = 128


def _diagonal_mask(size=_SIZE):
    m = np.zeros((size, size), np.uint8)
    cv2.line(m, (20, 30), (100, 90), 1, thickness=7)
    return m


def _textured_image(size=_SIZE):
    rng = np.random.default_rng(0)
    return rng.integers(1, 255, size=(size, size, 3), dtype=np.uint8)


def test_disabled_by_default_in_config():
    from detectron2.config import get_cfg

    cfg = get_cfg()
    config_mod.add_maskdino_config(cfg)
    assert cfg.INPUT.PERSPECTIVE.PROB == 0.0


def test_zero_prob_and_zero_distortion_are_no_ops():
    img = _textured_image()
    assert isinstance(
        pe.RandomPerspective(0.05, prob=0.0).get_transform(img), NoOpTransform
    )
    assert isinstance(
        pe.RandomPerspective(0.0, prob=1.0).get_transform(img), NoOpTransform
    )


def test_warp_changes_pixels_but_keeps_shape_and_dtype():
    img = _textured_image()
    np.random.seed(0)
    out = pe.RandomPerspective(0.08, prob=1.0).get_transform(img).apply_image(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_no_empty_corners_are_introduced():
    # Every output pixel must come from real source content. A constant-fill corner
    # would be a free "this frame was augmented" cue for the model to latch onto.
    img = np.full((_SIZE, _SIZE, 3), 200, np.uint8)
    for seed in range(20):
        np.random.seed(seed)
        out = pe.RandomPerspective(0.1, prob=1.0).get_transform(img).apply_image(img)
        assert out.min() == 200, "border fill leaked into the frame"


def test_mask_follows_image_content():
    # Encode the mask AS the image, warp both through the same transform, and check
    # they still describe the same region.
    mask = _diagonal_mask()
    img = np.repeat((mask * 255)[:, :, None], 3, axis=2)
    np.random.seed(2)
    tfm = pe.RandomPerspective(0.08, prob=1.0).get_transform(img)

    warped_img = tfm.apply_image(img)[:, :, 0] > 127
    warped_mask = tfm.apply_segmentation(mask).astype(bool)

    inter = np.logical_and(warped_img, warped_mask).sum()
    union = np.logical_or(warped_img, warped_mask).sum()
    assert union > 0
    assert inter / union > 0.95, "mask and image disagree after the same warp"


def test_apply_coords_agrees_with_apply_image():
    # A single bright dot: where apply_coords says it goes is where it must appear.
    img = np.zeros((_SIZE, _SIZE, 3), np.uint8)
    pt = (40, 55)  # x, y
    img[pt[1], pt[0]] = 255
    np.random.seed(5)
    tfm = pe.RandomPerspective(0.08, prob=1.0).get_transform(img)

    out = tfm.apply_image(img)[:, :, 0]
    ys, xs = np.nonzero(out)
    observed = np.array([xs.mean(), ys.mean()])
    predicted = tfm.apply_coords(np.array([[float(pt[0]), float(pt[1])]]))[0]
    assert np.linalg.norm(observed - predicted) < 1.5


def test_segmentation_preserves_dtype_and_is_binary():
    mask = _diagonal_mask()
    np.random.seed(1)
    tfm = pe.RandomPerspective(0.08, prob=1.0).get_transform(mask[:, :, None])
    out = tfm.apply_segmentation(mask)
    assert out.dtype == mask.dtype
    assert set(np.unique(out)).issubset({0, 1})


def test_transform_is_deterministic_once_constructed():
    # A TransformList is re-applied per annotation after the image; redrawing would
    # scatter instances away from their own pixels.
    img = _textured_image()
    np.random.seed(4)
    tfm = pe.RandomPerspective(0.08, prob=1.0).get_transform(img)
    assert np.array_equal(tfm.apply_image(img), tfm.apply_image(img))


def test_inverse_roundtrips_coordinates():
    img = _textured_image()
    np.random.seed(6)
    tfm = pe.RandomPerspective(0.08, prob=1.0).get_transform(img)
    pts = np.array([[10.0, 20.0], [100.0, 90.0], [64.0, 64.0]])
    back = tfm.inverse().apply_coords(tfm.apply_coords(pts))
    assert np.allclose(back, pts, atol=1e-6)


@pytest.mark.parametrize("distortion", [0.02, 0.05, 0.1])
def test_distortion_scale_bounds_the_displacement(distortion):
    # "Moderate" has to mean something: no corner may move more than
    # distortion_scale of the frame, so the knob is the actual ceiling.
    img = _textured_image()
    corners = np.array([[0.0, 0.0], [_SIZE - 1, 0.0],
                        [_SIZE - 1, _SIZE - 1], [0.0, _SIZE - 1]])
    worst = 0.0
    for seed in range(30):
        np.random.seed(seed)
        tfm = pe.RandomPerspective(distortion, prob=1.0).get_transform(img)
        moved = tfm.inverse().apply_coords(corners)
        worst = max(worst, np.abs(moved - corners).max())
    assert worst <= distortion * (_SIZE - 1) + 1e-6
