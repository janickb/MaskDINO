"""Unit tests for maskdino/data/augmentations/copy_paste.py.

The module is loaded by file path (like tests/test_class_mapping.py) so the tests
don't trigger ``import maskdino`` (which registers datasets from absolute data
paths). It has no maskdino-internal imports of its own, so this is purely to stay
consistent with the rest of the test suite.
"""
import importlib.util
import os
import random
import sys

import cv2
import h5py
import numpy as np
import pytest
from pycocotools import mask as mask_util

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "data", "augmentations", "copy_paste.py")
_spec = importlib.util.spec_from_file_location("copy_paste", _MOD_PATH)
cp_mod = importlib.util.module_from_spec(_spec)
sys.modules["copy_paste"] = cp_mod
_spec.loader.exec_module(cp_mod)

CopyPasteCompositor = cp_mod.CopyPasteCompositor

# config.py has no maskdino-internal imports of its own (only detectron2), so like
# copy_paste.py above it can be loaded by file path without triggering `import
# maskdino` - needed below to check the actual default BLEND_MODES now that
# CopyPasteCompositor's constructor no longer has its own (see config.py's
# INPUT.COPY_PASTE, the single source of truth for these defaults).
_CONFIG_PATH = os.path.join(_HERE, "..", "maskdino", "config.py")
_config_spec = importlib.util.spec_from_file_location("maskdino_config", _CONFIG_PATH)
config_mod = importlib.util.module_from_spec(_config_spec)
sys.modules["maskdino_config"] = config_mod
_config_spec.loader.exec_module(config_mod)

add_maskdino_config = config_mod.add_maskdino_config


def _encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _make_ann(mask, category_id):
    rle = _encode(mask)
    return {
        "bbox": mask_util.toBbox(rle).tolist(),
        "bbox_mode": 1,
        "category_id": category_id,
        "segmentation": rle,
        "area": float(mask_util.area(rle)),
        "iscrowd": 0,
    }


def _write_source(tmp_path, name, image, mask, category_id):
    path = str(tmp_path / f"{name}.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("colors", data=image)
    return {"file_name": path, "annotations": [_make_ann(mask, category_id)]}


def _l_shape(size=24, arm=8):
    """An asymmetric L-tromino-style corner: a vertical bar + a horizontal bar
    meeting at the bottom-left. Its mirror image ("J") is not reachable by any
    rotation, so it can detect an accidental flip."""
    mask = np.zeros((size, size), dtype=bool)
    mask[:, :arm] = True  # vertical bar (full height, left side)
    mask[-arm:, :] = True  # horizontal bar (bottom, full width)
    return mask


def _solid_image(size, value):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_rect_mask(size=40, margin=4):
    mask = np.zeros((size, size), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return mask


def _signed_orientation(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest, oriented=True)


@pytest.fixture
def compositor_factory(tmp_path):
    def make(n_sources=3, **kwargs):
        sources = [
            _write_source(tmp_path, f"src{i}", _solid_image(24, 200), _l_shape(), 5 + i)
            for i in range(n_sources)
        ]
        defaults = dict(
            min_instances=1,
            max_instances=1,
            rotation_degrees=90.0,
            mask_interp=("nearest",),
            blend_modes=("none",),
            blur_kernel_range=(3, 7),
            feather_width_range=(2, 7),
            core_margin_px=2,
            image_format="BGR",
        )
        defaults.update(kwargs)
        return CopyPasteCompositor(sources, **defaults), sources

    return make


def test_paste_appends_annotation_with_matching_category(compositor_factory):
    random.seed(0)
    compositor, _ = compositor_factory()
    dest = _solid_image(64, 0)
    new_image, new_anns = compositor(dest, [])

    assert new_image.shape == dest.shape
    assert new_image.dtype == dest.dtype
    assert len(new_anns) == 1
    assert new_anns[0]["category_id"] in {5, 6, 7}
    mask = mask_util.decode(new_anns[0]["segmentation"])
    assert mask.sum() > 0
    ys, xs = np.where(mask)
    assert ys.min() >= 0 and ys.max() < 64
    assert xs.min() >= 0 and xs.max() < 64


def test_no_paste_leaves_destination_unaffected(compositor_factory):
    random.seed(1)
    compositor, _ = compositor_factory(min_instances=0, max_instances=0)
    dest = _solid_image(64, 0)
    dest_ann = _make_ann(np.ones((64, 64), dtype=bool), category_id=1)

    new_image, new_anns = compositor(dest, [dest_ann])

    assert len(new_anns) == 1
    assert new_anns[0].get("visibility_fraction", 1.0) == 1.0
    assert np.array_equal(new_image, dest)


def test_paste_reduces_destination_visibility_on_overlap(compositor_factory):
    random.seed(2)
    # destination mask covers the whole canvas -> any placement overlaps it fully.
    compositor, _ = compositor_factory()
    dest = _solid_image(64, 0)
    dest_ann = _make_ann(np.ones((64, 64), dtype=bool), category_id=1)

    _, new_anns = compositor(dest, [dest_ann])

    original = next(a for a in new_anns if a["category_id"] == 1)
    pasted = next(a for a in new_anns if a["category_id"] != 1)
    assert original["visibility_fraction"] < 1.0
    pasted_mask = mask_util.decode(pasted["segmentation"]).astype(bool)
    assert pasted_mask.sum() > 0

    # Modal, not amodal: the original instance's stored mask/bbox/area must shrink
    # to match what's actually still visible, not stay at its pre-occlusion extent.
    original_mask = mask_util.decode(original["segmentation"]).astype(bool)
    assert original_mask.sum() < 64 * 64
    assert original["area"] == float(original_mask.sum())
    assert not np.any(original_mask & pasted_mask)  # no double-claimed pixels


def test_modal_masks_never_overlap_across_paste_chronology(compositor_factory):
    # With several pastes in one call, later pastes routinely cover part of
    # earlier ones (destination-original or already-pasted). Every returned
    # mask must reflect only what's still visible after the whole chronology -
    # modal, not amodal - so no two instances' masks may share a pixel, and
    # each annotation's "area" must match its own mask's pixel count.
    compositor, _ = compositor_factory(n_sources=5, min_instances=3, max_instances=3)
    dest = _solid_image(64, 0)

    random.seed(9)
    _, new_anns = compositor(dest, [])

    masks = [mask_util.decode(a["segmentation"]).astype(bool) for a in new_anns]
    for ann, mask in zip(new_anns, masks):
        assert ann["area"] == float(mask.sum())
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not np.any(masks[i] & masks[j])


def test_transform_never_flips(compositor_factory):
    compositor, _ = compositor_factory()
    original = _l_shape()
    original_sign = np.sign(_signed_orientation(original))

    random.seed(3)
    for _ in range(30):
        patch, patch_mask, _valid = compositor._transform_instance(
            _solid_image(original.shape[0], 200), original
        )
        assert np.sign(_signed_orientation(patch_mask)) == original_sign


def test_transform_uses_consistent_interpolation(compositor_factory):
    # Regression test: an earlier version used INTER_LINEAR for the image but
    # INTER_NEAREST for the mask, so their boundaries disagreed at non-90-degree
    # angles - visible as spurious notches biting into a pasted instrument's
    # silhouette. Both must use INTER_NEAREST, so every warped image pixel is an
    # exact copy of an original pixel (or the border fill) - never a blended
    # in-between value - and wherever the mask says "object", the image is
    # guaranteed to be real object color, never border-fill leaking in.
    compositor, _ = compositor_factory()
    original_mask = _l_shape()
    original_image = _solid_image(original_mask.shape[0], 200)
    allowed_values = set(np.unique(original_image).tolist()) | {0}  # 0 = border fill

    random.seed(6)
    for _ in range(40):
        patch, patch_mask, _valid = compositor._transform_instance(original_image, original_mask)
        assert set(np.unique(patch).tolist()) <= allowed_values
        assert set(np.unique(patch[patch_mask]).tolist()) <= {200}


@pytest.mark.parametrize("interp", ["linear", "bicubic", "lanczos"])
def test_transform_subpixel_interp_produces_smooth_edges(compositor_factory, interp):
    # Each subpixel mode warps both image and mask with the same flag (mask
    # thresholded at 0.5 afterward) - unlike "nearest" (which only ever copies
    # exact source/border values, see test above), this must produce intermediate
    # blended pixel values at the boundary, confirming the edge is actually
    # anti-aliased/smooth rather than blocky.
    compositor, _ = compositor_factory(mask_interp=(interp,))
    original_mask = _l_shape()
    original_image = _solid_image(original_mask.shape[0], 200)

    random.seed(12)
    saw_intermediate_value = False
    for _ in range(20):
        patch, patch_mask, valid_footprint = compositor._transform_instance(
            original_image, original_mask
        )
        assert patch_mask.dtype == bool
        assert patch_mask.shape == valid_footprint.shape
        if np.any((patch > 5) & (patch < 195)):
            saw_intermediate_value = True
    assert saw_intermediate_value


def test_blend_never_bleeds_border_fill(compositor_factory):
    # Regression test: the rotation canvas is larger than the rotated crop, so its
    # corners are synthetic borderValue=0 (black) padding, not real image content.
    # patch_mask already excludes that zone, but a blur alpha ramp doesn't - it
    # can pull the synthetic black into the blend near a thin/elongated object's
    # tip, where the true boundary sits close to those padded corners. Only real
    # (rotated crop) pixels may ever contribute to the blend.
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(21, 21),
        rotation_degrees=45.0,
    )
    thin_mask = np.zeros((60, 60), dtype=bool)
    for i in range(60):
        thin_mask[i, i] = True
        thin_mask[i, min(i + 1, 59)] = True  # a thin diagonal line corner-to-corner
    crop = _solid_image(60, 200)
    dest_value = 100

    random.seed(11)
    for _ in range(20):
        patch, patch_mask, valid_footprint = compositor._transform_instance(crop, thin_mask)
        ph, pw = patch_mask.shape[:2]
        composite = _solid_image(ph + pw + 100, dest_value)
        py, px = 50, 50

        compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

        region = composite[py : py + ph, px : px + pw]
        assert region.min() >= min(dest_value, 200)


def test_blend_none_is_hard_edged(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("none",))
    dest = _solid_image(64, 0)

    random.seed(4)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)

    pixel_values = set(new_image[..., 0][mask.astype(bool)].tolist())
    background_values = set(new_image[..., 0][~mask].tolist())
    assert pixel_values <= {200}
    assert background_values <= {0}


def test_blend_gaussian_softens_edge(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(9, 9),
    )
    dest = _solid_image(64, 0)

    random.seed(5)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & (
        ~mask
    ).astype(np.uint8)

    boundary_values = new_image[..., 0][boundary.astype(bool)]
    assert np.any((boundary_values > 5) & (boundary_values < 195))


def test_blend_box_blur_softens_edge(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("box_blur",),
        blur_kernel_range=(9, 9),
    )
    dest = _solid_image(64, 0)

    random.seed(7)
    new_image, new_anns = compositor(dest, [])
    mask = mask_util.decode(new_anns[0]["segmentation"]).astype(bool)
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) & (
        ~mask
    ).astype(np.uint8)

    boundary_values = new_image[..., 0][boundary.astype(bool)]
    assert np.any((boundary_values > 5) & (boundary_values < 195))


def test_core_margin_protects_object_interior(compositor_factory):
    # gaussian_blur_edge/box_blur/alpha_feather all ramp symmetrically, so a large
    # enough kernel/width would otherwise blend a chunk of the object's own
    # interior with the destination background. core_margin_px must force alpha=1
    # more than that many px inside the mask, regardless of kernel size.
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(15, 15),
        core_margin_px=2,
    )
    patch_mask = _solid_rect_mask(size=40, margin=4)  # true region is rows/cols [4,36)
    patch = _solid_image(40, 200)
    composite = _solid_image(64, 0)
    valid_footprint = np.ones_like(patch_mask, dtype=bool)
    py, px = 10, 10

    compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

    # eroded by 2px -> core is [6,34); [8,32) is safely inside it regardless of
    # the (much larger) 15px blur kernel used for the outward side.
    deep_interior = np.zeros_like(patch_mask)
    deep_interior[8:-8, 8:-8] = True
    region = composite[py : py + 40, px : px + 40]
    assert set(np.unique(region[deep_interior]).tolist()) == {200}


def test_core_margin_zero_disables_protection(compositor_factory):
    compositor, _ = compositor_factory(
        blend_modes=("gaussian_blur_edge",),
        blur_kernel_range=(31, 31),
        core_margin_px=0,
    )
    patch_mask = _solid_rect_mask(size=40, margin=4)
    patch = _solid_image(40, 200)
    composite = _solid_image(64, 0)
    valid_footprint = np.ones_like(patch_mask, dtype=bool)
    py, px = 10, 10

    compositor._blend(composite, patch, patch_mask, valid_footprint, py, px)

    # without protection, a wide enough blur reaches even fairly deep pixels
    deep_interior = np.zeros_like(patch_mask)
    deep_interior[8:-8, 8:-8] = True
    region = composite[py : py + 40, px : px + 40]
    assert not np.all(region[deep_interior] == 200)


def test_poisson_not_in_default_blend_modes():
    # "poisson" must be implemented but opt-in only (Cut, Paste and Learn's own
    # ablation found it hurts on average) - a regression guard on cfg.INPUT.COPY_PASTE
    # (CopyPasteCompositor's constructor has no default of its own - see its __init__)
    # so an experiment addition never silently becomes the shipped default.
    from detectron2.config import get_cfg

    cfg = get_cfg()
    add_maskdino_config(cfg)
    default_blend_modes = cfg.INPUT.COPY_PASTE.BLEND_MODES
    assert "poisson" not in default_blend_modes
    assert "gaussian_blur_edge" in default_blend_modes
    assert "box_blur" in default_blend_modes


def test_blend_poisson_applies_when_selected(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    composite = _solid_image(128, 50)
    patch = _solid_image(24, 200)
    patch_mask = _l_shape()
    py, px = 40, 40  # well inside the 128x128 canvas - padded patch won't spill

    applied = compositor._blend_poisson(composite, patch, patch_mask, py, px)

    assert applied
    assert not np.array_equal(composite, _solid_image(128, 50))


def test_blend_poisson_falls_back_near_edge(compositor_factory):
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    composite = _solid_image(32, 50)
    patch = _solid_image(24, 200)
    patch_mask = _l_shape()
    py, px = 0, 0  # padded patch spills past the top-left edge

    applied = compositor._blend_poisson(composite, patch, patch_mask, py, px)

    assert not applied
    assert np.array_equal(composite, _solid_image(32, 50))  # untouched


def test_poisson_blend_mode_falls_back_gracefully_in_full_call(compositor_factory):
    # Regardless of where random placement lands (including right at an edge,
    # where _blend_poisson declines), _blend must fall back to a hard paste
    # rather than raising.
    compositor, _ = compositor_factory(blend_modes=("poisson",))
    dest = _solid_image(64, 0)

    random.seed(8)
    new_image, new_anns = compositor(dest, [])

    assert len(new_anns) == 1
    assert mask_util.decode(new_anns[0]["segmentation"]).sum() > 0
