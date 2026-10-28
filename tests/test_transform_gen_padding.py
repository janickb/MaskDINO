"""Regression test for the FixedSizeCrop seg_pad_value bug.

detectron2's FixedSizeCrop defaults to seg_pad_value=255 - an *ignore* label, which
is correct for semantic segmentation and wrong here. This mapper carries each
instance as a binary bitmask (mask_format="bitmask", so
transform_instance_annotations routes it through apply_segmentation) and BitMasks
casts its input to bool, so a 255 pad turned every padded pixel into foreground for
EVERY instance in the frame at once: at MIN_SCALE 0.8 each gt_mask silently grew by
a 205px L-shaped band (36% of a 1024 canvas) while its box stayed correct.

These tests pin the fix at the level that actually matters - the masks the model is
trained on - rather than just asserting a constructor argument.
"""
import numpy as np
import pytest
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from pycocotools import mask as mask_util

from maskdino.config import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    build_transform_gen,
)

_SIZE = 128
_SCALE = 0.5  # -> 64x64 content, 64px of padding on the right and bottom


@pytest.fixture
def cfg():
    cfg = get_cfg()
    add_maskdino_config(cfg)
    cfg.INPUT.IMAGE_SIZE = _SIZE
    cfg.INPUT.MIN_SCALE = _SCALE
    cfg.INPUT.MAX_SCALE = _SCALE
    cfg.INPUT.RANDOM_ROTATION = False  # keep the geometry deterministic
    return cfg


def _ann(mask, category_id=0):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {
        "bbox": mask_util.toBbox(rle).tolist(),
        "bbox_mode": 1,
        "category_id": category_id,
        "segmentation": rle,
        "iscrowd": 0,
    }


def _apply(cfg, masks):
    image = np.full((_SIZE, _SIZE, 3), 60, np.uint8)
    annos = [_ann(m, i) for i, m in enumerate(masks)]
    out, tfms = T.apply_transform_gens(build_transform_gen(cfg, True), image)
    transformed = [
        utils.transform_instance_annotations(a, tfms, out.shape[:2]) for a in annos
    ]
    instances = utils.annotations_to_instances(
        transformed, out.shape[:2], mask_format="bitmask"
    )
    return out, instances


def test_build_transform_gen_sets_seg_pad_value_zero(cfg):
    crops = [a for a in build_transform_gen(cfg, True) if isinstance(a, T.FixedSizeCrop)]
    assert len(crops) == 1
    assert crops[0].seg_pad_value == 0


def test_padded_region_is_not_foreground(cfg):
    mask = np.zeros((_SIZE, _SIZE), np.uint8)
    mask[10:30, 10:50] = 1  # 800 px, well inside the kept region
    _out, instances = _apply(cfg, [mask])

    gt = instances.gt_masks.tensor.numpy()[0]
    assert gt.sum() == pytest.approx(800 * _SCALE**2, rel=0.1)
    # Nothing may be marked foreground in the padded band.
    kept = int(_SIZE * _SCALE)
    assert not gt[kept:, :].any()
    assert not gt[:, kept:].any()


def test_pad_band_is_not_shared_between_instances(cfg):
    # The decisive symptom of the bug: two instances of different classes both
    # claiming the identical pad band, each under its own category_id.
    a = np.zeros((_SIZE, _SIZE), np.uint8)
    a[10:30, 10:50] = 1
    b = np.zeros((_SIZE, _SIZE), np.uint8)
    b[40:60, 10:30] = 1
    _out, instances = _apply(cfg, [a, b])

    masks = instances.gt_masks.tensor.numpy()
    assert len(masks) == 2
    assert np.logical_and(masks[0], masks[1]).sum() == 0


def test_mask_and_box_agree_after_padding(cfg):
    # Boxes come from transformed corners and were always right; the bug made the
    # mask disagree with the box by ~30x. Mask area must stay well under box area.
    mask = np.zeros((_SIZE, _SIZE), np.uint8)
    mask[10:30, 10:50] = 1
    _out, instances = _apply(cfg, [mask])

    x0, y0, x1, y1 = instances.gt_boxes.tensor.numpy()[0]
    box_area = (x1 - x0) * (y1 - y0)
    assert instances.gt_masks.tensor.numpy()[0].sum() <= box_area * 1.05
