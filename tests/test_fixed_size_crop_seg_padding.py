"""Regression test for the FixedSizeCrop segmentation-padding bug.

detectron2's T.FixedSizeCrop defaults to seg_pad_value=255 - the semantic-segmentation
"ignore" label. That is harmless on the polygon path (upstream Mask2Former's COCO
mapper), but this fork feeds RLE masks through mask_format="bitmask", and
BitMasks.__init__ casts every non-zero value to True. So whenever ResizeScale picked a
scale < 1.0, the image got padded up to IMAGE_SIZE and the padding region silently
became part of EVERY instance's ground-truth mask.

Measured impact before the fix, on train_seta_mm at MIN/MAX_SCALE 0.8/1.2: 44% of
frames affected, each instance's mask covering ~23% of the image instead of the
instrument (true areas were 500-6000 px; corrupted areas ~250,000 px).
"""
import numpy as np
import pytest
import torch
from detectron2.data import transforms as T
from detectron2.structures import BitMasks

from maskdino.config import build_base_cfg
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    build_transform_gen,
)


def test_bitmasks_casts_255_to_true():
    """The mechanism that made a 255 pad value destructive rather than ignored."""
    m = np.zeros((4, 4), dtype=np.uint8)
    m[0, 0] = 1
    m[3, 3] = 255
    assert int(BitMasks(torch.from_numpy(m)[None]).tensor.sum()) == 2


@pytest.mark.parametrize(
    "config_name",
    [
        "maskdino_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
        "maskformer2_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
        "maskdino_R50_surgical_tools_reclassify",
        "maskformer2_R50_surgical_tools_reclassify",
    ],
)
def test_fixed_size_crop_pads_segmentation_with_zero(config_name):
    cfg = build_base_cfg()
    cfg.merge_from_file(f"configs/coco/instance-segmentation/{config_name}.yaml")
    crops = [
        t for t in build_transform_gen(cfg, True) if isinstance(t, T.FixedSizeCrop)
    ]
    assert crops, "expected a FixedSizeCrop in the augmentation list"
    for crop in crops:
        assert crop.seg_pad_value == 0, (
            "FixedSizeCrop must pad segmentation with 0, not detectron2's default 255: "
            "BitMasks casts non-zero to True, so 255 padding becomes part of every "
            "instance's GT mask"
        )


def test_padding_does_not_inflate_instance_masks():
    """End-to-end on a synthetic undersized image: the padded mask must keep exactly
    the instrument's own area."""
    image = np.zeros((895, 895, 3), dtype=np.uint8)
    seg = np.zeros((895, 895), dtype=np.uint8)
    seg[10:20, 10:20] = 1  # 100 px "instrument"

    crop = T.FixedSizeCrop(crop_size=(1024, 1024), seg_pad_value=0)
    out = crop(T.AugInput(image.copy(), sem_seg=seg.copy()))
    padded = out.apply_segmentation(seg.copy())

    assert padded.shape == (1024, 1024)
    assert set(np.unique(padded).tolist()) <= {0, 1}
    assert int((padded != 0).sum()) == 100
    # And through BitMasks, which is what the mapper actually builds.
    assert int(BitMasks(torch.from_numpy(padded)[None]).tensor.sum()) == 100
