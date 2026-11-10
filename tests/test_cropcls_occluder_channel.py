"""Step 11: the occluder channel.

The assertions the spec's Verify clause names. The theme throughout is that every way this
feature can go wrong is SILENT - a constant-zero channel 4, a channel that drifted out of
alignment with the pixels it describes, or a crop path that disagrees with the others about
the layout. None of those raise on their own; they just report a worse number.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import timm
import torch
from PIL import Image

from cropcls.config import CropClsConfig
from cropcls.crop_folder import load_crop, occ_path, save_crop
from cropcls.crops import letterbox_crop, rotate_k90, to_model_input
from cropcls.modeling.classifier import CropClassifier, add_mask_input_channel

SIZE = 64
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


def _frame(h=96, w=96):
    g = torch.Generator().manual_seed(0)
    return torch.rand((3, h, w), generator=g) * 255.0


def _blob(h=96, w=96, y=10, x=10, dy=30, dx=40):
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y : y + dy, x : x + dx] = True
    return m


# --- the channel is real, and it is the occluder ------------------------------------------


def test_occluder_channel_is_the_letterboxed_union_not_zeros():
    # The occluder must OVERLAP the crop box to appear: the box is the instance's tight box,
    # so at pad_frac 0 nothing outside it is in view (see the pad_frac test below).
    img, mine, others = _frame(), _blob(), _blob(y=10, x=40, dy=30, dx=20)
    out = letterbox_crop(img, mine, SIZE, others=others)
    x = to_model_input(out, MEAN, STD, 0, mask_channel=True, occluder_channel=True)
    assert x.shape == (5, SIZE, SIZE)
    ch4 = x[4]
    assert set(torch.unique(ch4).tolist()) <= {0.0, 1.0}
    assert ch4.any(), "channel 4 is all zeros; the occluder never arrived"
    # and it is NOT a copy of the instance's own mask
    assert not torch.equal(ch4, x[3])


def test_occluder_channel_is_zero_for_an_instance_alone_in_its_frame():
    img, mine = _frame(), _blob()
    empty = torch.zeros_like(mine)
    out = letterbox_crop(img, mine, SIZE, others=empty)
    x = to_model_input(out, MEAN, STD, 0, mask_channel=True, occluder_channel=True)
    assert not x[4].any()


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_occluder_channel_rotates_with_the_pixels(deg):
    img, mine, others = _frame(), _blob(), _blob(y=10, x=40, dy=30, dx=20)
    out = letterbox_crop(img, mine, SIZE, others=others)
    x = to_model_input(out, MEAN, STD, deg, mask_channel=True, occluder_channel=True)
    ref = rotate_k90(out[2].unsqueeze(0).float(), deg).squeeze(0)
    assert torch.equal(x[4], ref), "channel 4 drifted out of alignment under rotation"


def test_occluder_shares_the_crop_geometry_with_the_rgb():
    """An occluder that covers the whole frame must fill exactly the content box - the same
    region the instance mask's letterbox lands in - not the padded canvas."""
    img, mine = _frame(), _blob()
    everywhere = torch.ones_like(mine)
    out = letterbox_crop(img, mine, SIZE, others=everywhere)
    crop_mask, crop_others = out[1], out[2]
    # the content box is where the resized patch landed; the mask is inside it
    rows = crop_others.any(1).nonzero().flatten()
    assert crop_others[rows.min() : rows.max() + 1].all(0).any()
    assert crop_others[crop_mask].all(), "own mask sits outside the occluder's content box"


def test_an_adjacent_occluder_is_only_visible_once_pad_frac_widens_the_view():
    """The spec pins pad_frac for this arm because the two are entangled: a neighbour that
    does not overlap the instance's tight box reaches the crop only through the padded rim."""
    img, mine = _frame(), _blob(y=10, x=10, dy=30, dx=30)      # cols 10..40
    adjacent = _blob(y=10, x=42, dy=30, dx=20)                 # starts just outside
    tight = letterbox_crop(img, mine, SIZE, others=adjacent, pad_frac=0.0)
    wide = letterbox_crop(img, mine, SIZE, others=adjacent, pad_frac=0.5)
    assert not tight[2].any(), "an adjacent occluder must not appear at pad_frac 0"
    assert wide[2].any(), "pad_frac 0.5 should bring the neighbour into view"


# --- the silent-failure guards ------------------------------------------------------------


def test_missing_occluder_raster_raises_instead_of_feeding_zeros():
    img, mine = _frame(), _blob()
    out = letterbox_crop(img, mine, SIZE)            # no others=
    assert out[2] is None
    with pytest.raises(ValueError, match="crop_others=None"):
        to_model_input(out, MEAN, STD, 0, mask_channel=True, occluder_channel=True)


def test_load_crop_raises_when_the_sidecar_is_absent(tmp_path):
    p = tmp_path / "c.png"
    save_crop(p, torch.zeros((3, 8, 8), dtype=torch.uint8), _blob(8, 8, 1, 1, 3, 3))
    assert not occ_path(p).is_file()
    with pytest.raises(FileNotFoundError, match="occluder raster"):
        load_crop(p, want_others=True)
    rgb, _, others = load_crop(p, want_others=False)
    assert others is None and rgb.shape == (3, 8, 8)


def test_sidecar_round_trips(tmp_path):
    p = tmp_path / "c.png"
    mine, oth = _blob(8, 8, 1, 1, 3, 3), _blob(8, 8, 5, 5, 2, 2)
    save_crop(p, torch.zeros((3, 8, 8), dtype=torch.uint8), mine, oth)
    _, mask, others = load_crop(p, want_others=True)
    assert torch.equal(mask, mine)
    assert torch.equal(others, oth)


def test_sidecar_shape_mismatch_raises(tmp_path):
    p = tmp_path / "c.png"
    save_crop(p, torch.zeros((3, 8, 8), dtype=torch.uint8), _blob(8, 8, 1, 1, 3, 3))
    Image.fromarray(np.zeros((4, 4), np.uint8), "L").save(occ_path(p))
    with pytest.raises(ValueError, match="patch is"):
        load_crop(p, want_others=True)


# --- the config guards --------------------------------------------------------------------


def _cfg(**kw):
    base = {"detector_checkpoint": __file__, "detector_config": __file__}
    base.update(kw)
    return CropClsConfig(**base)


def test_in_chans_counts_both_extra_channels():
    assert _cfg().in_chans == 3
    assert _cfg(mask_channel=True).in_chans == 4
    assert _cfg(mask_channel=True, occluder_channel=True).in_chans == 5


def test_occluder_without_mask_channel_is_rejected():
    with pytest.raises(ValueError, match="without mask_channel"):
        _cfg(occluder_channel=True).validate()


def test_occluder_with_freeze_trunk_is_rejected():
    with pytest.raises(ValueError, match="freeze_trunk"):
        _cfg(occluder_channel=True, mask_channel=True, freeze_trunk=True).validate()


# --- the model ----------------------------------------------------------------------------


@pytest.mark.parametrize("trunk", ["resnet50", "convnext_tiny"])
def test_the_5ch_trunk_is_the_4ch_trunk_at_init_and_then_learns(trunk):
    """Zero-init means the arm measures the channel, not a perturbed trunk.

    Deep-copies the 4-channel trunk and adds one more channel, exactly as
    test_cropcls_mask_channel.py does for 3->4: constructing two CropClassifiers instead
    would compare their HEADS, because a 5-channel conv draws more RNG than a 4-channel one
    and shifts the stream before nn.Linear is built. Exact equality, not allclose - a zero
    weight contributes an exactly-zero term.
    """
    torch.manual_seed(0)
    t4 = add_mask_input_channel(
        timm.create_model(trunk, pretrained=False, num_classes=0, global_pool="avg"), 1
    ).eval()
    t5 = add_mask_input_channel(copy.deepcopy(t4), 1).eval()

    rgbm = torch.randn(2, 4, SIZE, SIZE)
    occ = (torch.rand(2, 1, SIZE, SIZE) > 0.5).float()      # any raster: the weights are zero
    with torch.no_grad():
        assert torch.equal(t4(rgbm), t5(torch.cat([rgbm, occ], dim=1)))

    conv = t5.get_submodule(t5.pretrained_cfg["first_conv"])
    assert conv.in_channels == 5
    assert not conv.weight[:, 4].any(), "channel 4 must start at zero"

    t5.train()
    opt = torch.optim.SGD(t5.parameters(), lr=1.0)
    t5(torch.cat([rgbm, occ], dim=1)).sum().backward()
    opt.step()
    assert conv.weight[:, 4].any(), "one step left channel 4 at zero; it is not learning"


def test_one_rebuild_not_two_for_both_channels():
    """CropClassifier adds mask+occluder in a single rebuild. Two successive rebuilds would
    also work, but the conv must end up with exactly 5 inputs either way."""
    torch.manual_seed(0)
    m = CropClassifier(
        num_classes=8, trunk="resnet50", pretrained=False,
        mask_channel=True, occluder_channel=True,
    )
    conv = m.trunk.get_submodule(m.trunk.pretrained_cfg["first_conv"])
    assert conv.in_channels == 5
    assert not conv.weight[:, 3:].any(), "both new channels must start at zero"


def test_crop_folder_meta_without_occluders_is_rejected(tmp_path):
    """A folder extracted before the sidecar existed must raise, not feed a zero channel."""
    from cropcls.crop_folder import CropFolder

    (tmp_path / "forcep03_mm").mkdir()
    save_crop(
        tmp_path / "forcep03_mm" / "f_p000.png",
        torch.zeros((3, 8, 8), dtype=torch.uint8),
        _blob(8, 8, 1, 1, 4, 4),
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"thing_classes": ["forcep03_mm"], "pad_frac": 0.2})
    )
    with pytest.raises(ValueError, match="does not record occluders"):
        CropFolder(tmp_path, _cfg(mask_channel=True, occluder_channel=True, pad_frac=0.1))
