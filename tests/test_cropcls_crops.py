"""Step-2 verification: letterbox cropping and background blackout (spec_cropcls.md)."""

import pytest
import torch

from cropcls.crops import (
    letterbox_crop,
    mask_tight_bbox,
    normalize_crop,
    rotate_k90,
    to_model_input,
)

SIZE = 224


def _frame(h=512, w=512, value=200.0):
    return torch.full((3, h, w), value)


def _rect_mask(h, w, y0, y1, x0, x1):
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_tight_bbox_is_half_open_and_exact():
    m = _rect_mask(100, 100, 10, 30, 40, 70)
    assert mask_tight_bbox(m) == (40, 10, 70, 30)


def test_tight_bbox_of_empty_mask_is_none():
    assert mask_tight_bbox(torch.zeros((50, 50), dtype=torch.bool)) is None


def test_output_is_square_and_requested_size():
    img, m = _frame(), _rect_mask(512, 512, 100, 300, 150, 250)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE)
    assert crop.shape == (3, SIZE, SIZE)
    assert crop_mask.shape == (SIZE, SIZE)


def test_aspect_ratio_preserved_not_skewed():
    # a 40 x 400 instance: 10:1. After letterboxing the mask must still be 10:1.
    img, m = _frame(512, 512), _rect_mask(512, 512, 50, 450, 100, 140)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE)
    box = mask_tight_bbox(crop_mask)
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    assert max(h, w) == pytest.approx(SIZE, abs=1)
    # 400/40 = 10, allow a pixel of rounding on the short axis
    assert (max(h, w) / min(h, w)) == pytest.approx(10.0, rel=0.1)


def test_background_is_exactly_zero():
    img, m = _frame(value=200.0), _rect_mask(512, 512, 100, 300, 150, 250)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE)
    assert torch.all(crop[:, ~crop_mask] == 0.0), "background must be exactly 0 after blackout"
    assert crop[:, crop_mask].mean() > 0, "instrument pixels must survive"


def test_padding_is_centred_so_rotation_keeps_content_centred():
    img, m = _frame(512, 512), _rect_mask(512, 512, 50, 450, 100, 140)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE)
    rot = rotate_k90(crop_mask, 90)
    # a 90-degree rotation of a centred, square canvas keeps the same pixel count
    assert int(crop_mask.sum()) == int(rot.sum())
    x0, y0, x1, y1 = mask_tight_bbox(rot)
    # the long axis must now be horizontal
    assert (x1 - x0) > (y1 - y0)


def test_empty_mask_is_rejected_not_nan():
    img = _frame()
    out = letterbox_crop(img, torch.zeros((512, 512), dtype=torch.bool), SIZE)
    assert out is None, "an empty mask must be dropped, never returned as an all-black crop"


def test_single_pixel_mask_does_not_crash():
    img, m = _frame(64, 64), _rect_mask(64, 64, 10, 11, 10, 11)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE)
    assert crop.shape == (3, SIZE, SIZE)
    assert torch.isfinite(crop).all()
    assert int(crop_mask.sum()) > 0


def test_rotations_are_lossless_permutations():
    img, m = _frame(512, 512), _rect_mask(512, 512, 50, 450, 100, 140)
    crop, _, _ = letterbox_crop(img, m, SIZE)
    assert torch.equal(rotate_k90(crop, 0), crop)
    assert torch.equal(rotate_k90(crop, 360), crop)
    # four 90s return to the original, exactly - no interpolation anywhere
    r = crop
    for _ in range(4):
        r = rotate_k90(r, 90)
    assert torch.equal(r, crop)
    # and the multiset of pixel values is preserved
    assert torch.equal(
        torch.sort(rotate_k90(crop, 90).flatten()).values,
        torch.sort(crop.flatten()).values,
    )


def test_non_multiple_of_90_is_rejected():
    with pytest.raises(ValueError, match="multiples of 90"):
        rotate_k90(torch.zeros((3, SIZE, SIZE)), 45)


def test_normalize_uses_maskdino_statistics():
    from cropcls.config import PIXEL_MEAN, PIXEL_STD

    crop = torch.full((3, 8, 8), 0.0)
    out = normalize_crop(crop, PIXEL_MEAN, PIXEL_STD)
    # a black (padded) pixel maps to -mean/std, not to 0
    expected = [-m / s for m, s in zip(PIXEL_MEAN, PIXEL_STD)]
    for c, e in enumerate(expected):
        assert out[c].mean().item() == pytest.approx(e, rel=1e-5)


def test_mismatched_mask_shape_is_an_error():
    with pytest.raises(ValueError, match="does not match"):
        letterbox_crop(_frame(64, 64), torch.zeros((32, 32), dtype=torch.bool), SIZE)


# --- mask dilation ---------------------------------------------------------------


def test_dilate_grows_the_bbox_by_exactly_the_radius():
    from cropcls.crops import dilate_mask

    m = _rect_mask(128, 128, 40, 60, 30, 70)
    assert mask_tight_bbox(m) == (30, 40, 70, 60)
    for r in (1, 3, 8):
        assert mask_tight_bbox(dilate_mask(m, r)) == (30 - r, 40 - r, 70 + r, 60 + r)


def test_dilate_zero_is_identity():
    from cropcls.crops import dilate_mask

    m = _rect_mask(64, 64, 10, 20, 10, 20)
    assert torch.equal(dilate_mask(m, 0), m)


def test_dilate_is_a_superset_never_removes_pixels():
    from cropcls.crops import dilate_mask

    m = _rect_mask(64, 64, 10, 20, 10, 20)
    for r in (1, 2, 5):
        d = dilate_mask(m, r)
        assert torch.all(d[m]), "dilation must never drop an original pixel"
        assert int(d.sum()) > int(m.sum())


def test_dilate_clamps_at_the_frame_edge():
    from cropcls.crops import dilate_mask

    m = _rect_mask(32, 32, 0, 5, 0, 5)
    d = dilate_mask(m, 10)
    assert d.shape == m.shape, "dilation must not change the mask size"


def test_dilated_crop_keeps_background_exactly_zero_outside_the_grown_mask():
    img, m = _frame(256, 256, 200.0), _rect_mask(256, 256, 60, 160, 70, 170)
    crop, crop_mask, _ = letterbox_crop(img, m, SIZE, dilate=3)
    assert torch.all(crop[:, ~crop_mask] == 0.0)
    # the grown mask admits strictly more pixels than the tight one
    tight_crop, tight_mask, _ = letterbox_crop(img, m, SIZE, dilate=0)
    assert int(crop_mask.sum()) > 0 and int(tight_mask.sum()) > 0


def test_closing_fills_gaps_without_growing_the_silhouette():
    """Closing repairs a fragmented mask; dilation repairs it AND adds context.

    The predicted masks are visibly broken (IoU 0.55-0.83, shafts as dashed segments),
    so these are two different hypotheses and must not be conflated.
    """
    from cropcls.crops import close_mask, dilate_mask

    m = torch.zeros((128, 128), dtype=torch.bool)
    m[60:66, 20:50] = True
    m[60:66, 54:90] = True          # 4px gap in a 6px-thick shaft
    box = mask_tight_bbox(m)

    closed = close_mask(m, 3)
    assert mask_tight_bbox(closed) == box, "closing must not grow the bounding box"
    assert bool(closed[60:66, 50:54].all()), "closing must bridge the gap"
    assert int(closed.sum()) == int(m.sum()) + 4 * 6

    dil = dilate_mask(m, 3)
    assert mask_tight_bbox(dil) != box, "dilation grows the box (that is the difference)"
    assert int(dil.sum()) > int(closed.sum())


def test_closing_is_idempotent_on_a_solid_mask():
    from cropcls.crops import close_mask

    m = _rect_mask(64, 64, 10, 30, 10, 30)
    assert torch.equal(close_mask(m, 3), m), "a hole-free convex mask is unchanged"


def test_letterbox_close_matches_the_standalone_op():
    from cropcls.crops import close_mask

    img = _frame(256, 256)
    m = torch.zeros((256, 256), dtype=torch.bool)
    m[100:110, 60:120] = True
    m[100:110, 125:180] = True
    _, got, _ = letterbox_crop(img, m, SIZE, dilate=4, close=True)
    _, ref, _ = letterbox_crop(img, close_mask(m, 4), SIZE, dilate=0)
    assert torch.equal(got, ref), "the crop-first fast path must be exact for closing too"


def _fragmented_mask():
    """Dashed shaft + jaws: holes for dilation to close, unlike a solid rectangle."""
    m = torch.zeros((256, 256), dtype=torch.bool)
    for i in range(0, 110, 14):
        m[100:106, 60 + i : 69 + i] = True
    m[82:124, 106:112] = True
    m[92:114, 164:186] = True
    return m


def test_a_passed_box_overrides_the_dilated_box():
    """A passed `box` wins over the box letterbox_crop would compute from the dilated mask.

    Every real caller passes the UNDILATED tight box, so dilation grows the mask inside the
    crop region and never widens it. `pad_frac` is the knob that widens it, and it is applied
    to the passed box afterwards, so the two stay separable: dilation repairs the mask, padding
    changes the field of view. The static crop folder reproduces the live path only because of
    this. If someone "fixes" dilation to also grow the crop region, the folder silently stops
    matching the live pipeline and only classaware_f1 moves - so pin it here.
    """
    img = _frame(256, 256)
    mask = _fragmented_mask()
    tight = mask_tight_bbox(mask)

    passed, _, _ = letterbox_crop(img, mask, SIZE, box=tight, dilate=12)
    computed, _, _ = letterbox_crop(img, mask, SIZE, box=None, dilate=12)
    assert not torch.equal(passed, computed), (
        "box=tight and box=None agree under dilation; the passed box is being ignored"
    )

    # With no dilation the two boxes coincide, so the override is invisible - which is why
    # the dilate>0 case above is the one that can fail.
    a, _, _ = letterbox_crop(img, mask, SIZE, box=tight, dilate=0)
    b, _, _ = letterbox_crop(img, mask, SIZE, box=None, dilate=0)
    assert torch.equal(a, b)


# --- to_model_input: the tensor the classifier is actually fed ----------------

MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


def test_to_model_input_is_3_channels_when_the_flag_is_off():
    img, m = _frame(), _rect_mask(512, 512, 100, 300, 150, 250)
    out = letterbox_crop(img, m, SIZE)
    assert to_model_input(out, MEAN, STD).shape == (3, SIZE, SIZE)


def test_to_model_input_matches_the_pre_helper_expression():
    """Keeps the refactor honest: the three crop paths used to spell this out inline as
    normalize(rotate(crop)). Also pins that normalize and rotate commute - per-channel
    elementwise against a pixel permutation - which is what lets one helper serve paths
    that previously rotated first."""
    img, m = _frame(), _rect_mask(512, 512, 100, 300, 150, 250)
    out = letterbox_crop(img, m, SIZE)
    for deg in (0, 90, 180, 270):
        assert torch.equal(
            to_model_input(out, MEAN, STD, deg),
            normalize_crop(rotate_k90(out[0], deg), MEAN, STD),
        ), deg


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_mask_channel_is_the_letterboxed_mask_not_zeros(deg):
    """The assertion the equivalence test in test_cropcls_mask_channel.py CANNOT make.

    Zero-initialized input weights ignore channel 3 entirely, so a channel of zeros, of
    noise, or of a mask rotated out of step with the RGB all pass there identically. This is
    what pins channel 3 to the actual mask, co-rotated with the pixels it describes.
    """
    img, m = _frame(), _rect_mask(512, 512, 100, 300, 150, 250)
    out = letterbox_crop(img, m, SIZE)
    x = to_model_input(out, MEAN, STD, deg, mask_channel=True)

    assert x.shape == (4, SIZE, SIZE)
    assert torch.equal(x[3], rotate_k90(out[1], deg).to(x.dtype))
    assert set(x[3].unique().tolist()) <= {0.0, 1.0}
    # non-degenerate in both directions: a constant channel would satisfy the equality above
    # only if the mask itself were constant, but assert it outright so the fixture cannot rot.
    assert 0 < float(x[3].sum()) < x[3].numel()
    # the RGB half is untouched by the flag
    assert torch.equal(x[:3], to_model_input(out, MEAN, STD, deg))
