"""`stem_stride`: re-stride the trunk's input conv to keep early detail.

Spec: spec_stem_stride.md. Ground truth: requirements_stem_stride.md.

The thing under test is the EARLY reduction. `output_stride` cannot reach it - dilating
stage 4 moves only the last feature map while the stem and stages 1-3 are untouched - which
is why these tests assert every stage, not just the final one.

`pretrained=False` throughout: these are shape, stride and wiring properties, and downloading
ImageNet weights would make the suite need the network.
"""

from __future__ import annotations

import typing

import pytest
import torch

from cropcls.config import CropClsConfig
from cropcls.modeling.classifier import (
    CropClassifier,
    build_classifier,
    first_conv,
    set_stem_stride,
)

TRUNK = "convnext_tiny"


def _stage_sizes(model: CropClassifier, size: int = 224, chans: int = 3) -> list[int]:
    """Spatial edge after the stem and after each stage."""
    trunk = model.trunk
    seen: list[int] = []
    hooks = [trunk.stem.register_forward_hook(
        lambda m, i, o: seen.append(int(o.shape[-1])))]
    hooks += [s.register_forward_hook(lambda m, i, o: seen.append(int(o.shape[-1])))
              for s in trunk.stages]
    with torch.no_grad():
        model(torch.zeros(1, chans, size, size))
    for h in hooks:
        h.remove()
    return seen


def _model(**kw) -> CropClassifier:
    return CropClassifier(num_classes=8, trunk=TRUNK, pretrained=False, **kw).eval()


def test_stem_stride_halves_every_stage():
    """The whole point: every stage doubles, not just the last one.

    convnext_tiny's stem is a single 4x4/stride-4 patchify, so at crop_size 224 the four
    stages run 56/28/14/7. Halving the stem stride gives 112/56/28/14 - crop_size 448's
    spatial resolution with no upsampling. An `output_stride` change would leave the first
    three of those numbers alone, which is the distinction this pins.
    """
    assert _stage_sizes(_model()) == [56, 56, 28, 14, 7]
    assert _stage_sizes(_model(stem_stride=2)) == [112, 112, 56, 28, 14]


def test_parameter_count_is_unchanged():
    """Identical parameters: this costs compute, not capacity.

    If it changed the parameter count the arm would confound resolution with capacity, and
    `cropcls-results.md` already records resnet18-vs-resnet50 at only -2.2 points, so capacity
    is not the binding constraint and must not be the thing accidentally varied.
    """
    a = sum(p.numel() for p in _model().parameters())
    b = sum(p.numel() for p in _model(stem_stride=2).parameters())
    assert a == b


def test_stem_weights_are_not_resampled():
    """Only `.stride` and `.padding` move; the kernel keeps its shape and its values.

    The alternative - a smaller kernel, or timm's `in_chans` style re-tiling - would alter the
    pretrained RGB path and make the arm a measurement of a perturbed trunk instead of of the
    stride. Same argument as `add_mask_input_channel` using zeros.
    """
    m = _model()
    before = first_conv(m.trunk)[1].weight.detach().clone()
    set_stem_stride(m.trunk, 2)
    conv = first_conv(m.trunk)[1]
    assert torch.equal(before, conv.weight)
    assert tuple(conv.weight.shape) == (96, 3, 4, 4)
    assert conv.stride == (2, 2)
    assert conv.padding == (1, 1)


def test_padding_keeps_the_stem_output_even():
    """`(kernel - stride) // 2`. Measured: 4x4/s2/p0 gives 111 and odd sizes all the way down.

    Odd feature maps are not wrong, but they make every downstream size a different number
    from the stride-4 run's halves, so a size assertion can no longer tell a correct
    re-stride from an off-by-one.
    """
    m = _model(stem_stride=2)
    assert _stage_sizes(m) == [112, 112, 56, 28, 14]
    conv = first_conv(m.trunk)[1]
    assert conv.padding == (1, 1)


def test_native_stride_is_a_no_op():
    """stem_stride=0 and stem_stride=4 are the same convnext_tiny, bit for bit.

    So the control arm can state the field explicitly instead of omitting it, and the
    checkpoint guard can compare effective strides without rejecting a legitimate pairing.
    """
    a, b = _model(stem_stride=0), _model(stem_stride=4)
    assert a.stem_stride == b.stem_stride == 4
    assert _stage_sizes(a) == _stage_sizes(b) == [56, 56, 28, 14, 7]
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)


def test_a_stride_wider_than_the_kernel_raises():
    """Past the kernel size the conv skips pixels entirely - a different operation."""
    m = _model()
    with pytest.raises(ValueError, match="exceeds the kernel size"):
        set_stem_stride(m.trunk, 8)


def test_a_trunk_without_first_conv_raises():
    """Located by `pretrained_cfg['first_conv']`, never a per-trunk table."""

    class Bare(torch.nn.Module):
        pretrained_cfg: typing.ClassVar[dict] = {}

    with pytest.raises(ValueError, match="no pretrained_cfg"):
        set_stem_stride(Bare(), 2)


def test_stem_stride_survives_the_mask_channel_surgery():
    """Order matters and nothing would raise if it were wrong.

    `add_mask_input_channel` REBUILDS the conv, copying `stride=old.stride` and
    `padding=old.padding`. Re-striding first means the rebuild inherits it; re-striding after
    would be discarded silently. mask_channel is the shipped setting for every mask4 arm, so
    this is the combination that actually runs.
    """
    m = _model(stem_stride=2, mask_channel=True)
    conv = first_conv(m.trunk)[1]
    assert conv.in_channels == 4
    assert conv.stride == (2, 2) and conv.padding == (1, 1)
    assert _stage_sizes(m, chans=4)[0] == 112
    # the added channel is still zero-initialized, i.e. step-0 identical to 3 channels
    assert torch.count_nonzero(conv.weight[:, 3:]) == 0


def test_build_classifier_passes_the_config_field():
    """A field the builder drops is a silent no-op, and the run would look like a control."""
    cfg = CropClsConfig(
        detector_checkpoint="/fake", detector_config="/fake",
        trunk=TRUNK, trunk_pretrained=False, stem_stride=2,
    )
    assert build_classifier(cfg, 8).stem_stride == 2


def test_native_stem_stride_is_recorded_for_legacy_checkpoints():
    """A checkpoint predating the field is read as the trunk's native stride.

    That is what every existing run trained at, so the guard must not reject them.
    """
    assert _model().native_stem_stride == 4
    assert _model(stem_stride=2).native_stem_stride == 4


def test_loading_a_mismatched_stem_stride_raises(tmp_path):
    """The silent-wrongness case this whole guard exists for.

    `.stride` is not in `state_dict`: a stride-2 and a stride-4 convnext_tiny have identical
    keys AND identical tensor shapes, so `load_state_dict` accepts the wrong one without
    error and then runs a 7x7 feature map where the weights expect 14x14. Unlike
    `mask_channel`, whose conv changes shape so torch raises, nothing here would.
    """
    strided = _model(stem_stride=2)
    plain = _model()
    # the hazard itself: torch is perfectly happy
    plain.load_state_dict(strided.state_dict())
    assert _stage_sizes(plain)[0] == 56          # ran at the wrong resolution, no error

    # so the guard has to be ours, on the recorded effective stride
    ckpt = tmp_path / "model_final.pth"
    torch.save({"model": strided.state_dict(), "iteration": 4000,
                "stem_stride": strided.stem_stride}, ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    fresh = _model()
    ckpt_stride = state.get("stem_stride", fresh.native_stem_stride)
    assert ckpt_stride == 2 and fresh.stem_stride == 4
    assert ckpt_stride != fresh.stem_stride      # what train_cropcls.py raises on

    # and a legacy checkpoint with no recorded stride pairs with a default build
    torch.save({"model": plain.state_dict(), "iteration": 4000}, ckpt)
    legacy = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert legacy.get("stem_stride", fresh.native_stem_stride) == fresh.stem_stride


@pytest.mark.parametrize(("trunk", "native_stride"),
                         [("convnext_tiny", 4), ("resnet50", 2), ("resnet50d", 2)])
def test_native_stride_is_a_no_op_on_every_trunk(trunk, native_stride):
    """The no-op guarantee has to hold for trunks the arm does not use.

    The ask is convnext-only but the code is not, and `(k - stride) // 2` is correct only for
    an EVEN kernel: it reproduces convnext_tiny's 4x4/s4/p0 but turns resnet50's 7x7/s2/p3 into
    p2 (stem 111, not 112) and resnet50d's 3x3/s2/p1 into p0. Stating the native stride would
    then silently change the model. Measured, hence `ceil((k - stride) / 2)` plus an early
    return.
    """
    import timm
    ref = timm.create_model(trunk, pretrained=False, num_classes=0, global_pool="")
    ref_conv = first_conv(ref)[1]
    m = timm.create_model(trunk, pretrained=False, num_classes=0, global_pool="")
    assert set_stem_stride(m, native_stride) == native_stride
    conv = first_conv(m)[1]
    assert conv.stride == ref_conv.stride
    assert conv.padding == ref_conv.padding, "stating the native stride changed the padding"
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(ref.forward_features(x), m.forward_features(x))


@pytest.mark.parametrize("trunk", ["convnext_tiny", "resnet50", "resnet50d"])
def test_stride_two_keeps_the_stem_output_even_on_every_trunk(trunk):
    """stride 2 must give 112 at input 224 regardless of kernel parity."""
    import timm
    m = timm.create_model(trunk, pretrained=False, num_classes=0, global_pool="")
    set_stem_stride(m, 2)
    conv = first_conv(m)[1]
    h = (224 + 2 * conv.padding[0] - conv.kernel_size[0]) // 2 + 1
    assert h == 112, f"{trunk}: stem {h}"
