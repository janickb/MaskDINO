"""Step-10 verification: the mask channel (spec_cropcls.md, requirements_mask_channel.md).

At step 0 the 4-channel model must BE the 3-channel model, and still be learnable.

`pretrained=False` throughout - the invariant is about the surgery, not about any particular
ImageNet weights, so the suite needs no network and no HF cache. `pretrained_cfg["first_conv"]`
is populated either way, which is what makes that possible.

The equivalence test alone is NOT sufficient and must never be the only one here: zero weights
ignore whatever is in channel 3, so it passes just as happily on a channel of garbage. Pair it
with the gradient test below, and with the data-side assertions in test_cropcls_crops.py that
pin channel 3 to the actual letterboxed mask.
"""

import copy
import inspect

import pytest
import timm
import torch
from torch import nn

from cropcls.modeling import CropClassifier, add_mask_input_channel

TRUNKS = ["resnet50", "resnet50d", "convnext_tiny"]


def _trunk(name):
    return timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")


def _first_conv(trunk):
    return trunk.get_submodule(trunk.pretrained_cfg["first_conv"])


@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_4ch_model_is_the_3ch_model_at_init(trunk):
    """Catches, in one assertion: timm's 3/4 rescale in adapt_input_conv, a wrong first_conv
    path, and a swapped RGB/mask channel order. Exact equality, not allclose - a zero weight
    contributes an exactly-zero term. Run on CPU: TF32 on CUDA may reassociate the reduction.
    """
    torch.manual_seed(0)
    m3 = _trunk(trunk).eval()
    m4 = add_mask_input_channel(copy.deepcopy(m3)).eval()

    rgb = torch.randn(2, 3, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()   # any mask: the new weights are zero
    with torch.no_grad():
        assert torch.equal(m3(rgb), m4(torch.cat([rgb, mask], dim=1)))


@pytest.mark.parametrize("trunk", TRUNKS)
def test_one_step_moves_the_mask_weights_off_zero(trunk):
    """The channel has to be *learnable*, not merely harmless. Without this, a detached or
    accidentally-frozen channel 3 would sail through the equivalence test above - the arm
    would train for 4000 iterations and reproduce the 3-channel result exactly.
    """
    torch.manual_seed(0)
    m = add_mask_input_channel(_trunk(trunk)).train()
    conv = _first_conv(m)
    assert conv.weight[:, 3:].eq(0).all(), "precondition: mask weights start at zero"

    x = torch.cat([torch.randn(2, 3, 64, 64), (torch.rand(2, 1, 64, 64) > 0.5).float()], 1)
    m(x).square().mean().backward()
    grad = conv.weight.grad[:, 3:]
    assert grad is not None and grad.abs().max() > 0, "no gradient reaches the mask channel"


@pytest.mark.parametrize("trunk", TRUNKS)
def test_rgb_weights_are_copied_bit_exactly_and_mask_weights_are_zero(trunk):
    """If this fails with RGB scaled by 0.75 and channel 3 equal to channel 0, someone
    replaced the surgery with timm.create_model(in_chans=4)."""
    torch.manual_seed(0)
    before = _trunk(trunk)
    old_w = _first_conv(before).weight.detach().clone()
    conv = _first_conv(add_mask_input_channel(before))

    assert torch.equal(conv.weight[:, :3], old_w)
    assert conv.weight[:, 3:].eq(0).all()


@pytest.mark.parametrize("trunk", TRUNKS)
def test_every_conv_attribute_is_preserved(trunk):
    torch.manual_seed(0)
    before = _trunk(trunk)
    old = _first_conv(before)
    attrs = {
        k: getattr(old, k)
        for k in ("out_channels", "kernel_size", "stride", "padding", "dilation",
                  "groups", "padding_mode")
    }
    old_bias = None if old.bias is None else old.bias.detach().clone()

    new = _first_conv(add_mask_input_channel(before))
    assert new.in_channels == old.in_channels + 1
    for k, v in attrs.items():
        assert getattr(new, k) == v, k
    # convnext_tiny's stem.0 carries a bias; both resnets' first conv does not.
    assert (new.bias is None) == (old_bias is None)
    if old_bias is not None:
        assert torch.equal(new.bias.detach(), old_bias)


def test_timm_in_chans_4_is_not_zero_init():
    """Pins the trap itself, so nobody 'simplifies' add_mask_input_channel into in_chans=4.
    Pure function over a weight tensor: no model, no download."""
    from timm.models._manipulate import adapt_input_conv

    w = adapt_input_conv(4, torch.ones(8, 3, 3, 3))
    assert w.shape[1] == 4
    assert not w[:, 3:].eq(0).any(), "timm tiles RGB; channel 3 is a copy of red, not zeros"
    assert torch.allclose(w, torch.full_like(w, 0.75)), "timm also rescales RGB by 3/4"


def test_a_trunk_without_first_conv_raises_clearly():
    m = _trunk("resnet50")
    m.pretrained_cfg = {}
    with pytest.raises(ValueError, match="first_conv"):
        add_mask_input_channel(m)


def test_a_non_conv_first_conv_raises():
    m = _trunk("resnet50")
    m.pretrained_cfg = {"first_conv": "global_pool"}
    with pytest.raises(TypeError, match="not nn.Conv2d"):
        add_mask_input_channel(m)


@pytest.mark.parametrize("trunk,key", [
    ("resnet50", "trunk.conv1.weight"),
    ("resnet50d", "trunk.conv1.0.weight"),
    ("convnext_tiny", "trunk.stem.0.weight"),
])
def test_a_3ch_checkpoint_into_a_4ch_model_fails_loudly(trunk, key):
    """Checkpoints do not cross the flag, by design (requirements_mask_channel.md). The guard
    is strict load_state_dict's own size mismatch; nothing papers over it."""
    torch.manual_seed(0)
    state = CropClassifier(num_classes=8, trunk=trunk, pretrained=False).state_dict()
    m4 = CropClassifier(num_classes=8, trunk=trunk, pretrained=False, mask_channel=True)
    with pytest.raises(RuntimeError, match=f"size mismatch for {key.replace('.', chr(92)+'.')}"):
        m4.load_state_dict(state)


def test_in_chans_is_not_a_constructor_param():
    """It was dead plumbing and the only route to adapt_input_conv. Keep it gone."""
    assert "in_chans" not in inspect.signature(CropClassifier.__init__).parameters


@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_classifier_accepts_3_channels(trunk):
    """Step 6's forward-shape check. The 3-channel contract is the default and stays the
    default; this is here so Step 6's Verify names a test that exists."""
    torch.manual_seed(0)
    m = CropClassifier(num_classes=8, trunk=trunk, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 8)
    assert _first_conv(m.trunk).in_channels == 3


@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_classifier_accepts_4_channels_end_to_end(trunk):
    torch.manual_seed(0)
    m = CropClassifier(num_classes=8, trunk=trunk, pretrained=False, mask_channel=True).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 4, 64, 64)).shape == (2, 8)
    assert isinstance(_first_conv(m.trunk), nn.Conv2d)
