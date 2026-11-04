"""Grad-CAM over the classifier trunk (cropcls/gradcam.py).

`pretrained=False` and 64px crops throughout - the invariants are about the hook, the
gradient and the normalization, not about any particular ImageNet weights, so the suite needs
no network and no HF cache. `feature_info` is populated either way, which is what makes that
possible.

The two assertions that matter most are the ones a passing CAM would otherwise hide:
`test_a_frozen_trunk_still_yields_a_map` (an UNFROZEN trunk works even without the
`requires_grad_` line, so testing only the default would let the trap back in) and
`test_the_map_is_not_constant` (an all-zero map renders black and reads as "the model looks
nowhere" rather than as a bug).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cropcls.gradcam import CropGradCAM, resolve_cam_layer
from cropcls.modeling import CropClassifier
from cropcls.viz import cam_overlay, denorm_crop

TRUNKS = ["resnet50", "resnet50d", "convnext_tiny"]
LAST_STAGE = {"resnet50": "layer4", "resnet50d": "layer4", "convnext_tiny": "stages.3"}


def _model(trunk="resnet50", **kw):
    torch.manual_seed(0)
    return CropClassifier(num_classes=8, trunk=trunk, pretrained=False, **kw).eval()


def _crops(n=2, c=3, s=64):
    torch.manual_seed(0)
    return torch.randn(n, c, s, s)


@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_layer_resolves_to_the_last_feature_stage(trunk):
    """Located from feature_info, not from a per-trunk table that goes stale on `trunk=`."""
    m = _model(trunk)
    name = resolve_cam_layer(m.trunk)
    assert name == LAST_STAGE[trunk]
    assert m.trunk.get_submodule(name) is CropGradCAM(m).layer


def test_a_trunk_without_feature_info_is_rejected_loudly():
    """Not a silent fall back to some other module: a map of the wrong tensor still renders."""
    m = _model()
    del m.trunk.feature_info
    with pytest.raises(ValueError, match="feature_info"):
        resolve_cam_layer(m.trunk)


@pytest.mark.parametrize("trunk", TRUNKS)
def test_the_map_covers_the_crop_and_is_normalized(trunk):
    cam, logits = CropGradCAM(_model(trunk))(_crops(n=3))
    assert cam.shape == (3, 64, 64)
    assert logits.shape == (3, 8)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) == pytest.approx(1.0)
    # Detached: a map carrying grad_fn would pin the whole graph for every crop a sheet keeps.
    assert not cam.requires_grad and not logits.requires_grad


@pytest.mark.parametrize("trunk", TRUNKS)
def test_a_frozen_trunk_still_yields_a_map(trunk):
    """`freeze_trunk` makes every trunk parameter a non-grad leaf, so without requires_grad_
    on the INPUT nothing in the activation's history requires grad and autograd.grad raises
    "One of the differentiated Tensors does not require grad". Delete that line and this is
    the only test that fails.
    """
    cam, _ = CropGradCAM(_model(trunk, freeze_trunk=True))(_crops())
    # Deliberately NOT `== 1.0`: normalization is the other test's invariant, and asserting it
    # here made a dropped normalization fail six tests instead of three - which hides which
    # invariant actually broke. Without requires_grad_ this line is never reached anyway: the
    # call raises.
    assert float(cam.max()) > 0.0


def test_the_map_is_not_constant():
    """Rules out the two silent failures: all-zero (renders black) and uniform (renders flat).

    Deliberately only "not constant" and not a variance floor: with random weights the spread
    is trunk-dependent (0.02 to 0.32 measured), and a floor tuned to that would be a test of
    the initialization rather than of the CAM.
    """
    cam, _ = CropGradCAM(_model())(_crops())
    assert float(cam.max()) > float(cam.min())
    assert float(cam.sum()) > 0.0


def test_the_map_depends_on_the_target_class():
    """Otherwise the class selection could be ignored and every panel would be the same map."""
    f = CropGradCAM(_model())
    x = _crops()
    a, _ = f(x, targets=torch.zeros(2, dtype=torch.long))
    b, _ = f(x, targets=torch.ones(2, dtype=torch.long))
    assert not torch.allclose(a, b)
    # Same target twice is the same map: the hook must not accumulate across calls.
    assert torch.equal(a, f(x, targets=torch.zeros(2, dtype=torch.long))[0])


@pytest.mark.parametrize("training", [True, False])
def test_the_pass_leaves_no_grads_no_hook_and_the_original_mode(training):
    """The model object is shared with the optimizer and the in-training val monitor.

    A written parameter `.grad` would land in the next optimizer step, a leaked forward hook
    would accumulate activations for the rest of the run, and a forward left in train mode
    would move the trunk's BatchNorm running stats - none of which raises.
    """
    m = _model().train(training)
    f = CropGradCAM(m)
    f(_crops())
    assert all(p.grad is None for p in m.parameters())
    assert len(f.layer._forward_hooks) == 0
    assert m.training is training


@pytest.mark.parametrize("trunk", TRUNKS)
def test_a_four_channel_crop_is_accepted(trunk):
    """mask_channel runs: the hook sits past the stem, so the extra channel changes nothing."""
    m = _model(trunk, mask_channel=True)
    x = torch.cat([_crops(), (torch.rand(2, 1, 64, 64) > 0.5).float()], dim=1)
    cam, _ = CropGradCAM(m)(x)
    assert cam.shape == (2, 64, 64)


def test_an_explicit_layer_overrides_the_last_stage():
    f = CropGradCAM(_model(), layer="layer3")
    assert f.layer_name == "layer3"
    cam, _ = f(_crops())
    assert cam.shape == (2, 64, 64)


def test_cam_overlay_matches_the_crop_and_does_not_mutate_it():
    rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
    before = rgb.copy()
    out = cam_overlay(rgb, torch.zeros(8, 8))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert np.array_equal(rgb, before), "the input crop was modified in place"
    # A hot map and a cold map must not render the same, or the overlay is dropping the cam.
    assert not np.array_equal(out, cam_overlay(rgb, torch.ones(8, 8)))
    with pytest.raises(ValueError, match="does not match"):
        cam_overlay(rgb, torch.zeros(4, 4))


def test_denorm_crop_inverts_the_pipeline_normalization_and_drops_channel_3():
    """One copy of this, shared with tools/viz_misclassified.py: a renderer that de-normalizes
    with other statistics than the pipeline used does not raise, it just draws a tinted tool.
    """
    from cropcls.config import PIXEL_MEAN, PIXEL_STD
    from cropcls.crops import normalize_crop

    raw = torch.randint(0, 256, (3, 8, 8)).float()
    back = denorm_crop(normalize_crop(raw, PIXEL_MEAN, PIXEL_STD), PIXEL_MEAN, PIXEL_STD)
    assert np.abs(back.astype(np.int16) - raw.permute(1, 2, 0).numpy().astype(np.int16)).max() <= 1

    four = torch.cat([normalize_crop(raw, PIXEL_MEAN, PIXEL_STD), torch.ones(1, 8, 8)], dim=0)
    assert denorm_crop(four, PIXEL_MEAN, PIXEL_STD).shape == (8, 8, 3)


def test_chunking_matches_one_pass_and_slices_the_targets_with_the_crops():
    """`cam_in_chunks` exists so a 448px frame batch cannot OOM the diagnostic right after a
    successful eval. It must be an optimization only - and the targets have to be sliced
    alongside the crops, which is the off-by-one a reader would not notice in a rendered map.
    """
    from cropcls.gradcam import cam_in_chunks

    f = CropGradCAM(_model())
    x = _crops(n=5)
    targets = torch.tensor([0, 3, 1, 7, 2])
    whole, logits = f(x, targets=targets)
    for chunk in (1, 2, 5, 99):
        got, got_logits = cam_in_chunks(f, x, x.device, chunk, targets=targets)
        assert torch.allclose(got, whole, atol=1e-6), f"chunk={chunk} changed the map"
        assert torch.allclose(got_logits, logits, atol=1e-6)
    # Mis-sliced targets would still return the right SHAPE, so compare against a shuffle.
    rolled, _ = cam_in_chunks(f, x, x.device, 2, targets=targets.roll(1))
    assert not torch.allclose(rolled, whole, atol=1e-6)


def test_the_cam_crop_is_cut_from_the_undilated_box():
    """`data.py` passes `box=mask_tight_bbox(...)` and `letterbox_crop` treats a passed box as
    overriding the dilated one (`test_a_passed_box_overrides_the_dilated_box`). Dropping it
    here would widen the CAM's field of view at `mask_dilate > 0` while the training input
    stays tight, so the map would explain a crop the classifier never sees.
    """
    from cropcls.config import CropClsConfig
    from cropcls.crops import letterbox_crop, mask_tight_bbox, to_model_input
    from cropcls.viz import _model_input

    cfg = CropClsConfig(detector_checkpoint="/fake/det.pth", crop_size=32, mask_dilate=8)
    torch.manual_seed(0)
    image = torch.rand(3, 64, 64) * 255
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[26:38, 26:38] = True

    def _crop(**kw):
        out = letterbox_crop(
            image, mask, cfg.crop_size, dilate=cfg.mask_dilate, blackout=cfg.blackout, **kw
        )
        return to_model_input(out, cfg.pixel_mean, cfg.pixel_std, 0, cfg.mask_channel)

    assert torch.equal(_model_input(cfg, image, mask), _crop(box=mask_tight_bbox(mask)))
    assert not torch.equal(_model_input(cfg, image, mask), _crop()), "box= changed nothing"
