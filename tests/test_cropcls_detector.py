"""Step-1 verification: the frozen detector loads and is genuinely class-agnostic."""

import time
from pathlib import Path

import pytest
import torch

from cropcls.config import load_config

CONFIG = "configs/cropcls/phase1_seta.yaml"
EIGHT_CLASS_RUN = Path(
    "/home/janick.bilang/dev/MaskDINO/runs/20261006204751_phase1_training_seta_v1-1"
)

cfg = load_config(CONFIG)

needs_ckpt = pytest.mark.skipif(
    not Path(cfg.detector_checkpoint).is_file(),
    reason="detector checkpoint not present on this machine",
)
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@needs_ckpt
def test_checkpoint_class_embed_is_single_class():
    """The (1, 256) check from spec step 1, without building the model.

    class_embed is (num_classes, HIDDEN_DIM), so a class-agnostic frontend is (1, 256).
    """
    sd = torch.load(cfg.detector_checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    w = sd["sem_seg_head.predictor.class_embed.weight"]
    assert tuple(w.shape) == (1, 256), (
        f"expected a single-class head (1, HIDDEN_DIM=256), got {tuple(w.shape)}"
    )


@needs_ckpt
@needs_gpu
def test_detector_loads_frozen_and_in_eval_mode():
    from cropcls.detector import FrozenSingleClassDetector

    det = FrozenSingleClassDetector(cfg, device="cuda")
    assert det.num_classes == 1
    assert not det.model.training
    assert not any(p.requires_grad for p in det.model.parameters())


@needs_ckpt
@needs_gpu
def test_forward_returns_masks_in_the_input_coordinate_frame(capsys):
    """Masks must come back at the augmented resolution so they share a frame with GT."""
    from cropcls.detector import FrozenSingleClassDetector

    det = FrozenSingleClassDetector(cfg, device="cuda")
    frame = torch.randint(0, 255, (3, 1024, 1024), dtype=torch.uint8)

    det.detect([frame])  # warm up cuda / cudnn autotune
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = det.detect([frame])
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    assert len(out) == 1
    d = out[0]
    assert d.masks.dtype == torch.bool
    assert d.masks.shape[-2:] == (1024, 1024), "masks must be at the input resolution"
    assert d.scores.shape == (len(d),)
    assert torch.all(d.scores >= cfg.detector_score_thresh)
    with capsys.disabled():
        print(
            f"\n  [step 1] 1 frame @1024x1024 -> {len(d)} instances "
            f"(score >= {cfg.detector_score_thresh}), forward {dt * 1000:.0f} ms"
        )


@needs_gpu
@pytest.mark.skipif(
    not (EIGHT_CLASS_RUN / "model_final.pth").is_file(),
    reason="8-class run not present",
)
def test_multiclass_checkpoint_is_rejected():
    """Pointing the config at an 8-class run must fail loudly, not silently.

    Nothing would crash otherwise - the detector would just start emitting 8 class logits
    and the "always predicts surgical_instrument" premise would break invisibly.
    """
    from cropcls.config import CropClsConfig
    from cropcls.detector import FrozenSingleClassDetector

    bad = CropClsConfig(
        detector_checkpoint=str(EIGHT_CLASS_RUN / "model_final.pth"),
        detector_config=str(EIGHT_CLASS_RUN / "config.yaml"),
    )
    with pytest.raises(ValueError, match="not the class-agnostic frontend"):
        FrozenSingleClassDetector(bad, device="cuda")
