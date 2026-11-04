"""Grad-CAM over the classifier's trunk - "where did the model look to call this a forcep03?".

The map explains the **predicted** class on the crop as the model saw it: blacked out,
letterboxed, normalized, and 4-channel when `mask_channel` is on. It is a diagnostic only -
nothing here produces a number, and no run is ranked by it (ranking stays `classaware_f1`
with `score_mode=detector`).

Resolution is coarse by construction: the last feature stage is stride 32, so a 224 crop
gives a 7x7 map and a 448 crop 14x14, upsampled back to the crop. Pass an earlier stage
(`layer3`, `stages.2`) for a 14x14 read at 224.

Stock PyTorch, no `pytorch-grad-cam` dependency (spec_cropcls.md step 6: "Stock PyTorch ops
only"). Nothing in this file touches MaskDINO.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["CropGradCAM", "cam_in_chunks", "resolve_cam_layer"]


def resolve_cam_layer(trunk: torch.nn.Module) -> str:
    """Submodule name of the trunk's last feature stage.

    `layer4` for resnet50 and resnet50d, `stages.3` for convnext_tiny - read from timm's
    `feature_info` rather than from a per-trunk table, for the same reason
    `add_mask_input_channel` reads `pretrained_cfg["first_conv"]`: a table goes stale the first
    time someone passes a different `trunk=` on the command line. timm populates
    `feature_info` on a classification-mode model too, so `features_only=True` is not needed.
    """
    info = getattr(trunk, "feature_info", None)
    entries = getattr(info, "info", info)      # FeatureInfo wraps the same list of dicts
    last = entries[-1] if entries else None
    name = last.get("module") if isinstance(last, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{type(trunk).__name__} declares no usable feature_info[-1]['module'], so its "
            "last feature stage cannot be located; pass the layer explicitly "
            "(--gradcam-layer <submodule name>)"
        )
    trunk.get_submodule(name)                  # AttributeError here beats a silent wrong map
    return name


class CropGradCAM:
    """Grad-CAM for one `CropClassifier`. Reusable across batches; holds no state between calls.

    The forward hook is attached and removed per call, so an exception cannot leave the trunk
    carrying a hook that quietly accumulates activations for the rest of the run.
    """

    def __init__(self, model, layer: str | None = None):
        self.model = model
        self.layer_name = layer or resolve_cam_layer(model.trunk)
        self.layer = model.trunk.get_submodule(self.layer_name)

    def __call__(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(crops) -> (cam (B, S, S) float in [0, 1], logits (B, C)), both detached.

        `targets` picks the class each map explains; None means the argmax - where the model
        looked to say what it said.
        """
        store: dict[str, torch.Tensor] = {}
        handle = self.layer.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("act", out)
        )
        # eval() for the duration even if the caller forgot: timm trunks use real BatchNorm2d,
        # so a forward in train mode would move running_mean/var - and this same model object
        # is the one the training loop's monitor and the optimizer hold.
        was_training = self.model.training
        try:
            self.model.eval()
            # requires_grad on the INPUT, not just on the parameters. Under `freeze_trunk` every
            # trunk parameter is a non-grad leaf, so nothing in the activation's history requires
            # grad and autograd.grad raises "One of the differentiated Tensors does not require
            # grad". An unfrozen trunk works either way, which is exactly what hides it.
            x = x.detach().float().requires_grad_(True)
            with torch.enable_grad():
                # Deliberately outside autocast: the hooked activation is then already fp32, so
                # the gradient target is the tensor in the graph and not a cast copy of it (a
                # cast copy raises "unused tensor"). fp16 was measured to give the same map, so
                # this is determinism, not a correctness fix.
                logits = self.model(x).float()
                act = store["act"]
                if targets is None:
                    targets = logits.argmax(1)
                selected = logits.gather(1, targets.reshape(-1, 1).to(logits.device)).sum()
                # autograd.grad, not .backward(): no parameter .grad is written, so running this
                # mid-training cannot pollute the next optimizer step.
                grads = torch.autograd.grad(selected, act)[0]

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act.float()).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)
            # Per-crop normalization is not cosmetic: the raw peak is ~1.7e-3 at init, so an
            # unnormalized map renders black and reads as "the model looks nowhere". clamp_min
            # keeps an all-negative (hence all-zero after relu) map at zero instead of NaN.
            peak = cam.flatten(1).amax(1).reshape(-1, 1, 1)
            cam = cam / peak.clamp_min(torch.finfo(cam.dtype).eps)
            return cam.detach(), logits.detach()
        finally:
            handle.remove()
            self.model.train(was_training)


def cam_in_chunks(
    cam: CropGradCAM,
    crops: torch.Tensor,
    device,
    chunk: int,
    targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`cam` over `crops` in slices of `chunk`, results on the CPU.

    A CAM pass costs far more memory than the scoring pass it follows: grad-enabled and fp32,
    where `evaluate()` is `no_grad` and fp16. A whole `ims_per_batch` frame batch of 448px
    crops can therefore OOM immediately after a successful eval, which looks like the eval
    broke rather than the diagnostic. `crops_per_batch` is the config's own notion of how many
    crops fit at once.
    """
    heats, logits = [], []
    for s0 in range(0, len(crops), max(1, chunk)):
        sl = slice(s0, s0 + max(1, chunk))
        h, lg = cam(crops[sl].to(device, non_blocking=True),
                    targets=None if targets is None else targets[sl])
        heats.append(h.cpu())
        logits.append(lg.cpu())
    if not heats:
        return torch.zeros((0,)), torch.zeros((0,))
    return torch.cat(heats), torch.cat(logits)
