"""The classifier: a CNN trunk + global average pool + linear head.

Classification only, no dense output. The segmentation already comes from MaskDINO
and all this stage adds is a class, so there is no expanding path and nothing
U-Net-shaped about it (spec_cropcls.md section 2) - it is an ImageNet image
classifier applied to a cropped, blacked-out instance.

The trunk is a timm backbone so it can start from ImageNet weights, and it is
pluggable: `resnet50` is the default, not the architecture. That pretraining is
also why the crop size is 224.

Nothing in this file touches MaskDINO. Its pixel decoder and transformer decoder are
frozen for the lifetime of the run; the only trainable parameters in the system are
this trunk and this head.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ..config import CropClsConfig

__all__ = ["CropClassifier", "add_mask_input_channel", "build_classifier"]


def add_mask_input_channel(trunk: nn.Module) -> nn.Module:
    """Give the trunk's first conv one extra, ZERO-initialized input channel.

    NOT `timm.create_model(in_chans=4)`. That routes the pretrained weights through
    `adapt_input_conv` (timm/models/_manipulate.py), which tiles the RGB kernels to
    (R,G,B,R) and rescales all four by 3/4 - so the pretrained RGB path is altered and the
    new channel starts as a copy of red. Zeros instead leave the 4-channel model
    bit-identical to the 3-channel one at step 0, which is what makes the arm a measurement
    of the channel rather than of a perturbed trunk.

    The conv is located via `pretrained_cfg["first_conv"]` - `conv1`, `conv1.0`, `stem.0`
    for resnet50 / resnet50d / convnext_tiny - because a per-trunk table goes stale the
    first time someone passes a different `trunk=` on the command line.
    """
    path = getattr(trunk, "pretrained_cfg", {}).get("first_conv")
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"{type(trunk).__name__} declares no pretrained_cfg['first_conv'], so its input "
            "conv cannot be located; mask_channel needs a trunk that names one"
        )
    old = trunk.get_submodule(path)
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"first_conv {path!r} is {type(old).__name__}, not nn.Conv2d")

    new = nn.Conv2d(
        old.in_channels + 1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight[:, : old.in_channels] = old.weight
        new.weight[:, old.in_channels :] = 0.0
        if old.bias is not None:      # convnext_tiny's stem.0 has one; the resnets do not
            new.bias.copy_(old.bias)
    parent, _, name = path.rpartition(".")
    setattr(trunk.get_submodule(parent) if parent else trunk, name, new)
    return trunk


class CropClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        trunk: str = "resnet50",
        pretrained: bool = True,
        freeze_trunk: bool = False,
        mask_channel: bool = False,
    ):
        super().__init__()
        # num_classes=0 + global_pool="avg" gives pooled features and no head, so the
        # head stays explicit and swappable (linear / mlp capacity is a reported finding,
        # not an implementation detail).
        # No in_chans= here on purpose: it is the only route to timm's adapt_input_conv,
        # which rescales the pretrained RGB kernels. See add_mask_input_channel.
        self.trunk = timm.create_model(
            trunk,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        if mask_channel:
            # Before the freeze below, so a frozen trunk freezes the new conv too rather
            # than leaving it the one trainable tensor in a "frozen" run.
            add_mask_input_channel(self.trunk)
        self.feature_dim = int(self.trunk.num_features)
        self.head = nn.Linear(self.feature_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.freeze_trunk = freeze_trunk
        if freeze_trunk:
            for p in self.trunk.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep a frozen trunk in eval mode so its BatchNorm running stats cannot drift.

        timm backbones use real BatchNorm2d (unlike detectron2's FrozenBN), so a frozen
        trunk left in train mode would still update running_mean/var on every forward -
        silently changing a module whose weights are pinned.
        """
        super().train(mode)
        if self.freeze_trunk:
            self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def param_groups(self, base_lr: float, weight_decay: float):
        """Decay on weights, none on norms and biases."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_classifier(cfg: CropClsConfig, num_classes: int) -> CropClassifier:
    return CropClassifier(
        trunk=cfg.trunk,
        num_classes=num_classes,
        pretrained=cfg.trunk_pretrained,
        freeze_trunk=cfg.freeze_trunk,
        mask_channel=cfg.mask_channel,
    )
