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

__all__ = ["CropClassifier", "build_classifier"]


class CropClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        trunk: str = "resnet50",
        pretrained: bool = True,
        freeze_trunk: bool = False,
        in_chans: int = 3,
    ):
        super().__init__()
        # num_classes=0 + global_pool="avg" gives pooled features and no head, so the
        # head stays explicit and swappable (linear / mlp capacity is a reported finding,
        # not an implementation detail).
        self.trunk = timm.create_model(
            trunk,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
        )
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
    )
