# Copyright (c) Facebook, Inc. and its affiliates.
# Vendored from facebookresearch/Mask2Former @ 9b0651c - see ../VENDORED.md.
#
# Upstream also does `from .backbone.swin import D2SwinTransformer` here. We must NOT:
# maskdino/modeling/__init__.py already registers that exact name in detectron2's
# global BACKBONE_REGISTRY, and a second registration of the same key raises at import
# time. Swin configs for either architecture resolve to the maskdino copy, so nothing
# is lost. Upstream's per_pixel_baseline import is also dropped - those are
# semantic-only heads this fork never builds.
from .pixel_decoder.fpn import BasePixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .meta_arch.mask_former_head import MaskFormerHead

__all__ = ["BasePixelDecoder", "MSDeformAttnPixelDecoder", "MaskFormerHead"]
