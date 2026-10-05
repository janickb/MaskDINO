# Copyright (c) Facebook, Inc. and its affiliates.
"""Vendored subset of facebookresearch/Mask2Former @ 9b0651c.

Deliberately NOT a full copy. Dataset registration, dataset mappers, evaluation, TTA,
the Swin backbone and the MSDeformAttn ops package are all taken from the maskdino/
package instead, so both architectures share one harness byte-for-byte - that is what
makes a MaskDINO-vs-Mask2Former AP delta attributable to the architecture rather than
to the surrounding pipeline.

See VENDORED.md for the file-by-file provenance and the full patch list.
"""
from . import modeling  # registers MaskFormerHead / MSDeformAttnPixelDecoder / decoders
from .config import add_mask2former_config
from .maskformer_model import MaskFormer

__all__ = ["MaskFormer", "add_mask2former_config"]
