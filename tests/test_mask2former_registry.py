"""Import-time contracts for the vendored Mask2Former package.

These are cheap but guard the two failure modes that are hardest to debug after an
upstream re-sync: a duplicate detectron2 registry key (hard crash at import) and a
second MSDeformAttn implementation that would reintroduce the CUDA-stream corruption
this fork works around.
"""
import ast
import inspect
import textwrap

import maskdino  # noqa: F401  - must be imported first; see mask2former/modeling/__init__.py
import mask2former  # noqa: F401


def test_both_packages_import_without_registry_collision():
    """maskdino/modeling/__init__.py registers D2SwinTransformer. Upstream
    mask2former/modeling/__init__.py registers the same key, which raises - hence the
    deleted backbone/ directory."""
    import importlib

    importlib.reload(mask2former.modeling)


def test_meta_architectures_registered():
    from detectron2.modeling import META_ARCH_REGISTRY

    for name in ("MaskDINO", "MaskFormer"):
        assert name in META_ARCH_REGISTRY, name


def test_sem_seg_heads_registered():
    from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

    for name in (
        "MaskDINOHead",
        "MaskDINOEncoder",
        "MaskFormerHead",
        "MSDeformAttnPixelDecoder",
    ):
        assert name in SEM_SEG_HEADS_REGISTRY, name


def test_transformer_decoder_registered():
    from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import (
        TRANSFORMER_DECODER_REGISTRY,
    )

    assert "MultiScaleMaskedTransformerDecoder" in TRANSFORMER_DECODER_REGISTRY


def test_msdeformattn_is_the_shared_patched_implementation():
    """mask2former must use maskdino's ops package, not a vendored second copy."""
    import maskdino.modeling.pixel_decoder.ops.modules as shared
    import mask2former.modeling.pixel_decoder.msdeformattn as m2f

    assert m2f.MSDeformAttn is shared.MSDeformAttn


def test_msdeformattn_dispatches_to_pure_pytorch_kernel():
    """The compiled CUDA kernel corrupts the CUDA stream on this torch/CUDA build - it
    surfaces as "invalid resource handle" on the next, unrelated CUDA call, so a
    try/except around the call site does not catch it. forward() must therefore call
    ms_deform_attn_core_pytorch unconditionally.

    Checks the parsed call graph rather than the raw source text, because the
    explanatory comment in forward() mentions MSDeformAttnFunction by name.
    """
    from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn

    tree = ast.parse(textwrap.dedent(inspect.getsource(MSDeformAttn.forward)))
    called = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "ms_deform_attn_core_pytorch" in called, called
    assert not any("MSDeformAttnFunction" in name for name in called), called


def test_no_second_ops_package_vendored():
    import os

    assert not os.path.exists("mask2former/modeling/pixel_decoder/ops"), (
        "a second MSDeformAttn ops/ tree would let someone run its make.sh and "
        "reintroduce the CUDA-stream corruption"
    )


def test_class_aware_mask_nms_is_one_shared_object():
    from maskdino.maskdino import class_aware_mask_nms as from_meta_arch
    from maskdino.modeling.postprocess import class_aware_mask_nms as canonical
    from mask2former.maskformer_model import class_aware_mask_nms as from_m2f

    assert canonical is from_meta_arch is from_m2f


def test_derived_pred_boxes_land_on_the_mask_device():
    """BitMasks.get_bounding_boxes() allocates with torch.zeros(N, 4) and no device, so
    it returns CPU boxes for CUDA masks. instance_inference must move them, or the
    class_aware_mask_nms slice indexes a CPU box tensor with a CUDA keep-mask and
    raises. Pinned statically so it is caught without a GPU."""
    import inspect

    from mask2former.maskformer_model import MaskFormer

    src = inspect.getsource(MaskFormer.instance_inference)
    line = next(l for l in src.splitlines() if "get_bounding_boxes()" in l)
    assert ".to(" in line or ".to(" in src.split("get_bounding_boxes()")[1][:120], (
        "derived pred_boxes must be moved onto the mask device"
    )
