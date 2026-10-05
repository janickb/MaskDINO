"""Cross-architecture harness invariants for the MaskDINO vs Mask2Former comparison.

The whole point of vendoring Mask2Former into this repo behind one train_net.py is that
an AP delta between the two arms should be attributable to the architecture, not to the
surrounding pipeline. These tests turn that from a code-review promise into an enforced
invariant:

  * the two arms build the IDENTICAL augmentation list, and
  * their fully-resolved configs differ ONLY inside each architecture's own namespace
    (plus WEIGHTS / OUTPUT_DIR / the head module names), and
  * the mapper hands both arms byte-identical GT in the layout Mask2Former's
    prepare_targets requires.

If a test here fails, the fix is almost always the YAML, not the test.
"""
import glob
import os

import numpy as np
import pytest
import torch
import yaml

from maskdino import arch_ns, build_base_cfg
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    build_transform_gen,
)

CFG_DIR = "configs/coco/instance-segmentation"

# Config pairs that must describe the same experiment on the two architectures.
PAIRS = [
    pytest.param(
        "maskdino_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
        "maskformer2_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
        id="phase1-multiclass",
    ),
    pytest.param(
        "maskdino_R50_surgical_tools_reclassify",
        "maskformer2_R50_surgical_tools_reclassify",
        id="phase2-reclassify",
    ),
    pytest.param(
        "maskdino_R50_surgical_tools_finetune_singleclass",
        "maskformer2_R50_surgical_tools_finetune_singleclass",
        id="singleclass",
    ),
]

# Keys the two arms are ALLOWED to differ on. Everything else is harness.
#
# The two MODEL.<ARCH> namespaces are allowed because they hold genuinely
# architecture-specific knobs (loss weights, decoder depth, denoising) - but NOT
# wholesale: the sub-keys that add_surgical_arch_config injects into both namespaces are
# this fork's shared harness, so they are pulled back out below and compared.
_ALLOWED_PREFIXES = (
    "MODEL.META_ARCHITECTURE",
    "MODEL.WEIGHTS",
    "OUTPUT_DIR",
    "MODEL.MaskDINO",
    "MODEL.MASK_FORMER",
    # Head/pixel-decoder module names and the MaskDINO-only feature-level plumbing.
    "MODEL.SEM_SEG_HEAD.NAME",
    "MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME",
    "MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD",
    "MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS",
    "MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS",
    "MODEL.SEM_SEG_HEAD.FEATURE_ORDER",
)

# Injected into BOTH MODEL.<ARCH> namespaces by add_surgical_arch_config, so they are
# shared harness and must match even though they sit under an allowed prefix. An
# asymmetric TEST.NMS_IOU in particular directly moves precision: duplicates count as
# false positives in the Hungarian evaluator.
_SHARED_ARCH_SUFFIXES = (
    "TEST.NMS_IOU",
    "TEST.HUNGARIAN_EVAL.ENABLED",
    "TEST.HUNGARIAN_EVAL.SCORE_THRESH",
    "TEST.HUNGARIAN_EVAL.IOU_THRESH",
    "TEST.HUNGARIAN_EVAL.BOX_PREFILTER",
    "TEST.HUNGARIAN_EVAL.PERIOD",
    "TEST.VAL_LOSS.ENABLED",
    "RECLASSIFY_FINETUNE.ENABLED",
    "RECLASSIFY_FINETUNE.UNFREEZE_DECODER",
    "RECLASSIFY_FINETUNE.UNFREEZE_ENCODER",
    "RECLASSIFY_FINETUNE.ENCODER_LR_MULTIPLIER",
)


def _resolve(name):
    cfg = build_base_cfg()
    cfg.merge_from_file(os.path.join(CFG_DIR, f"{name}.yaml"))
    return cfg


def _flatten(cfg):
    """cfg -> {"A.B.C": value} via its own dump, so this sees exactly what gets archived."""
    out = {}

    def walk(node, prefix=""):
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, path + ".")
            else:
                out[path] = value

    walk(yaml.safe_load(cfg.dump()))
    return out


@pytest.mark.parametrize("maskdino_cfg,m2f_cfg", PAIRS)
def test_augmentation_list_is_identical(maskdino_cfg, m2f_cfg):
    """Both arms call the SAME build_transform_gen, so this guards against YAML drift in
    INPUT.{IMAGE_SIZE,MIN_SCALE,MAX_SCALE,RANDOM_ROTATION,ROTATION_ANGLES,ROTATION_EXPAND}
    - the most likely silent confound in the comparison."""
    a = [repr(t) for t in build_transform_gen(_resolve(maskdino_cfg), True)]
    b = [repr(t) for t in build_transform_gen(_resolve(m2f_cfg), True)]
    assert a == b, f"augmentation pipelines diverge:\n  maskdino={a}\n  m2f     ={b}"


@pytest.mark.parametrize("maskdino_cfg,m2f_cfg", PAIRS)
def test_resolved_configs_differ_only_by_architecture(maskdino_cfg, m2f_cfg):
    a, b = _flatten(_resolve(maskdino_cfg)), _flatten(_resolve(m2f_cfg))
    divergences = []
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key, "<absent>"), b.get(key, "<absent>")
        if av == bv:
            continue
        # Both MODEL.<ARCH> namespaces are allowed here in full. Comparing
        # MODEL.MaskDINO.* across the two FILES would be meaningless: each config
        # populates only its own architecture's namespace and leaves the other at
        # defaults. The fork-specific keys that live in both namespaces are compared
        # between each arm's ACTIVE namespace by
        # test_shared_arch_namespace_keys_match below.
        if key.startswith(_ALLOWED_PREFIXES):
            continue
        divergences.append(f"  {key}: maskdino={av!r} m2f={bv!r}")
    assert not divergences, (
        "harness divergence between the two arms - fix the YAML, not this test:\n"
        + "\n".join(divergences)
    )


@pytest.mark.parametrize("maskdino_cfg,m2f_cfg", PAIRS)
def test_shared_arch_namespace_keys_match(maskdino_cfg, m2f_cfg):
    """The fork-specific keys add_surgical_arch_config injects must carry the same values
    in both arms - they are harness that merely happens to live under MODEL.<ARCH>."""
    a, b = _resolve(maskdino_cfg), _resolve(m2f_cfg)
    for suffix in _SHARED_ARCH_SUFFIXES:
        av, bv = arch_ns(a), arch_ns(b)
        for part in suffix.split("."):
            av, bv = av[part], bv[part]
        assert av == bv, f"{suffix}: maskdino={av!r} m2f={bv!r}"


def test_arch_ns_resolves_for_every_surgical_config():
    """A config whose META_ARCHITECTURE has no registered namespace would silently lose
    the Hungarian evaluator and the validation-loss hook, so arch_ns must resolve."""
    names = sorted(glob.glob(os.path.join(CFG_DIR, "*surgical*.yaml")))
    assert names, "no surgical configs found"
    for path in names:
        cfg = build_base_cfg()
        cfg.merge_from_file(path)
        ns = arch_ns(cfg)
        assert "HUNGARIAN_EVAL" in ns.TEST, path


@pytest.mark.parametrize("maskdino_cfg,m2f_cfg", PAIRS)
def test_mapper_gt_is_identical_and_plain_tensor(maskdino_cfg, m2f_cfg, tmp_path):
    """Same frame through both arms' mappers must yield identical GT, and gt_masks must be
    a plain tensor - Mask2Former's prepare_targets slices gt_masks.shape[0..2] rather than
    going through BitMasks, so a BitMasks here would break it."""
    h5py = pytest.importorskip("h5py")
    import json

    from detectron2.structures import BoxMode
    from pycocotools import mask as mask_util

    from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
        Hdf5CocoInstanceDatasetMapper,
    )

    # One synthetic 64x64 frame with a single square instance.
    size = 64
    colors = np.zeros((size, size, 3), dtype=np.uint8)
    colors[16:48, 16:48] = 200
    seg = np.zeros((size, size), dtype=np.uint8)
    seg[16:48, 16:48] = 1
    rle = mask_util.encode(np.asfortranarray(seg))
    rle["counts"] = rle["counts"].decode("utf-8")
    anns = [
        {
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "segmentation": rle,
            "area": float(mask_util.area(rle)),
            "bbox": mask_util.toBbox(rle).tolist(),
            "bbox_mode": BoxMode.XYWH_ABS,
            "iscrowd": 0,
            "visibility_fraction": 1.0,
        }
    ]
    path = tmp_path / "frame.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("colors", data=colors)
        f.create_dataset("coco_annotations", data=json.dumps(anns))

    def run(cfg_name):
        cfg = _resolve(cfg_name)
        # Copy-paste needs the full dataset-dict list; not what this test is about.
        cfg.INPUT.COPY_PASTE.ENABLED = False
        mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
        np.random.seed(0)
        torch.manual_seed(0)
        return mapper(
            {
                "file_name": str(path),
                "image_id": 0,
                "height": size,
                "width": size,
                "annotations": anns,
            }
        )

    da, db = run(maskdino_cfg), run(m2f_cfg)
    ia, ib = da["instances"], db["instances"]
    assert torch.equal(ia.gt_classes, ib.gt_classes)
    assert torch.equal(ia.gt_masks, ib.gt_masks)
    assert torch.equal(da["image"], db["image"])
    for inst in (ia, ib):
        assert isinstance(inst.gt_masks, torch.Tensor), type(inst.gt_masks)
        assert inst.gt_masks.ndim == 3
