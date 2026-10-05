"""The two-architecture config contract.

maskdino.config.arch_ns lets one train_net.py and one HungarianInstanceEvaluator drive
both meta-architectures by resolving MODEL.META_ARCHITECTURE to the MODEL.<X> node that
holds this fork's custom keys. These tests pin that contract, and - just as important -
that adding Mask2Former's namespace did not move any key outside it.
"""
import glob

import pytest
from detectron2.config import CfgNode as CN

from maskdino import add_surgical_arch_config, arch_ns, build_base_cfg

# Every key add_surgical_arch_config injects, with its documented default. A change here
# should be deliberate: these are read by train_net.py and the Hungarian evaluator, and
# every archived runs/*/config.yaml carries them under MODEL.MaskDINO.
SHARED_DEFAULTS = {
    "TEST.NMS_IOU": 0.0,
    "TEST.HUNGARIAN_EVAL.ENABLED": False,
    "TEST.HUNGARIAN_EVAL.SCORE_THRESH": 0.5,
    "TEST.HUNGARIAN_EVAL.IOU_THRESH": 0.5,
    "TEST.HUNGARIAN_EVAL.BOX_PREFILTER": True,
    "TEST.HUNGARIAN_EVAL.PERIOD": 0,
    "TEST.VAL_LOSS.ENABLED": False,
    "RECLASSIFY_FINETUNE.ENABLED": False,
    "RECLASSIFY_FINETUNE.TRAINABLE_PARAM_PREFIXES": ["sem_seg_head.predictor.class_embed"],
    "RECLASSIFY_FINETUNE.UNFREEZE_DECODER": False,
    "RECLASSIFY_FINETUNE.UNFREEZE_ENCODER": False,
    "RECLASSIFY_FINETUNE.ENCODER_LR_MULTIPLIER": 0.1,
}


def _get(node, dotted):
    for part in dotted.split("."):
        node = node[part]
    return node


def test_arch_ns_resolves_both_architectures():
    cfg = build_base_cfg()
    cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
    assert arch_ns(cfg) is cfg.MODEL.MaskDINO
    cfg.MODEL.META_ARCHITECTURE = "MaskFormer"
    assert arch_ns(cfg) is cfg.MODEL.MASK_FORMER


@pytest.mark.parametrize("bad", ["", "maskdino", "Mask2Former", "Nonsense"])
def test_arch_ns_raises_on_unknown_architecture(bad):
    """Must raise rather than defaulting: a config that silently resolved to the wrong
    namespace would read HUNGARIAN_EVAL.ENABLED=False and VAL_LOSS.ENABLED=False and
    produce a whole run with no confusion matrix and no validation loss."""
    cfg = build_base_cfg()
    cfg.MODEL.META_ARCHITECTURE = bad
    with pytest.raises(KeyError):
        arch_ns(cfg)


@pytest.mark.parametrize("dotted,expected", sorted(SHARED_DEFAULTS.items()))
def test_shared_defaults_are_identical_in_both_namespaces(dotted, expected):
    cfg = build_base_cfg()
    a = _get(cfg.MODEL.MaskDINO, dotted)
    b = _get(cfg.MODEL.MASK_FORMER, dotted)
    assert a == expected, f"MaskDINO.{dotted} default changed: {a!r}"
    assert b == expected, f"MASK_FORMER.{dotted} default changed: {b!r}"


def test_add_surgical_arch_config_is_self_contained():
    """It must only need an existing .TEST node, so a third architecture can opt in."""
    node = CN()
    node.TEST = CN()
    add_surgical_arch_config(node)
    for dotted, expected in SHARED_DEFAULTS.items():
        assert _get(node, dotted) == expected


def test_add_mask2former_config_does_not_touch_shared_keys():
    """Upstream's add_maskformer2_config would re-default these to Mask2Former's own
    values, changing behaviour for every MaskDINO config that relies on a default.
    The trimmed add_mask2former_config must not."""
    cfg = build_base_cfg()
    assert cfg.INPUT.DATASET_MAPPER_NAME == "MaskDINO_semantic"
    assert cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME == "MaskDINOEncoder"
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == -1
    assert cfg.MODEL.SEM_SEG_HEAD.MASK_DIM == 256
    assert cfg.MODEL.SWIN.DROP_PATH_RATE == 0.3
    assert cfg.MODEL.SWIN.OUT_FEATURES == ["res2", "res3", "res4", "res5"]
    assert cfg.INPUT.IMAGE_SIZE == 1024
    assert cfg.INPUT.MIN_SCALE == 0.1
    assert cfg.INPUT.MAX_SCALE == 2.0
    assert cfg.SOLVER.BACKBONE_MULTIPLIER == 0.1
    assert cfg.SOLVER.OPTIMIZER == "ADAMW"


def test_mask2former_namespace_exists_and_is_separate():
    cfg = build_base_cfg()
    assert "MASK_FORMER" in cfg.MODEL
    assert cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME == "MultiScaleMaskedTransformerDecoder"
    # MaskDINO-only keys must NOT have leaked into the Mask2Former namespace.
    for maskdino_only in ("DN", "TWO_STAGE", "INITIALIZE_BOX_TYPE", "BOX_WEIGHT"):
        assert maskdino_only not in cfg.MODEL.MASK_FORMER, maskdino_only


def test_every_shipped_config_merges():
    paths = sorted(glob.glob("configs/**/*.yaml", recursive=True))
    assert paths, "no configs found"
    for path in paths:
        cfg = build_base_cfg()
        cfg.merge_from_file(path)  # raises on an unknown key


def test_archived_run_configs_still_merge():
    """Every runs/*/config.yaml is the --eval-only entry point for a finished
    experiment, so adding the Mask2Former namespace must not break them.

    NOTE: 14 of these already fail on this fork for unrelated reasons - earlier commits
    renamed MODEL.MaskDINO.CLASSIFIER_RETRAIN to RECLASSIFY_FINETUNE (d26e97a), dropped
    RECLASSIFY_FINETUNE.DECODER_LR_MULTIPLIER (1597243) and INPUT.COPY_PASTE.
    CLUSTER_PROB. This test asserts only that the count does not GROW, so it catches a
    regression from the arch_ns work without pretending the pre-existing breakage is
    fixed. Re-evaluating those runs needs the removed keys re-added or passed via opts.
    """
    paths = sorted(glob.glob("runs/*/config.yaml"))
    if not paths:
        pytest.skip("no archived runs in this checkout")
    known_broken_keys = (
        "CLASSIFIER_RETRAIN",
        "DECODER_LR_MULTIPLIER",
        "CLUSTER_PROB",
    )
    unexpected = []
    for path in paths:
        cfg = build_base_cfg()
        try:
            cfg.merge_from_file(path)
        except Exception as exc:  # noqa: BLE001 - we classify it below
            if not any(k in str(exc) for k in known_broken_keys):
                unexpected.append((path, str(exc)[:160]))
    assert not unexpected, "archived configs newly broken:\n" + "\n".join(
        f"  {p}: {e}" for p, e in unexpected
    )
