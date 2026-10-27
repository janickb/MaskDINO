"""Guards the two traps in the evaluation path (see cropcls/export.py)."""

import pytest

from maskdino.evaluation.hungarian_instance_evaluation import HungarianInstanceEvaluator
from cropcls.export import _eval_cfg


def _detector_cfg():
    """A stock MaskDINO cfg as the single-class run's config.yaml resolves it."""
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config

    from maskdino import add_maskdino_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1      # the frontend is class-agnostic
    return cfg


def test_detector_cfg_would_silently_disable_misclassification():
    """The trap: scoring with the detector's own cfg reports every misclassified_* as 0.0."""
    cfg = _detector_cfg()
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1
    # This is the evaluator's own rule, reproduced so the test fails loudly if it changes.
    assert (cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES > 1) is False


def test_eval_cfg_clone_restores_the_class_count():
    cfg = _eval_cfg(_detector_cfg(), "val_seta_mm", num_classes=8, stride=4)
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 8, (
        "the eval cfg must carry the CLASSIFIER's class count, not the detector's 1, "
        "or HungarianInstanceEvaluator sets _check_classes=False and silently reports "
        "misclassification as 0.0"
    )
    assert (cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES > 1) is True
    assert cfg.DATASETS.TEST == ("val_seta_mm",)
    assert cfg.DATASETS.TEST_SAMPLE_STRIDE == 4


def test_eval_cfg_does_not_mutate_the_detector_cfg():
    base = _detector_cfg()
    _eval_cfg(base, "val_seta_mm", num_classes=8, stride=4)
    assert base.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1, "the detector's cfg must stay untouched"


def test_stride_helper_is_applied_at_most_once_per_dataset():
    """apply_test_sample_stride is not idempotent: a second call strides 4 -> 16."""
    import cropcls.export as export

    export.STRIDE_APPLIED.clear()
    export.STRIDE_APPLIED.add("val_seta_mm")
    assert "val_seta_mm" in export.STRIDE_APPLIED
    export.STRIDE_APPLIED.clear()
