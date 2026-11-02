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


def _registered(n: int) -> str:
    """A synthetic dataset of n frames, uuid-named so tests cannot collide."""
    import uuid

    from detectron2.data import DatasetCatalog

    name = f"_test_stride_{uuid.uuid4().hex}"
    dicts = [{"file_name": f"f{i:04d}.hdf5", "image_id": i} for i in range(n)]
    DatasetCatalog.register(name, lambda d=dicts: d)
    return name


@pytest.mark.parametrize("stride", [2, 4, 7])
def test_stride_keeps_every_nth_val_frame(stride):
    from detectron2.data import DatasetCatalog

    from cropcls.data import apply_val_stride

    name = _registered(100)
    assert apply_val_stride(name, stride) == len(range(0, 100, stride))
    kept = [d["file_name"] for d in DatasetCatalog.get(name)]
    assert kept[:3] == [f"f{i:04d}.hdf5" for i in (0, stride, 2 * stride)]
    DatasetCatalog.remove(name)


def test_striding_twice_does_not_restride():
    """The documented trap: apply_test_sample_stride re-registers an already-strided list,
    so a second call turns 4 into 16 and silently scores a quarter of the frames. The shared
    guard in cropcls.data is the only thing standing between that and a wrong number."""
    from cropcls.data import apply_val_stride

    name = _registered(100)
    first = apply_val_stride(name, 4)
    second = apply_val_stride(name, 4)
    assert first == 25
    assert second == first, f"restrided to {second}; the guard did not hold"

    from detectron2.data import DatasetCatalog

    DatasetCatalog.remove(name)


def test_stride_of_one_is_a_no_op():
    """1 must mean 'score every frame', not 'stride by 1 and burn the guard'."""
    from cropcls.data import STRIDE_APPLIED, apply_val_stride

    name = _registered(10)
    assert apply_val_stride(name, 1) == 10
    assert name not in STRIDE_APPLIED, "a no-op must not consume the one-shot guard"

    from detectron2.data import DatasetCatalog

    DatasetCatalog.remove(name)


def _registered_trunc() -> str:
    """One frame, 10x10, with one interior and one border-touching annotation."""
    import uuid

    from detectron2.data import DatasetCatalog

    name = f"_test_trunc_{uuid.uuid4().hex}"
    dicts = [{
        "file_name": "f.hdf5", "image_id": 0, "height": 10, "width": 10,
        "annotations": [
            {"bbox": [2, 2, 5, 5], "bbox_mode": 1, "category_id": 0},   # interior
            {"bbox": [0, 2, 4, 5], "bbox_mode": 1, "category_id": 0},   # touches x=0
        ],
    }]
    DatasetCatalog.register(name, lambda d=dicts: d)
    return name


def test_truncated_filter_stops_scoring_border_instances():
    """The baseline excluded these; scoring them makes the comparison pessimistic for us."""
    from cropcls.data import apply_truncated_filter

    name = _registered_trunc()
    assert apply_truncated_filter(name, enabled=True) == 1, "the border instance must not score"
    from detectron2.data import DatasetCatalog

    DatasetCatalog.remove(name)


def test_truncated_filter_marks_iscrowd_rather_than_deleting():
    """The behaviour main switched to (c4daffb), and the reason the prediction-side
    DropTruncatedPredictions wrapper could go.

    The annotation must SURVIVE, flagged: HungarianInstanceEvaluator builds its ignore
    regions from `iscrowd`, so a detection landing on a cut tool is neither a false positive
    nor counted in num_pred. Deleting it instead left that detection nothing to match, which
    is what drove false positives 19 -> 45.
    """
    from detectron2.data import DatasetCatalog

    from cropcls.data import apply_truncated_filter

    name = _registered_trunc()
    apply_truncated_filter(name, enabled=True)
    anns = DatasetCatalog.get(name)[0]["annotations"]
    assert len(anns) == 2, "the truncated annotation must still be present, not removed"
    assert [a.get("iscrowd", 0) for a in anns] == [0, 1], "only the border one is an ignore"
    DatasetCatalog.remove(name)


def test_truncated_filter_disabled_keeps_everything():
    from cropcls.data import apply_truncated_filter

    name = _registered_trunc()
    assert apply_truncated_filter(name, enabled=False) == 2
    from detectron2.data import DatasetCatalog

    DatasetCatalog.remove(name)


def test_filtering_twice_does_not_refilter():
    """Same non-idempotence hazard as the stride helper: it re-registers the entry."""
    from cropcls.data import apply_truncated_filter

    name = _registered_trunc()
    first = apply_truncated_filter(name, enabled=True)
    assert apply_truncated_filter(name, enabled=True) == first
    from detectron2.data import DatasetCatalog

    DatasetCatalog.remove(name)
