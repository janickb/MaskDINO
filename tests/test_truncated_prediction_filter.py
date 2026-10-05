"""Unit tests for maskdino/evaluation/truncated_prediction_filter.py - the
prediction-side counterpart of apply_truncated_instance_filter.
"""
import torch
from detectron2.structures import Boxes, Instances

from maskdino.evaluation.truncated_prediction_filter import DropTruncatedPredictions


def _instances(boxes):
    inst = Instances((100, 100))
    inst.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
    inst.scores = torch.ones(len(boxes))
    return inst


def test_drop_truncated_keeps_interior_drops_edge_touching():
    out = {
        "instances": _instances(
            [
                [10.0, 10.0, 20.0, 20.0],  # interior - kept
                [0.0, 5.0, 10.0, 10.0],  # touches left - dropped
                [90.0, 5.0, 100.0, 10.0],  # touches right - dropped
            ]
        )
    }
    filtered = DropTruncatedPredictions._drop_truncated(
        {"height": 100, "width": 100}, out
    )
    kept = filtered["instances"].pred_boxes.tensor
    assert kept.shape == (1, 4)
    assert kept[0].tolist() == [10.0, 10.0, 20.0, 20.0]


def test_drop_truncated_noop_without_instances_key():
    out = {"sem_seg": torch.zeros(1)}
    assert DropTruncatedPredictions._drop_truncated({"height": 10, "width": 10}, out) is out


def test_drop_truncated_noop_on_empty_instances():
    out = {"instances": _instances([])}
    filtered = DropTruncatedPredictions._drop_truncated(
        {"height": 100, "width": 100}, out
    )
    assert filtered is out


# ---------------------------------------------------------------------------
# Mask2Former arm: this filter reads instances.pred_boxes, and Mask2Former has no box
# branch. The two tests below pin why mask2former/maskformer_model.py must derive
# pred_boxes from the predicted masks.
# ---------------------------------------------------------------------------


def _m2f_instances(masks):
    """Mask2Former-shaped Instances: masks + scores + classes, with pred_boxes derived
    from the masks exactly as the patched instance_inference does."""
    from detectron2.structures import BitMasks

    m = torch.stack(masks).bool()
    inst = Instances((100, 100))
    inst.pred_masks = m.float()
    inst.scores = torch.ones(len(masks))
    inst.pred_classes = torch.zeros(len(masks), dtype=torch.int64)
    inst.pred_boxes = BitMasks(m).get_bounding_boxes()
    return inst


def test_mask_derived_boxes_keep_interior_and_drop_edge_touching():
    interior = torch.zeros(100, 100, dtype=torch.bool)
    interior[20:40, 20:40] = True
    edge = torch.zeros(100, 100, dtype=torch.bool)
    edge[0:20, 30:50] = True  # touches the top edge

    out = {"instances": _m2f_instances([interior, edge])}
    filtered = DropTruncatedPredictions._drop_truncated(
        {"height": 100, "width": 100}, out
    )
    assert len(filtered["instances"]) == 1, (
        "mask-derived boxes should behave exactly like MaskDINO's learned boxes here"
    )
    assert filtered["instances"].pred_masks[0].bool().equal(interior.float().bool())


def test_zero_pred_boxes_drop_every_prediction():
    """Regression guard for the AP-0.0 blocker.

    Upstream Mask2Former's instance_inference sets pred_boxes to Boxes(zeros(N, 4)).
    A [0, 0, 0, 0] box touches both the left and the top edge, so with
    INPUT.EXCLUDE_TRUNCATED_INSTANCES True - the DEFAULT - this filter drops EVERY
    prediction and the run reports segm AP 0.0 with nothing in the log to explain it.
    That reads as "the architecture cannot do this task" rather than as a plumbing bug,
    which is why mask2former/maskformer_model.py derives real boxes from the masks.

    If this test ever starts failing because the filter changed, re-check that patch.
    """
    inst = Instances((100, 100))
    inst.pred_masks = torch.ones(3, 100, 100)
    inst.scores = torch.ones(3)
    inst.pred_classes = torch.zeros(3, dtype=torch.int64)
    inst.pred_boxes = Boxes(torch.zeros(3, 4))

    filtered = DropTruncatedPredictions._drop_truncated(
        {"height": 100, "width": 100}, {"instances": inst}
    )
    assert len(filtered["instances"]) == 0


def test_patched_m2f_inference_does_not_emit_zero_boxes():
    """The patch must still be in mask2former/maskformer_model.py - an upstream re-sync
    would silently reintroduce the zero boxes."""
    import inspect

    from mask2former.maskformer_model import MaskFormer

    src = inspect.getsource(MaskFormer.instance_inference)
    assert "get_bounding_boxes()" in src
    assert "Boxes(torch.zeros(" not in src
