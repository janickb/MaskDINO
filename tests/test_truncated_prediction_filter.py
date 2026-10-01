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
