"""Unit tests for apply_truncated_instance_filter / _is_truncated in
maskdino/data/datasets/register_hdf5_instance.py.

Same style as tests/test_hungarian_instance_evaluation.py: a real `import
maskdino` (registers real datasets from absolute data paths, tolerating
missing/unready sources), throwaway DatasetCatalog entries for the actual
test logic.
"""
import types
import uuid

import pytest
from detectron2.data import DatasetCatalog

from maskdino.data.datasets.register_hdf5_instance import (
    _is_truncated,
    apply_truncated_instance_filter,
    touches_frame_edge,
)


def _ann(bbox):
    return {"bbox": list(bbox), "bbox_mode": 1, "category_id": 0, "iscrowd": 0}


# --------------------------------------------------------------------------- _is_truncated


@pytest.mark.parametrize(
    "bbox",
    [
        (0.0, 5.0, 10.0, 10.0),  # touches left
        (5.0, 0.0, 10.0, 10.0),  # touches top
        (90.0, 5.0, 10.0, 10.0),  # touches right (90 + 10 == width)
        (5.0, 90.0, 10.0, 10.0),  # touches bottom (90 + 10 == height)
    ],
)
def test_is_truncated_edge_cases(bbox):
    assert _is_truncated(_ann(bbox), height=100, width=100) is True


def test_is_truncated_interior_bbox_is_not_truncated():
    assert _is_truncated(_ann((10.0, 10.0, 20.0, 20.0)), height=100, width=100) is False


# --------------------------------------------------------------------------- touches_frame_edge (XYXY, shared by GT + predictions)


def test_touches_frame_edge_xyxy_interior_and_edge():
    assert touches_frame_edge(10.0, 10.0, 30.0, 30.0, height=100, width=100) is False
    assert touches_frame_edge(0.0, 10.0, 30.0, 30.0, height=100, width=100) is True
    assert touches_frame_edge(10.0, 10.0, 100.0, 30.0, height=100, width=100) is True


# --------------------------------------------------------------------------- apply_truncated_instance_filter


def _fake_cfg(dataset_name, exclude_truncated):
    return types.SimpleNamespace(
        DATASETS=types.SimpleNamespace(TEST=(dataset_name,)),
        INPUT=types.SimpleNamespace(EXCLUDE_TRUNCATED_INSTANCES=exclude_truncated),
    )


@pytest.fixture
def registered_test_dataset():
    name = f"_test_truncfilter_{uuid.uuid4().hex}"
    dicts = [
        {
            "file_name": "a.hdf5",
            "image_id": 0,
            "height": 100,
            "width": 100,
            "annotations": [
                _ann((10.0, 10.0, 20.0, 20.0)),  # interior - kept
                _ann((0.0, 5.0, 10.0, 10.0)),  # touches left edge - dropped
            ],
        },
        {
            "file_name": "b.hdf5",
            "image_id": 1,
            "height": 100,
            "width": 100,
            "annotations": [
                _ann((5.0, 90.0, 10.0, 10.0)),  # touches bottom edge - dropped
            ],
        },
    ]
    DatasetCatalog.register(name, lambda: dicts)
    yield name, dicts
    DatasetCatalog.remove(name)


def test_apply_truncated_instance_filter_drops_edge_touching_annotations(
    registered_test_dataset,
):
    name, original = registered_test_dataset
    cfg = _fake_cfg(name, exclude_truncated=True)
    apply_truncated_instance_filter(cfg)

    result = DatasetCatalog.get(name)
    assert [len(d["annotations"]) for d in result] == [1, 0]
    assert result[0]["annotations"][0]["bbox"] == [10.0, 10.0, 20.0, 20.0]
    # image-level fields untouched
    assert result[0]["file_name"] == "a.hdf5"
    assert result[0]["height"] == 100 and result[0]["width"] == 100
    assert result[1]["file_name"] == "b.hdf5"


def test_apply_truncated_instance_filter_is_noop_when_disabled(registered_test_dataset):
    name, original = registered_test_dataset
    cfg = _fake_cfg(name, exclude_truncated=False)
    apply_truncated_instance_filter(cfg)

    result = DatasetCatalog.get(name)
    assert result is original  # never re-registered at all
