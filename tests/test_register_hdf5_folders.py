"""Unit tests for register_all_hdf5_instances in
maskdino/data/datasets/register_hdf5_instance.py: every subfolder of the images
root registers under its folder name (+ "_singleclass"), "_pool_" folders as live
pools, and a folder that isn't a usable set is skipped instead of failing.

Same style as tests/test_truncated_instance_filter.py: a real `import maskdino`,
then throwaway folders (uuid-prefixed, so they can't clash with real sets) under
tmp_path, removed from the catalogs again afterwards.
"""
import json
import uuid

import h5py
import numpy as np
import pytest
from detectron2.data import DatasetCatalog, MetadataCatalog
from sgdata import schema

from maskdino.data.datasets import register_hdf5_instance as reg

CLASSES = ["background", "forcep01_mm", "scissors01_mm"]


def _write_frame(path, annotations):
    with h5py.File(path, "w") as f:
        f[schema.COLORS] = np.zeros((4, 6, 3), dtype=np.uint8)
        f[schema.COCO_ANNOTATIONS] = json.dumps(annotations)
        f[schema.INSTRUMENT_CLASSES] = json.dumps(CLASSES)


def _ann(category_id):
    return {"bbox": [1.0, 1.0, 2.0, 2.0], "bbox_mode": 1, "category_id": category_id, "iscrowd": 0}


@pytest.fixture
def root(tmp_path):
    prefix = uuid.uuid4().hex[:8]
    yield tmp_path, prefix
    for name in list(DatasetCatalog.list()):
        if name.startswith(prefix):
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)


def test_static_folder_registers_under_its_name(root):
    tmp_path, prefix = root
    d = tmp_path / f"{prefix}_val_set_x_mm"
    d.mkdir()
    _write_frame(d / "a.hdf5", [_ann(1), _ann(2)])
    _write_frame(d / "b.hdf5", [_ann(2)])

    reg.register_all_hdf5_instances(str(tmp_path))

    dicts = DatasetCatalog.get(d.name)
    assert [x["file_name"] for x in dicts] == [str(d / "a.hdf5"), str(d / "b.hdf5")]
    assert (dicts[0]["height"], dicts[0]["width"]) == (4, 6)
    assert list(MetadataCatalog.get(d.name).thing_classes) == ["forcep01_mm", "scissors01_mm"]
    assert list(MetadataCatalog.get(d.name + "_singleclass").thing_classes) == [
        "surgical-instruments"
    ]
    # the single-class view lists the same frames, every instrument on class 0
    single = DatasetCatalog.get(d.name + "_singleclass")
    assert {a["category_id"] for x in single for a in x["annotations"]} == {0}


def test_pool_folder_goes_through_live_pool(root, monkeypatch):
    tmp_path, prefix = root
    pool = tmp_path / f"{prefix}_train_phase1_pool_x_mm"
    pool.mkdir()
    _write_frame(pool / "a.hdf5", [_ann(1)])
    calls = []
    monkeypatch.setattr(
        reg,
        "register_hdf5_pool_instances",
        lambda name, pool_dir, *a: calls.append((name, pool_dir)),
    )

    reg.register_all_hdf5_instances(str(tmp_path))

    assert calls == [
        (pool.name, str(pool)),
        (pool.name + "_singleclass", str(pool)),
    ]


def test_unusable_folders_are_skipped(root):
    tmp_path, prefix = root
    (tmp_path / f"{prefix}_empty").mkdir()  # still rendering
    no_classes = tmp_path / f"{prefix}_no_classes"
    no_classes.mkdir()
    with h5py.File(no_classes / "a.hdf5", "w") as f:  # predates instrument_classes
        f[schema.COLORS] = np.zeros((4, 6, 3), dtype=np.uint8)
    (tmp_path / f"{prefix}_notes.txt").write_text("")  # not a folder
    ok = tmp_path / f"{prefix}_ok"
    ok.mkdir()
    _write_frame(ok / "a.hdf5", [_ann(1)])

    reg.register_all_hdf5_instances(str(tmp_path))

    registered = {n for n in DatasetCatalog.list() if n.startswith(prefix)}
    assert registered == {ok.name, ok.name + "_singleclass"}
