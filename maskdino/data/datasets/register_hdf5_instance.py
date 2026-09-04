# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import os

import h5py

from detectron2.data import DatasetCatalog, MetadataCatalog
from sgdata import schema

from .register_hdf5_pool_instance import (
    _POOL_DIR,
    _POOL_MIN_FILES,
    _POOL_VIRTUAL_SIZE,
    register_hdf5_pool_instances,
)

_PREDEFINED_SPLITS = {
    # name: dirname
    # absolute paths, so os.path.join(root, dirname) below returns them as-is regardless
    # of DETECTRON2_DATASETS/root
    "train": "/home/janick.bilang/dev/scene_generator/output/20260808203602_1024x1024_train",
    "val": "/home/janick.bilang/dev/scene_generator/output/20260808210656_1024x1024_valid",
}


def list_hdf5_dicts(hdf5_dir):
    paths = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))

    dataset_dicts = []
    for image_id, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            height, width = f[schema.COLORS].shape[:2]
            annotations = json.loads(f[schema.COCO_ANNOTATIONS][()])
        dataset_dicts.append(
            {
                "file_name": path,
                "image_id": image_id,
                "height": height,
                "width": width,
                "annotations": annotations,
            }
        )
    return dataset_dicts


def instrument_classes_from_hdf5(hdf5_dir):
    """category_id -> obj_name, read straight from any one .hdf5 file's embedded
    `instrument_classes` key (see sgdata/coco.py's instrument_classes_from_config,
    written at render time - run sgdata.backfill over older directories that
    predate this). Every file in a directory carries the same list, so reading one
    is enough.

    Used directly as the class index (no COCO-JSON-style remap happens in this
    custom loader): index 0 is background (filtered out of every frame's
    coco_annotations, never has ground truth), and any category_id with no live
    config.yaml `objects:` entry at generation time gets a placeholder name since
    it never spawns in this dataset but still needs to occupy its slot.
    """
    sample_path = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))[0]
    with h5py.File(sample_path, "r") as f:
        return json.loads(f[schema.INSTRUMENT_CLASSES][()])


def register_hdf5_instances(name, hdf5_dir):
    DatasetCatalog.register(name, lambda: list_hdf5_dicts(hdf5_dir))
    thing_classes = instrument_classes_from_hdf5(hdf5_dir)
    MetadataCatalog.get(name).set(thing_classes=thing_classes, evaluator_type="coco")


def register_all_hdf5_instances(root):
    for key, dirname in _PREDEFINED_SPLITS.items():
        if key == "train" and _POOL_DIR:
            register_hdf5_pool_instances(
                key, _POOL_DIR, _POOL_VIRTUAL_SIZE, _POOL_MIN_FILES
            )
        else:
            register_hdf5_instances(key, os.path.join(root, dirname))


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hdf5_instances(_root)
