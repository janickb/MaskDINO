# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import os

import h5py

from detectron2.data import DatasetCatalog, MetadataCatalog
from sgdata import schema

_PREDEFINED_SPLITS = {
    # name: (dirname, fraction of files to actually use)
    # 15k/5k versatile-tools dataset (scene_generator run 20260808*) - absolute
    # paths, so os.path.join(root, dirname) below returns them as-is regardless
    # of DETECTRON2_DATASETS/root.
    "surgical_tools_train": ("/home/janick.bilang/dev/scene_generator/output/20260808203602_1024x1024_train", 1.0),
    "surgical_tools_val": ("/home/janick.bilang/dev/scene_generator/output/20260808210656_1024x1024_valid", 0.25),
}


def list_hdf5_dicts(hdf5_dir, fraction=1.0):
    paths = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))
    if fraction < 1.0:
        # Evenly spaced stride so the kept files span the whole directory
        # (filenames are grouped by simulation run) rather than clustering
        # at the start. Files on disk are untouched; the rest are just not
        # read for this split.
        stride = round(1 / fraction)
        paths = paths[::stride]

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


def register_hdf5_instances(name, hdf5_dir, fraction=1.0):
    DatasetCatalog.register(name, lambda: list_hdf5_dicts(hdf5_dir, fraction))
    MetadataCatalog.get(name).set(thing_classes=["surgical_tool"], evaluator_type="coco")


def register_all_hdf5_instances(root):
    for key, (dirname, fraction) in _PREDEFINED_SPLITS.items():
        register_hdf5_instances(key, os.path.join(root, dirname), fraction)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hdf5_instances(_root)
