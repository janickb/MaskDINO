# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import os

import h5py

from detectron2.data import DatasetCatalog, MetadataCatalog

_PREDEFINED_SPLITS = {
    # name: (dirname, fraction of files to actually use)
    "surgical_tools_train": ("surgical_tools/train", 1.0),
    "surgical_tools_val": ("surgical_tools/val", 0.25),
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
            height, width = f["colors"].shape[:2]
            annotations = json.loads(f["coco_annotations"][()])
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
