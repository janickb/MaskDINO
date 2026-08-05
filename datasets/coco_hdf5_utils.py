#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for embedding COCO-style instance annotations directly into BlenderProc-style
.hdf5 frames, as a `coco_annotations` dataset key (JSON string), instead of exporting
separate COCO JSON + image files.

Framework-agnostic on purpose (only numpy / h5py / pycocotools): this module is meant
to be copied into (or imported by) the data-generation script as well, so newly
generated .hdf5 files can carry the `coco_annotations` key from the start.

Schema written under the `coco_annotations` key (JSON array, one element per instance):
    {
        "bbox": [x, y, w, h],          # absolute pixel coords, XYWH
        "bbox_mode": 1,                 # detectron2 BoxMode.XYWH_ABS
        "category_id": 0,               # single class: "surgical_tool"
        "segmentation": {"size": [h, w], "counts": "<ascii RLE>"},
        "area": <float>,
        "iscrowd": 0,
    }
"""
import json

import h5py
import numpy as np
import pycocotools.mask as mask_util

COCO_ANNOTATIONS_KEY = "coco_annotations"


def build_coco_annotations(instance_segmaps: np.ndarray, instance_attribute_maps_json: bytes) -> list:
    """Turn one frame's `instance_segmaps` + `instance_attribute_maps` into a list of
    single-class COCO-style annotation dicts (RLE segmentation).

    Every entry in `instance_attribute_maps` whose `category_id != 0` is treated as a
    positive "surgical_tool" instance; `category_id == 0` (e.g. background/table
    "Plane") is skipped. Instances with an empty mask (e.g. fully occluded) are
    skipped too.
    """
    attributes = json.loads(instance_attribute_maps_json)

    annotations = []
    for entry in attributes:
        if entry["category_id"] == 0:
            continue

        mask = instance_segmaps == entry["idx"]
        if not mask.any():
            continue

        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        bbox = mask_util.toBbox(rle).tolist()
        area = float(mask_util.area(rle))

        annotations.append(
            {
                "bbox": bbox,
                "bbox_mode": 1,  # BoxMode.XYWH_ABS
                "category_id": 0,
                "segmentation": rle,
                "area": area,
                "iscrowd": 0,
            }
        )

    return annotations


def write_coco_annotations(h5_path: str, annotations: list, key: str = COCO_ANNOTATIONS_KEY, overwrite: bool = False) -> bool:
    """Write `annotations` into the .hdf5 file at `h5_path` under `key`, in place.

    Returns True if the file was written, False if `key` already existed and
    `overwrite` is False.
    """
    with h5py.File(h5_path, "a") as f:
        if key in f:
            if not overwrite:
                return False
            del f[key]
        f.create_dataset(key, data=json.dumps(annotations))
    return True
