#!/usr/bin/env python
"""
Convert a detectron2 COCOEvaluator prediction cache (<OUTPUT_DIR>/inference/
instances_predictions.pth, written by every MaskDINO or Mask2Former eval) into the JSON
Lines file tools/score_predictions.py reads, so detectron2 runs are scored through the
same path as models from other frameworks.

The cache keys frames by image_id - the frame's index in the registered dataset
(register_hdf5_instance.list_hdf5_dicts) - and classes by contiguous id (it is saved
before COCOEvaluator maps them back to dataset ids). Both are resolved against
--dataset as it is registered now, so its folder must hold the same frames as when the
eval ran, and a single-class run needs the *_singleclass name. If the run directory
(the parent of inference/) holds the class_mapping.json every training run writes, its
classes must equal the dataset's.

Usage:
    ./.venv/bin/python tools/export_predictions_jsonl.py \
        --predictions runs/<run>/inference/instances_predictions.pth \
        --dataset val_set_a_mm \
        --output runs/<run>/predictions/val_set_a_mm.jsonl
"""
import argparse
import json
import os
import sys

import torch
from detectron2.data import DatasetCatalog, MetadataCatalog

# repo root, so `maskdino` imports when this runs by path from any directory
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maskdino import ClassMapping
from maskdino.data.class_mapping import CLASS_MAPPING_FILENAME


def _check_run_classes(predictions, thing_classes):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(predictions)))
    path = os.path.join(run_dir, CLASS_MAPPING_FILENAME)
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        run_classes = ClassMapping.from_json_dict(json.load(fh)).thing_classes
    if run_classes != thing_classes:
        raise ValueError(
            f"{path} has classes {run_classes}, the dataset has {thing_classes}; "
            f"wrong dataset (or its *_singleclass view)?"
        )


def export(predictions, dataset, output):
    """Write `output` (JSON Lines) from the COCOEvaluator cache `predictions` ->
    the number of frames written."""
    if dataset not in DatasetCatalog.list():
        raise ValueError(f"dataset {dataset!r} is not registered")
    frames = {d["image_id"]: d for d in DatasetCatalog.get(dataset)}
    thing_classes = list(MetadataCatalog.get(dataset).thing_classes)
    _check_run_classes(predictions, thing_classes)

    cache = torch.load(predictions, map_location="cpu", weights_only=False)
    records = {}
    for p in cache:
        image_id = p["image_id"]
        d = frames.get(image_id)
        if d is None:
            raise ValueError(f"image_id {image_id} is not in {dataset} ({len(frames)} frames)")
        if image_id in records:
            raise ValueError(f"image_id {image_id} appears twice in {predictions}")
        h, w = d["height"], d["width"]
        instances = []
        for det in p.get("instances", []):
            cid = int(det["category_id"])
            if not 0 <= cid < len(thing_classes):
                raise ValueError(
                    f"image_id {image_id}: category_id {cid} is outside {dataset}'s "
                    f"{len(thing_classes)} classes"
                )
            seg = det["segmentation"]
            if list(seg["size"]) != [h, w]:
                raise ValueError(
                    f"image_id {image_id}: mask size {list(seg['size'])}, frame is [{h}, {w}]"
                )
            counts = seg["counts"]
            instances.append(
                {
                    "class_name": thing_classes[cid],
                    "score": float(det["score"]),
                    "segmentation": {
                        "size": [h, w],
                        "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
                    },
                }
            )
        records[image_id] = {
            "file_name": os.path.basename(d["file_name"]),
            "height": h,
            "width": w,
            "instances": instances,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as fh:
        for image_id in sorted(records):
            fh.write(json.dumps(records[image_id]) + "\n")
    return len(records)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", required=True, help="inference/instances_predictions.pth")
    ap.add_argument("--dataset", required=True, help="the dataset the eval ran on")
    ap.add_argument("--output", required=True, help="JSON Lines file to write")
    args = ap.parse_args()
    try:
        n = export(args.predictions, args.dataset, args.output)
    except ValueError as exc:
        sys.exit(f"export_predictions_jsonl: {exc}")
    print(f"wrote {n} frames to {args.output}")


if __name__ == "__main__":
    main()
