#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch existing BlenderProc-style .hdf5 frames in place with a `coco_annotations` key
(see coco_hdf5_utils.py for the schema), derived from their `instance_segmaps` +
`instance_attribute_maps`. No new files are created — each .hdf5 file is modified
in place.

Usage:
    python datasets/backfill_coco_annotations.py <dir> [--overwrite] [--dry-run]
"""
import argparse
import glob
import os

import h5py

from coco_hdf5_utils import COCO_ANNOTATIONS_KEY, build_coco_annotations, write_coco_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory to (recursively) scan for *.hdf5 files")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite an existing coco_annotations key")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing anything")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, "**", "*.hdf5"), recursive=True))
    if not paths:
        print(f"No .hdf5 files found under {args.directory}")
        return

    for path in paths:
        with h5py.File(path, "r") as f:
            already_present = COCO_ANNOTATIONS_KEY in f
            if already_present and not args.overwrite:
                print(f"[skip]  {path} (already has {COCO_ANNOTATIONS_KEY})")
                continue
            annotations = build_coco_annotations(f["instance_segmaps"][()], f["instance_attribute_maps"][()])

        if args.dry_run:
            print(f"[would-write] {path}: {len(annotations)} instances")
            continue

        write_coco_annotations(path, annotations, overwrite=args.overwrite)
        print(f"[write] {path}: {len(annotations)} instances")


if __name__ == "__main__":
    main()
