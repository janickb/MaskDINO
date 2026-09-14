# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import logging
import os
import re

import h5py

from detectron2.data import DatasetCatalog
from sgdata import schema
from sgdata.coco import build_coco_annotations

from ..class_mapping import (
    apply_class_mapping_to_metadata,
    derive_class_mapping,
    remap_gt_category_ids,
)
from .register_hdf5_pool_instance import (
    _POOL_MIN_FILES,
    _POOL_VIRTUAL_SIZE,
    pool_has_frames,
    register_hdf5_pool_instances,
)

# --- Named data-set variants (e.g. seta/setb) ----------------------------------
# Each key here is the exact name a config puts in DATASETS.TRAIN/TEST to select
# that variant - no code change needed to switch between existing ones. Add a
# new set by adding a dict entry (its name doesn't have to follow any pattern,
# e.g. a val-only combined set could just be VAL_DIRS["val_setab"] = ...). A
# variant with no rendered frames yet is skipped with a warning instead of
# blocking/crashing `import maskdino`.
_TRAIN_POOL_DIRS = {
    "train_seta": "/home/janick.bilang/training/images/pool_1024x1024_seta_train",
    "train_setb": "/home/janick.bilang/training/images/pool_1024x1024_setb_train",
}
_VAL_DIRS = {
    "val_seta": "/home/janick.bilang/training/images/20260808_1024x1024_valid_1000_seta",
    "val_setb": "/home/janick.bilang/training/images/20260808_1024x1024_valid_1000_setb",
}

# --- Phase-2 classifier-retrain splits ("reclassification mode") --------------
# One scene_generator classification-mode set holding BOTH instrument sets
# (deterministic single-instrument renders: 2 sides x 10 rotation steps = 20
# imgs/instrument, filenames "<instrument>_<A|B>_<NNN>deg.hdf5"). It already
# carries a unified `instrument_classes` list, so no merge / restamp. Classifier
# mode omits `coco_annotations`, so they are built on the fly from
# instance_segmaps + instance_attribute_maps (same output as sgdata.backfill).
# `reclass_train` / `reclass_val` are an angle-parity split of these frames
# (both A/B sides land in each half).
_RECLASS_DIR = "/home/janick.bilang/training/images/20260909_classification_setab"


def _deg_of(path):
    m = re.search(r"_(\d+)deg", os.path.basename(path))
    return int(m.group(1)) if m else None


def _first_readable_hdf5(hdf5_dir):
    for p in sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5"))):
        if os.path.getsize(p) > 0:
            return p
    return None


def _list_hdf5_dicts_filtered(hdf5_dir, cm, keep=None):
    """list_hdf5_dicts over one dir, optionally filtering file paths through
    `keep(path) -> bool`. 0-byte / unreadable frames are skipped with a warning.
    Frames without `coco_annotations` (classification mode omits them) get them
    built on the fly from instance_segmaps + instance_attribute_maps. Raw
    `category_id`s are remapped to `cm`'s contiguous label space."""
    log = logging.getLogger(__name__)
    dicts = []
    for path in sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5"))):
        if keep is not None and not keep(path):
            continue
        if os.path.getsize(path) == 0:
            continue
        try:
            with h5py.File(path, "r") as f:
                height, width = f[schema.COLORS].shape[:2]
                if schema.COCO_ANNOTATIONS in f:
                    annotations = json.loads(f[schema.COCO_ANNOTATIONS][()])
                else:
                    annotations = build_coco_annotations(
                        f[schema.INSTANCE_SEGMAPS][()],
                        f[schema.INSTANCE_ATTRIBUTE_MAPS][()],
                    )
        except (OSError, KeyError) as exc:
            log.warning("[reclass] skipping unreadable frame %s: %s", path, exc)
            continue
        remap_gt_category_ids(annotations, cm)
        dicts.append(
            {
                "file_name": path,
                "image_id": len(dicts),
                "height": height,
                "width": width,
                "annotations": annotations,
            }
        )
    if not dicts:
        log.warning("[reclass] 0 usable frames from %s (filter/keep too strict?)", hdf5_dir)
    return dicts


def register_hdf5_slice(name, hdf5_dir, keep=None):
    """register_hdf5_instances for one dir + an optional path filter."""
    with h5py.File(_first_readable_hdf5(hdf5_dir), "r") as f:
        instrument_classes = json.loads(f[schema.INSTRUMENT_CLASSES][()])
    cm = derive_class_mapping(instrument_classes)
    DatasetCatalog.register(name, lambda: _list_hdf5_dicts_filtered(hdf5_dir, cm, keep))
    apply_class_mapping_to_metadata(name, cm)


def register_reclass_splits():
    """Opt-in: only registers if _RECLASS_DIR already holds rendered frames, so a
    not-yet-rendered set can't break `import maskdino`."""
    log = logging.getLogger(__name__)
    if _first_readable_hdf5(_RECLASS_DIR) is None:
        log.warning(
            "[reclass] no readable frames under %s; skipping reclass_* dataset "
            "registration (set _RECLASS_DIR / render the set first)",
            _RECLASS_DIR,
        )
        return
    # angle steps are 0,36,72,...,324; %72==0 -> {0,72,144,216,288}, the other
    # five otherwise. Side (A/B) is independent of angle, so both land in each.
    even = lambda p: (_deg_of(p) or 0) % 72 == 0
    odd = lambda p: (_deg_of(p) or 0) % 72 != 0
    register_hdf5_slice("reclass_train", _RECLASS_DIR, keep=odd)
    register_hdf5_slice("reclass_val", _RECLASS_DIR, keep=even)


def list_hdf5_dicts(hdf5_dir, cm):
    paths = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))

    dataset_dicts = []
    for image_id, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            height, width = f[schema.COLORS].shape[:2]
            annotations = json.loads(f[schema.COCO_ANNOTATIONS][()])
        remap_gt_category_ids(annotations, cm)
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

    List index == canonical category_id: index 0 is background (filtered out of
    every frame's coco_annotations, never has ground truth), and any category_id
    with no live config.yaml `objects:` entry at generation time gets a
    `unused_<i>` placeholder. `derive_class_mapping` turns this into the compact
    contiguous label space the model actually trains on (background + every
    `unused_*` dropped); `remap_gt_category_ids` rewrites each annotation's raw
    `category_id` into that space at dataset-listing time.
    """
    sample_path = sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5")))[0]
    with h5py.File(sample_path, "r") as f:
        return json.loads(f[schema.INSTRUMENT_CLASSES][()])


def register_hdf5_instances(name, hdf5_dir):
    cm = derive_class_mapping(instrument_classes_from_hdf5(hdf5_dir))
    DatasetCatalog.register(name, lambda: list_hdf5_dicts(hdf5_dir, cm))
    apply_class_mapping_to_metadata(name, cm)


def register_all_hdf5_instances(root):
    """Registers every _TRAIN_POOL_DIRS/_VAL_DIRS entry under its dict key,
    skipping (with a warning) whichever variant has no rendered frames yet -
    e.g. a set that's still being generated - so that one not-yet-ready
    variant can't break `import maskdino` for the others."""
    log = logging.getLogger(__name__)
    for name, pool_dir in _TRAIN_POOL_DIRS.items():
        if not pool_has_frames(pool_dir):
            log.warning(
                "[%s] no frames yet in pool %s; skipping registration", name, pool_dir
            )
            continue
        register_hdf5_pool_instances(name, pool_dir, _POOL_VIRTUAL_SIZE, _POOL_MIN_FILES)
    for name, val_dirname in _VAL_DIRS.items():
        val_dir = os.path.join(root, val_dirname)
        if _first_readable_hdf5(val_dir) is None:
            log.warning(
                "[%s] no readable frames under %s; skipping registration", name, val_dir
            )
            continue
        register_hdf5_instances(name, val_dir)
    register_reclass_splits()


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hdf5_instances(_root)
