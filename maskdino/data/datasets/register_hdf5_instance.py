# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import json
import logging
import os

import h5py
from detectron2.data import DatasetCatalog
from sgdata import schema
from sgdata.coco import build_coco_annotations

from ..class_mapping import (
    apply_class_mapping_to_metadata,
    derive_class_mapping,
    derive_single_class_mapping,
    remap_gt_category_ids,
)
from .images_root import ENV_VAR, resolve_images_root
from .register_hdf5_pool_instance import (
    _POOL_MIN_FILES,
    _POOL_VIRTUAL_SIZE,
    pool_has_frames,
    register_hdf5_pool_instances,
)

# --- Data sets = folders ------------------------------------------------------
# Every subfolder of this server's images root (~/training/images_<server>, see
# images_root.py) is registered under its own folder name, so a config names the
# folder directly in DATASETS.TRAIN/TEST - adding a set is copying it there (or
# symlinking it in). Each folder registers twice:
#   <folder>              one class per instrument (derive_class_mapping)
#   <folder>_singleclass  same frames, every instrument collapsed onto one
#                         "surgical-instruments" class (derive_single_class_mapping)
# A folder whose name contains "_pool_" is a live sgdata image pool (scene_generator
# keeps replenishing it during training) and is registered as a LivePoolDataset;
# every other folder is a static set. A folder with no readable frames yet (still
# rendering/copying), or with unreadable ones, is skipped with a warning instead
# of blocking/crashing `import maskdino`.
_SINGLE_CLASS_SUFFIX = "_singleclass"
_POOL_MARKER = "_pool_"


def _first_readable_hdf5(hdf5_dir):
    for p in sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5"))):
        if os.path.getsize(p) > 0:
            return p
    return None


_dataset_dicts_cache = {}


def _cached(name, build_fn):
    """Wrap a zero-arg dataset-dict builder so DatasetCatalog.get(name) only pays the
    HDF5 metadata scan once per process, instead of once per caller (test-loader build,
    COCOEvaluator's json conversion, HungarianInstanceEvaluator's own GT index) on every
    single periodic eval tick - DatasetCatalog.get() is not memoized by detectron2."""

    def get():
        if name not in _dataset_dicts_cache:
            _dataset_dicts_cache[name] = build_fn()
        return _dataset_dicts_cache[name]

    return get


def apply_test_sample_stride(cfg):
    """Re-register each cfg.DATASETS.TEST dataset to keep only every
    cfg.DATASETS.TEST_SAMPLE_STRIDE-th entry (sorted/cached order), in place under the
    SAME name - every consumer (test loader, COCOEvaluator, HungarianInstanceEvaluator)
    resolves GT purely via DatasetCatalog.get(dataset_name), so this is the one place
    that needs to apply the stride for it to take effect everywhere. No-op when stride
    <= 1 (the default)."""
    stride = cfg.DATASETS.TEST_SAMPLE_STRIDE
    if stride <= 1:
        return
    for name in cfg.DATASETS.TEST:
        full = DatasetCatalog.get(name)
        sampled = full[::stride]
        DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda sampled=sampled: sampled)


def touches_frame_edge(x1, y1, x2, y2, height, width):
    """True if box [x1, y1, x2, y2] (XYXY, absolute pixels) touches the frame
    boundary."""
    return x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height


def _is_truncated(ann, height, width):
    """True if ann's bbox touches the frame edge - bbox is a tight fit around
    the segmentation mask (sgdata.coco.build_coco_annotations)."""
    x, y, w, h = ann["bbox"]
    return touches_frame_edge(x, y, x + w, y + h, height, width)


def apply_truncated_instance_filter(cfg):
    """Re-register each cfg.DATASETS.TEST dataset with frame-border-truncated GT
    instances marked iscrowd=1, when cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES is set.

    iscrowd turns a cut tool into an IGNORE region rather than deleting it: neither
    a miss when undetected nor a false positive when detected. COCOEvaluator gets
    that from pycocotools' crowd handling (detectron2's COCO conversion carries the
    flag), HungarianInstanceEvaluator from its own crowd handling. Deleting the GT
    instead left every detection of a cut tool with nothing to match, i.e. a false
    positive - and the higher the zoom, the more tools the frame edge cuts. The
    is_train mappers skip iscrowd annotations, so the validation loss still never
    sees them."""
    if not cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES:
        return
    for name in cfg.DATASETS.TEST:
        full = DatasetCatalog.get(name)
        marked = []
        for d in full:
            h, w = d["height"], d["width"]
            cut = [_is_truncated(a, h, w) for a in d["annotations"]]
            if not any(cut):
                marked.append(d)
                continue
            # copy, never mutate: the source dicts are _cached and shared
            marked.append(
                {
                    **d,
                    "annotations": [
                        {**a, "iscrowd": 1} if c else a
                        for a, c in zip(d["annotations"], cut)
                    ],
                }
            )
        DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda marked=marked: marked)


def list_hdf5_dicts(hdf5_dir, cm):
    """One dataset dict per frame in `hdf5_dir`, raw `category_id`s remapped to
    `cm`'s contiguous label space. 0-byte / unreadable frames are skipped with a
    warning. Frames without `coco_annotations` (scene_generator's classification
    mode omits them) get them built on the fly from instance_segmaps +
    instance_attribute_maps (same output as sgdata.backfill)."""
    log = logging.getLogger(__name__)
    dicts = []
    for path in sorted(glob.glob(os.path.join(hdf5_dir, "*.hdf5"))):
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
            log.warning("skipping unreadable frame %s: %s", path, exc)
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
        log.warning("0 usable frames from %s", hdf5_dir)
    return dicts


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
    with h5py.File(_first_readable_hdf5(hdf5_dir), "r") as f:
        return json.loads(f[schema.INSTRUMENT_CLASSES][()])


def register_hdf5_instances(name, hdf5_dir, class_mapping_fn=derive_class_mapping):
    cm = class_mapping_fn(instrument_classes_from_hdf5(hdf5_dir))
    DatasetCatalog.register(name, _cached(name, lambda: list_hdf5_dicts(hdf5_dir, cm)))
    apply_class_mapping_to_metadata(name, cm)


def register_all_hdf5_instances(root):
    """Registers every subfolder of `root` under its folder name, plus its
    _singleclass view (see "Data sets = folders" at the top). Train/val pairing
    needs no table either: each set carries its own instrument_classes, and
    train_net.setup() calls assert_train_test_class_mapping_consistent() to fail
    fast if a config pairs two sets whose class mappings don't match."""
    log = logging.getLogger(__name__)
    for folder in sorted(os.listdir(root)):
        hdf5_dir = os.path.join(root, folder)
        if not os.path.isdir(hdf5_dir):
            continue
        is_pool = _POOL_MARKER in folder
        if not (pool_has_frames(hdf5_dir) if is_pool else _first_readable_hdf5(hdf5_dir)):
            log.warning(
                "[%s] no readable frames under %s; skipping registration", folder, hdf5_dir
            )
            continue
        try:
            for name, class_mapping_fn in (
                (folder, derive_class_mapping),
                (folder + _SINGLE_CLASS_SUFFIX, derive_single_class_mapping),
            ):
                if is_pool:
                    register_hdf5_pool_instances(
                        name, hdf5_dir, _POOL_VIRTUAL_SIZE, _POOL_MIN_FILES, class_mapping_fn
                    )
                else:
                    register_hdf5_instances(name, hdf5_dir, class_mapping_fn)
        except (OSError, KeyError, ValueError) as exc:
            log.warning("[%s] skipping registration, unreadable set: %s", folder, exc)


_root = resolve_images_root()
if _root is None:
    logging.getLogger(__name__).warning(
        "no ~/training/images_* dir and %s unset; skipping surgical dataset "
        "registration",
        ENV_VAR,
    )
else:
    register_all_hdf5_instances(_root)
