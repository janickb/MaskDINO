# Copyright (c) Facebook, Inc. and its affiliates.
"""Live-pool counterpart of register_hdf5_instance.py: registers a "train_<set>"
split as a LivePoolDataset over sgdata's continuously-replenished image pool
(see scene_generator's src/sgdata/pool.py) instead of a static glob'd
directory. See register_hdf5_instance.py's _TRAIN_POOL_DIRS / register_all_hdf5_instances
for the set-name -> pool-dir table and the opt-in registration loop.
"""

import json
import logging
import time

import h5py
from detectron2.data import DatasetCatalog
from sgdata import pool, schema

from ..class_mapping import apply_class_mapping_to_metadata, derive_class_mapping
from .live_pool_dataset import LivePoolDataset

# "val" always stays static/pinned regardless of the pool - scoring against a
# shifting pool would make eval runs incomparable.
_POOL_VIRTUAL_SIZE = (
    1024  # order-of-magnitude match to scene_generator's image_pool.cap
)
_POOL_MIN_FILES = (
    50  # block dataset registration until the pool has at least this many frames
)


def pool_has_frames(pool_dir):
    """Non-blocking peek, used to skip registering a not-yet-rendered pool
    variant instead of paying _wait_for_one_frame's blocking wait for it."""
    return bool(pool.list_pool_frames(pool_dir))


def _wait_for_one_frame(pool_dir, timeout_s=1800.0):
    """Block until at least one frame exists in the pool, then return its path.
    thing_classes has to be set on the MetadataCatalog at registration time,
    and every frame carries the same `instrument_classes` list, so one is
    enough."""
    deadline = time.monotonic() + timeout_s
    last_warn = 0.0
    while True:
        frames = sorted(str(p) for p in pool.list_pool_frames(pool_dir))
        if frames:
            return frames[0]
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"no .hdf5 frames in pool {pool_dir} after {timeout_s:.0f}s"
            )
        now = time.monotonic()
        if now - last_warn > 30.0:
            logging.getLogger(__name__).warning(
                "[register_hdf5_pool_instances] waiting for first frame in pool %s",
                pool_dir,
            )
            last_warn = now
        time.sleep(2.0)


def register_hdf5_pool_instances(name, pool_dir, virtual_size, min_files):
    sample_path = _wait_for_one_frame(pool_dir)
    with h5py.File(sample_path, "r") as f:
        instrument_classes = json.loads(f[schema.INSTRUMENT_CLASSES][()])
    cm = derive_class_mapping(instrument_classes)
    DatasetCatalog.register(
        name,
        lambda: LivePoolDataset(
            pool_dir, virtual_size, min_files=min_files, class_mapping=cm
        ),
    )
    apply_class_mapping_to_metadata(name, cm)
