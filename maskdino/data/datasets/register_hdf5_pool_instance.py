# Copyright (c) Facebook, Inc. and its affiliates.
"""Live-pool counterpart of register_hdf5_instance.py: registers the "train"
split as a LivePoolDataset over sgdata's continuously-replenished image pool
(see scene_generator's src/sgdata/pool.py) instead of a static glob'd
directory. Opt-in via _POOL_DIR below - see register_hdf5_instance.py's
register_all_hdf5_instances for the dispatch between this and the static path.
"""

import json
import logging
import time

import h5py
from detectron2.data import DatasetCatalog, MetadataCatalog
from sgdata import pool, schema

from .live_pool_dataset import LivePoolDataset

# Live-pool training set (opt-in): set _POOL_DIR below to enable it - the
# "train" split is then registered as a LivePoolDataset over that
# continuously-replenished directory instead of the static path in
# register_hdf5_instance.py. None disables live-pool training entirely (the
# default). "val" always stays static/pinned regardless of this - scoring
# against a shifting pool would make eval runs incomparable.
_POOL_DIR = "/home/janick.bilang/training/images/pool_1024x1024_setb_train"
_POOL_VIRTUAL_SIZE = (
    1024  # order-of-magnitude match to scene_generator's image_pool.cap
)
_POOL_MIN_FILES = (
    50  # block dataset registration until the pool has at least this many frames
)


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
    DatasetCatalog.register(
        name,
        lambda: LivePoolDataset(pool_dir, virtual_size, min_files=min_files),
    )
    sample_path = _wait_for_one_frame(pool_dir)
    with h5py.File(sample_path, "r") as f:
        thing_classes = json.loads(f[schema.INSTRUMENT_CLASSES][()])
    MetadataCatalog.get(name).set(thing_classes=thing_classes, evaluator_type="coco")
