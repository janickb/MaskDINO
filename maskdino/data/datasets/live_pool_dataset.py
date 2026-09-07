"""LivePoolDataset - a map-style ``torch`` dataset over sgdata's live image pool
(a single capped directory of ``.hdf5`` frames that concurrent blenderproc
workers keep replenished). Registered in place of register_hdf5_instance's
static file list so MaskDINO trains against the pool as it churns, with no
restarts.

Why a Dataset and not a list: detectron2's ``get_detection_dataset_dicts`` and
``build_detection_train_loader`` pass a ``torch.utils.data.Dataset`` straight
through (no ``DatasetFromList`` freeze) and wrap it in the usual
``MapDataset`` + ``TrainingSampler``. So this class is the only new piece on the
read side - ``__len__`` gives the sampler a stable index space and
``__getitem__`` resolves an index against the *current* file list, re-scanned
every ``refresh_interval_s``.

DataLoader workers fork, so each worker inherits this object's file cache and
then refreshes it independently - no cross-worker coordination.
"""

from __future__ import annotations

import json
import logging
import random
import time

import h5py
import torch.utils.data
from sgdata import pool, schema

logger = logging.getLogger(__name__)


class LivePoolDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pool_dir,
        virtual_size,
        *,
        refresh_interval_s: float = 5.0,
        min_files: int = 50,
        min_files_timeout_s: float = 1800.0,
        max_read_retries: int = 5,
    ):
        self.pool_dir = str(pool_dir)
        self.virtual_size = int(virtual_size)
        self.refresh_interval_s = refresh_interval_s
        self.max_read_retries = max_read_retries

        self._cache: list[str] = []
        self._cache_time = 0.0
        self._wait_for_min_files(min_files, min_files_timeout_s)

    # -- pool file listing ---------------------------------------------------
    def _scan(self) -> list[str]:
        self._cache = [str(p) for p in pool.list_pool_frames(self.pool_dir)]
        self._cache_time = time.monotonic()
        return self._cache

    def _current_files(self) -> list[str]:
        if (
            not self._cache
            or (time.monotonic() - self._cache_time) >= self.refresh_interval_s
        ):
            self._scan()
        return self._cache

    def _wait_for_min_files(self, min_files: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_warn = 0.0
        while True:
            files = self._scan()
            if len(files) >= min_files:
                logger.info(
                    "[LivePoolDataset] pool %s ready: %d frames (>= %d)",
                    self.pool_dir,
                    len(files),
                    min_files,
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"[LivePoolDataset] pool {self.pool_dir} has {len(files)} frames "
                    f"after waiting {timeout_s:.0f}s for at least {min_files}"
                )
            now = time.monotonic()
            if now - last_warn > 30.0:
                logger.warning(
                    "[LivePoolDataset] waiting for pool %s to fill: %d/%d frames",
                    self.pool_dir,
                    len(files),
                    min_files,
                )
                last_warn = now
            time.sleep(2.0)

    # -- torch Dataset protocol -------------------------------------------
    def __len__(self) -> int:
        return self.virtual_size

    @staticmethod
    def _read_dict(path: str, image_id: int) -> dict:
        with h5py.File(path, "r") as f:
            height, width = f[schema.COLORS].shape[:2]
            annotations = json.loads(f[schema.COCO_ANNOTATIONS][()])
        return {
            "file_name": path,
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": annotations,
        }

    def __getitem__(self, idx: int) -> dict:
        last_exc: BaseException | None = None
        for attempt in range(self.max_read_retries):
            files = self._current_files()
            if not files:
                files = self._scan()
            if not files:
                last_exc = RuntimeError(f"pool {self.pool_dir} is empty")
                time.sleep(0.5)
                continue
            # First try maps the sampler index into the current listing; later
            # tries just pick another file, since the culprit was an evicted /
            # torn frame rather than a bad index.
            path = files[idx % len(files)] if attempt == 0 else random.choice(files)
            try:
                return self._read_dict(path, image_id=int(idx))
            except (OSError, KeyError, ValueError) as exc:
                last_exc = exc
                try:
                    self._cache.remove(path)
                except ValueError:
                    pass
        raise RuntimeError(
            f"[LivePoolDataset] could not read any frame from pool {self.pool_dir} "
            f"after {self.max_read_retries} attempts"
        ) from last_exc
