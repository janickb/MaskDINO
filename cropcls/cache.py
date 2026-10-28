"""Static crop cache: one detector forward per frame, then ordinary static training.

With `augment_frames: false` the detector sees the clean frame and its masks are
deterministic per frame (see `data.py`), so the live pipeline's per-batch forward recomputes
identical output every time a frame is drawn. This extracts once into a directory, and the
classifier then trains from it through a plain `DataLoader` - global shuffle, fixed batch size
in crops, one rotation sampled per draw. See spec_static_crops.md.

What is stored is the *native-resolution patch* (the mask's tight box grown by `margin`) plus
its mask, pre-blackout - not the finished letterbox. A stored patch is therefore just a small
frame, so `letterbox_crop` applies to it unchanged and `blackout`, `mask_only`, `mask_dilate`,
`mask_close`, rotation and photometric all stay load-time choices.

The patch must carry `margin` px of real context because `letterbox_crop` dilates over
`tight +- dilate * (2 if close else 1)`: without it the dilation would read patch-edge zeros
instead of true neighbours and the crop would not match the live path.

Extraction is a pre-pass, never a `DataLoader` worker: workers fork, which would mean a CUDA
detector per worker and concurrent manifest writers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image

from .assign import assign_gt_classes
from .augment import PhotometricJitter
from .config import CropClsConfig
from .crops import letterbox_crop, mask_tight_bbox, normalize_crop, rotate_k90
from .data import CropPipeline, CropRecord

__all__ = ["CacheMismatch", "CachedCropDataset", "build_loader", "extract"]

MANIFEST = "manifest.json"
RECORDS = "records.jsonl"
PATCHES = "patches"


class CacheMismatch(RuntimeError):
    """The cache on disk cannot serve what the config asks for.

    Always raised, never warned: a cache built by a different detector, or loaded with a
    `mask_dilate` wider than its margin, trains perfectly happily on the wrong pixels.
    """


def _pool_provenance(dataset_name: str) -> dict:
    """Identify the *data*, not just the dataset name.

    `training/images` is a per-host symlink, so two hosts can serve different renders of the
    same scene set under one name. Record the resolved path and a digest of the frame names.
    """
    dicts = DatasetCatalog.get(dataset_name)
    names = sorted({os.path.basename(dicts[i]["file_name"]) for i in range(len(dicts))})
    paths = {os.path.realpath(os.path.dirname(dicts[i]["file_name"])) for i in range(len(dicts))}
    return {
        "dataset": dataset_name,
        "pool_realpath": sorted(paths),
        "frame_digest": hashlib.md5("\n".join(names).encode()).hexdigest(),
        "n_frames": len(names),
        "thing_classes": list(MetadataCatalog.get(dataset_name).thing_classes),
    }


def _distinct_frames(dicts) -> list[tuple[int, str]]:
    """(index, file_name) for each DISTINCT frame reachable through the index space.

    `LivePoolDataset.__len__` is `virtual_size` while `__getitem__` maps
    `files[idx % len(files)]`, so indexing 0..len-1 can hit the same frame repeatedly (or
    leave frames unreachable). One forward per *frame* means deduplicating by name.
    """
    seen: dict[str, int] = {}
    for i in range(len(dicts)):
        name = dicts[i]["file_name"]
        if name not in seen:
            seen[name] = i
    return [(i, n) for n, i in seen.items()]


def _patch_bbox(mask: torch.Tensor, margin: int) -> tuple[int, int, int, int] | None:
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    H, W = mask.shape
    x0, y0, x1, y1 = tight
    return (max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin))


def _save_patch(path: Path, patch: torch.Tensor, mask: torch.Tensor) -> None:
    """RGBA PNG: RGB is the raw patch, alpha is the mask. Lossless for uint8."""
    rgb = patch.permute(1, 2, 0).numpy()
    a = (mask.numpy().astype(np.uint8)) * 255
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(path, optimize=True)


def _load_patch(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.array(Image.open(path).convert("RGBA"))
    patch = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1)
    mask = torch.from_numpy(arr[..., 3].copy()) > 127
    return patch, mask


@torch.no_grad()
def extract(
    cfg: CropClsConfig,
    detector,
    dataset_name: str,
    out_dir: str | Path,
    max_frames: int | None = None,
) -> dict:
    """The pre-pass. Returns the manifest it wrote.

    Skips any frame already in the cache, so re-running costs no forwards. Writes a patch
    only for detections that pass the assignment rule, and a metadata row for every
    detection - rows are bytes and they carry the assignment-rate monitor.
    """
    out = Path(out_dir)
    (out / PATCHES).mkdir(parents=True, exist_ok=True)

    prov = _pool_provenance(dataset_name)
    class_names = prov["thing_classes"]
    pipe = CropPipeline(cfg, detector, dataset_name, is_train=False)
    frames = _distinct_frames(pipe.dicts)
    if max_frames is not None:
        frames = frames[:max_frames]

    done = set()
    rec_path = out / RECORDS
    if rec_path.is_file():
        with rec_path.open() as fh:
            done = {json.loads(line)["file_name"] for line in fh if line.strip()}

    counts: dict[str, int] = {}
    n_patch = n_row = 0
    with rec_path.open("a") as fh:
        for frame_i, name in frames:
            if name in done:
                continue
            mapped = pipe.mapper(copy.deepcopy(pipe.dicts[frame_i]))
            image = mapped["image"]
            if image.dtype != torch.uint8:
                raise CacheMismatch(
                    f"frame {name} mapped to {image.dtype}; the cache stores PNG and is only "
                    "lossless for uint8. A float frame would quantize and break bit-identity "
                    "with the live path."
                )
            inst = mapped.get("instances", None)
            if inst is None or not inst.has("gt_masks"):
                continue
            gt = inst.gt_masks
            gt = gt if torch.is_tensor(gt) else gt.tensor

            det = pipe.detector.detect([image], amp=cfg.amp)[0]
            pred = det.masks.cpu()
            a = assign_gt_classes(
                pred, gt, inst.gt_classes,
                iou_floor=cfg.iou_floor, runner_up_ceiling=cfg.runner_up_ceiling,
            )
            for v, c in a.counts().items():
                counts[v] = counts.get(v, 0) + c

            for k in range(len(a)):
                keep = bool(a.keep[k])
                rel = None
                if keep:
                    box = _patch_bbox(pred[k], cfg.margin)
                    if box is None:
                        continue
                    x0, y0, x1, y1 = box
                    rel = f"{PATCHES}/{Path(name).stem}_p{k:03d}.png"
                    _save_patch(out / rel, image[:, y0:y1, x0:x1], pred[k][y0:y1, x0:x1])
                    n_patch += 1
                label = int(a.labels[k]) if keep else -1
                row = asdict(CropRecord(
                    frame_index=frame_i, file_name=name, pred_index=k, rotation=0,
                    label=label, class_name=class_names[label] if keep else "",
                    best_iou=float(a.best_iou[k]), runner_up_iou=float(a.runner_up_iou[k]),
                    verdict=a.verdict[k], score=float(det.scores[k]),
                ))
                row.pop("rotation")        # sampled per draw, never stored
                row["patch"] = rel
                fh.write(json.dumps(row) + "\n")
                n_row += 1

    manifest = {
        **prov,
        "detector_checkpoint": cfg.detector_checkpoint,
        "detector_config": cfg.detector_config,
        "detector_score_thresh": cfg.detector_score_thresh,
        "margin": cfg.margin,
        "iou_floor": cfg.iou_floor,
        "runner_up_ceiling": cfg.runner_up_ceiling,
        "n_frames_extracted": len(frames),
        "n_patches": n_patch,
        "n_rows": n_row,
        "verdict_counts": counts,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


class CachedCropDataset(torch.utils.data.Dataset):
    """Serves crops from a cache. Validates that the cache can answer this config."""

    def __init__(self, cfg: CropClsConfig, dataset_name: str, is_train: bool = True):
        root = Path(cfg.crop_cache)
        man_path = root / MANIFEST
        if not man_path.is_file():
            raise CacheMismatch(f"no cache manifest at {man_path}; run the extraction pre-pass")
        self.manifest = json.loads(man_path.read_text())
        self.root = root
        self.cfg = cfg
        self._check(cfg, dataset_name)

        self.class_names = list(self.manifest["thing_classes"])
        idx = {n: i for i, n in enumerate(self.class_names)}
        rows = []
        with (root / RECORDS).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("patch"):
                    continue                      # dropped: row only, no patch
                if r["best_iou"] < cfg.iou_floor or r["runner_up_iou"] >= cfg.runner_up_ceiling:
                    continue                      # tightened at load time
                r["label"] = idx[r["class_name"]]  # name is authoritative, int is derived
                rows.append(r)
        self.rows = rows
        self.photometric = PhotometricJitter() if (is_train and cfg.photometric) else None
        self.rotations = list(cfg.rotations) if is_train else [0]

    def _check(self, cfg: CropClsConfig, dataset_name: str) -> None:
        m = self.manifest
        if os.path.realpath(cfg.detector_checkpoint) != os.path.realpath(m["detector_checkpoint"]):
            raise CacheMismatch(
                f"cache was built with detector {m['detector_checkpoint']} but the config says "
                f"{cfg.detector_checkpoint}. Masks from a different detector are different data."
            )
        live = list(MetadataCatalog.get(dataset_name).thing_classes)
        if live != list(m["thing_classes"]):
            raise CacheMismatch(
                f"label space differs from the cache's: {live} vs {m['thing_classes']}. A "
                "permutation of the same names would silently remap every crop."
            )
        if cfg.mask_dilate > m["margin"]:
            raise CacheMismatch(
                f"mask_dilate={cfg.mask_dilate} exceeds the cache's margin={m['margin']}; the "
                "patches do not carry enough context and dilation would clip silently."
            )
        if cfg.iou_floor < m["iou_floor"] or cfg.runner_up_ceiling > m["runner_up_ceiling"]:
            raise CacheMismatch(
                f"cannot loosen thresholds at load time: cache was extracted at "
                f"iou_floor={m['iou_floor']}, runner_up_ceiling={m['runner_up_ceiling']}; the "
                f"config asks for {cfg.iou_floor} / {cfg.runner_up_ceiling}. Re-extract."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        patch, mask = _load_patch(self.root / r["patch"])
        cfg = self.cfg
        # box= the UNDILATED tight box, exactly as data.py:218-225 does: build_batch passes
        # the undilated box, which overrides letterbox_crop's dilated one. Reproducing the
        # live path's actual behaviour, not crops.py's docstring (they disagree).
        tight = mask_tight_bbox(mask)
        if tight is None:
            raise CacheMismatch(f"{r['patch']} has an empty mask; the cache is corrupt")
        out = letterbox_crop(
            patch.float(), mask, cfg.crop_size, box=tight,
            photometric=self.photometric, mask_only=cfg.mask_only, blackout=cfg.blackout,
            dilate=cfg.mask_dilate, close=cfg.mask_close,
        )
        if out is None:
            raise CacheMismatch(f"{r['patch']} produced no crop; the cache is corrupt")
        deg = self.rotations[torch.randint(len(self.rotations), ()).item()]
        crop = rotate_k90(out[0], deg)
        return normalize_crop(crop, cfg.pixel_mean, cfg.pixel_std), r["label"]


def build_loader(cfg: CropClsConfig, dataset_name: str, is_train: bool = True, seed: int = 0):
    """Ordinary static batching: global shuffle, constant batch size in crops."""
    ds = CachedCropDataset(cfg, dataset_name, is_train=is_train)
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.crops_per_batch,
        shuffle=is_train,
        drop_last=is_train,
        generator=g if is_train else None,
        num_workers=0,
    )
