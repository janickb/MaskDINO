#!/usr/bin/env python
"""Write the classifier's training crops to a folder, one detector forward per frame.

The detector is frozen and `augment_frames: false` makes it see the clean frame, so its
masks are deterministic per frame - the live pipeline recomputes identical output every time
a frame is drawn. ~600 ms/frame, so a 1024-frame pool is ~10 minutes. Cheap enough that
changing a mask setting means re-running this rather than storing every variant.

    PYTHONPATH=. python tools/extract_crops.py --config configs/cropcls/phase1_seta.yaml \
        --out crops/train
    PYTHONPATH=. python tools/extract_crops.py --config ... --split val --out crops/val

`PYTHONPATH=.` is not optional in this worktree: `maskdino` is not installed, so it resolves
only through the current directory, and a script's sys.path[0] is its own directory. Without
it every tool here dies on `import maskdino`.

Writes `<out>/<class_name>/<frame>_p<k>.png` (RGBA, alpha is the mask), plus meta.json and
crops.csv. The stored patch is the mask's tight box grown by `pad_frac` per axis, and it is a
CEILING: the loader crops to its own `cfg.pad_frac` within it, so extracting at 0.20 and
training at 0.10 costs disk and buys the sweep, including the tight control at 0.0. Only detections passing the assignment rule get a crop; the rest appear in the CSV
with the verdict that explains why, which is what the assignment rate is read from.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.assign import assign_gt_classes
from cropcls.config import load_config
from cropcls.crop_folder import write_frame_crops
from cropcls.data import CropPipeline, dataset_class_names
from cropcls.detector import FrozenSingleClassDetector

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="train", choices=["train", "val"])
ap.add_argument("--frames", type=int, default=None, help="cap, for a smoke run")
ap.add_argument("--device", default="cuda")
ap.add_argument(
    "--pad-frac", type=float, default=None,
    help="override cfg.pad_frac for the STORED patch only. The stored fraction is a ceiling, "
         "not the training setting: the loader crops to its own cfg.pad_frac within it. Extract "
         "above what you intend to train at and one folder serves the whole sweep, the tight "
         "control included, with no second detector pass.",
)
a = ap.parse_args()

cfg = load_config(a.config)
if a.pad_frac is not None:
    cfg = replace(cfg, pad_frac=a.pad_frac)
    cfg.validate()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
split = cfg.train_dataset if a.split == "train" else cfg.val_dataset
class_names = dataset_class_names(split)

det = FrozenSingleClassDetector(cfg, device=a.device)
pipe = CropPipeline(cfg, det, split, is_train=False)

# LivePoolDataset.__len__ is a virtual size while __getitem__ maps files[idx % len(files)],
# so indexing 0..len-1 can hit one frame repeatedly or leave frames unreachable. One forward
# per *frame* means going by name.
seen: dict[str, int] = {}
for i in range(len(pipe.dicts)):
    seen.setdefault(pipe.dicts[i]["file_name"], i)
frames = list(seen.items())[: a.frames]

stems: set[str] = set()
rows: list[dict] = []
counts: dict[str, int] = {}
n_crop = 0
t0 = time.perf_counter()

for name, frame_i in frames:
    mapped = pipe.mapper(copy.deepcopy(pipe.dicts[frame_i]))
    inst = mapped.get("instances", None)
    if inst is None or not inst.has("gt_masks"):
        counts["no_gt_masks"] = counts.get("no_gt_masks", 0) + 1
        continue
    gt = inst.gt_masks
    gt = gt if torch.is_tensor(gt) else gt.tensor

    image = mapped["image"]
    pred = det.detect([image], amp=cfg.amp)[0].masks.cpu()
    asg = assign_gt_classes(
        pred, gt, inst.gt_classes,
        iou_floor=cfg.iou_floor, runner_up_ceiling=cfg.runner_up_ceiling,
    )
    for v, c in asg.counts().items():
        counts[v] = counts.get(v, 0) + c

    stem = Path(name).stem
    if stem in stems:
        raise SystemExit(
            f"two frames share the basename {stem!r}; crop filenames would collide and "
            "overwrite each other silently. Give the pool unique basenames."
        )
    stems.add(stem)

    frame_rows = write_frame_crops(out, stem, image, pred, asg, class_names, cfg.pad_frac)
    n_crop += sum(1 for r in frame_rows if r["crop"])
    rows += frame_rows

dt = time.perf_counter() - t0
with (out / "crops.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["frame"])
    w.writeheader()
    w.writerows(rows)

# The folder name is the label, so nothing here is needed to train. It records which data
# this came from: `training/images*` folder names do not identify a render set - two 1024-frame
# renders of the same scene set exist and a folder was repointed between them - so pin the
# frame-name digest.
# The thinnest border actually stored, over the sides that are not at the frame edge. The
# `mask_close` guard is checked against this rather than against pad_frac, because pad_frac is
# relative and clipped at the frame: a nominal 10% is not a promise of any particular pixel
# count. None when every crop was frame-clipped on all four sides.
borders = [r["border_px"] for r in rows if r["border_px"] != ""]
min_border = min(borders) if borders else None

basenames = sorted(os.path.basename(n) for n, _ in frames)
(out / "meta.json").write_text(json.dumps({
    "dataset": split,
    "pool_realpath": sorted({os.path.realpath(os.path.dirname(n)) for n, _ in frames}),
    "frame_digest": hashlib.md5("\n".join(basenames).encode()).hexdigest(),
    "n_frames": len(frames),
    "thing_classes": class_names,
    "detector_checkpoint": os.path.realpath(cfg.detector_checkpoint),
    "detector_score_thresh": cfg.detector_score_thresh,
    "pad_frac": cfg.pad_frac,
    "min_border_px": min_border,
    "iou_floor": cfg.iou_floor,
    "runner_up_ceiling": cfg.runner_up_ceiling,
    "augment_frames": cfg.augment_frames,
    "n_crops": n_crop,
    "verdict_counts": counts,
    "seconds": round(dt, 1),
}, indent=2))

assigned = counts.get("assigned", 0)
total = sum(v for k, v in counts.items() if k != "no_gt_masks")
print(f"\n{n_crop} crops from {len(frames)} frames in {dt / 60:.1f} min "
      f"({1000 * dt / max(len(frames), 1):.0f} ms/frame)")
print(f"assignment rate {assigned}/{total} = {assigned / max(total, 1):.1%}")
for v, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {v:26s} {c}")
print(f"\n-> {out}   (name the machine and its load when quoting the ms/frame)")
