#!/usr/bin/env python
"""Merge the crops_shard*.csv / meta_shard*.json of a sharded extract_crops.py run into the
crops.csv / meta.json that CropFolder reads."""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
metas = [json.loads(p.read_text()) for p in sorted(out.glob("meta_shard*.json"))]
rows = []
for p in sorted(out.glob("crops_shard*.csv")):
    with p.open() as fh:
        rows += list(csv.DictReader(fh))
n = int(metas[0]["shard"].split("/")[1])
if sorted(m["shard"] for m in metas) != sorted(f"{i}/{n}" for i in range(n)):
    raise SystemExit(f"incomplete shards: {sorted(m['shard'] for m in metas)}")

same = ["dataset", "thing_classes", "detector_checkpoint", "detector_score_thresh", "pad_frac",
        "occluders", "iou_floor", "runner_up_ceiling", "augment_frames",
        "copy_paste_per_frame", "copy_paste_seed", "copy_paste_cfg"]
for k in same:
    if len({json.dumps(m.get(k), sort_keys=True) for m in metas}) != 1:
        raise SystemExit(f"shards disagree on {k}")

meta = {k: metas[0].get(k) for k in same}
counts = {}
for m in metas:
    for v, c in m["verdict_counts"].items():
        counts[v] = counts.get(v, 0) + c
borders = [m["min_border_px"] for m in metas if m["min_border_px"] is not None]
meta.update({
    "pool_realpath": sorted({p for m in metas for p in m["pool_realpath"]}),
    "n_frames": sum(m["n_frames"] for m in metas),
    "n_composites": sum(m["n_composites"] for m in metas),
    "n_crops": sum(m["n_crops"] for m in metas),
    "min_border_px": min(borders) if borders else None,
    "verdict_counts": counts,
    "seconds_per_shard": [m["seconds"] for m in metas],
    "shards": n,
})
with (out / "crops.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"{meta['n_crops']} crops, {meta['n_composites']} composites, {meta['n_frames']} frames")
print(counts)
