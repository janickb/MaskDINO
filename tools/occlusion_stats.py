"""Per-prediction occlusion statistics, from the predicted masks alone.

    PYTHONPATH=. python tools/occlusion_stats.py --run runs/20261103122513_mask4_cxt_c448

Answers two questions about feeding the occluding instances as an extra input channel,
before anything is built:

1. Does boundary censoring separate the misclassifications? If an instance's silhouette is
   cut by another instance, the evidence that distinguishes it may be gone - that is the
   "censored measurement" signal, and it should show up as a higher error rate.
2. Does censoring leak class identity? Occlusion correlates with scene clutter, which may
   correlate with which instruments co-occur. If class is predictable from occlusion alone,
   the channel is a shortcut rather than a censoring cue and the arm measures nothing.

Detector only - no classifier forward. The classifier's own outputs are joined from
`<run>/confidence/crops.csv` on (frame, pred_index), and the join is verified against
`det_score` so a silent misalignment cannot pass.

Everything here uses ONLY the detector's own predictions, never GT, so every statistic is
available at inference. Writes `<run>/occlusion/stats.csv`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.config import load_config
from cropcls.data import CropPipeline, apply_truncated_filter, apply_val_stride
from cropcls.detector import FrozenSingleClassDetector

RING = 4          # px: the boundary band tested for what terminates the silhouette


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--frames-per-forward", type=int, default=2)
    ap.add_argument("--ring", type=int, default=RING)
    ap.add_argument("--val-stride", type=int, default=None,
                    help="override cfg.val_stride; 1 = every val frame. The run's "
                         "own value is the 8-class baseline protocol, so metrics "
                         "compared against it must keep it.")
    ap.add_argument("--conf-csv", default=None,
                    help="classifier outputs to join (default <run>/confidence/crops.csv). "
                         "Must be the SAME val_stride as this run or the join silently covers "
                         "only the overlapping frames.")
    ap.add_argument("--out", default=None, help="default <run>/occlusion")
    return ap.parse_args()


def dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Binary dilation by a (2r+1) square, via max-pool."""
    x = m[None, None].float()
    return F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)[0, 0] > 0.5


@torch.no_grad()
def collect(pipe, detector, cfg, max_frames=None, step=2, ring=RING):
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    rows = []
    for start in range(0, n, step):
        idx = list(range(start, min(start + step, n)))
        mapped = pipe._map_frames(idx)
        dets = detector.detect([m["image"] for m in mapped], amp=cfg.amp)

        for m, det in zip(mapped, dets):
            if len(det) == 0:
                continue
            masks = det.masks.cpu().bool()
            H, W = masks.shape[-2:]
            # Frame-interior test: a silhouette that runs into the image border is censored
            # by the field of view rather than by another tool. Kept separate because the
            # two have different fixes (one is a capture problem, one is a scene problem).
            interior = torch.zeros((H, W), dtype=torch.bool)
            interior[ring:H - ring, ring:W - ring] = True

            for k in range(len(det)):
                mine = masks[k]
                area = int(mine.sum())
                if area == 0:
                    continue
                others = torch.zeros((H, W), dtype=torch.bool)
                for j in range(len(det)):
                    if j != k:
                        others |= masks[j]

                grown = dilate(mine, ring)
                band = grown & ~mine                      # the outward boundary band
                nb = int(band.sum())
                # What terminates the silhouette, as a fraction of its boundary band:
                cens = int((band & others).sum()) / max(nb, 1)      # another instance
                edge = int((band & ~interior).sum()) / max(nb, 1)   # the frame border
                rows.append({
                    "frame": Path(str(m.get("file_name", ""))).stem,
                    "pred_index": k,
                    "det_score": float(det.scores[k]),
                    "area": area,
                    "boundary_px": nb,
                    "censored_frac": cens,
                    "edge_frac": edge,
                    "overlap_frac": int((mine & others).sum()) / area,
                    "n_instances": len(det),
                })
    return rows


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    xs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1
        for t in range(i, j + 1):
            rk[t] = mid
        i = j + 1
    rp = sum(r for r, (_, l) in zip(rk, xs) if l == 1)
    a, b = len(pos), len(neg)
    return (rp - a * (a + 1) / 2) / (a * b)


def main():
    args = parse_args()
    run = Path(args.run)
    cfg = load_config(run / "config.json")
    out = Path(args.out or run / "occlusion")
    out.mkdir(parents=True, exist_ok=True)

    apply_val_stride(cfg.val_dataset, (cfg.val_stride if args.val_stride is None else args.val_stride))
    apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    det = FrozenSingleClassDetector(cfg, device=args.device)
    pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)

    rows = collect(pipe, det, cfg, args.frames, args.frames_per_forward, args.ring)
    print(f"{len(rows)} predictions with occlusion stats (ring={args.ring} px)")

    # --- join the classifier's own outputs, verifying with det_score ---
    conf_path = Path(args.conf_csv) if args.conf_csv else run / "confidence" / "crops.csv"
    conf = {}
    if conf_path.is_file():
        with open(conf_path) as f:
            for r in csv.DictReader(f):
                conf[(r["frame"], int(r["pred_index"]))] = r
    # Every geometry row is written, joined or not: dropping the unjoined ones silently
    # shrinks the file to the overlap of two strides, which is how this produced a 1221-row
    # file for a 4765-prediction pass. Unjoined rows carry correct=-2 / msp=-1.0.
    joined, bad = [], 0
    for r in rows:
        c = conf.get((r["frame"], r["pred_index"]))
        r = dict(r)
        if c is not None and abs(float(c["det_score"]) - r["det_score"]) > 1e-4:
            bad += 1
            c = None
        r["correct"] = int(c["correct"]) if c else -2
        r["msp"] = float(c["msp"]) if c else -1.0
        r["gt_class"] = c["gt_class"] if c else ""
        r["pred_class"] = c["pred_class"] if c else ""
        # Carried so a claim comparing occlusion against mask quality or size has ONE file
        # behind it. These live in the confidence CSV; quoting a ratio across the two from an
        # unsaved ad-hoc join leaves the number with nothing to open (preflight finding).
        for k in ("best_iou", "mask_long", "mask_area"):
            r[k] = float(c[k]) if c and k in c else -1.0
        joined.append(r)
    print(f"joined {len(joined)} rows to {conf_path.name} ({bad} det_score mismatches dropped)")

    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joined[0]) if joined else list(rows[0]))
        w.writeheader()
        w.writerows(joined)

    print(f"  (rows with no classifier join: {sum(1 for r in joined if r['correct'] == -2)})")
    matched = [r for r in joined if r["correct"] >= 0]
    ok = [r for r in matched if r["correct"] == 1]
    bad_r = [r for r in matched if r["correct"] == 0]
    print(f"\nmatched {len(matched)}, errors {len(bad_r)}")

    print("\n--- Q1: does censoring separate the errors? ---")
    for key in ("censored_frac", "edge_frac", "overlap_frac"):
        mc = statistics.median(r[key] for r in ok)
        mw = statistics.median(r[key] for r in bad_r)
        # higher censoring should mean MORE likely wrong -> score the negated value
        a = auroc([-r[key] for r in ok], [-r[key] for r in bad_r])
        print(f"  {key:<15} median correct {mc:.4f}  wrong {mw:.4f}   AUROC(error) {a:.4f}")
    a_msp = auroc([r["msp"] for r in ok], [r["msp"] for r in bad_r])
    print(f"  {'msp':<15} AUROC(error) {a_msp:.4f}   <- for reference")

    print("\n  error rate by censored_frac:")
    for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)):
        sel = [r for r in matched if lo <= r["censored_frac"] < hi]
        if sel:
            e = sum(1 for r in sel if r["correct"] == 0)
            mm = statistics.median(r["msp"] for r in sel)
            print(f"    [{lo:.2f},{hi:.2f})  n={len(sel):4d} errors={e:2d}  "
                  f"{100 * e / len(sel):5.1f}%   median msp {mm:.5f}")

    print("\n--- Q2: does censoring leak class identity? (shortcut check) ---")
    by = defaultdict(list)
    for r in matched:
        by[r["gt_class"]].append(r)
    print(f"    {'gt class':<18} {'n':>5} {'med censored':>14} {'med edge':>12}")
    for c in sorted(by):
        v = by[c]
        mcf = statistics.median(r["censored_frac"] for r in v)
        mef = statistics.median(r["edge_frac"] for r in v)
        print(f"    {c:<18} {len(v):5d} {mcf:14.4f} {mef:12.4f}")
    # the pair that matters: can censoring alone tell forcep03 from forcep04?
    for a, b in (("forcep03_mm", "forcep04_mm"), ("forcep01_mm", "forcep02_mm")):
        A, B = by.get(a, []), by.get(b, [])
        if A and B:
            lk = auroc([r["censored_frac"] for r in A], [r["censored_frac"] for r in B])
            print(f"    {a} vs {b} from censored_frac alone: AUROC {lk:.4f} (0.5 = no leak)")

    print(f"\nwrote {out}/stats.csv")


if __name__ == "__main__":
    main()
