#!/usr/bin/env python
"""Compare focal levels at each level's best score threshold, over the per-run
files tools/scaling/sweep_focal_thresholds.py writes (sw_<level>_seed<seed>.npz).

    bash tools/scaling/analyze_focal_sweep.sh               # runs this too
    ./.venv/bin/python tools/scaling/analyze_focal_thresholds.py runs/_analysis/focal_sweep_newval

Prints:
  1. seed-mean class-aware F1 vs score threshold, for all tools and for
     forcep03+04 (precision there = correct 03/04 labels / all 03/04 labels);
  2. each level at its best threshold (chosen on the seed-mean curve), then
     at the fixed thresholds --at, with one-way forcep03 / forcep04 outcomes
     (right, labelled as the other one, missed);
  3. best vs best against --ref, 95 % CI from a bootstrap over simulations with
     the threshold RE-CHOSEN inside every resample, so picking the best threshold
     does not inflate the difference. The CI covers val sampling only, not
     training-seed noise;
  4. threshold-free 03-vs-04 separability on the objects kept at every level:
     margin = best 03-labelled score - best 04-labelled score on the object's
     mask; AUC of that margin, accuracy of the top label, and the best balanced
     accuracy reachable with one fixed offset on the margin (chosen in-sample,
     so optimistic).

All-GT numbers (1-3) use each level's own kept objects; 4 is paired.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default="1.0x")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--at", default="0.3,0.4,0.5", help="fixed thresholds for the one-way table")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    found = defaultdict(dict)
    for p in glob.glob(os.path.join(args.dir, "sw_*_seed*.npz")):
        m = re.fullmatch(r"sw_(.+)_seed(\d+)\.npz", os.path.basename(p))
        if m:
            found[m.group(1)][m.group(2)] = p
    levels = sorted(found, key=lambda l: float(l.rstrip("x")))
    seeds = sorted(set.intersection(*(set(found[l]) for l in levels)), key=int)
    Z = {(l, s): np.load(found[l][s]) for l in levels for s in seeds}
    z0 = Z[(levels[0], seeds[0])]
    names = list(z0["names"])
    c3, c4 = names.index("forcep03_mm"), names.index("forcep04_mm")
    T = np.round(z0["thresh"], 2)
    nT = len(T)
    ref = levels.index(args.ref)
    n_img = len(z0["img"])

    # per-simulation counts [sim, level, seed, T, metric]:
    #   0 n_gt, 1 correct, 2 mis (matched, wrong class), 3 fp_all,
    #   4 n_0304, 5 correct_0304, 6 swap_0304, 7 missed_0304,
    #   8 predictions labelled 03/04 on a kept GT, 9 false positives labelled 03/04
    sims = sorted({f.split("_")[0] for f in z0["img"]})
    six = {s: i for i, s in enumerate(sims)}
    C = np.zeros((len(sims), len(levels), len(seeds), nT, 10))
    for li, l in enumerate(levels):
        for si, s in enumerate(seeds):
            z = Z[(l, s)]
            gt, pr = z["gt"], z["pred"].astype(np.int64)  # pred [n_gt, T], -1 = missed
            ins = np.array([six[f.split("_")[0]] for f in z["frame"]])
            is34 = np.isin(gt, [c3, c4])[:, None]
            hit = pr >= 0
            cols = {
                0: np.ones_like(pr), 1: pr == gt[:, None], 2: hit & (pr != gt[:, None]),
                4: np.broadcast_to(is34, pr.shape), 5: is34 & (pr == gt[:, None]),
                6: is34 & hit & np.isin(pr, [c3, c4]) & (pr != gt[:, None]), 7: is34 & ~hit,
                8: np.isin(pr, [c3, c4]),
            }
            for k, v in cols.items():
                np.add.at(C[:, li, si, :, k], ins, v.astype(float))
            img_s = np.array([six[f.split("_")[0]] for f in z["img"]])
            np.add.at(C[:, li, si, :, 3], img_s, z["fp"].sum(-1).astype(float))
            np.add.at(C[:, li, si, :, 9], img_s, z["fp"][..., [c3, c4]].sum(-1).astype(float))

    def curves(Cs):
        n, cor, mis, fp = Cs[..., 0], Cs[..., 1], Cs[..., 2], Cs[..., 3]
        P = cor / np.maximum(cor + mis + fp, 1)
        R = cor / np.maximum(n, 1)
        n4, c4_, sw, mi4, pl, fp4 = (Cs[..., k] for k in (4, 5, 6, 7, 8, 9))
        P4 = c4_ / np.maximum(pl + fp4, 1)
        R4 = c4_ / np.maximum(n4, 1)
        return {
            "F1": 2 * P * R / np.maximum(P + R, 1e-9), "P": P, "R": R,
            "F1_34": 2 * P4 * R4 / np.maximum(P4 + R4, 1e-9), "P_34": P4, "R_34": R4,
            "swap_34": sw / np.maximum(n4, 1), "miss_34": mi4 / np.maximum(n4, 1),
            "fp_per_img": fp / n_img,
        }

    pt = curves(C.sum(0))  # [level, seed, T]
    W = rng.multinomial(len(sims), np.ones(len(sims)) / len(sims), size=args.boot)
    bt = curves(np.einsum("bs,s...->b...", W, C))  # [B, level, seed, T]

    print(f"levels {levels}, seeds {seeds}; one label per mask, cut tools = iscrowd ignore regions")
    for key, title in (("F1", "class-aware F1, all tools"),
                       ("F1_34", "forcep03+04 F1 (P = correct 03/04 labels / all 03/04 labels)")):
        print(f"\n== {title}: seed-mean curve")
        print("  thresh " + " ".join(f"{t:5.2f}" for t in T))
        for li, l in enumerate(levels):
            m = pt[key][li].mean(0)
            print(f"  {l:6s} " + " ".join(f"{x:5.3f}" for x in m) + f"   best {m.max():.3f} @ {T[m.argmax()]:.2f}")

    print("\n== best threshold per level (chosen on the seed-mean curve), everything at that threshold")
    for key in ("F1", "F1_34"):
        for li, l in enumerate(levels):
            m = pt[key][li].mean(0)
            ti = int(m.argmax())
            per_seed_t = " ".join(f"{T[int(pt[key][li, si].argmax())]:.2f}" for si in range(len(seeds)))
            sd = pt[key][li, :, ti].std(ddof=1)
            extras = "  ".join(f"{k} {pt[k][li, :, ti].mean():.3f}" for k in
                               (("P", "R", "fp_per_img") if key == "F1" else ("P_34", "R_34", "miss_34", "swap_34")))
            print(f"  {key:6s} {l:6s} t*={T[ti]:.2f} (per-seed optimum {per_seed_t})  "
                  f"{key} {m[ti]:.3f} ± {sd:.3f} (seed sd)  {extras}")

    print("\n== fixed thresholds, seed mean (03: share of forcep03 GT labelled 03 / labelled 04 / missed; 04 likewise)")
    print("  thresh level     F1     P     R fp/img |  F1_34 miss_34 swap_34 | 03 right  ->04  miss | 04 right  ->03  miss")
    for t in (float(x) for x in args.at.split(",")):
        ti = int(np.argmin(np.abs(T - t)))
        for li, l in enumerate(levels):
            agg = " ".join(f"{pt[k][li, :, ti].mean():5.3f}" for k in ("F1", "P", "R", "fp_per_img"))
            a34 = "   ".join(f"{pt[k][li, :, ti].mean():5.3f}" for k in ("F1_34", "miss_34", "swap_34"))
            oneway = []
            for s in seeds:
                z = Z[(l, s)]
                gt, pr = z["gt"], z["pred"][:, ti].astype(np.int64)
                row = []
                for c, other in ((c3, c4), (c4, c3)):
                    p = pr[gt == c]
                    row += [(p == c).mean(), (p == other).mean(), (p < 0).mean()]
                oneway.append(row)
            o = np.mean(oneway, 0)
            print(f"  {T[ti]:6.2f} {l:6s} {agg} |  {a34} |    {o[0]:.3f} {o[1]:.3f} {o[2]:.3f} |"
                  f"    {o[3]:.3f} {o[4]:.3f} {o[5]:.3f}")

    print(f"\n== best vs best (threshold re-chosen inside every bootstrap resample), 95% CI, vs {args.ref}")
    for key in ("F1", "F1_34"):
        peak = bt[key].mean(2).max(-1)  # [B, level]
        for li, l in enumerate(levels):
            if li == ref:
                continue
            d = peak[:, li] - peak[:, ref]
            dpt = pt[key][li].mean(0).max() - pt[key][ref].mean(0).max()
            lo, hi = np.percentile(d, [2.5, 97.5])
            per_seed = " ".join(f"{pt[key][li, si].max() - pt[key][ref, si].max():+.3f}" for si in range(len(seeds)))
            print(f"  {key:6s} {l} - {args.ref}: {dpt:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}"
                  f"   per-seed best-vs-best [{per_seed}]")

    print("\n== 03 vs 04 separability, threshold-free (paired objects kept at every level)")
    print("   margin = best 03-labelled score - best 04-labelled score on the object's mask (IoU >= 0.5, scores >= 0.05)")
    keys = {l: set(zip(Z[(l, seeds[0])]["frame"], Z[(l, seeds[0])]["obj"])) for l in levels}
    common = set.intersection(*keys.values())
    for l in levels:
        rows = []
        for s in seeds:
            z = Z[(l, s)]
            sel = np.array([(f, o) in common for f, o in zip(z["frame"], z["obj"])]) & np.isin(z["gt"], [c3, c4])
            gt, s3, s4 = z["gt"][sel], z["s3"][sel], z["s4"][sel]
            cov = (s3 > 0) | (s4 > 0)
            g, mg = gt[cov], (s3 - s4)[cov]
            pos, neg = mg[g == c3], mg[g == c4]
            auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
            acc03, acc04 = (pos > 0).mean(), (neg < 0).mean()
            cands = np.unique(mg)
            bal = [((pos > b).mean() + (neg <= b).mean()) / 2 for b in cands]
            rows.append((cov.mean(), auc, acc03, acc04, (acc03 + acc04) / 2, max(bal), cands[int(np.argmax(bal))]))
        r = np.array(rows)
        print(f"  {l:6s} localized {r[:,0].mean():.3f}  AUC {r[:,1].mean():.3f} ± {r[:,1].std(ddof=1):.3f}"
              f"  | top label: 03 right {r[:,2].mean():.3f}, 04 right {r[:,3].mean():.3f}, balanced {r[:,4].mean():.3f}"
              f"  | best offset: balanced {r[:,5].mean():.3f} at b={np.mean(r[:,6]):+.2f}")


if __name__ == "__main__":
    main()
