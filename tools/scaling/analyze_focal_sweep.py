#!/usr/bin/env python
"""Focal-length sweep analysis (docs/experiments/scaling_experiment.md) over the
per-run files tools/scaling/replay_focal_instances.py writes.

    bash tools/scaling/analyze_focal_sweep.sh                      # replay every run, then this
    ./.venv/bin/python tools/scaling/analyze_focal_sweep.py runs/_analysis/focal_sweep_newval

DIR holds inst_<level>_seed<seed>.npz; levels and seeds are discovered (only seeds
present at every level are used) and every level is compared against --ref.
Per score threshold the replay holds, it prints:

  1. all-GT rates per level and seed - each level's own official eval set;
  2. the PAIRED comparison - only instances kept (uncut by the frame edge) at
     every level, so all levels are scored on the same physical objects. Rates
     are seed means; 95% CIs come from a bootstrap over simulations (both frames
     of a drop resampled together) applied jointly to every level and seed, so
     each delta is paired. Per-seed deltas are listed beside it;
  3. (first threshold only) per-class missed rates, and object size / mask IoU /
     anatomy of misses for forcep03+04;
  4. COCO segm AP / AP50 / AP75 per level, mean +/- sd over seeds (each level's
     own GT set, not the paired one).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

F3, F4 = "forcep03_mm", "forcep04_mm"
AP_CLASSES = ("_all", F3, F4, "forcep01_mm", "forcep02_mm", "forcep11_mm", "tweezer11_mm")


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ref", default="1.0x", help="control level every other level is compared to")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    found = defaultdict(dict)
    for p in glob.glob(os.path.join(args.dir, "inst_*_seed*.npz")):
        m = re.fullmatch(r"inst_(.+)_seed(\d+)\.npz", os.path.basename(p))
        if m:
            found[m.group(1)][m.group(2)] = p
    levels = sorted(found, key=lambda l: float(l.rstrip("x")))
    seeds = sorted(set.intersection(*(set(found[l]) for l in levels)), key=int)
    for l in levels:
        missing = set(found[l]) - set(seeds)
        if missing:
            print(f"[warn] {l}: seed(s) {sorted(missing)} not present at every level - ignored")
    assert args.ref in levels, f"reference level {args.ref} not in {levels}"
    ref = levels.index(args.ref)
    others = [i for i in range(len(levels)) if i != ref]
    print(f"levels {levels}, seeds {seeds}, reference {args.ref}")

    Z = {(l, s): np.load(found[l][s]) for l in levels for s in seeds}
    z0 = Z[(args.ref, seeds[0])]
    names = list(z0["names"])
    c3, c4 = names.index(F3), names.index(F4)
    threshs = sorted({k.split("_")[0][1:] for k in z0.files if re.match(r"s[\d.]+_gt$", k)},
                     key=float, reverse=True)
    lv = " / ".join(levels)

    def outcomes(z, t):
        gt, pr = z[f"s{t}_gt"], z[f"s{t}_pred"]
        miss = pr == -1
        correct = pr == gt
        swap = ((gt == c3) & (pr == c4)) | ((gt == c4) & (pr == c3))
        return gt, miss, correct, swap

    for ti, t in enumerate(threshs):
        rng = np.random.default_rng(0)
        print(f"\n################ score >= {t} ################")

        # ---- 1. official all-GT numbers, per seed
        print(f"\n== all GT (official eval set per level) - rates per seed {seeds} -> mean")
        for grp, sel in ((F3, [c3]), (F4, [c4]), ("03+04", [c3, c4]),
                         ("other 14", [i for i in range(len(names)) if i not in (c3, c4)])):
            for l in levels:
                per = []
                for s in seeds:
                    gt, miss, cor, swp = outcomes(Z[(l, s)], t)
                    m = np.isin(gt, sel)
                    per.append((m.sum(), miss[m].mean(), swp[m].mean(),
                                cor[m].sum() / max((~miss[m]).sum(), 1)))
                per = np.array(per)
                print(f"{grp:12s} {l:6s} n={int(per[0,0]):4d}  missed {' '.join(fmt(x) for x in per[:,1])} -> {fmt(per[:,1].mean())}"
                      f"   swap {' '.join(fmt(x) for x in per[:,2])} -> {fmt(per[:,2].mean())}"
                      f"   correct|det {fmt(per[:,3].mean())}")

        # ---- 2. paired: instances kept at every level
        keys = {l: list(zip(Z[(l, seeds[0])][f"s{t}_frame"], Z[(l, seeds[0])][f"s{t}_obj"])) for l in levels}
        for l in levels:  # GT never depends on the model, so every seed must agree
            for s in seeds[1:]:
                assert (Z[(l, s)][f"s{t}_obj"] == Z[(l, seeds[0])][f"s{t}_obj"]).all(), (l, s)
        common = set(keys[levels[0]])
        for l in levels[1:]:
            common &= set(keys[l])
        common = sorted(common)
        sims = sorted({f.split("_")[0] for f, _ in common})
        sim_ix = {s: i for i, s in enumerate(sims)}
        inst_sim = np.array([sim_ix[f.split("_")[0]] for f, _ in common])
        idx = {}
        for l in levels:
            pos = {k: i for i, k in enumerate(keys[l])}
            idx[l] = np.array([pos[k] for k in common])
        gt_c = z0[f"s{t}_gt"][idx[args.ref]]
        for l in levels:  # the same physical object must carry the same class at every level
            assert (Z[(l, seeds[0])][f"s{t}_gt"][idx[l]] == gt_c).all(), l

        groups = {F3: gt_c == c3, F4: gt_c == c4, "03+04": np.isin(gt_c, [c3, c4]),
                  "other 14": ~np.isin(gt_c, [c3, c4])}
        for i, n in enumerate(names):
            if i not in (c3, c4):
                groups[n] = gt_c == i
        G = list(groups)
        # C[sim, level, seed, group, (gt, missed, swap, correct)]
        C = np.zeros((len(sims), len(levels), len(seeds), len(G), 4))
        for li, l in enumerate(levels):
            for si, s in enumerate(seeds):
                _, miss, cor, swp = (x[idx[l]] for x in outcomes(Z[(l, s)], t))
                for gi, g in enumerate(G):
                    m = groups[g]
                    for mi, v in enumerate((np.ones_like(miss), miss, swp, cor)):
                        np.add.at(C[:, li, si, gi, mi], inst_sim[m], v[m].astype(float))
        W = rng.multinomial(len(sims), np.ones(len(sims)) / len(sims), size=args.boot)
        Cb = np.einsum("bs,s...->b...", W, C)
        rate_b = Cb[..., 1:] / np.maximum(Cb[..., :1], 1)                 # [B, level, seed, group, 3]
        rate_pt = C.sum(0)[..., 1:] / np.maximum(C.sum(0)[..., :1], 1)  # [level, seed, group, 3]
        mean_b, mean_pt = rate_b.mean(2), rate_pt.mean(1)               # seed means
        det_pt = rate_pt[..., 2] / np.maximum(1 - rate_pt[..., 0], 1e-9)  # correct | detected

        print(f"\n== paired: {len(common)} instances uncut at all levels, {len(sims)} simulations")
        print(f"   rate = mean over {len(seeds)} seeds; Δ vs {args.ref} with 95% CI (simulation bootstrap); per-seed Δ in []")
        for g in (F3, F4, "03+04", "other 14"):
            gi = G.index(g)
            line = [f"{g:9s} n={int(C.sum(0)[0, 0, gi, 0]):4d}"]
            for mi, mname in enumerate(("missed", "swap", "correct")):
                line.append(f"{mname} {' / '.join(fmt(mean_pt[li, gi, mi]) for li in range(len(levels)))}")
            line.append(f"correct|det {' / '.join(fmt(det_pt[li, :, gi].mean()) for li in range(len(levels)))}")
            print("  ".join(line) + f"   ({lv})")
            for li in others:
                parts = []
                for mi, mname in enumerate(("missed", "swap", "correct")):
                    d_b = mean_b[:, li, gi, mi] - mean_b[:, ref, gi, mi]
                    d = mean_pt[li, gi, mi] - mean_pt[ref, gi, mi]
                    lo, hi = np.percentile(d_b, [2.5, 97.5])
                    per_seed = rate_pt[li, :, gi, mi] - rate_pt[ref, :, gi, mi]
                    star = "*" if lo > 0 or hi < 0 else " "
                    parts.append(f"{mname} {d:+.3f} [{lo:+.3f},{hi:+.3f}]{star} seeds[{' '.join(f'{x:+.3f}' for x in per_seed)}]")
                print(f"   {levels[li]}-{args.ref}  " + "  ".join(parts))

        if ti == 0:
            print(f"\n== paired, per class: missed rate (seed mean) {lv}, and Δ vs {args.ref} (* = 95% CI excludes 0)")
            for g in G[4:] + [F3, F4]:
                gi = G.index(g)
                deltas = []
                for li in others:
                    d_b = mean_b[:, li, gi, 0] - mean_b[:, ref, gi, 0]
                    star = "*" if np.percentile(d_b, 2.5) > 0 or np.percentile(d_b, 97.5) < 0 else " "
                    deltas.append(f"Δ{levels[li]} {mean_pt[li, gi, 0] - mean_pt[ref, gi, 0]:+.3f}{star}")
                mark = "<<" if g in (F3, F4) else "  "
                print(f"{mark}{g:16s} n={int(C.sum(0)[0, 0, gi, 0]):4d}  "
                      f"{' / '.join(fmt(mean_pt[li, gi, 0]) for li in range(len(levels)))}   " + "   ".join(deltas))

            # ---- 3. size, mask quality and anatomy of misses on the paired 03/04 set
            print("\n== paired 03/04: object size, mask IoU of correct matches, anatomy of misses (seeds pooled)")
            m34 = groups["03+04"]
            for l in levels:
                area = Z[(l, seeds[0])][f"s{t}_area"][idx[l]][m34]
                ious, best_of_miss = [], []
                for s in seeds:
                    z = Z[(l, s)]
                    pr, gt = z[f"s{t}_pred"][idx[l]][m34], z[f"s{t}_gt"][idx[l]][m34]
                    ious.append(z[f"s{t}_iou"][idx[l]][m34][pr == gt])
                    best_of_miss.append(z[f"s{t}_bestiou"][idx[l]][m34][pr == -1])
                ious, bm = np.concatenate(ious), np.concatenate(best_of_miss)
                print(f"  {l:6s} area median {int(np.median(area)):6d}px  | IoU median {np.median(ious):.3f}  >=0.75 {np.mean(ious>=0.75):.2f}"
                      f"  | misses {bm.size:3d}: near-miss mask (best IoU 0.3-0.5) {np.mean((bm>=0.3)&(bm<0.5)):.2f}, nothing (<0.1) {np.mean(bm<0.1):.2f}")

    # ---- 4. COCO segm AP, threshold-free
    print("\n== mask AP (each level's own GT, COCO segm), mean ± sd over seeds:  AP | AP50 | AP75")

    def ap_of(l, s, cls):
        z = Z[(l, s)]
        if cls == "_all":
            return z["ap_all"]
        return z["ap_per_class"][list(z["ap_classes"]).index(cls)]

    for cls in AP_CLASSES:
        row = []
        for l in levels:
            v = np.array([ap_of(l, s, cls) for s in seeds])
            row.append(" | ".join(f"{v[:, i].mean():5.1f}±{v[:, i].std(ddof=1):3.1f}" for i in range(3)))
        print(f"  {cls:14s} " + "    ".join(f"{l}: {r}" for l, r in zip(levels, row)))
    rest = [c for c in z0["ap_classes"] if c not in (F3, F4)]
    for l in levels:
        v = np.array([[ap_of(l, s, c) for c in rest] for s in seeds])
        print(f"  other-14 mean {l}: AP {v[..., 0].mean():.1f}  AP50 {v[..., 1].mean():.1f}  AP75 {v[..., 2].mean():.1f}")


if __name__ == "__main__":
    main()
