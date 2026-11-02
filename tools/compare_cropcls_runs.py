#!/usr/bin/env python
"""Tabulate cropcls runs against the 8-class MaskDINO baseline.

    python tools/compare_cropcls_runs.py runs/cropcls_*/

Reads each run's config.json + eval.json (crop top-1) and hungarian_eval.json when
present, and prints the step-8 table. The two metrics answer different questions and are
kept apart deliberately:

  top-1 on matched masks  - confound-free. Both sides condition on an instance having
                            been localized, so the frontends' differing class-agnostic
                            recall (0.9792 vs 0.9276) cannot flatter either.
  classaware_f1           - end-to-end, and therefore inherits that recall difference.
                            Always read it next to classagnostic_f1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# 20261006204751_phase1_training_seta_v1-1, val_seta_mm, stride 4, excl-truncated, 1201 GT
BASELINE = {
    "name": "8-class MaskDINO (v1-1)",
    "top1_matched": 1083 / 1114,      # 0.9722
    "classaware_f1": 0.9256,
    "classagnostic_f1": 0.9521,
    # Per-class top-1 ON MATCHED masks = classaware_recall / classagnostic_recall.
    # NOT classaware_recall itself: that divides by all GT including instances the
    # detector never localized, while our runs divide by matched crops only. Comparing
    # the two would flatter the classifier on every row.
    "per_class": {
        "forcep01_mm": 0.964,
        "forcep02_mm": 0.9425,
        "forcep03_mm": 0.9322,
        "forcep04_mm": 0.9237,
        "scalpel01_mm": 0.9938,
        "hammer01_mm": 1.0,
        "sharpspoon01_mm": 1.0,
        "scarstick01_mm": 1.0,
    },
}
# knobs worth showing per run; everything else is identical across them
KNOBS = ("trunk", "mask_only", "photometric", "augment_frames",
         "base_lr", "max_iter", "ims_per_batch", "crop_size", "freeze_trunk")


def load(run: Path) -> dict | None:
    cfg = run / "config.json"
    # eval.json is what a training run writes; eval_evalonly.json is a later --eval-only
    # re-score of the same checkpoint, which deliberately does not overwrite it.
    ev = next((p for p in (run / "eval.json", run / "eval_evalonly.json") if p.is_file()), None)
    if ev is None:
        return None
    out = {"name": run.name, "cfg": json.loads(cfg.read_text()) if cfg.is_file() else {}}
    out.update(json.loads(ev.read_text()))
    # One file per score mode (hungarian_eval_<mode>.json), plus the legacy unsuffixed
    # name used before the mode was recorded in the filename. Prefer detector_x_class:
    # it is the mode every historical run was scored with, so it is the comparable one.
    found = {}
    for p in sorted(run.glob("hungarian_eval*.json")):
        mode = p.stem.replace("hungarian_eval", "").lstrip("_") or "unspecified"
        found[mode] = json.loads(p.read_text())
    for mode in ("detector_x_class", "unspecified", "detector"):
        if mode in found:
            out["hungarian"], out["hungarian_mode"] = found[mode], mode
            break
    else:
        if found:
            mode, data = next(iter(found.items()))
            out["hungarian"], out["hungarian_mode"] = data, mode
    return out


def labels(runs: list[dict]) -> list[str]:
    """Column headings: each run name with its 14-digit timestamp prefix dropped.

    Do not shorten these by truncating to a fixed width. The distinguishing token sits at
    the FRONT of the config part, so keeping a fixed-length tail collapses runs that share
    a long suffix - `..._cxt_c448_dil0_lr1e4` and `..._r50_c448_dil0_lr1e4` both end in
    `448_dil0_lr1e4` and printed as two identically-headed columns.
    """
    stems = [re.sub(r"^\d{14}_", "", r["name"]) for r in runs]
    # Two runs of the same config at different times collide once the prefix is gone.
    return stems if len(set(stems)) == len(stems) else [r["name"] for r in runs]


def main(argv: list[str]) -> int:
    runs = [r for r in (load(Path(p)) for p in argv) if r]
    if not runs:
        print("no runs with an eval.json found")
        return 1

    # Full names here (the timestamp is provenance); size the column so knobs stay aligned.
    n_w = max(28, max(len(r["name"]) for r in runs) + 2)
    print(f"\n{'run':<{n_w}}{'top1':>8}{'vs base':>9}{'val_loss':>10}{'n':>7}   knobs")
    print(f"{BASELINE['name']:<{n_w}}{BASELINE['top1_matched']:>8.4f}{'--':>9}{'--':>10}{1114:>7}")
    print("-" * (n_w + 68))
    for r in sorted(runs, key=lambda x: -x.get("val_top1", 0)):
        c = r["cfg"]
        knobs = " ".join(
            f"{k}={c[k]}" for k in KNOBS if k in c and k not in ("crop_size", "ims_per_batch")
        )
        print(
            f"{r['name']:<{n_w}}{r.get('val_top1', float('nan')):>8.4f}"
            f"{r.get('val_top1', 0) - BASELINE['top1_matched']:>+9.4f}"
            f"{r.get('val_loss', float('nan')):>10.4f}{r.get('val_n', 0):>7}   {knobs}"
        )

    classes = list(BASELINE["per_class"])
    head = labels(runs)
    w = max(16, max(len(h) for h in head) + 2)
    print(f"\nper-class top-1\n{'class':<18}{'baseline':>10}", end="")
    for h in head:
        print(f"{h:>{w}}", end="")
    print()
    print("-" * (28 + w * len(runs)))
    for cl in classes:
        print(f"{cl:<18}{BASELINE['per_class'][cl]:>10.4f}", end="")
        for r in runs:
            v = r.get("per_class", {}).get(cl)
            print(f"{v:>{w}.4f}" if v is not None else f"{'-':>{w}}", end="")
        print()

    hruns = [r for r in runs if "hungarian" in r]
    if hruns:
        hhead = labels(hruns)
        print("\nHungarian evaluator (same code as the baseline)")
        for key, base in (("classagnostic_f1", BASELINE["classagnostic_f1"]),
                          ("classaware_f1", BASELINE["classaware_f1"])):
            print(f"  {key:<22} baseline {base:.4f}", end="")
            for h, r in zip(hhead, hruns):
                v = r["hungarian"].get(key)
                print(f" | {h} {v:.4f} ({v - base:+.4f})" if v else "", end="")
            print()
        modes = {r.get("hungarian_mode") for r in hruns}
        if modes != {"detector_x_class"}:
            # A mixed table is not comparable: score_mode changes which predictions clear
            # SCORE_THRESH, so it moves classaware_f1 on its own.
            print("  score_mode per run: "
                  + ", ".join(f"{h}={r.get('hungarian_mode')}"
                              for h, r in zip(hhead, hruns)))
    else:
        print("\n(no hungarian_eval_<mode>.json yet - run with --eval-only --hungarian for "
              "the classaware_f1 comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
