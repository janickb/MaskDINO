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
    ev = run / "eval.json"
    if not ev.is_file():
        return None
    out = {"name": run.name, "cfg": json.loads(cfg.read_text()) if cfg.is_file() else {}}
    out.update(json.loads(ev.read_text()))
    h = run / "hungarian_eval.json"
    if h.is_file():
        out["hungarian"] = json.loads(h.read_text())
    return out


def main(argv: list[str]) -> int:
    runs = [r for r in (load(Path(p)) for p in argv) if r]
    if not runs:
        print("no runs with an eval.json found")
        return 1

    print(f"\n{'run':<28}{'top1':>8}{'vs base':>9}{'val_loss':>10}{'n':>7}   knobs")
    print(f"{BASELINE['name']:<28}{BASELINE['top1_matched']:>8.4f}{'--':>9}{'--':>10}{1114:>7}")
    print("-" * 96)
    for r in sorted(runs, key=lambda x: -x.get("val_top1", 0)):
        c = r["cfg"]
        knobs = " ".join(
            f"{k}={c[k]}" for k in KNOBS if k in c and k not in ("crop_size", "ims_per_batch")
        )
        print(
            f"{r['name']:<28}{r.get('val_top1', float('nan')):>8.4f}"
            f"{r.get('val_top1', 0) - BASELINE['top1_matched']:>+9.4f}"
            f"{r.get('val_loss', float('nan')):>10.4f}{r.get('val_n', 0):>7}   {knobs}"
        )

    classes = list(BASELINE["per_class"])
    print(f"\nper-class top-1\n{'class':<18}{'baseline':>10}", end="")
    for r in runs:
        print(f"{r['name'][-14:]:>16}", end="")
    print()
    print("-" * (28 + 16 * len(runs)))
    for cl in classes:
        print(f"{cl:<18}{BASELINE['per_class'][cl]:>10.4f}", end="")
        for r in runs:
            v = r.get("per_class", {}).get(cl)
            print(f"{v:>16.4f}" if v is not None else f"{'-':>16}", end="")
        print()

    hruns = [r for r in runs if "hungarian" in r]
    if hruns:
        print(f"\nHungarian evaluator (same code as the baseline)")
        for key, base in (("classagnostic_f1", BASELINE["classagnostic_f1"]),
                          ("classaware_f1", BASELINE["classaware_f1"])):
            print(f"  {key:<22} baseline {base:.4f}", end="")
            for r in hruns:
                v = r["hungarian"].get(key)
                print(f" | {r['name'][-14:]} {v:.4f} ({v - base:+.4f})" if v else "", end="")
            print()
    else:
        print("\n(no hungarian_eval.json yet - run with --eval-only --hungarian for "
              "the classaware_f1 comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
