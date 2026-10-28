#!/usr/bin/env python
"""Compare N training runs the way scale_jitter_ab_phase2.md compares two.

Prints the three tables that write-up is built from, for any number of arms:

  1. headline per eval tick  - precision / recall / class-aware recall / f1 / mean IoU
  2. per-class class-aware recall at the final tick, with deltas vs the baseline arm
  3. one confusion row per arm for a chosen GT class (default forcep03_mm), read
     from inference/hungarian_confusion/iter_*/instance_matching_confusion_rownorm.csv

Reads metrics.json (written by EventStorage) and the per-iter confusion CSVs; it
never loads a checkpoint, so it is safe to run against a job that is still training.

Example::

    ./.venv/bin/python tools/compare_runs.py runs/*_aug_control_seed4* runs/*_aug_abl_jitteronly_*
    ./.venv/bin/python tools/compare_runs.py --gt-class forcep03_mm --baseline 0 runs/...

When arms include a seed-replicate of the same condition, pass it like any other
arm: its delta column IS the noise floor, and any per-class difference smaller than
it should be read as nothing.
"""
import argparse
import csv
import glob
import json
import os

_HEADLINE = [
    ("precision", "instance_matching/classagnostic_precision"),
    ("recall", "instance_matching/classagnostic_recall"),
    ("class-aware recall", "instance_matching/classaware_recall"),
    ("f1", "instance_matching/classagnostic_f1"),
    ("mean IoU", "instance_matching/mean_iou_matched"),
]
_PER_CLASS_PREFIX = "instance_matching/classaware_recall_per_class/"


def read_metrics(run_dir):
    """[(iteration, {metric: value})] for every tick that carries eval metrics."""
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return []
    ticks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line while the run is still writing
            if any(k.startswith("instance_matching/") for k in row):
                ticks.append((int(row.get("iteration", -1)), row))
    return ticks


def read_confusion_row(run_dir, gt_class):
    """{iter: {pred_class: fraction}} for one GT row, newest-last."""
    out = {}
    pattern = os.path.join(run_dir, "inference", "hungarian_confusion", "iter_*")
    for d in sorted(glob.glob(pattern)):
        csv_path = os.path.join(d, "instance_matching_confusion_rownorm.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = next(iter(row))  # the "gt\\pred" column, name varies by writer
                if row[key] == gt_class:
                    out[int(os.path.basename(d).split("_")[-1])] = {
                        k: float(v) for k, v in row.items() if k != key
                    }
    return out


OBJECTIVE_KEY = "instance_matching/classaware_f1"
OBJECTIVE_TICKS = 2


def _configured_max_iter(run_dir):
    """SOLVER.MAX_ITER from the run's own dumped config, or None if unreadable."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip().startswith("MAX_ITER:"):
                try:
                    return int(line.split(":", 1)[1])
                except ValueError:
                    return None
    return None


def objective_value(run_dir, key=OBJECTIVE_KEY, ticks=OBJECTIVE_TICKS):
    """The study's single scalar score for a run: `key` averaged over its last
    `ticks` eval ticks.

    Defined once, here, so every phase scores runs identically - a noise-floor
    replicate and a ladder arm must not be compared through two different
    definitions. Averaging the last two ticks damps single-eval noise, which
    matters because the whole study is read against a seed-to-seed sigma.

    Returns None - never a number - for a run that has not FINISHED. An earlier
    version just averaged whatever ticks existed, so a run still training at iter
    3999 scored 0.38 against a finished run's 0.65: a plausible-looking value
    computed from different iterations than everything it would be compared with.
    A half-trained run must be absent from a table, not silently near the bottom
    of it.
    """
    rows = [(it, row[key]) for it, row in read_metrics(run_dir)
            if key in row and row[key] is not None]
    if len(rows) < ticks:
        return None
    max_iter = _configured_max_iter(run_dir)
    if max_iter is not None and rows[-1][0] < max_iter - 1:
        return None  # still training / died early
    return sum(v for _it, v in rows[-ticks:]) / ticks


def _short(name, limit=20):
    return name if len(name) <= limit else ".." + name[-(limit - 2):]


def fmt(v, width=6):
    return "-".rjust(width) if v is None else f"{v:.3f}".rjust(width)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--gt-class", default="forcep03_mm")
    ap.add_argument("--baseline", type=int, default=0,
                    help="index into run_dirs that per-class deltas are taken against")
    args = ap.parse_args()

    # Run dirs are long and share a timestamp prefix, so the distinctive part is
    # the tail ("..._aug_abl_jitteronly_seed42"); keep that and elide the front, or the delta
    # columns collide with the value columns in the per-class table.
    names = [_short(os.path.basename(d.rstrip("/"))) for d in args.run_dirs]
    width = max(len(n) for n in names) + 3
    per_run = [read_metrics(d) for d in args.run_dirs]

    print("\n== headline per eval tick ==")
    print("iter".rjust(7) + "arm".rjust(width)
          + "".join(label.rjust(14) for label, _ in _HEADLINE))
    for name, ticks in zip(names, per_run):
        if not ticks:
            print(f"{'-':>7}{name:>{width}}   (no eval metrics yet)")
            continue
        for it, row in ticks:
            cells = "".join(fmt(row.get(key), 14) for _, key in _HEADLINE)
            print(f"{it:>7}{name:>{width}}{cells}")

    print(f"\n== final-tick class-aware recall per class (delta vs {names[args.baseline]}) ==")
    finals = [t[-1][1] if t else {} for t in per_run]
    base = finals[args.baseline]
    classes = sorted({k[len(_PER_CLASS_PREFIX):] for f in finals for k in f
                      if k.startswith(_PER_CLASS_PREFIX)})
    if classes:
        print("class".ljust(16) + "".join(n.rjust(width) for n in names)
              + "".join(("d:" + n).rjust(width + 2) for i, n in enumerate(names)
                        if i != args.baseline))
        rows = []
        for c in classes:
            vals = [f.get(_PER_CLASS_PREFIX + c) for f in finals]
            b = vals[args.baseline]
            deltas = [None if (v is None or b is None) else v - b
                      for i, v in enumerate(vals) if i != args.baseline]
            sort_key = max((abs(d) for d in deltas if d is not None), default=0.0)
            rows.append((sort_key, c, vals, deltas))
        for _, c, vals, deltas in sorted(rows, reverse=True):
            print(c.ljust(16)
                  + "".join(fmt(v, width) for v in vals)
                  + "".join(("" if d is None else f"{d:+.3f}").rjust(width + 2)
                            for d in deltas))
    else:
        print("  (no per-class metrics found)")

    print(f"\n== confusion row for GT {args.gt_class} ==")
    for name, d in zip(names, args.run_dirs):
        rows = read_confusion_row(d, args.gt_class)
        if not rows:
            print(f"{name}: (no hungarian_confusion CSVs yet)")
            continue
        it = max(rows)
        row = rows[it]
        correct = row.get(args.gt_class, 0.0)
        fn = row.get("(false negative)", 0.0)
        detected = 1.0 - fn
        leaks = sorted(((v, k) for k, v in row.items()
                        if k not in (args.gt_class, "(false negative)") and v > 0.01),
                       reverse=True)[:3]
        leak_s = ", ".join(f"->{k} {v:.3f}" for v, k in leaks) or "none >0.01"
        share = f"{correct / detected:.0%}" if detected > 0 else "-"
        print(f"{name} @iter {it}: correct {correct:.3f}, FN {fn:.3f}, "
              f"correct/detected {share} | {leak_s}")
    print()


if __name__ == "__main__":
    main()
