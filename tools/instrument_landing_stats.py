#!/usr/bin/env python
"""Class-independent landing heatmaps + dataset-level co-occurrence/duplicate stats.

Scans one or more folders of ``.hdf5`` frames written in the ``sgdata`` schema
(see ``scene_generator``'s ``docs/hdf5_schema.md``) and, per folder, produces:

* a **landing heatmap** -- where instruments end up on the mat, class-independent:
  - pixel coverage: fraction of frames in which each pixel is covered by *any*
    instrument instance (decoded from the embedded ``coco_annotations`` RLE masks),
  - centroid density: 2D histogram of per-instance mask centroids,
  - coverage support: discrete never / <0.1% / 0.1-1% / 1-10% / >10%-of-frames
    levels, i.e. which parts of the frame instruments essentially never reach.
* a **co-occurrence matrix** over classes: off-diagonal = number of frames in which
  both classes appear, diagonal = number of frames in which that class appears
  *more than once* (i.e. duplicates).
* **duplicate statistics**: per-class multiplicity histograms, how often a frame
  contains any duplicated class, instances-per-frame distribution.

Only the small ``coco_annotations`` / ``instrument_classes`` / ``metadata_json``
payloads are read -- never the RGB/depth arrays -- so a 1k-frame set scans in
seconds.

Frames rendered from the same physics simulation (same ``simulation_hash`` in
``metadata_json``, typically 2 camera-jittered frames per drop) share a tool
composition, so every count that describes composition rather than pixels is
reported twice: per *frame* and per *simulation* (deduplicated).

The coverage panel is log-scaled by default (the tail spans three decades);
``--linear-coverage`` switches back. ``--cmap`` picks the density colormap
(inferno/magma/viridis/cividis/plasma, or ``blue`` for a single-hue ramp) -- pair
the matplotlib ones with ``--theme dark``.

Usage
-----
    python tools/instrument_landing_stats.py \\
        ~/train_images/val_set_a_mm ~/train_images/val_set_ab_mm \\
        --out ~/dev/MaskDINO/output/landing_stats --theme dark --cmap inferno

Outputs per dataset (``<out>/<dataset-name>_*``): ``landing_heatmap.png``,
``cooccurrence.png``, ``instances_per_frame.png``, ``stats.json``,
``cooccurrence.csv``, ``class_stats.csv``, ``coverage.npy`` (raw accumulator, for
re-plotting), plus a shared ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from pycocotools import mask as mask_util

try:  # schema constants, when the sgdata package is importable
    from sgdata import schema
except ImportError:  # standalone fallback -- keys are stable, see docs/hdf5_schema.md

    class schema:  # mimics the module namespace
        COCO_ANNOTATIONS = "coco_annotations"
        INSTRUMENT_CLASSES = "instrument_classes"
        METADATA_ATTR = "metadata_json"
        BACKGROUND_CATEGORY_ID = 0


# ---------------------------------------------------------------------------
# palette (sequential single-hue blue ramp, light -> dark; zero recedes to surface)
# ---------------------------------------------------------------------------

SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "grid": "#dedcd5"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7", "grid": "#3a3a37"},
}


# Colormaps for the density heatmaps. All are single-progression (dark->bright or
# light->dark) and perceptually ordered -- no rainbow/jet, which invents false
# boundaries in a continuous density. "blue" keeps zero at the chart surface;
# the matplotlib ones put zero at their own dark end, which reads best on --theme dark.
HEATMAP_CMAPS = ("inferno", "magma", "viridis", "cividis", "plasma", "blue")


def density_cmap(name: str, surface: str):
    """Colormap for a density field; the lowest step is anchored to the chart surface."""
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if name == "blue":
        return LinearSegmentedColormap.from_list("seq_blue", [surface, *SEQUENTIAL_BLUE])
    return mpl.colormaps[name]


def sequential_cmap(surface: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq_blue", [surface, *SEQUENTIAL_BLUE])


# ---------------------------------------------------------------------------
# per-file scan (runs in worker processes)
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """Everything one frame contributes that is not a pixel accumulator."""

    path: str
    sim_hash: str | None
    image_index: int | None
    class_counts: dict[int, int] = field(default_factory=dict)
    areas: list[tuple[int, float]] = field(default_factory=list)  # (category_id, mask area px)


def _decode_text(value) -> str:
    return value.decode() if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value)


def _scan_chunk(paths: list[str], bins: int):
    """Scan a list of frames; return (coverage, centroid_hist, shape, records, errors)."""
    coverage: np.ndarray | None = None
    centroid_hist = np.zeros((bins, bins), dtype=np.int64)
    shape: tuple[int, int] | None = None
    records: list[FrameRecord] = []
    errors: list[tuple[str, str]] = []

    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                anns = json.loads(_decode_text(f[schema.COCO_ANNOTATIONS][()]))
                meta_raw = f.attrs.get(schema.METADATA_ATTR)
                meta = json.loads(_decode_text(meta_raw)) if meta_raw is not None else {}
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the scan
            errors.append((path, f"{type(exc).__name__}: {exc}"))
            continue

        rec = FrameRecord(
            path=path,
            sim_hash=meta.get("simulation_hash"),
            image_index=meta.get("image_index"),
        )
        counts: Counter[int] = Counter()

        for ann in anns:
            cid = int(ann["category_id"])
            if cid == schema.BACKGROUND_CATEGORY_ID:
                continue
            seg = ann["segmentation"]
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode()
            m = mask_util.decode(rle).astype(bool)

            if shape is None:
                shape = m.shape
                coverage = np.zeros(shape, dtype=np.float32)
            elif m.shape != shape:
                errors.append((path, f"frame size {m.shape} != {shape}, skipped instance"))
                continue

            coverage += m
            ys, xs = np.nonzero(m)
            if ys.size:
                cy, cx = ys.mean(), xs.mean()
                iy = min(int(cy / shape[0] * bins), bins - 1)
                ix = min(int(cx / shape[1] * bins), bins - 1)
                centroid_hist[iy, ix] += 1
            counts[cid] += 1
            rec.areas.append((cid, float(m.sum())))

        rec.class_counts = dict(counts)
        records.append(rec)

    return coverage, centroid_hist, shape, records, errors


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _multiplicity_stats(compositions: list[dict[int, int]], class_ids: list[int]) -> dict:
    """Co-occurrence / duplicate statistics over a list of per-unit class->count maps."""
    n = len(compositions)
    idx = {cid: i for i, cid in enumerate(class_ids)}
    k = len(class_ids)

    cooc = np.zeros((k, k), dtype=np.int64)  # off-diag: both present; diag: present >= 2x
    present = np.zeros(k, dtype=np.int64)
    instances = np.zeros(k, dtype=np.int64)
    multiplicity = {cid: Counter() for cid in class_ids}
    per_unit_totals = Counter()
    units_with_any_duplicate = 0

    for comp in compositions:
        live = [cid for cid, c in comp.items() if c > 0]
        total = sum(comp.values())
        per_unit_totals[total] += 1
        has_dup = False
        for cid in live:
            i = idx[cid]
            present[i] += 1
            instances[i] += comp[cid]
            multiplicity[cid][comp[cid]] += 1
            if comp[cid] >= 2:
                cooc[i, i] += 1
                has_dup = True
        if has_dup:
            units_with_any_duplicate += 1
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                i, j = idx[live[a]], idx[live[b]]
                cooc[i, j] += 1
                cooc[j, i] += 1

    # lift: P(i,j) / (P(i) P(j)) -- >1 means the pair shows up together more than
    # independent sampling of the two classes would predict.
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            both = int(cooc[i, j])
            exp = present[i] * present[j] / n if n and present[i] and present[j] else 0.0
            pairs.append(
                {
                    "class_a": class_ids[i],
                    "class_b": class_ids[j],
                    "frames_both": both,
                    "expected_if_independent": round(exp, 2),
                    "lift": round(both / exp, 3) if exp else None,
                }
            )
    pairs.sort(key=lambda p: -p["frames_both"])

    totals_hist = {int(t): int(c) for t, c in sorted(per_unit_totals.items())}
    counts_flat = np.repeat(
        np.array(list(totals_hist.keys()) or [0]), np.array(list(totals_hist.values()) or [0])
    )

    return {
        "units": n,
        "cooccurrence": cooc,
        "present": present,
        "instances": instances,
        "multiplicity": {cid: {int(m): int(c) for m, c in sorted(v.items())} for cid, v in multiplicity.items()},
        "units_with_any_duplicate": units_with_any_duplicate,
        "frac_units_with_any_duplicate": round(units_with_any_duplicate / n, 4) if n else 0.0,
        "instances_per_unit_hist": totals_hist,
        "instances_per_unit": {
            "mean": round(float(counts_flat.mean()), 3) if counts_flat.size else 0.0,
            "median": float(np.median(counts_flat)) if counts_flat.size else 0.0,
            "min": int(counts_flat.min()) if counts_flat.size else 0,
            "max": int(counts_flat.max()) if counts_flat.size else 0,
            "empty_units": int(totals_hist.get(0, 0)),
        },
        "top_pairs": pairs,
    }


@dataclass
class DatasetResult:
    name: str
    root: Path
    n_files: int
    coverage: np.ndarray
    centroid_hist: np.ndarray
    shape: tuple[int, int]
    class_names: dict[int, str]
    class_ids: list[int]
    per_frame: dict
    per_sim: dict
    area_stats: dict
    support: dict
    errors: list[tuple[str, str]]


def scan_dataset(root: Path, bins: int, workers: int, max_files: int | None) -> DatasetResult:
    files = sorted(str(p) for p in root.glob("*.hdf5"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"no .hdf5 frames found in {root}")

    class_names = read_class_names(files)

    chunks = [files[i::workers] for i in range(workers)] if workers > 1 else [files]
    chunks = [c for c in chunks if c]
    if len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            results = list(pool.map(_scan_chunk, chunks, [bins] * len(chunks)))
    else:
        results = [_scan_chunk(chunks[0], bins)]

    coverage = None
    centroid_hist = np.zeros((bins, bins), dtype=np.int64)
    shape = None
    records: list[FrameRecord] = []
    errors: list[tuple[str, str]] = []
    for cov, hist, shp, recs, errs in results:
        if cov is not None:
            coverage = cov if coverage is None else coverage + cov
            shape = shp
        centroid_hist += hist
        records.extend(recs)
        errors.extend(errs)
    if coverage is None or shape is None:
        raise SystemExit(f"{root}: no decodable instances found in {len(files)} frames")

    class_ids = sorted(class_names)
    frame_comps = [r.class_counts for r in records]
    per_frame = _multiplicity_stats(frame_comps, class_ids)

    # one composition per simulation (frames of the same drop share tools)
    by_sim: dict[str, dict[int, int]] = {}
    for i, r in enumerate(records):
        key = r.sim_hash or f"__file_{i}"
        # frames of a sim can differ by occlusion filtering; keep the richest one
        prev = by_sim.get(key)
        if prev is None or sum(r.class_counts.values()) > sum(prev.values()):
            by_sim[key] = r.class_counts
    per_sim = _multiplicity_stats(list(by_sim.values()), class_ids)

    areas: dict[int, list[float]] = {cid: [] for cid in class_ids}
    for r in records:
        for cid, a in r.areas:
            areas.setdefault(cid, []).append(a)
    px = float(shape[0] * shape[1])
    area_stats = {
        cid: {
            "n": len(v),
            "median_px": round(float(np.median(v)), 1),
            "mean_px": round(float(np.mean(v)), 1),
            "median_frac_of_image": round(float(np.median(v)) / px, 5),
        }
        for cid, v in areas.items()
        if v
    }

    return DatasetResult(
        name=root.name,
        root=root,
        n_files=len(records),
        coverage=coverage,
        centroid_hist=centroid_hist,
        shape=shape,
        class_names=class_names,
        class_ids=class_ids,
        per_frame=per_frame,
        per_sim=per_sim,
        area_stats=area_stats,
        support=coverage_support_stats(coverage, len(records)),
        errors=errors,
    )


def read_class_names(files: list[str]) -> dict[int, str]:
    """``instrument_classes`` is index-aligned: index == category_id (0 == background)."""
    for path in files[:16]:
        try:
            with h5py.File(path, "r") as f:
                if schema.INSTRUMENT_CLASSES not in f:
                    continue
                names = json.loads(_decode_text(f[schema.INSTRUMENT_CLASSES][()]))
        except Exception:  # noqa: BLE001, S112 - try the next file instead
            continue
        return {i: n for i, n in enumerate(names) if i != schema.BACKGROUND_CATEGORY_ID}
    return {}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _style(theme: dict):
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["primary"],
            "axes.labelcolor": theme["secondary"],
            "axes.edgecolor": theme["grid"],
            "xtick.color": theme["secondary"],
            "ytick.color": theme["secondary"],
            "font.size": 10,
        }
    )


SUPPORT_LEVELS = (0.0, 0.001, 0.01, 0.10)  # fraction-of-frames boundaries
SUPPORT_LABELS = (
    "never covered",
    "< 0.1% of frames",
    "0.1 - 1%",
    "1 - 10%",
    "> 10% of frames",
)


def largest_centered_square(coverage: np.ndarray, min_count: float) -> int:
    """Side length of the largest image-centered square whose every pixel was covered
    at least ``min_count`` times. Monotone in the side length, so binary-searched."""
    h, w = coverage.shape
    cy, cx = h // 2, w // 2
    lo, hi = 0, min(cy, cx, h - cy, w - cx)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        box = coverage[cy - mid : cy + mid, cx - mid : cx + mid]
        if box.min() >= min_count:
            lo = mid
        else:
            hi = mid - 1
    return 2 * lo


def coverage_support_stats(coverage: np.ndarray, n_frames: int) -> dict:
    """How much of the frame the instruments actually reach, and how thin the tail is."""
    n = max(n_frames, 1)
    frac = coverage / n
    total_px = coverage.size
    h, w = coverage.shape
    never = coverage == 0

    # how much of the never-covered area sits in the outer border ring vs. the interior
    border = np.zeros_like(never)
    bh, bw = h // 8, w // 8
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True

    levels = {}
    prev = 0.0
    for i, hi in enumerate((*SUPPORT_LEVELS[1:], 1.0001)):
        sel = (frac > prev) & (frac <= hi)
        levels[SUPPORT_LABELS[i + 1]] = round(float(sel.mean()), 5)
        prev = hi

    covered = frac[~never]
    return {
        "frames": n_frames,
        "image_px": int(total_px),
        "never_covered_px": int(never.sum()),
        "never_covered_frac": round(float(never.mean()), 5),
        "never_covered_in_border_eighth_frac": (
            round(float((never & border).sum() / never.sum()), 4) if never.any() else None
        ),
        "covered_at_most_once_frac": round(float((coverage <= 1).mean()), 5),
        "level_area_fractions": {SUPPORT_LABELS[0]: round(float(never.mean()), 5), **levels},
        "coverage_frac_percentiles_over_covered_px": {
            f"p{p}": round(float(np.percentile(covered, p)), 6) for p in (5, 25, 50, 75, 95, 99)
        }
        if covered.size
        else {},
        "max_coverage_frac": round(float(frac.max()), 5),
        "largest_centered_square_px": {
            "covered_at_least_once": largest_centered_square(coverage, 1),
            "covered_in_0.1pct_of_frames": largest_centered_square(coverage, 0.001 * n),
            "covered_in_1pct_of_frames": largest_centered_square(coverage, 0.01 * n),
        },
    }


def plot_landing_heatmap(
    res: DatasetResult, out: Path, theme: dict, dpi: int, cmap_name: str, log_scale: bool
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch

    cmap = density_cmap(cmap_name, theme["surface"])
    never_color = "#4a4a46" if theme is THEMES["dark"] else "#d9d8d2"
    n = max(res.n_files, 1)
    cov = res.coverage / n  # fraction of frames in which the pixel is covered

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.9))

    # --- panel 1: coverage field (log scale by default -- the tail spans 3+ decades)
    if log_scale:
        data = np.ma.masked_where(res.coverage == 0, cov)
        norm = LogNorm(vmin=max(1.0 / n, cov[cov > 0].min() if (cov > 0).any() else 1.0 / n), vmax=cov.max())
        panel_cmap = cmap.with_extremes(bad=never_color)
        scale_note = "log scale; grey = never covered"
    else:
        data, norm, panel_cmap = cov, None, cmap
        scale_note = "linear scale"
    im0 = axes[0].imshow(data, cmap=panel_cmap, norm=norm, interpolation="nearest", origin="upper")
    axes[0].set_title(f"Pixel coverage ({scale_note})", color=theme["primary"], pad=8)
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("fraction of frames covered by any instrument", color=theme["secondary"])
    cb0.outline.set_edgecolor(theme["grid"])

    # --- panel 2: centroid density
    bins = res.centroid_hist.shape[0]
    im1 = axes[1].imshow(
        res.centroid_hist,
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        extent=(0, res.shape[1], res.shape[0], 0),
    )
    axes[1].set_title(f"Centroid density ({bins}x{bins} bins)", color=theme["primary"], pad=8)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("instance centroids per bin", color=theme["secondary"])
    cb1.outline.set_edgecolor(theme["grid"])

    # --- panel 3: discrete support levels -- answers "what is never/almost never covered?"
    steps = [cmap(v) for v in (0.35, 0.55, 0.75, 0.95)]
    level_cmap = ListedColormap([never_color, *steps])
    bounds = [-1e-9, *SUPPORT_LEVELS[1:], 1.0001]
    im2 = axes[2].imshow(
        cov,
        cmap=level_cmap,
        norm=BoundaryNorm(bounds, level_cmap.N),
        interpolation="nearest",
        origin="upper",
    )
    im2.set_clim(0, 1)
    support = res.support
    axes[2].set_title(
        f"Coverage support - {100 * support['never_covered_frac']:.1f}% of the image never covered",
        color=theme["primary"],
        pad=8,
    )
    axes[2].legend(
        handles=[
            Patch(facecolor=c, edgecolor=theme["surface"], linewidth=1.5,
                  label=f"{lab}  ({100 * support['level_area_fractions'][lab]:.1f}%)")
            for c, lab in zip([never_color, *steps], SUPPORT_LABELS)
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=9,
        labelcolor=theme["secondary"],
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    total = int(res.per_frame["instances"].sum())
    fig.suptitle(
        f"Where instruments land - {res.name}  (class-independent)",
        color=theme["primary"],
        fontsize=14,
        y=0.98,
    )
    sq = support["largest_centered_square_px"]
    fig.text(
        0.5,
        0.015,
        f"{res.n_files} frames - {total} instances - "
        f"{res.per_frame['instances_per_unit']['mean']} instances/frame mean - "
        f"largest centred square covered at least once: {sq['covered_at_least_once']}px, "
        f"in >=1% of frames: {sq['covered_in_1pct_of_frames']}px",
        ha="center",
        color=theme["secondary"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 0.995, 0.93))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_cooccurrence(res: DatasetResult, stats: dict, out: Path, theme: dict, dpi: int, unit: str) -> None:
    import matplotlib.pyplot as plt

    live = [i for i, cid in enumerate(res.class_ids) if stats["present"][i] > 0]
    if not live:
        return
    ids = [res.class_ids[i] for i in live]
    labels = [res.class_names.get(cid, f"id {cid}") for cid in ids]
    m = stats["cooccurrence"][np.ix_(live, live)].astype(float)

    cmap = sequential_cmap(theme["surface"])
    vmax = max(m.max(), 1.0)
    size = max(6.0, 0.62 * len(ids) + 3.2)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(ids)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ids)), labels)
    ax.set_xticks(np.arange(-0.5, len(ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
    # 2px surface gap between cells
    ax.grid(which="minor", color=theme["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(len(ids)):
        for j in range(len(ids)):
            v = int(m[i, j])
            if v == 0:
                continue
            dark_cell = m[i, j] / vmax > 0.55
            ax.text(
                j,
                i,
                v,
                ha="center",
                va="center",
                fontsize=8,
                color=theme["surface"] if dark_cell else theme["primary"],
                fontweight="bold" if i == j else "normal",
            )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"{unit}s", color=theme["secondary"])
    cb.outline.set_edgecolor(theme["grid"])

    ax.set_title(
        f"Class co-occurrence - {res.name}  (per {unit})",
        color=theme["primary"],
        pad=12,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        f"off-diagonal: {unit}s containing both classes  -  "
        f"diagonal (bold): {unit}s containing 2+ of that class (duplicates)  -  "
        f"{stats['units']} {unit}s total",
        ha="center",
        color=theme["secondary"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_instances_per_frame(res: DatasetResult, out: Path, theme: dict, dpi: int) -> None:
    import matplotlib.pyplot as plt

    hist = res.per_frame["instances_per_unit_hist"]
    dup = res.per_frame["multiplicity"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    xs = sorted(hist)
    axes[0].bar(xs, [hist[x] for x in xs], color=SEQUENTIAL_BLUE[7], width=0.7)
    axes[0].set_title("Instances per frame", color=theme["primary"], pad=8)
    axes[0].set_xlabel("annotated instances in frame")
    axes[0].set_ylabel("frames")
    axes[0].set_xticks(xs)

    # per-class duplicate rate: share of frames-with-this-class that hold 2+ of it
    rows = []
    for i, cid in enumerate(res.class_ids):
        present = int(res.per_frame["present"][i])
        if not present:
            continue
        dups = sum(c for m, c in dup[cid].items() if m >= 2)
        rows.append((res.class_names.get(cid, f"id {cid}"), 100.0 * dups / present, dups, present))
    rows.sort(key=lambda r: -r[1])
    if rows:
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        axes[1].barh(range(len(rows)), vals, color=SEQUENTIAL_BLUE[7], height=0.68)
        axes[1].set_yticks(range(len(rows)), names, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("% of frames containing the class that contain 2+ of it")
        axes[1].set_title("Duplicate rate per class", color=theme["primary"], pad=8)
        for i, (_, v, d, p) in enumerate(rows):
            axes[1].text(v + max(vals) * 0.015, i, f"{v:.0f}%  ({d}/{p})", va="center", fontsize=8,
                         color=theme["secondary"])
        axes[1].set_xlim(0, max(vals) * 1.28 if max(vals) else 1)

    for ax in axes:
        ax.grid(axis="x" if ax is axes[1] else "y", color=theme["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(f"Composition & duplicates - {res.name}", color=theme["primary"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def _stats_to_json(res: DatasetResult, stats: dict) -> dict:
    names = res.class_names
    return {
        "units": stats["units"],
        "instances_total": int(stats["instances"].sum()),
        "instances_per_unit": stats["instances_per_unit"],
        "instances_per_unit_hist": stats["instances_per_unit_hist"],
        "units_with_any_duplicate": stats["units_with_any_duplicate"],
        "frac_units_with_any_duplicate": stats["frac_units_with_any_duplicate"],
        "per_class": {
            names.get(cid, f"id {cid}"): {
                "category_id": cid,
                "instances": int(stats["instances"][i]),
                "units_present": int(stats["present"][i]),
                "units_with_duplicates": int(stats["cooccurrence"][i, i]),
                "multiplicity_hist": stats["multiplicity"][cid],
                "mean_count_when_present": (
                    round(float(stats["instances"][i] / stats["present"][i]), 3)
                    if stats["present"][i]
                    else 0.0
                ),
            }
            for i, cid in enumerate(res.class_ids)
        },
        "cooccurrence_matrix": {
            "class_ids": res.class_ids,
            "class_names": [names.get(cid, f"id {cid}") for cid in res.class_ids],
            "note": "off-diagonal = units containing both; diagonal = units containing 2+ of that class",
            "counts": stats["cooccurrence"].tolist(),
        },
        "top_pairs": [
            {
                **p,
                "class_a_name": names.get(p["class_a"], f"id {p['class_a']}"),
                "class_b_name": names.get(p["class_b"], f"id {p['class_b']}"),
            }
            for p in stats["top_pairs"][:40]
        ],
    }


def write_csvs(res: DatasetResult, out_dir: Path, prefix: str) -> None:
    import csv

    names = [res.class_names.get(cid, f"id {cid}") for cid in res.class_ids]
    with (out_dir / f"{prefix}_cooccurrence.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", *names])
        for i, n in enumerate(names):
            w.writerow([n, *res.per_frame["cooccurrence"][i].tolist()])

    with (out_dir / f"{prefix}_class_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "class", "category_id", "instances", "frames_present", "frames_with_duplicates",
                "mean_count_when_present", "max_count_in_frame", "sims_present",
                "sims_with_duplicates", "median_mask_area_px",
            ]
        )
        for i, cid in enumerate(res.class_ids):
            pf, ps = res.per_frame, res.per_sim
            mult = pf["multiplicity"][cid]
            w.writerow(
                [
                    res.class_names.get(cid, f"id {cid}"),
                    cid,
                    int(pf["instances"][i]),
                    int(pf["present"][i]),
                    int(pf["cooccurrence"][i, i]),
                    round(float(pf["instances"][i] / pf["present"][i]), 3) if pf["present"][i] else 0.0,
                    max(mult) if mult else 0,
                    int(ps["present"][i]),
                    int(ps["cooccurrence"][i, i]),
                    res.area_stats.get(cid, {}).get("median_px", ""),
                ]
            )


def summary_md(results: list[DatasetResult]) -> str:
    lines = ["# Instrument landing & composition statistics", ""]
    for res in results:
        pf, ps = res.per_frame, res.per_sim
        cov_any = float((res.coverage > 0).mean())
        cov_mean = float(res.coverage.mean() / max(res.n_files, 1))
        sup = res.support
        lines += [
            f"## {res.name}",
            "",
            f"- source: `{res.root}`",
            f"- frames scanned: **{res.n_files}** ({ps['units']} distinct simulations)",
            (
                f"- instances: **{int(pf['instances'].sum())}**, "
                f"{pf['instances_per_unit']['mean']} per frame on average "
                f"(median {pf['instances_per_unit']['median']:g}, max {pf['instances_per_unit']['max']})"
            ),
            f"- empty frames (no annotated instrument): **{pf['instances_per_unit']['empty_units']}**",
            (
                f"- frames with at least one duplicated class: **{pf['units_with_any_duplicate']}** "
                f"({100 * pf['frac_units_with_any_duplicate']:.1f}%); per simulation: "
                f"{ps['units_with_any_duplicate']} ({100 * ps['frac_units_with_any_duplicate']:.1f}%)"
            ),
            (
                f"- image area ever covered by an instrument: **{100 * cov_any:.1f}%**; "
                f"mean per-frame coverage: {100 * cov_mean:.2f}%"
            ),
            (
                f"- never covered: **{100 * sup['never_covered_frac']:.1f}%** of the image "
                f"({sup['never_covered_px']} px, {100 * (sup['never_covered_in_border_eighth_frac'] or 0):.0f}% of "
                f"them in the outer border eighth); covered in <0.1% of frames: "
                f"{100 * sup['level_area_fractions']['< 0.1% of frames']:.1f}%; "
                f"in >10% of frames: {100 * sup['level_area_fractions']['> 10% of frames']:.1f}%"
            ),
            (
                f"- largest image-centred square covered at least once: "
                f"**{sup['largest_centered_square_px']['covered_at_least_once']}px**; "
                f"covered in >=0.1% of frames: "
                f"{sup['largest_centered_square_px']['covered_in_0.1pct_of_frames']}px; "
                f"in >=1%: {sup['largest_centered_square_px']['covered_in_1pct_of_frames']}px"
            ),
            "",
            "### Per-class (per frame)",
            "",
            "| class | instances | frames present | frames with 2+ | mean count when present | median mask px |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        order = sorted(range(len(res.class_ids)), key=lambda i: -pf["instances"][i])
        for i in order:
            cid = res.class_ids[i]
            if pf["present"][i] == 0:
                continue
            lines.append(
                f"| {res.class_names.get(cid, f'id {cid}')} | {int(pf['instances'][i])} | "
                f"{int(pf['present'][i])} | {int(pf['cooccurrence'][i, i])} | "
                f"{pf['instances'][i] / pf['present'][i]:.2f} | "
                f"{res.area_stats.get(cid, {}).get('median_px', '-')} |"
            )
        lines += ["", "### Most frequent class pairs (per frame)", "",
                  "| pair | frames together | expected if independent | lift |", "|---|---:|---:|---:|"]
        for p in pf["top_pairs"][:12]:
            if not p["frames_both"]:
                continue
            a = res.class_names.get(p["class_a"], f"id {p['class_a']}")
            b = res.class_names.get(p["class_b"], f"id {p['class_b']}")
            lift = "-" if p["lift"] is None else f"{p['lift']:.2f}"
            lines.append(f"| {a} + {b} | {p['frames_both']} | {p['expected_if_independent']:.1f} | {lift} |")
        lines += ["", f"- instances per frame histogram: `{pf['instances_per_unit_hist']}`", ""]
        if res.errors:
            lines += [f"- **{len(res.errors)} frames/instances were skipped** (see stats.json)", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", type=Path, help="folders of .hdf5 frames")
    ap.add_argument("--out", type=Path, required=True, help="output folder for plots/stats")
    ap.add_argument("--bins", type=int, default=64, help="centroid-histogram bins per axis (default 64)")
    ap.add_argument("--workers", type=int, default=8, help="parallel scan workers (default 8)")
    ap.add_argument("--max-files", type=int, default=None, help="scan only the first N frames (smoke test)")
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    ap.add_argument(
        "--cmap",
        choices=HEATMAP_CMAPS,
        default="inferno",
        help="density-heatmap colormap (default inferno; pair the matplotlib ones with --theme dark)",
    )
    ap.add_argument(
        "--linear-coverage",
        action="store_true",
        help="plot the coverage panel on a linear scale (default: log, which reveals the thin tail)",
    )
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument(
        "--cooccurrence-unit",
        choices=("frame", "simulation"),
        default="frame",
        help="unit for the co-occurrence plot; both are always in stats.json (default frame)",
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    theme = THEMES[args.theme]
    _style(theme)

    results: list[DatasetResult] = []
    for root in args.datasets:
        root = root.expanduser().resolve()
        print(f"[scan] {root}", flush=True)
        res = scan_dataset(root, args.bins, max(1, args.workers), args.max_files)
        prefix = res.name
        print(
            f"       {res.n_files} frames, {int(res.per_frame['instances'].sum())} instances, "
            f"{len(res.area_stats)} classes seen"
            + (f", {len(res.errors)} issues" if res.errors else ""),
            flush=True,
        )

        plot_landing_heatmap(
            res,
            args.out / f"{prefix}_landing_heatmap.png",
            theme,
            args.dpi,
            args.cmap,
            not args.linear_coverage,
        )
        stats = res.per_frame if args.cooccurrence_unit == "frame" else res.per_sim
        plot_cooccurrence(
            res, stats, args.out / f"{prefix}_cooccurrence.png", theme, args.dpi, args.cooccurrence_unit
        )
        plot_instances_per_frame(res, args.out / f"{prefix}_instances_per_frame.png", theme, args.dpi)

        np.save(args.out / f"{prefix}_coverage.npy", res.coverage)
        np.save(args.out / f"{prefix}_centroid_hist.npy", res.centroid_hist)
        write_csvs(res, args.out, prefix)
        (args.out / f"{prefix}_stats.json").write_text(
            json.dumps(
                {
                    "dataset": res.name,
                    "root": str(res.root),
                    "frames_scanned": res.n_files,
                    "image_shape": list(res.shape),
                    "class_names": {str(k): v for k, v in res.class_names.items()},
                    "coverage": {
                        "pixels_ever_covered_frac": round(float((res.coverage > 0).mean()), 5),
                        "mean_per_frame_coverage_frac": round(
                            float(res.coverage.mean() / max(res.n_files, 1)), 6
                        ),
                        "support": res.support,
                    },
                    "mask_area_by_class": {
                        res.class_names.get(cid, f"id {cid}"): v for cid, v in res.area_stats.items()
                    },
                    "per_frame": _stats_to_json(res, res.per_frame),
                    "per_simulation": _stats_to_json(res, res.per_sim),
                    "skipped": [{"path": p, "error": e} for p, e in res.errors[:200]],
                },
                indent=2,
            )
        )
        results.append(res)

    (args.out / "summary.md").write_text(summary_md(results))
    print(f"[done] wrote {len(results)} dataset report(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
