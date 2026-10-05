#!/usr/bin/env python
"""Where does the model MISS instruments? -- spatial false-negative analysis.

Runs a trained MaskDINO checkpoint over a registered HDF5 validation set, matches
predictions to ground truth exactly the way ``HungarianInstanceEvaluator`` does
(class-agnostic mask-IoU Hungarian assignment at the config's SCORE_THRESH /
IOU_THRESH, same MIN_VISIBILITY GT filter, same truncated-instance handling), and
accumulates every GT instance that came back **unmatched** into a heatmap.

Produced side by side with ``tools/instrument_landing_stats.py``'s landing heatmap,
this answers the question that motivates it: are the misses simply spread wherever
instruments happen to land (miss *density* tracks GT density, miss *rate* flat), or
is the model systematically worse in some part of the frame (miss rate structured)?
Both are reported -- the density correlation and the rate correlation -- because the
first one is nearly always high and on its own means nothing.

Non-spatial correlates are computed from the same matching pass, since they are the
usual real explanation for a spatial pattern: miss rate by instance mask area, by
GT visibility fraction (occlusion), by GT-density decile, and per class.

Usage
-----
    ./.venv/bin/python tools/missed_instrument_heatmap.py \\
        --config-file runs/<run>/config.yaml \\
        --weights runs/<run>/model_final.pth \\
        --dataset val_setab_mm \\
        --out output/landing_stats --theme dark --cmap inferno

``--dataset`` is a name registered in ``maskdino.data.datasets.register_hdf5_instance``
(``val_seta_mm``, ``val_setab_mm``, ``val_set_home``, ...). Outputs, per dataset:
``<ds>_missed_heatmap.png`` (GT / missed / miss-rate maps + radial profile),
``<ds>_miss_correlates.png`` (size, visibility, density, per-class), ``<ds>_miss_stats.json``,
``<ds>_miss_instances.csv`` (one row per GT instance), ``<ds>_miss_coverage.npy``, and
``missed_summary.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pycocotools.mask as mask_util
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from instrument_landing_stats import (
    HEATMAP_CMAPS,
    THEMES,
    _style,
    coverage_support_stats,
    density_cmap,
)
from scipy.stats import pearsonr, spearmanr

from maskdino import add_maskdino_config, set_num_classes_from_metadata
from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
    Hdf5CocoInstanceDatasetMapper,
)
from maskdino.data.datasets.register_hdf5_instance import (
    apply_truncated_instance_filter,
    touches_frame_edge,
)
from maskdino.evaluation.hungarian_instance_evaluation import (
    hungarian_match,
    mask_iou_matrix,
)

# ---------------------------------------------------------------------------
# inference + matching
# ---------------------------------------------------------------------------


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    # Archived run configs (runs/<run>/config.yaml) can carry keys that later schema
    # changes removed from add_maskdino_config; this is a read-only analysis tool, so
    # accept them instead of refusing to load an older run. --opts stays strict.
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = (args.dataset,)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    if args.device:
        cfg.MODEL.DEVICE = args.device
    set_num_classes_from_metadata(cfg, args.dataset)
    # Must precede apply_truncated_instance_filter: that helper iterates the dataset,
    # which never ends on a live pool (see freeze_live_pool_snapshot).
    freeze_live_pool_snapshot(args.dataset)
    # Same GT the evaluator sees: border-truncated instances dropped when configured.
    apply_truncated_instance_filter(cfg)
    cfg.freeze()
    return cfg


def freeze_live_pool_snapshot(dataset: str) -> None:
    """Materialize a live-pool train set into a plain list before anything iterates it.

    A pool split (LivePoolDataset, e.g. train_seta_mm) is a torch Dataset whose
    __getitem__ wraps the index into the pool's *current* file list
    (``files[idx % len(files)]``), so it never raises IndexError -- a bare
    ``for d in dataset`` over it never terminates. apply_truncated_instance_filter does
    exactly that, so without this the tool hangs, accumulating dicts until the machine
    runs out of memory. Taking exactly len(ds) items also pins one snapshot, so the
    truncation filter, the GT index and the loader all see identical frames even while
    renderers keep replenishing the pool.
    """
    ds = DatasetCatalog.get(dataset)
    if isinstance(ds, list):
        return
    snapshot = [ds[i] for i in range(len(ds))]
    DatasetCatalog.remove(dataset)
    DatasetCatalog.register(dataset, lambda frozen=snapshot: frozen)
    print(f"[pool] froze live-pool snapshot of {dataset}: {len(snapshot)} frames", flush=True)


def gt_masks_from_anns(anns, h, w, min_visibility):
    """Decode the GT the evaluator would score against -> (masks[M,h,w] bool, meta list)."""
    masks, meta = [], []
    for ann in anns:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("visibility_fraction", 1.0) < min_visibility:
            continue
        seg = ann["segmentation"]
        counts = seg["counts"]
        if isinstance(counts, str):
            seg = {"size": seg["size"], "counts": counts.encode("utf-8")}
        m = mask_util.decode(seg)
        if m.shape[:2] != (h, w):
            raise RuntimeError(f"GT mask {m.shape[:2]} != image {(h, w)}")
        masks.append(torch.from_numpy(np.ascontiguousarray(m)).bool())
        meta.append(
            {
                "category_id": int(ann["category_id"]),
                "visibility_fraction": float(ann.get("visibility_fraction", float("nan"))),
                "bbox": [float(v) for v in ann["bbox"]],
            }
        )
    stacked = torch.stack(masks, 0) if masks else torch.zeros((0, h, w), dtype=torch.bool)
    return stacked, meta


def drop_truncated_predictions(instances, h, w):
    """Prediction-side counterpart of the GT truncation filter (see
    maskdino/evaluation/truncated_prediction_filter.py)."""
    if len(instances) == 0:
        return instances
    keep = torch.tensor(
        [not touches_frame_edge(*box, h, w) for box in instances.pred_boxes.tensor.tolist()],
        dtype=torch.bool,
    )
    return instances[keep]


def run_matching(cfg, args):
    """Run the model over the dataset and accumulate GT / missed-GT pixel evidence."""
    model = build_model(cfg)
    model.eval()
    incompatible = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    # A class-head shape mismatch (checkpoint trained on a different class set than
    # --dataset declares) leaves that layer randomly initialized: scores, and therefore
    # the score threshold and every "miss", would be meaningless. Refuse rather than
    # quietly produce a wrong heatmap.
    bad = list(getattr(incompatible, "incorrect_shapes", []) or [])
    if bad and not args.allow_weight_mismatch:
        detail = ", ".join(f"{k}: ckpt {tuple(s2)} vs model {tuple(s1)}" for k, s1, s2 in bad)
        raise SystemExit(
            f"checkpoint does not fit this dataset's class space ({detail}). The model was "
            f"trained on a different instrument set than {args.dataset!r} declares - use a "
            f"checkpoint trained for it, or pass --allow-weight-mismatch if you really mean to."
        )

    mapper = Hdf5CocoInstanceDatasetMapper(cfg, False)
    loader = build_detection_test_loader(cfg, args.dataset, mapper=mapper)

    dicts = DatasetCatalog.get(args.dataset)

    gt_by_id = {d["image_id"]: d.get("annotations", []) for d in dicts}
    hw_by_id = {d["image_id"]: (d["height"], d["width"]) for d in dicts}
    file_by_id = {d["image_id"]: d.get("file_name", "") for d in dicts}

    min_visibility = float(cfg.INPUT.MIN_VISIBILITY)
    score_thresh = args.score_thresh
    iou_thresh = args.iou_thresh
    drop_truncated_preds = bool(cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES)
    num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    check_classes = num_classes > 1

    gt_cov = miss_cov = fp_cov = None
    shape = None
    records: list[dict] = []
    totals = Counter()
    n_images = 0

    with torch.no_grad():
        for batch in loader:
            if args.max_images and n_images >= args.max_images:
                break
            for inp, out in zip(batch, model(batch)):
                image_id = inp["image_id"]
                h, w = hw_by_id[image_id]
                if shape is None:
                    shape = (h, w)
                    gt_cov = np.zeros(shape, np.float32)
                    miss_cov = np.zeros(shape, np.float32)
                    fp_cov = np.zeros(shape, np.float32)

                gt_masks, gt_meta = gt_masks_from_anns(gt_by_id.get(image_id, []), h, w, min_visibility)

                instances = out["instances"].to("cpu")
                instances = instances[instances.scores >= score_thresh]
                if drop_truncated_preds:
                    instances = drop_truncated_predictions(instances, h, w)
                pred_masks = instances.pred_masks.bool()
                pred_classes = instances.pred_classes.numpy().astype(np.int64)

                m, n = gt_masks.shape[0], pred_masks.shape[0]
                iou = mask_iou_matrix(pred_masks, gt_masks, box_prefilter=True)
                row_ind, col_ind, matched_iou = hungarian_match(iou, iou_thresh)
                accepted = matched_iou >= iou_thresh
                acc_pred, acc_gt = row_ind[accepted], col_ind[accepted]

                iou_np = iou.numpy() if iou.numel() else np.zeros((n, m), np.float32)
                best_iou_per_gt = iou_np.max(axis=0) if n else np.zeros((m,), np.float32)
                match_by_gt = dict(zip(acc_gt.tolist(), acc_pred.tolist()))

                for j in range(m):
                    mask = gt_masks[j].numpy()
                    gt_cov += mask
                    meta = gt_meta[j]
                    pred_idx = match_by_gt.get(j)
                    mislabeled = bool(
                        pred_idx is not None
                        and check_classes
                        and pred_classes[pred_idx] != meta["category_id"]
                    )
                    # class-agnostic by default (matches classagnostic_recall); with
                    # --class-aware a mask-matched but mislabeled GT also counts as missed
                    missed = pred_idx is None or (args.class_aware and mislabeled)
                    if missed:
                        miss_cov += mask
                    ys, xs = np.nonzero(mask)
                    records.append(
                        {
                            "image_id": image_id,
                            "file_name": file_by_id.get(image_id, ""),
                            "category_id": meta["category_id"],
                            "area_px": int(mask.sum()),
                            "cy": float(ys.mean()) if ys.size else float("nan"),
                            "cx": float(xs.mean()) if xs.size else float("nan"),
                            "visibility_fraction": meta["visibility_fraction"],
                            "matched": pred_idx is not None,
                            "mislabeled": mislabeled,
                            "missed": missed,
                            "best_iou": float(best_iou_per_gt[j]) if m else 0.0,
                        }
                    )

                matched_preds = set(acc_pred.tolist())
                for i in range(n):
                    if i not in matched_preds:
                        fp_cov += pred_masks[i].numpy()

                totals["gt"] += m
                totals["pred"] += n
                totals["matched"] += int(accepted.sum())
                totals["false_pos"] += n - int(accepted.sum())
                n_images += 1

            if args.max_images and n_images >= args.max_images:
                break
            if n_images % 100 == 0:
                print(f"       {n_images} images...", flush=True)

    if shape is None:
        raise SystemExit(f"{args.dataset}: no images processed")
    return {
        "shape": shape,
        "gt_cov": gt_cov,
        "miss_cov": miss_cov,
        "fp_cov": fp_cov,
        "records": records,
        "totals": totals,
        "n_images": n_images,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def cell_sums(arr: np.ndarray, bins: int) -> np.ndarray:
    """Sum an (H, W) array into a (bins, bins) grid (edge cells absorb the remainder)."""
    h, w = arr.shape
    ys = np.linspace(0, h, bins + 1).astype(int)
    xs = np.linspace(0, w, bins + 1).astype(int)
    out = np.zeros((bins, bins), np.float64)
    for i in range(bins):
        for j in range(bins):
            out[i, j] = arr[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].sum()
    return out


def rate_by_bucket(records, key, n_buckets=10, value_fn=None):
    """Miss rate per quantile bucket of `key` -> list of dicts (edges, n, rate)."""
    vals = np.array([value_fn(r) if value_fn else r[key] for r in records], float)
    missed = np.array([r["missed"] for r in records], bool)
    ok = np.isfinite(vals)
    vals, missed = vals[ok], missed[ok]
    if vals.size == 0:
        return []
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_buckets + 1)))
    if edges.size < 2:
        return []
    idx = np.clip(np.digitize(vals, edges[1:-1], right=True), 0, edges.size - 2)
    out = []
    for b in range(edges.size - 1):
        sel = idx == b
        if not sel.any():
            continue
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(sel.sum()),
                "missed": int(missed[sel].sum()),
                "miss_rate": round(float(missed[sel].mean()), 4),
            }
        )
    return out


def radial_profile(gt_cov, miss_cov, n_rings=16):
    """Miss rate (missed GT px / GT px) as a function of distance from the image centre."""
    h, w = gt_cov.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2)
    r_max = min(h, w) / 2
    edges = np.linspace(0, r_max, n_rings + 1)
    rows = []
    for i in range(n_rings):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        g = float(gt_cov[sel].sum())
        rows.append(
            {
                "r_lo": float(edges[i]),
                "r_hi": float(edges[i + 1]),
                "gt_px": g,
                "miss_px": float(miss_cov[sel].sum()),
                "miss_rate": round(float(miss_cov[sel].sum() / g), 4) if g else None,
            }
        )
    return rows


def analyse(res, bins: int, min_cell_gt: float):
    gt_cov, miss_cov = res["gt_cov"], res["miss_cov"]
    records = res["records"]
    gt_cells = cell_sums(gt_cov, bins)
    miss_cells = cell_sums(miss_cov, bins)

    # density correlation: nearly always high -- misses occur where instruments are.
    live = gt_cells > 0
    dens_p = dens_s = None
    if live.sum() > 2:
        dens_p = float(pearsonr(gt_cells[live], miss_cells[live]).statistic)
        dens_s = float(spearmanr(gt_cells[live], miss_cells[live]).statistic)

    # rate correlation: the one that actually says whether location matters.
    solid = gt_cells >= min_cell_gt
    rate_cells = np.full_like(gt_cells, np.nan)
    rate_cells[solid] = miss_cells[solid] / gt_cells[solid]
    rate_p = rate_s = None
    if solid.sum() > 2:
        rate_p = float(pearsonr(gt_cells[solid], rate_cells[solid]).statistic)
        rate_s = float(spearmanr(gt_cells[solid], rate_cells[solid]).statistic)

    # per-instance density lookup: GT density of the cell each instance's centroid is in
    h, w = gt_cov.shape
    for r in records:
        if np.isfinite(r["cy"]):
            i = min(int(r["cy"] / h * bins), bins - 1)
            j = min(int(r["cx"] / w * bins), bins - 1)
            r["cell_gt_density"] = float(gt_cells[i, j])
            r["radius_px"] = float(np.hypot(r["cy"] - (h - 1) / 2, r["cx"] - (w - 1) / 2))
        else:
            r["cell_gt_density"] = float("nan")
            r["radius_px"] = float("nan")

    per_class = defaultdict(lambda: {"n": 0, "missed": 0})
    for r in records:
        pc = per_class[r["category_id"]]
        pc["n"] += 1
        pc["missed"] += int(r["missed"])
    for pc in per_class.values():
        pc["miss_rate"] = round(pc["missed"] / pc["n"], 4) if pc["n"] else 0.0

    n_missed = sum(r["missed"] for r in records)
    return {
        "gt_cells": gt_cells,
        "miss_cells": miss_cells,
        "rate_cells": rate_cells,
        "instance_miss_rate": round(n_missed / len(records), 4) if records else 0.0,
        "n_instances": len(records),
        "n_missed": n_missed,
        "pixel_miss_rate": round(float(miss_cov.sum() / gt_cov.sum()), 4) if gt_cov.sum() else 0.0,
        "correlation": {
            "note": (
                "density: GT px vs missed px per cell (high by construction -- misses "
                "happen where instruments are). rate: GT px vs local miss rate per cell "
                "-- this is the one that says whether position itself predicts failure."
            ),
            "bins": bins,
            "min_cell_gt_px": min_cell_gt,
            "cells_used_density": int(live.sum()),
            "cells_used_rate": int(solid.sum()),
            "density_pearson_r": None if dens_p is None else round(dens_p, 4),
            "density_spearman_r": None if dens_s is None else round(dens_s, 4),
            "rate_vs_density_pearson_r": None if rate_p is None else round(rate_p, 4),
            "rate_vs_density_spearman_r": None if rate_s is None else round(rate_s, 4),
        },
        "radial": radial_profile(res["gt_cov"], res["miss_cov"]),
        "by_area": rate_by_bucket(records, "area_px"),
        "by_visibility": rate_by_bucket(records, "visibility_fraction"),
        "by_cell_density": rate_by_bucket(records, "cell_gt_density"),
        "by_radius": rate_by_bucket(records, "radius_px"),
        "per_class": {int(k): v for k, v in sorted(per_class.items())},
    }


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def _compact(v: float) -> str:
    """Short axis label for a bucket edge: 1.2k / 3.4M / 0.75."""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}k"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "miss_div",
        ["#184f95", "#2a78d6", "#9ec5f4", "#8a8a84", "#ec835a", "#d03b3b", "#7d1f1f"],
    )


def plot_maps(res, ana, name, out, theme, dpi, cmap_name, class_names):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    cmap = density_cmap(cmap_name, theme["surface"])
    never = "#4a4a46" if theme is THEMES["dark"] else "#d9d8d2"
    h, w = res["shape"]
    n_img = res["n_images"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.0))
    (ax_gt, ax_miss), (ax_rate, ax_prof) = axes

    for ax, field, title, label in (
        (ax_gt, res["gt_cov"], "GT instruments (all)", "GT instances covering the pixel"),
        (ax_miss, res["miss_cov"], "Missed instruments (false negatives)", "missed instances covering the pixel"),
    ):
        data = np.ma.masked_where(field == 0, field / max(n_img, 1))
        norm = LogNorm(vmin=1.0 / max(n_img, 1), vmax=max(field.max() / max(n_img, 1), 2.0 / max(n_img, 1)))
        im = ax.imshow(data, cmap=cmap.with_extremes(bad=never), norm=norm, interpolation="nearest")
        ax.set_title(title, color=theme["primary"], pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(f"{label} (per frame, log)", color=theme["secondary"])
        cb.outline.set_edgecolor(theme["grid"])

    # miss RATE per cell, diverging around the dataset-wide pixel miss rate
    base = ana["pixel_miss_rate"] or 1e-6
    rate = ana["rate_cells"]
    rate_max = float(np.nanmax(rate)) if np.isfinite(rate).any() else base * 2
    im = ax_rate.imshow(
        np.ma.masked_invalid(rate),
        cmap=diverging_cmap().with_extremes(bad=never),
        norm=TwoSlopeNorm(vmin=0.0, vcenter=base, vmax=max(rate_max, base * 1.5 + 1e-6)),
        interpolation="nearest",
        extent=(0, w, h, 0),
    )
    ax_rate.set_title(
        f"Local miss rate (grey = too little GT support)\ncentred on the dataset rate {100 * base:.1f}%",
        color=theme["primary"],
        pad=8,
    )
    cb = fig.colorbar(im, ax=ax_rate, fraction=0.046, pad=0.03)
    cb.set_label("missed GT px / GT px in cell", color=theme["secondary"])
    cb.outline.set_edgecolor(theme["grid"])

    # radial profile
    rings = [r for r in ana["radial"] if r["miss_rate"] is not None]
    if rings:
        xs = [(r["r_lo"] + r["r_hi"]) / 2 for r in rings]
        ys = [100 * r["miss_rate"] for r in rings]
        ax_prof.plot(xs, ys, color="#eb6834", linewidth=2, marker="o", markersize=5)
        ax_prof.axhline(100 * base, color=theme["secondary"], linewidth=1.2, linestyle="--")
        ax_prof.text(
            xs[0], 100 * base, f" dataset rate {100 * base:.1f}%", va="bottom",
            color=theme["secondary"], fontsize=8,
        )
        for k in (0, len(rings) // 2, len(rings) - 1):
            ax_prof.annotate(
                f"{rings[k]['gt_px'] / 1e6:.1f}M GT px",
                (xs[k], ys[k]), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=theme["secondary"],
            )
    ax_prof.set_title("Miss rate vs. distance from image centre", color=theme["primary"], pad=8)
    ax_prof.set_xlabel("radius from centre (px)")
    ax_prof.set_ylabel("missed GT px / GT px (%)")
    ax_prof.grid(color=theme["grid"], linewidth=0.6)
    ax_prof.set_axisbelow(True)
    for side in ("top", "right"):
        ax_prof.spines[side].set_visible(False)

    for ax in (ax_gt, ax_miss, ax_rate):
        ax.set_aspect("equal")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    corr = ana["correlation"]
    fig.suptitle(f"Missed instruments - {name}", color=theme["primary"], fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.012,
        f"{n_img} frames - {ana['n_instances']} GT instances - {ana['n_missed']} missed "
        f"({100 * ana['instance_miss_rate']:.1f}% of instances, {100 * ana['pixel_miss_rate']:.1f}% of GT px)   |   "
        f"miss density vs GT density: r={corr['density_pearson_r']} (rho={corr['density_spearman_r']})   |   "
        f"miss RATE vs GT density: r={corr['rate_vs_density_pearson_r']} (rho={corr['rate_vs_density_spearman_r']})",
        ha="center",
        color=theme["secondary"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_correlates(ana, name, out, theme, dpi, class_names):
    import matplotlib.pyplot as plt

    base = ana["instance_miss_rate"]
    # Only panels with data: GT without visibility_fraction (most val sets) would
    # otherwise leave an empty axes in the row.
    panels = [
        (key, xlabel, title)
        for key, xlabel, title in (
            ("by_area", "GT mask area (px)", "Miss rate by instrument size"),
            ("by_visibility", "GT visibility fraction", "Miss rate by occlusion"),
            ("by_radius", "distance from image centre (px)", "Miss rate by eccentricity"),
            ("by_cell_density", "GT density of the instance's cell (px)", "Miss rate by local crowding"),
        )
        if ana.get(key)
    ]
    n_panels = len(panels) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 5.2))

    for ax, (key, xlabel, title) in zip(axes, panels):
        rows = ana[key]
        xs = range(len(rows))
        ax.bar(xs, [100 * r["miss_rate"] for r in rows], color="#eb6834", width=0.72)
        ax.axhline(100 * base, color=theme["secondary"], linewidth=1.2, linestyle="--")
        ax.set_xticks(
            list(xs),
            [f"{_compact(r['lo'])}-{_compact(r['hi'])}" for r in rows],
            fontsize=7,
            rotation=45,
            ha="right",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("miss rate (%)")
        ax.set_title(title, color=theme["primary"], pad=8)
        for i, r in enumerate(rows):
            ax.text(i, 100 * r["miss_rate"], f"{r['n']}", ha="center", va="bottom",
                    fontsize=7, color=theme["secondary"])

    rows = sorted(ana["per_class"].items(), key=lambda kv: -kv[1]["miss_rate"])
    ax = axes[-1]
    if rows:
        labels = [class_names.get(cid, str(cid)) for cid, _ in rows]
        vals = [100 * v["miss_rate"] for _, v in rows]
        ax.barh(range(len(rows)), vals, color="#eb6834", height=0.7)
        ax.set_yticks(range(len(rows)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(100 * base, color=theme["secondary"], linewidth=1.2, linestyle="--")
        ax.set_xlabel("miss rate (%)")
        ax.set_title("Miss rate by class", color=theme["primary"], pad=8)
        for i, (_, v) in enumerate(rows):
            ax.text(vals[i] + 1, i, f"{v['missed']}/{v['n']}", va="center", fontsize=7,
                    color=theme["secondary"])
        ax.set_xlim(0, max(vals) * 1.25 + 2)

    for ax in axes:
        ax.grid(axis="y" if ax is not axes[-1] else "x", color=theme["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        f"What the misses correlate with - {name}  (dashed = overall {100 * base:.1f}%)",
        color=theme["primary"],
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------


def summary_md(name, res, ana, cfg, args, class_names, support) -> str:
    corr = ana["correlation"]
    t = res["totals"]
    lines = [
        f"# Missed-instrument analysis - {name}",
        "",
        f"- model: `{cfg.MODEL.WEIGHTS}`  (config `{args.config_file}`)",
        (
            f"- matching: class-{'aware' if args.class_aware else 'agnostic'} Hungarian mask-IoU, "
            f"score>={args.score_thresh}, IoU>={args.iou_thresh}, "
            f"MIN_VISIBILITY={cfg.INPUT.MIN_VISIBILITY}, truncated instances "
            f"{'excluded' if cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES else 'kept'}"
        ),
        (
            f"- frames: **{res['n_images']}** - GT instances: **{ana['n_instances']}** - "
            f"predictions kept: {t['pred']} - matched: {t['matched']} - "
            f"false positives: {t['false_pos']}"
        ),
        (
            f"- **missed: {ana['n_missed']} ({100 * ana['instance_miss_rate']:.1f}% of instances, "
            f"{100 * ana['pixel_miss_rate']:.1f}% of GT pixels)**"
        ),
        "",
        "## Do the misses correlate with where instruments land?",
        "",
        (
            f"- missed-pixel density vs GT density, per {corr['bins']}x{corr['bins']} cell: "
            f"**r = {corr['density_pearson_r']}**, rho = {corr['density_spearman_r']} "
            f"({corr['cells_used_density']} cells)"
        ),
        (
            f"- local miss *rate* vs GT density: **r = {corr['rate_vs_density_pearson_r']}**, "
            f"rho = {corr['rate_vs_density_spearman_r']} ({corr['cells_used_rate']} cells with "
            f">= {corr['min_cell_gt_px']:g} GT px)"
        ),
        "",
        "The first number is high almost by construction - you cannot miss an instrument where",
        "no instrument ever lands. The second is the real test: ~0 means the model fails at a",
        "constant rate everywhere and the miss map is just the landing map re-scaled; clearly",
        "non-zero means position itself predicts failure.",
        "",
        "## Miss rate by radius from image centre",
        "",
        "| radius (px) | GT px | miss rate |",
        "|---|---:|---:|",
    ]
    for r in ana["radial"]:
        if r["miss_rate"] is None:
            continue
        lines.append(f"| {r['r_lo']:.0f} - {r['r_hi']:.0f} | {r['gt_px'] / 1e6:.2f}M | {100 * r['miss_rate']:.1f}% |")

    for key, title, fmt in (
        ("by_area", "Miss rate by GT mask area (px)", "{:.0f}"),
        ("by_visibility", "Miss rate by GT visibility fraction", "{:.3f}"),
        ("by_cell_density", "Miss rate by local GT density", "{:.0f}"),
    ):
        rows = ana[key]
        if not rows:
            continue
        lines += ["", f"## {title}", "", "| bucket | instances | missed | miss rate |", "|---|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {fmt.format(r['lo'])} - {fmt.format(r['hi'])} | {r['n']} | {r['missed']} | "
                f"{100 * r['miss_rate']:.1f}% |"
            )

    lines += ["", "## Miss rate by class", "", "| class | instances | missed | miss rate |", "|---|---:|---:|---:|"]
    for cid, v in sorted(ana["per_class"].items(), key=lambda kv: -kv[1]["miss_rate"]):
        lines.append(f"| {class_names.get(cid, cid)} | {v['n']} | {v['missed']} | {100 * v['miss_rate']:.1f}% |")

    lines += [
        "",
        "## Spatial support of the missed-instrument map",
        "",
        f"- missed instruments never touch **{100 * support['never_covered_frac']:.1f}%** of the image",
        (
            f"- largest image-centred square with at least one miss everywhere: "
            f"{support['largest_centered_square_px']['covered_at_least_once']}px"
        ),
        "",
    ]
    return "\n".join(lines)


def load_previous_run(out: Path, dataset: str) -> dict:
    """Rebuild the scan result from a previous run's saved artifacts, so the plots can
    be re-rendered (different --cmap/--theme/--bins) without paying for inference again."""
    gt = np.load(out / f"{dataset}_gt_coverage_model.npy")
    miss = np.load(out / f"{dataset}_miss_coverage.npy")
    fp_path = out / f"{dataset}_fp_coverage.npy"
    stats = json.loads((out / f"{dataset}_miss_stats.json").read_text())
    records = []
    with (out / f"{dataset}_miss_instances.csv").open() as fh:
        for row in csv.DictReader(fh):
            records.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "category_id": int(row["category_id"]),
                    "area_px": int(row["area_px"]),
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "visibility_fraction": float(row["visibility_fraction"]),
                    "matched": row["matched"] == "True",
                    "mislabeled": row["mislabeled"] == "True",
                    "missed": row["missed"] == "True",
                    "best_iou": float(row["best_iou"]),
                }
            )
    return {
        "shape": gt.shape,
        "gt_cov": gt,
        "miss_cov": miss,
        "fp_cov": np.load(fp_path) if fp_path.exists() else np.zeros_like(gt),
        "records": records,
        "totals": Counter(stats["counts"]),
        "n_images": stats["frames"],
        "num_classes": 0,
        "weights": stats["weights"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", help="required unless --replot")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="re-render the figures from a previous run's saved .npy/.csv in --out, no inference",
    )
    ap.add_argument("--weights", default=None, help="checkpoint (overrides MODEL.WEIGHTS)")
    ap.add_argument("--dataset", required=True, help="registered dataset name, e.g. val_setab_mm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-thresh", type=float, default=None, help="default: HUNGARIAN_EVAL.SCORE_THRESH")
    ap.add_argument("--iou-thresh", type=float, default=None, help="default: HUNGARIAN_EVAL.IOU_THRESH")
    ap.add_argument(
        "--class-aware",
        action="store_true",
        help="count a mask-matched but mislabeled GT as missed too (default: class-agnostic)",
    )
    ap.add_argument("--bins", type=int, default=32, help="grid for the rate map / correlations (default 32)")
    ap.add_argument(
        "--min-cell-gt",
        type=float,
        default=20000.0,
        help="GT px a cell needs before its miss rate is trusted/plotted (default 20000)",
    )
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--theme", choices=sorted(THEMES), default="light")
    ap.add_argument("--cmap", choices=HEATMAP_CMAPS, default="inferno")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument(
        "--allow-weight-mismatch",
        action="store_true",
        help="proceed even if the checkpoint's class head does not fit the dataset (scores become meaningless)",
    )
    ap.add_argument("--device", default=None, help="cuda / cuda:1 / cpu (default: config)")
    ap.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    if not args.replot and not args.config_file:
        ap.error("--config-file is required unless --replot is given")

    cfg = None
    if not args.replot:
        cfg = setup_cfg(args)
        he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
        if args.score_thresh is None:
            args.score_thresh = float(he.SCORE_THRESH)
        if args.iou_thresh is None:
            args.iou_thresh = float(he.IOU_THRESH)

    args.out.mkdir(parents=True, exist_ok=True)
    theme = THEMES[args.theme]
    _style(theme)

    meta = MetadataCatalog.get(args.dataset)
    class_names = {i: n for i, n in enumerate(getattr(meta, "thing_classes", []) or [])}

    if args.replot:
        print(f"[plot] re-rendering {args.dataset} from {args.out}", flush=True)
        res = load_previous_run(args.out, args.dataset)
    else:
        print(f"[run ] {args.dataset} with {cfg.MODEL.WEIGHTS}", flush=True)
        res = run_matching(cfg, args)
    ana = analyse(res, args.bins, args.min_cell_gt)
    support = coverage_support_stats(res["miss_cov"], res["n_images"])
    print(
        f"[stat] {ana['n_missed']}/{ana['n_instances']} instances missed "
        f"({100 * ana['instance_miss_rate']:.1f}%); miss-rate vs density rho="
        f"{ana['correlation']['rate_vs_density_spearman_r']}",
        flush=True,
    )

    prefix = args.dataset
    plot_maps(res, ana, args.dataset, args.out / f"{prefix}_missed_heatmap.png", theme, args.dpi, args.cmap, class_names)
    plot_correlates(ana, args.dataset, args.out / f"{prefix}_miss_correlates.png", theme, args.dpi, class_names)

    if args.replot:
        print(f"[done] re-rendered figures in {args.out}")
        return 0

    np.save(args.out / f"{prefix}_miss_coverage.npy", res["miss_cov"])
    np.save(args.out / f"{prefix}_gt_coverage_model.npy", res["gt_cov"])
    np.save(args.out / f"{prefix}_fp_coverage.npy", res["fp_cov"])

    with (args.out / f"{prefix}_miss_instances.csv").open("w", newline="") as fh:
        fields = [
            "image_id", "file_name", "category_id", "area_px", "cx", "cy", "radius_px",
            "visibility_fraction", "cell_gt_density", "matched", "mislabeled", "missed", "best_iou",
        ]
        wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(res["records"])

    (args.out / f"{prefix}_miss_stats.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "weights": cfg.MODEL.WEIGHTS,
                "config_file": args.config_file,
                "matching": {
                    "class_aware": args.class_aware,
                    "score_thresh": args.score_thresh,
                    "iou_thresh": args.iou_thresh,
                    "min_visibility": float(cfg.INPUT.MIN_VISIBILITY),
                    "exclude_truncated": bool(cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES),
                },
                "frames": res["n_images"],
                "counts": dict(res["totals"]),
                "instance_miss_rate": ana["instance_miss_rate"],
                "pixel_miss_rate": ana["pixel_miss_rate"],
                "correlation": ana["correlation"],
                "radial": ana["radial"],
                "by_area": ana["by_area"],
                "by_visibility": ana["by_visibility"],
                "by_cell_density": ana["by_cell_density"],
                "by_radius": ana["by_radius"],
                "per_class": {
                    class_names.get(k, str(k)): v for k, v in ana["per_class"].items()
                },
                "miss_map_support": support,
            },
            indent=2,
        )
    )
    (args.out / f"{prefix}_missed_summary.md").write_text(
        summary_md(args.dataset, res, ana, cfg, args, class_names, support)
    )
    print(f"[done] wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
