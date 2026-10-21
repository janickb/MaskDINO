# Copyright (c) Facebook, Inc. and its affiliates.
"""Render MaskDINO's DENSE encoder-level class map - "what class does the model
think is at each location?".

MaskDINO applies its one shared `class_embed` to every encoder memory token
before any query exists, uses the max over classes to pick the top-k queries,
and then discards the dense tensor (maskdino_decoder.py:403-407). This tool
captures it (see maskdino/visualization/class_map.py) and renders it.

Unlike tools/export_attention.py this needs no detection at all: the map exists
whether or not anything was detected, which is exactly what makes it useful for
"why did nothing get proposed here?".

Read the output as the classifier on ENCODER tokens - the decision that gates
which queries exist - NOT as an explanation of a final detection's label.

Usage:
    ./.venv/bin/python tools/export_class_map.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_v1_real.yaml \
        --input "path/to/frames/*.hdf5" \
        --output /tmp/classmap --top-k 6 \
        --opts MODEL.WEIGHTS runs/<run>/model_final.pth
"""
import argparse
import glob
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config, set_num_classes_from_metadata
from maskdino.visualization.class_map import capture_encoder_class_logits
from maskdino.visualization.class_map_panel import (
    DEFAULT_CMAP,
    INK,
    OVERLAY_CMAPS,
    draw_argmax,
    overlay_cmap,
)

HDF5_EXTS = (".hdf5", ".h5")



def load_image_bgr(path):
    """Returns an (H, W, 3) BGR uint8 array, as DefaultPredictor expects."""
    if path.lower().endswith(HDF5_EXTS):
        from sgdata.reader import read_image as read_hdf5_image

        rgb = read_hdf5_image(path)[:, :, :3]  # drop alpha if RGBA
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def render(cmap, rgb, out_path, classes, peaks, levels=None, reduce="max",
           show_argmax=False, cmap_name=DEFAULT_CMAP):
    """Write the diagnostic figure: the frame, the gate's score field, and one
    sequential small multiple per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors
    import matplotlib.pyplot as plt

    h, w = cmap.image_size
    base = np.asarray(rgb)[:, :, :3]
    hh, ww = min(h, base.shape[0]), min(w, base.shape[1])
    base = base[:hh, :ww]

    panels = 2 + len(classes) + (1 if show_argmax else 0)
    cols = min(4, panels)
    rows = int(np.ceil(panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows * hh / max(ww, 1)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()

    seq = overlay_cmap(cmap_name)
    # One shared score scale across every class panel. Per-panel autoscaling
    # would make a class that never fires look as confident as one that does.
    vmax = float(max(peaks[classes].max(), 1e-6))
    vmin = 0.0

    axes[0].imshow(base)
    axes[0].set_title("frame", fontsize=10, color=INK)

    # -- the gate's own score field, with the queries it actually placed ------
    conf = cmap.confidence_map(levels=levels, reduce=reduce)[:hh, :ww]
    axes[1].imshow(base)
    axes[1].imshow(conf, cmap=seq, vmin=vmin, vmax=max(vmax, float(conf.max())))
    pts = cmap.selected_points()
    inside = (pts[:, 0] < ww) & (pts[:, 1] < hh)
    axes[1].scatter(pts[inside, 0], pts[inside, 1], s=7, facecolors="none",
                    edgecolors="#ffffff", linewidths=0.7)
    axes[1].set_title(
        f"gate: max-over-classes + {int(inside.sum())} queries", fontsize=10, color=INK
    )

    # -- per-class small multiples ------------------------------------------
    for slot, cls in enumerate(classes):
        ax = axes[2 + slot]
        heat = cmap.class_heatmap(int(cls), levels=levels, reduce=reduce)[:hh, :ww]
        ax.imshow(base)
        ax.imshow(heat, cmap=seq, vmin=vmin, vmax=vmax)
        # direct label rather than a per-panel colorbar: the scale is shared, so
        # the only per-panel number worth printing is this class's own peak
        ax.set_title(f"{cmap.label(int(cls))}  peak {peaks[cls]:.2f}",
                     fontsize=10, color=INK)

    # -- optional argmax mosaic, capped at three named classes + other -------
    if show_argmax:
        ax = axes[2 + len(classes)]
        draw_argmax(ax, cmap, base, conf, [int(c) for c in classes[:3]], levels)
        ax.set_title("argmax (top 3 + other, faded by score)", fontsize=10, color=INK)

    mappable = plt.cm.ScalarMappable(cmap=seq)
    mappable.set_clim(vmin, vmax)
    bar = fig.colorbar(mappable, ax=axes.tolist(), fraction=0.02, pad=0.01)
    bar.set_label("class score (sigmoid), shared scale", fontsize=9, color=INK)

    fig.suptitle(
        f"{os.path.basename(cmap.source_frame)} - dense ENCODER class map "
        f"({cmap.logits.shape[0]} tokens x {cmap.num_classes} classes, "
        f"levels {[tuple(int(v) for v in s) for s in cmap.level_spatial_shapes]}, "
        f"reduce={reduce})",
        fontsize=10, color=INK,
    )
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def get_parser():
    p = argparse.ArgumentParser(description="Render MaskDINO's dense encoder class map")
    p.add_argument("--config-file", required=True)
    p.add_argument("--input", required=True, help="glob for images or .hdf5 frames")
    p.add_argument("--output", default=None, help="directory for the PNGs (default: alongside each frame)")
    p.add_argument("--top-k", type=int, default=6,
                   help="how many classes to render, ranked by peak score")
    p.add_argument("--classes", default=None,
                   help="comma-separated class ids or names to render instead of the top-k")
    p.add_argument("--levels", default=None,
                   help="comma-separated feature levels to use (default: all)")
    p.add_argument("--reduce", default="max", choices=["max", "mean"],
                   help="how to combine feature levels (default: max)")
    p.add_argument("--cmap", default=DEFAULT_CMAP, choices=OVERLAY_CMAPS,
                   help=f"colour ramp for the score overlay (default: {DEFAULT_CMAP})")
    p.add_argument("--argmax", action="store_true",
                   help="also draw the argmax mosaic, capped at 3 classes + other")
    p.add_argument("--npz", action="store_true",
                   help="also dump the raw logits next to each PNG (~1.4 MB per frame at 1024px)")
    p.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                   help="config overrides, e.g. MODEL.WEIGHTS path/to/model_final.pth")
    return p


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    set_num_classes_from_metadata(cfg, cfg.DATASETS.TEST[0])
    cfg.freeze()
    return cfg


def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = list(getattr(meta, "thing_classes", []) or [])

    levels = [int(v) for v in args.levels.split(",")] if args.levels else None
    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if not paths:
        raise SystemExit(f"no inputs matched {args.input!r}")
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    for path in paths:
        bgr = load_image_bgr(path)
        if bgr is None:
            print(f"  skip (unreadable): {path}")
            continue
        # DefaultPredictor is under torch.no_grad(), which is all this needs -
        # the dense map is a forward-pass artifact, not an attribution.
        with capture_encoder_class_logits(predictor.model) as cap:
            predictor(bgr)
        cmap = cap.record(
            # Render over the image as loaded. The encoder tokens span the whole
            # resized+padded input, so stretching them to the source frame is
            # exact when the padding is zero (SIZE_DIVISIBILITY=32 with the
            # 1024x1024 renders) and off by the padding otherwise - the same
            # approximation attention.py makes. Without this the map would be
            # rendered at the padded frame's size and the base image cropped
            # to its top-left corner.
            image_size=bgr.shape[:2],
            class_names=class_names,
            source_frame=path,
            checkpoint=cfg.MODEL.WEIGHTS,
            config_file=args.config_file,
        )

        if args.classes:
            chosen = np.array([cmap.class_index(
                c if not c.strip().isdigit() else int(c)) for c in args.classes.split(",")])
            _, peaks = cmap.rank_classes(levels=levels, reduce=args.reduce)
        else:
            order, peaks = cmap.rank_classes(levels=levels, reduce=args.reduce)
            chosen = order[: min(args.top_k, cmap.num_classes)]

        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = args.output or os.path.dirname(path)
        png = os.path.join(out_dir, stem + ".clsmap.png")
        render(cmap, bgr[:, :, ::-1], png, chosen, peaks,
               levels=levels, reduce=args.reduce, show_argmax=args.argmax,
               cmap_name=args.cmap)
        print(f"  {png}  (top class {cmap.label(int(chosen[0]))}, peak {peaks[chosen[0]]:.3f})")

        if args.npz:
            np.savez_compressed(
                os.path.join(out_dir, stem + ".clsmap.npz"),
                logits=cmap.logits,
                level_spatial_shapes=cmap.level_spatial_shapes,
                selected_tokens=cmap.selected_tokens,
                image_size=np.asarray(cmap.image_size, np.int32),
                class_names=np.asarray(cmap.class_names, dtype=object),
            )


if __name__ == "__main__":
    main()
