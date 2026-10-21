# Copyright (c) Facebook, Inc. and its affiliates.
"""
Export decoder cross-attention per detection, as `<frame>.attn.hdf5` sidecars.

Answers "where did the model look when it detected this instrument?" - see
maskdino/visualization/attention.py for what is captured and why it is sparse
(MaskDINO's decoder cross-attention is deformable: a few weighted sample points
per head per level, not a dense query->pixel map).

Writes a sidecar next to each frame rather than into it: the live pool evicts
and overwrites frames, sgdata.schema owns that key namespace, and attention is
a property of a *checkpoint* - you will want several per frame. The viewer
(hdf5_viewer) reads the sidecars; `--preview` additionally renders a PNG per
frame so the export can be eyeballed without the viewer.

Like demo/run_inference.py, this filters predictions by score itself:
instance_inference() always returns TEST.DETECTIONS_PER_IMAGE predictions
regardless of confidence.

Usage:
    python tools/export_attention.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune_singleclass.yaml \
        --input "path/to/frames/*.hdf5" \
        --output /tmp/attn --preview \
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
from sgdata.reader import read_image as read_hdf5_image

from maskdino import add_maskdino_config, set_num_classes_from_metadata
from maskdino.visualization.attention import (
    capture_decoder_attention,
    heatmap,
    sidecar_path,
    write_attention,
)

HDF5_EXTS = (".hdf5", ".h5")


def load_image_bgr(path: str) -> np.ndarray:
    """Returns an (H, W, 3) BGR uint8 array, as DefaultPredictor expects."""
    if path.lower().endswith(HDF5_EXTS):
        rgb = read_hdf5_image(path)[:, :, :3]  # drop alpha if RGBA
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def get_parser():
    parser = argparse.ArgumentParser(description="Export MaskDINO decoder attention per detection")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--input", required=True,
        help="glob pattern for input images or .hdf5 frames, e.g. 'dir/*.hdf5'",
    )
    parser.add_argument(
        "--output", default=None,
        help="directory for the .attn.hdf5 sidecars (default: alongside each input frame)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="minimum score to export a detection")
    parser.add_argument("--max-detections", type=int, default=10,
                        help="cap on exported detections per frame, highest score first")
    parser.add_argument("--dense", action="store_true",
                        help="also store each query's dense mask-logit response map "
                             "(1/4 resolution, fp16) - bulky, off by default")
    parser.add_argument("--preview", action="store_true",
                        help="also write a PNG overlay per frame for a sanity check")
    parser.add_argument("--preview-layers", default="all",
                        help="'all' (summed, default) or a layer index for the preview PNG")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                        help="modify config options, 'KEY VALUE' pairs")
    return parser


def write_preview(path, image_bgr, record, layers=None):
    """One PNG per frame: the image, then one heatmap panel per detection."""
    h, w = record.image_size
    base = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    panels = [base]
    for det in range(len(record)):
        heat = heatmap(record, det, layers=layers, downscale=2)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        peak = float(heat.max())
        if peak > 0:
            heat = heat / peak
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panel = cv2.addWeighted(base, 0.45, colored, 0.55, 0)

        # the detection's own mask outline, so "is the mass on the right object?"
        # is answerable at a glance - that is the whole point of the preview.
        if record.pred_masks is not None:
            mask = record.pred_masks[det].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
        cv2.putText(panel, record.label(det), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    # single row, downscaled so a 10-detection frame stays a viewable file
    scale = 0.5 if len(panels) > 4 else 1.0
    panels = [cv2.resize(p, None, fx=scale, fy=scale) for p in panels]
    cv2.imwrite(path, np.concatenate(panels, axis=1))


def main():
    args = get_parser().parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    set_num_classes_from_metadata(cfg, cfg.DATASETS.TEST[0])
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = list(metadata.get("thing_classes", []) or [])
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    preview_layers = None if args.preview_layers == "all" else [int(args.preview_layers)]

    paths = sorted(glob.glob(args.input))
    assert paths, f"No images matched {args.input}"

    for path in paths:
        img_bgr = load_image_bgr(path)
        with capture_decoder_attention(predictor.model, dense=args.dense) as cap:
            instances = predictor(img_bgr)["instances"].to("cpu")
        kept = instances[instances.scores > args.confidence_threshold]
        if len(kept) == 0:
            print(f"{os.path.basename(path)}: no detection above {args.confidence_threshold}, skipped")
            continue

        record = cap.record(
            kept,
            original_size=img_bgr.shape[:2],
            max_detections=args.max_detections,
            class_names=class_names,
            source_frame=os.path.abspath(path),
            checkpoint=cfg.MODEL.WEIGHTS,
            config_file=os.path.abspath(args.config_file),
        )
        out = sidecar_path(path, args.output)
        write_attention(out, record)
        print(f"{os.path.basename(path)}: {len(instances)} raw -> {len(kept)} kept "
              f"-> {len(record)} exported ({os.path.basename(out)})")

        if args.preview:
            write_preview(os.path.splitext(out)[0] + ".preview.png", img_bgr, record,
                          layers=preview_layers)


if __name__ == "__main__":
    main()
