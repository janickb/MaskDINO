# Copyright (c) Facebook, Inc. and its affiliates.
"""
Runs MaskDINO on a folder of images and, for the highest-scoring detection in
each, computes and visualizes its principal axis (maskdino.classification.
shape_features.principal_axis) - a red line overlay showing the major/minor
axes over the mask, plus a canonically-rotated crop where that axis is
horizontal. Useful for sanity-checking that the axis tracks instrument
rotation consistently across different renders of the same physical tool.

Usage:
    python tools/visualize_principal_axis.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune.yaml \
        --input "path/to/calibration/forcep03_a/*.hdf5" \
        --output out_dir \
        --opts MODEL.WEIGHTS output_v2/model_final.pth MODEL.DEVICE cuda
"""
import argparse
import glob
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer

from maskdino import add_maskdino_config
from maskdino.classification import EmbeddingPredictor, principal_axis, rotate_to_canonical, canonical_crop
from sgdata.reader import read_image as read_hdf5_image

HDF5_EXTS = (".hdf5", ".h5")


def load_image_bgr(path: str) -> np.ndarray:
    if path.lower().endswith(HDF5_EXTS):
        rgb = read_hdf5_image(path)[:, :, :3]
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def get_parser():
    parser = argparse.ArgumentParser(description="Visualize the principal axis of each image's top detection")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--input", required=True, help="glob pattern for input images or .hdf5 frames")
    parser.add_argument("--output", required=True, help="directory to save axis-overlay and canonical-crop PNGs")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="modify config options, 'KEY VALUE' pairs")
    return parser


def main():
    args = get_parser().parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    predictor = EmbeddingPredictor(cfg)
    os.makedirs(args.output, exist_ok=True)

    paths = sorted(glob.glob(args.input))
    assert paths, f"No images matched {args.input}"

    for path in paths:
        img_bgr = load_image_bgr(path)
        instances = predictor(img_bgr).to("cpu")
        name = os.path.splitext(os.path.basename(path))[0]

        if len(instances) == 0:
            print(f"{name}: no detections")
            continue
        best = int(instances.scores.argmax())
        if instances.scores[best] < args.confidence_threshold:
            print(f"{name}: top score {instances.scores[best]:.2f} < {args.confidence_threshold}, skipping")
            continue

        mask = instances.pred_masks[best].numpy() > 0.5
        axis = principal_axis(mask)
        print(f"{name}: angle={np.degrees(axis.angle):.1f} deg, "
              f"major={axis.major_length:.1f}, minor={axis.minor_length:.1f}, centroid={axis.centroid}")

        vis = Visualizer(img_bgr[:, :, ::-1], instance_mode=ColorMode.IMAGE)
        vis.draw_binary_mask(mask, color=(0.2, 0.7, 1.0), alpha=0.4)
        cx, cy = axis.centroid
        dx, dy = np.cos(axis.angle), np.sin(axis.angle)
        half = axis.major_length / 2
        p1 = (cx - dx * half, cy - dy * half)
        p2 = (cx + dx * half, cy + dy * half)
        vis.draw_line([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=2)
        vis.draw_text(f"angle={np.degrees(axis.angle):.1f}deg", (10, 10), color="white", horizontal_alignment="left")
        cv2.imwrite(os.path.join(args.output, f"{name}_axis.png"), vis.output.get_image()[:, :, ::-1])

        canonical = rotate_to_canonical(img_bgr, axis)
        cv2.imwrite(os.path.join(args.output, f"{name}_canonical.png"), canonical)

        cropped = canonical_crop(img_bgr, mask, axis)
        cv2.imwrite(os.path.join(args.output, f"{name}_crop.png"), cropped)

    print(f"\nSaved visualizations to {args.output}")


if __name__ == "__main__":
    main()
