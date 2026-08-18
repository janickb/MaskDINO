# Copyright (c) Facebook, Inc. and its affiliates.
"""
Run inference and save visualized predictions.

Unlike demo.py, this filters predictions by score before drawing them.
MaskDINO's instance_inference() always returns exactly
TEST.DETECTIONS_PER_IMAGE (100) predictions regardless of confidence, so
demo.py's --confidence-threshold flag (inherited from detectron2's older
ROI-head/RetinaNet models) has no effect here and every image shows all 100
raw queries. This script applies the threshold itself, after inference,
using each prediction's actual score.

Also accepts scene_generator's .hdf5 frames directly (via sgdata.reader),
so you don't need to pre-extract 'colors' to plain image files first.
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
from detectron2.utils.colormap import random_color
from detectron2.utils.visualizer import ColorMode, Visualizer, _create_text_labels

from maskdino import add_maskdino_config
from sgdata.reader import read_image as read_hdf5_image

HDF5_EXTS = (".hdf5", ".h5")


def load_image_bgr(path: str) -> np.ndarray:
    """Returns an (H, W, 3) BGR uint8 array, as DefaultPredictor expects."""
    if path.lower().endswith(HDF5_EXTS):
        rgb = read_hdf5_image(path)[:, :, :3]  # drop alpha if RGBA
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def get_parser():
    parser = argparse.ArgumentParser(description="MaskDINO inference with score-filtered visualization")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--input", required=True,
        help="glob pattern for input images or .hdf5 frames, e.g. 'dir/*.png' or 'dir/*.hdf5'",
    )
    parser.add_argument("--output", required=True, help="directory to save visualized predictions")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="minimum score to keep a prediction")
    parser.add_argument("--bbox", action="store_true", help="draw bounding boxes")
    parser.add_argument("--cat_name", action="store_true", help="show the category name alongside the score")
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

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    os.makedirs(args.output, exist_ok=True)

    paths = sorted(glob.glob(args.input))
    assert paths, f"No images matched {args.input}"

    for path in paths:
        img_bgr = load_image_bgr(path)
        instances = predictor(img_bgr)["instances"].to("cpu")
        kept = instances[instances.scores > args.confidence_threshold]
        print(f"{os.path.basename(path)}: {len(instances)} raw -> {len(kept)} kept "
              f"(score>{args.confidence_threshold})")

        # Score (percentage) and the pixel-wise mask are always drawn; --bbox and
        # --cat_name independently toggle the box outline and the class-name text.
        boxes = kept.pred_boxes if args.bbox and kept.has("pred_boxes") else None
        masks = np.asarray(kept.pred_masks) if kept.has("pred_masks") else None
        scores = kept.scores if kept.has("scores") else None
        classes = kept.pred_classes.tolist() if kept.has("pred_classes") else None
        labels = _create_text_labels(
            classes if args.cat_name else None, scores, metadata.get("thing_classes", None)
        )

        # overlay_instances() converts each mask to polygons via cv2.findContours and
        # fills every one of them, including hole boundaries (e.g. forceps/scissors
        # finger loops render as solid blobs instead of rings). draw_binary_mask()
        # handles holes correctly (pixel-wise alpha blend instead of naive polygon
        # fill), so draw everything through it/draw_box/draw_text directly instead.
        vis = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
        colors = [random_color(rgb=True, maximum=1) for _ in range(len(kept))]
        boxes_arr = boxes.tensor.numpy() if boxes is not None else None
        for i in range(len(kept)):
            label_i = labels[i] if labels is not None else None
            if masks is not None:
                vis.draw_binary_mask(masks[i], color=colors[i], alpha=0.5,
                                      text=label_i if boxes_arr is None else None)
            if boxes_arr is not None:
                vis.draw_box(boxes_arr[i], edge_color=colors[i])
                if label_i is not None:
                    vis.draw_text(label_i, (boxes_arr[i][0], boxes_arr[i][1]),
                                   color=colors[i], horizontal_alignment="left")
        out = vis.output
        out_name = os.path.splitext(os.path.basename(path))[0] + ".png"
        cv2.imwrite(os.path.join(args.output, out_name), out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    main()
