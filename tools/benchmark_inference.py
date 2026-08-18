# Copyright (c) Facebook, Inc. and its affiliates.
"""
Benchmark GPU inference latency/throughput for a trained MaskDINO checkpoint.

Times only the model forward pass (build_model + eval), not image loading or
disk I/O - preprocessing happens once up front, outside the timed loop. Uses
torch.cuda.synchronize() around each timed call since CUDA execution is
otherwise asynchronous and would under-report latency.

Usage:
    python tools/benchmark_inference.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune.yaml \
        --input "path/to/images/*.hdf5" \
        --opts MODEL.WEIGHTS output_v2/model_final.pth MODEL.DEVICE cuda
"""
import argparse
import glob
import os
import statistics
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import cv2
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config
from sgdata.reader import read_image as read_hdf5_image

HDF5_EXTS = (".hdf5", ".h5")


def load_image_bgr(path: str):
    if path.lower().endswith(HDF5_EXTS):
        rgb = read_hdf5_image(path)[:, :, :3]
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark MaskDINO inference speed")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--input", required=True, help="glob pattern for images/.hdf5 frames to sample from")
    parser.add_argument("--num-images", type=int, default=10, help="distinct images to cycle through")
    parser.add_argument("--warmup", type=int, default=10, help="untimed warmup forward passes")
    parser.add_argument("--iters", type=int, default=50, help="timed forward passes")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


def main():
    args = get_parser().parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    device = torch.device(cfg.MODEL.DEVICE)

    paths = sorted(glob.glob(args.input))[: args.num_images]
    assert paths, f"No images matched {args.input}"
    print(f"Sampling from {len(paths)} images")

    # Preprocess once, up front - only the forward pass itself is timed.
    inputs = []
    for path in paths:
        img_bgr = load_image_bgr(path)
        if cfg.INPUT.FORMAT == "RGB":
            img_bgr = img_bgr[:, :, ::-1]
        height, width = img_bgr.shape[:2]
        image = aug.get_transform(img_bgr).apply_image(img_bgr)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": image, "height": height, "width": width})
    print(f"Resized input shape: {tuple(inputs[0]['image'].shape)}")

    def run_one(i):
        with torch.no_grad():
            model([inputs[i % len(inputs)]])

    print(f"Warming up ({args.warmup} iters)...")
    for i in range(args.warmup):
        run_one(i)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Timing ({args.iters} iters)...")
    latencies_ms = []
    for i in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_one(i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    mean = statistics.mean(latencies_ms)
    print(f"\n--- Results over {args.iters} forward passes, batch size 1 ---")
    print(f"mean:   {mean:.1f} ms  ({1000 / mean:.2f} images/sec)")
    print(f"median: {statistics.median(latencies_ms):.1f} ms")
    print(f"stdev:  {statistics.stdev(latencies_ms):.1f} ms")
    print(f"min:    {min(latencies_ms):.1f} ms")
    print(f"max:    {max(latencies_ms):.1f} ms")


if __name__ == "__main__":
    main()
