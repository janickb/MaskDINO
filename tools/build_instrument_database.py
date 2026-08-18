# Copyright (c) Facebook, Inc. and its affiliates.
"""
Build a per-site instrument calibration database from a handful of images per
instrument type, for runtime few-shot classification on top of the (single-class)
surgical-instrument detector.

Expects one dominant instrument per calibration image, laid out as:
    calibration_dir/
        instrument_name_a/*.jpg  (10-20 images)
        instrument_name_b/*.jpg
        ...

For each image, the highest-scoring detection is taken as that instrument's
example embedding (MaskDINO's per-query decoder feature, L2-normalized). The
unnormalized feature is also saved (raw_embeddings) for diagnosing whether
magnitude carries any identity signal that normalization discards.

After building the database, runs a leave-one-out 1-NN check within the
calibration set itself: for every calibration embedding, is its nearest
neighbor among the *other* calibration embeddings the same instrument? This is
the cheapest available signal for whether the chosen embedding (which was only
ever trained to separate instrument-vs-background, not sub-types) actually
separates these specific instruments well enough to trust at runtime - low
per-class accuracy here means don't deploy yet; see the plan's fallback
(swap in a frozen pretrained appearance embedding on the masked crop).

Usage:
    python tools/build_instrument_database.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_finetune.yaml \
        --calibration-dir path/to/calibration \
        --output instrument_db.npz \
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
from maskdino.classification import EmbeddingPredictor
from sgdata.reader import read_image as read_hdf5_image

HDF5_EXTS = (".hdf5", ".h5")
IMAGE_GLOB_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.hdf5", "*.h5")


def load_image_bgr(path: str) -> np.ndarray:
    if path.lower().endswith(HDF5_EXTS):
        rgb = read_hdf5_image(path)[:, :, :3]
        return rgb[:, :, ::-1]
    return cv2.imread(path)


def get_parser():
    parser = argparse.ArgumentParser(description="Build a per-site instrument calibration database")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--calibration-dir", required=True,
        help="directory with one subfolder per instrument name, each holding 10-20 images",
    )
    parser.add_argument("--output", required=True, help="output .npz database path")
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="skip a calibration image if its top detection scores below this",
    )
    parser.add_argument("--knn-k", type=int, default=5, help="k for the leave-one-out diagnostic")
    parser.add_argument(
        "--debug-vis-dir", default=None,
        help="if set, save one annotated PNG per calibration image here (mirroring the "
        "calibration-dir's instrument subfolders) showing which detection was picked as "
        "that image's example embedding, so you can sanity-check the selection by eye",
    )
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="modify config options, 'KEY VALUE' pairs")
    return parser


def collect_calibration_images(calibration_dir):
    """Returns {instrument_name: [image_paths]} for each subfolder."""
    per_instrument = {}
    for entry in sorted(os.scandir(calibration_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        paths = []
        for pattern in IMAGE_GLOB_PATTERNS:
            paths.extend(glob.glob(os.path.join(entry.path, pattern)))
        paths = sorted(paths)
        if paths:
            per_instrument[entry.name] = paths
    return per_instrument


def save_debug_vis(debug_vis_dir, instrument_name, path, img_bgr, status, instances=None, best=None):
    """Saves one annotated PNG showing which detection (if any) was picked as this
    calibration image's example embedding, so a human can sanity-check the pick by eye."""
    out_dir = os.path.join(debug_vis_dir, instrument_name)
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(path))[0] + ".png"

    vis = Visualizer(img_bgr[:, :, ::-1], instance_mode=ColorMode.IMAGE)
    if instances is not None and best is not None:
        color = (0.2, 0.7, 1.0)
        if instances.has("pred_masks"):
            # draw_binary_mask (not overlay_instances) handles finger-loop holes correctly -
            # forceps/scissors would otherwise render as solid blobs.
            vis.draw_binary_mask(np.asarray(instances.pred_masks[best]), color=color, alpha=0.5)
        if instances.has("pred_boxes"):
            vis.draw_box(instances.pred_boxes.tensor[best].numpy(), edge_color=color)
    vis.draw_text(status, (10, 10), color="white", horizontal_alignment="left")
    out = vis.output
    cv2.imwrite(os.path.join(out_dir, out_name), out.get_image()[:, :, ::-1])


def build_embeddings(predictor, per_instrument, confidence_threshold, debug_vis_dir=None):
    embeddings, raw_embeddings, labels, sources = [], [], [], []
    for instrument_name, paths in per_instrument.items():
        kept = 0
        for path in paths:
            img_bgr = load_image_bgr(path)
            instances = predictor(img_bgr).to("cpu")
            if len(instances) == 0:
                print(f"  [skip] {path}: no detections")
                if debug_vis_dir:
                    save_debug_vis(debug_vis_dir, instrument_name, path, img_bgr, "SKIPPED: no detections")
                continue
            best = int(instances.scores.argmax())
            score = float(instances.scores[best])
            if score < confidence_threshold:
                print(f"  [skip] {path}: top score {score:.2f} < {confidence_threshold}")
                if debug_vis_dir:
                    save_debug_vis(
                        debug_vis_dir, instrument_name, path, img_bgr,
                        f"SKIPPED: score {score:.2f} < {confidence_threshold}", instances, best,
                    )
                continue
            embeddings.append(instances.pred_embeddings[best].numpy())
            raw_embeddings.append(instances.raw_embedding[best].numpy())
            labels.append(instrument_name)
            sources.append(path)
            kept += 1
            if debug_vis_dir:
                save_debug_vis(
                    debug_vis_dir, instrument_name, path, img_bgr,
                    f"{instrument_name}  score={score:.2f}", instances, best,
                )
        print(f"{instrument_name}: {kept}/{len(paths)} images used")
    return (
        np.stack(embeddings).astype(np.float32),
        np.stack(raw_embeddings).astype(np.float32),
        np.array(labels),
        sources,
    )


def leave_one_out_report(embeddings, labels, k):
    n = len(embeddings)
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # exclude self-match

    correct = 0
    per_class_total, per_class_correct = {}, {}
    confusion = {}
    for i in range(n):
        k_eff = min(k, n - 1)
        top_idx = np.argpartition(-sims[i], k_eff - 1)[:k_eff]
        top_labels = labels[top_idx]
        uniq, counts = np.unique(top_labels, return_counts=True)
        predicted = uniq[counts.argmax()]

        true_label = labels[i]
        per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
        if predicted == true_label:
            correct += 1
            per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1
        confusion.setdefault(true_label, {}).setdefault(predicted, 0)
        confusion[true_label][predicted] += 1

    print(f"\n--- Leave-one-out {k}-NN separability check (within calibration set) ---")
    print(f"Overall accuracy: {correct}/{n} = {correct / n:.1%}")
    print("Per-class accuracy:")
    for label in sorted(per_class_total):
        acc = per_class_correct.get(label, 0) / per_class_total[label]
        print(f"  {label}: {per_class_correct.get(label, 0)}/{per_class_total[label]} = {acc:.1%}")
    print("Confusion (true -> predicted counts):")
    for true_label in sorted(confusion):
        row = ", ".join(f"{pred}={cnt}" for pred, cnt in sorted(confusion[true_label].items()))
        print(f"  {true_label}: {row}")
    if correct / n < 0.9:
        print(
            "\nWARNING: leave-one-out accuracy is below 90% - the decoder query "
            "embedding may not separate these instruments reliably enough for "
            "deployment. Consider the appearance-embedding fallback before shipping."
        )


def main():
    args = get_parser().parse_args()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    per_instrument = collect_calibration_images(args.calibration_dir)
    assert per_instrument, f"No instrument subfolders with images found under {args.calibration_dir}"
    print(f"Found {len(per_instrument)} instrument(s): {sorted(per_instrument)}")

    if args.debug_vis_dir:
        os.makedirs(args.debug_vis_dir, exist_ok=True)
        print(f"Saving per-image debug visualizations to {args.debug_vis_dir}")

    predictor = EmbeddingPredictor(cfg)
    embeddings, raw_embeddings, labels, sources = build_embeddings(
        predictor, per_instrument, args.confidence_threshold, args.debug_vis_dir
    )
    assert len(embeddings) > 0, "No usable calibration embeddings were collected"

    leave_one_out_report(embeddings, labels, args.knn_k)

    centroid_labels = sorted(set(labels.tolist()))
    centroids = np.stack([embeddings[labels == name].mean(axis=0) for name in centroid_labels])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    np.savez(
        args.output,
        embeddings=embeddings,
        raw_embeddings=raw_embeddings,
        labels=labels,
        sources=np.array(sources),
        centroid_labels=np.array(centroid_labels),
        centroids=centroids,
        cfg_path=args.config_file,
    )
    print(f"\nSaved database with {len(embeddings)} embeddings across {len(centroid_labels)} instruments to {args.output}")


if __name__ == "__main__":
    main()
