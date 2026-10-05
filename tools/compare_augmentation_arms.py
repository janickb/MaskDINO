#!/usr/bin/env python
"""Side-by-side sanity check of the augmented TRAINING images the two architecture
arms actually receive, plus a pixel-exact diff.

Both arms are supposed to share one augmentation pipeline: the same
Hdf5CocoInstanceDatasetMapper, the same build_transform_gen() list, the same
CopyPasteCompositor. tests/test_mask2former_harness_parity.py asserts that on a
synthetic frame; this does it on real frames and lets a human look at the result
before committing GPU hours.

For each sampled frame it reseeds the global RNGs identically, runs the frame through
each arm's own mapper (built from that arm's own config), and writes one panel:

    [ MaskDINO | Mask2Former | abs diff ]

Because the augmentations draw from the global numpy/random/torch RNGs, reseeding
before each call makes the two arms' output byte-identical when the pipeline really is
shared. Any nonzero diff means the configs have drifted on one of
INPUT.{IMAGE_SIZE,MIN_SCALE,MAX_SCALE,RANDOM_ROTATION,ROTATION_ANGLES,ROTATION_EXPAND,
MIN_VISIBILITY,COPY_PASTE.*} - fix the YAML, the pipeline itself is one object.

NOTE: byte-identity here is a property of *reseeding*, not of a training run. In real
training the two arms consume the global RNG at different rates, so they see the same
augmentation DISTRIBUTION but not the same realized pixels. That is why the comparison
wants >= 2 seeds per arm.

Examples::

    # phase-1 pair (rotation + LSJ jitter + photometric, no copy-paste)
    ./.venv/bin/python tools/compare_augmentation_arms.py --pair phase1 --num-samples 6

    # phase-2 pair (copy-paste compositing on, scale pinned to 1.0)
    ./.venv/bin/python tools/compare_augmentation_arms.py --pair phase2 --num-samples 6

    # what actually feeds the network, without the mask overlay
    ./.venv/bin/python tools/compare_augmentation_arms.py --pair phase2 --no-overlay
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CFG_DIR = "configs/coco/instance-segmentation"

# (maskdino config, mask2former config) per experiment, as the parity test pairs them.
_PAIRS = {
    "phase1": (
        "maskdino_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
        "maskformer2_R50_surgical_tools_finetune_multiclass_seta_frozen_backbone",
    ),
    "phase2": (
        "maskdino_R50_surgical_tools_reclassify",
        "maskformer2_R50_surgical_tools_reclassify",
    ),
    "singleclass": (
        "maskdino_R50_surgical_tools_finetune_singleclass",
        "maskformer2_R50_surgical_tools_finetune_singleclass",
    ),
}

_PALETTE = [
    (66, 135, 245), (245, 90, 66), (66, 245, 129), (245, 215, 66),
    (188, 66, 245), (66, 245, 236), (245, 66, 175), (150, 245, 66),
]


def _seed_all(seed):
    """Reset every RNG the augmentations draw from, so the two arms start level."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_bgr(out, image_format):
    """mapper output -> HWC uint8 BGR, ready for cv2."""
    image = out["image"].permute(1, 2, 0).numpy()
    if image_format != "BGR":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image.astype(np.uint8))


def _overlay(image, instances, class_names, draw=True):
    vis = image.copy()
    rows = []
    for i in range(len(instances)):
        mask = instances.gt_masks[i].numpy().astype(bool)
        if draw:
            color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
            vis[mask] = (0.5 * vis[mask] + 0.5 * color).astype(np.uint8)
            ys, xs = np.where(mask)
            if len(ys):
                cv2.rectangle(
                    vis, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())),
                    tuple(int(c) for c in color), 1,
                )
        cid = int(instances.gt_classes[i])
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        rows.append((i, name, cid, int(mask.sum())))
    return vis, rows


def _label(image, text):
    """Caption strip above the panel, so a saved PNG is self-describing."""
    bar = np.zeros((28, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, image])


def _build(config_name, overlay):
    """Return (cfg, mapper, class_names, dataset_dicts) for one arm."""
    from detectron2.data import DatasetCatalog, MetadataCatalog

    from maskdino.config import build_base_cfg
    from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
        Hdf5CocoInstanceDatasetMapper,
    )

    cfg = build_base_cfg()
    cfg.merge_from_file(os.path.join(_CFG_DIR, config_name + ".yaml"))
    cfg.freeze()

    name = cfg.DATASETS.TRAIN[0]
    dicts = DatasetCatalog.get(name)
    class_names = list(MetadataCatalog.get(name).thing_classes)
    mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
    # Only does anything when INPUT.COPY_PASTE.ENABLED; needs the full dict list.
    mapper.set_copy_paste_sources(dicts)
    return cfg, mapper, class_names, dicts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pair", choices=sorted(_PAIRS), default="phase1")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="save the raw augmented image with no mask overlay - exactly what the "
             "network sees",
    )
    args = parser.parse_args()

    dino_name, m2f_name = _PAIRS[args.pair]
    out_dir = args.out_dir or f"/tmp/aug_compare_{args.pair}"
    os.makedirs(out_dir, exist_ok=True)

    dino_cfg, dino_mapper, dino_classes, dicts = _build(dino_name, not args.no_overlay)
    m2f_cfg, m2f_mapper, m2f_classes, _ = _build(m2f_name, not args.no_overlay)

    print(f"pair           : {args.pair}")
    print(f"  maskdino     : {dino_name}")
    print(f"  mask2former  : {m2f_name}")
    print(f"dataset        : {dino_cfg.DATASETS.TRAIN[0]}  ({len(dicts)} frames, "
          f"{len(dino_classes)} classes)")
    print(f"augmentation   : {[type(t).__name__ for t in dino_mapper.tfm_gens]}")
    print(f"  IMAGE_SIZE={dino_cfg.INPUT.IMAGE_SIZE} "
          f"MIN_SCALE={dino_cfg.INPUT.MIN_SCALE} MAX_SCALE={dino_cfg.INPUT.MAX_SCALE} "
          f"ROTATION={dino_cfg.INPUT.RANDOM_ROTATION}{list(dino_cfg.INPUT.ROTATION_ANGLES)} "
          f"MIN_VISIBILITY={dino_cfg.INPUT.MIN_VISIBILITY}")
    print(f"  COPY_PASTE.ENABLED={dino_cfg.INPUT.COPY_PASTE.ENABLED}"
          + (f"  instances={dino_cfg.INPUT.COPY_PASTE.MIN_INSTANCES}-"
             f"{dino_cfg.INPUT.COPY_PASTE.MAX_INSTANCES}"
             f"  blends={list(dino_cfg.INPUT.COPY_PASTE.BLEND_MODES)}"
             if dino_cfg.INPUT.COPY_PASTE.ENABLED else ""))
    if dino_classes != m2f_classes:
        print("  !! the two arms resolved DIFFERENT class name lists")
    print()

    # Index rather than random.sample(dicts, ...): the phase-1 train sets are a
    # LivePoolDataset (a map-style torch Dataset over a churning pool directory), which
    # supports len()/__getitem__ but is not a sequence.
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dicts)), min(args.num_samples, len(dicts)))
    sample = [dicts[j] for j in indices]

    n_identical = 0
    n_compared = 0
    for i, d in enumerate(sample):
        # Same frame, same RNG state, each arm's own mapper.
        _seed_all(args.seed + i)
        dino_out = dino_mapper(d)
        _seed_all(args.seed + i)
        m2f_out = m2f_mapper(d)

        if dino_out is None or m2f_out is None:
            print(f"[{i}] mapper returned None (frame evicted / unreadable), skipping")
            continue

        dino_img = _to_bgr(dino_out, dino_cfg.INPUT.FORMAT)
        m2f_img = _to_bgr(m2f_out, m2f_cfg.INPUT.FORMAT)
        dino_inst, m2f_inst = dino_out["instances"], m2f_out["instances"]

        pixel_identical = dino_img.shape == m2f_img.shape and np.array_equal(
            dino_img, m2f_img
        )
        gt_identical = (
            len(dino_inst) == len(m2f_inst)
            and dino_inst.gt_classes.equal(m2f_inst.gt_classes)
            and dino_inst.gt_masks.shape == m2f_inst.gt_masks.shape
            and dino_inst.gt_masks.equal(m2f_inst.gt_masks)
        )
        n_compared += 1
        n_identical += bool(pixel_identical and gt_identical)

        diff = cv2.absdiff(dino_img, m2f_img) if dino_img.shape == m2f_img.shape else None
        max_diff = int(diff.max()) if diff is not None else -1

        dino_vis, dino_rows = _overlay(
            dino_img, dino_inst, dino_classes, draw=not args.no_overlay
        )
        m2f_vis, _ = _overlay(
            m2f_img, m2f_inst, m2f_classes, draw=not args.no_overlay
        )
        panels = [
            _label(dino_vis, f"MaskDINO  {len(dino_inst)} inst"),
            _label(m2f_vis, f"Mask2Former  {len(m2f_inst)} inst"),
        ]
        if diff is not None:
            # Amplify so a non-zero-but-small difference is actually visible.
            panels.append(
                _label(
                    np.clip(diff.astype(np.int32) * 8, 0, 255).astype(np.uint8),
                    f"abs diff x8  max={max_diff}",
                )
            )
        panel = np.hstack(panels)

        out_path = os.path.join(out_dir, f"{args.pair}_{i:02d}.png")
        cv2.imwrite(out_path, panel)

        flag = "IDENTICAL" if (pixel_identical and gt_identical) else "DIFFERS"
        print(f"[{i}] {os.path.basename(d['file_name'])}  {flag}  "
              f"(pixels={'=' if pixel_identical else '!'} gt={'=' if gt_identical else '!'}"
              f" max_pixel_diff={max_diff})  -> {out_path}")
        for idx, name, cid, area in dino_rows:
            print(f"       [{idx}] {name} (id {cid}) area={area}")

    print()
    print(f"{n_identical}/{n_compared} frames byte-identical across the two arms")
    if n_compared and n_identical == n_compared:
        print("=> the two arms share one augmentation pipeline, confirmed on real frames")
    elif n_compared:
        print("=> MISMATCH: the configs have drifted; compare the resolved configs with")
        print("   ./.venv/bin/python -m pytest tests/test_mask2former_harness_parity.py")
    print(f"wrote {n_compared} panel(s) to {out_dir}")


if __name__ == "__main__":
    main()
