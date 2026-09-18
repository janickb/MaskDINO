#!/usr/bin/env python
"""Preview the reclassify-phase copy-paste compositing augmentation on real
reclass_train frames, without running any model/training.

Runs Hdf5CocoInstanceDatasetMapper (with INPUT.COPY_PASTE forced on) over a
handful of real reclass_train dicts and saves, per sample, a PNG with the
composited image and its kept instance masks overlaid in distinct colors -
plus prints each kept instance's category name and visibility_fraction, so a
human can eyeball that:
  - multiple instruments show up in a composite (not just the original one)
  - pasted instruments look plausibly rotated/scaled, never mirrored
  - the 2-3 blend styles are visibly distinguishable across the batch
  - low-visibility instances are actually getting dropped by MIN_VISIBILITY

Example::

    ./.venv/bin/python tools/preview_copy_paste.py --num-samples 15 --out-dir /tmp/cp_preview
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PALETTE = [
    (66, 135, 245), (245, 90, 66), (66, 245, 129), (245, 215, 66),
    (188, 66, 245), (66, 245, 236), (245, 66, 175), (150, 245, 66),
]


def _overlay_masks(image, instances, class_names, draw=True):
    vis = image.copy()
    lines = []
    for i in range(len(instances)):
        mask = instances.gt_masks[i].numpy().astype(bool)
        if draw:
            color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
            vis[mask] = (0.5 * vis[mask] + 0.5 * color).astype(np.uint8)

            ys, xs = np.where(mask)
            if len(ys):
                cv2.rectangle(
                    vis, (xs.min(), ys.min()), (xs.max(), ys.max()),
                    tuple(int(c) for c in color), 1,
                )
        cat_id = int(instances.gt_classes[i])
        name = class_names[cat_id] if 0 <= cat_id < len(class_names) else str(cat_id)
        lines.append(f"  [{i}] {name} (contiguous id {cat_id}), mask area {mask.sum()}")
    return vis, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--out-dir", default="/tmp/preview_copy_paste")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="save the raw composited image (no mask overlay/bbox drawing) - "
             "what actually feeds into training",
    )
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config

    from maskdino.config import add_maskdino_config
    from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
        Hdf5CocoInstanceDatasetMapper,
    )

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.INPUT.COPY_PASTE.ENABLED = True
    cfg.INPUT.COPY_PASTE.BLEND_MODES = ["gaussian_blur_edge"] #["none", "gaussian_blur_edge", "box_blur", "alpha_feather"]
    cfg.INPUT.COPY_PASTE.MASK_INTERP = ["bicubic"]  # randomized per instance
    cfg.freeze()

    train_dataset_name = cfg.DATASETS.TRAIN[0]
    dataset_dicts = DatasetCatalog.get(train_dataset_name)
    class_names = MetadataCatalog.get(train_dataset_name).thing_classes
    print(f"[preview_copy_paste] {train_dataset_name}: {len(dataset_dicts)} frames, "
          f"{len(class_names)} classes")

    mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
    mapper.set_copy_paste_sources(dataset_dicts)

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.num_samples, len(dataset_dicts))
    sample = random.sample(dataset_dicts, n)

    for i, d in enumerate(sample):
        out = mapper(d)
        if out is None:
            print(f"[{i}] mapper returned None (unreadable frame), skipping")
            continue

        image = out["image"].permute(1, 2, 0).numpy()
        if cfg.INPUT.FORMAT != "BGR":
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)

        instances = out["instances"]
        vis, lines = _overlay_masks(image, instances, class_names, draw=not args.no_overlay)

        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.png")
        cv2.imwrite(out_path, vis)

        print(f"[{i}] {os.path.basename(d['file_name'])} -> {out_path}, "
              f"{len(instances)} kept instance(s)")
        for line in lines:
            print(line)

    print(f"[preview_copy_paste] wrote {n} preview image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
