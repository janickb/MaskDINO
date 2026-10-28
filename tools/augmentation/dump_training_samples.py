#!/usr/bin/env python
"""Dump what the training mapper actually hands the model, each layer as its own file.

Unlike hdf5_viewer (which browses raw .hdf5 frames, i.e. *pre*-augmentation ground
truth), this runs the real Hdf5CocoInstanceDatasetMapper in training mode - copy-paste, rotation,
resize, pad, photometric augs, MIN_VISIBILITY filtering, all of it - and writes the
image, the instance map, the class map, every per-instance binary mask, the boxes and
a JSON report as *separate* files, so a mask can be inspected on its own.

Per sample::

    000_image.png                     final augmented RGB the model receives
    000_instances.png                 instance-id label map, one color per instance
    000_classes.png                   category-id label map, one color per class
    000_boxes.png                     gt_boxes on black - exposes box/mask disagreement
    000_overlay.png                   masks tinted on the image (only with --overlay)
    000_mask_00_forcep03_mm.png       one binary mask per instance (white = foreground)
    000_info.json                     per-instance areas/classes + per-frame transforms

Example::

    ./.venv/bin/python tools/augmentation/dump_training_samples.py \
        --config-file configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify.yaml \
        --num-samples 8 --out-dir /tmp/dump INPUT.MIN_SCALE 0.8 INPUT.MAX_SCALE 1.2

Trailing KEY VALUE pairs override the config, so one knob can be previewed across
settings without editing anything::

    for t in 8.0 18.0 40.0; do
      ./.venv/bin/python tools/augmentation/dump_training_samples.py --config-file <arm>.yaml \
          --seed 5 --out-dir /tmp/tilt_$t INPUT.COPY_PASTE.PERSPECTIVE.TILT_DEGREES $t
    done

(yacs type-checks against the existing value, so a float knob needs `40.0`, not `40`.)

With --compare-pad-fix the same seeded samples are dumped twice, into <out>/fixed/ and
<out>/broken/, differing only in FixedSizeCrop's seg_pad_value (0 vs detectron2's
default 255). Diffing the two *_instances.png / *_info.json is the direct way to see
the pad-value bug: with 255 every instance's bitmask additionally claims the whole
padded band, while its box stays correct.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

# repo root is three levels up now that this lives in tools/augmentation/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from _preview_common import build_preview_mapper
from detectron2.data import transforms as T

# Distinct, saturated colors for the instance map; cycled if a frame has more.
_PALETTE = [
    (66, 135, 245), (245, 90, 66), (66, 245, 129), (245, 215, 66),
    (188, 66, 245), (66, 245, 236), (245, 66, 175), (150, 245, 66),
    (245, 150, 66), (66, 90, 245), (120, 245, 180), (245, 66, 90),
]


def _class_color(contiguous_id, num_classes):
    """Evenly spaced hues, so two neighbouring class ids never look alike."""
    hue = int(179 * (contiguous_id % max(num_classes, 1)) / max(num_classes, 1))
    hsv = np.uint8([[[hue, 220, 245]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class _Recording(T.Augmentation):
    """Wraps one Augmentation and records the Transform it produces per call, so the
    dump can report the sampled scale and the exact pad geometry instead of guessing
    them back out of the pixels. Must be an Augmentation (AugmentationList asserts on
    the type), and its get_transform signature must stay image-only - that signature
    is what detectron2's _get_aug_input_args inspects to decide which AugInput fields
    to pass, and all three wrapped augs take just the image."""

    def __init__(self, inner, sink):
        super().__init__()
        self.inner = inner
        self.sink = sink

    def get_transform(self, image):
        tfm = self.inner.get_transform(image)
        self.sink.append(repr(tfm))
        return tfm

    def __repr__(self):
        return f"_Recording({self.inner!r})"


def _instrument_tfm_gens(pristine_gens, sink, seg_pad_value=None):
    """Returns a fresh tfm_gens list: the geometric augs wrapped in _Recording, and
    optionally FixedSizeCrop rebuilt with a different seg_pad_value (the pad-fix A/B).
    Builds from `pristine_gens` rather than mutating in place, so repeated variants
    never stack wrappers on top of each other."""
    gens = []
    for aug in pristine_gens:
        if seg_pad_value is not None and isinstance(aug, T.FixedSizeCrop):
            aug = T.FixedSizeCrop(
                crop_size=aug.crop_size, pad=aug.pad, pad_value=aug.pad_value,
                seg_pad_value=seg_pad_value,
            )
        if isinstance(aug, (T.ResizeScale, T.FixedSizeCrop, T.RandomRotation)):
            aug = _Recording(aug, sink)
        gens.append(aug)
    return gens


def _overlay(image, masks, boxes, classes, class_names):
    """Masks tinted onto the image with their boxes - the one-glance view.

    Absorbed from the former tools/preview_copy_paste.py. The overlay answers "does this look plausible?"; the separate
    per-layer files answer "is the ground truth correct?" - a mask drawn ON the
    image hides exactly the errors worth finding (the FixedSizeCrop pad band was
    invisible in an overlay and obvious in a bare instance map).
    """
    vis = image.copy()
    for i in range(len(masks)):
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        m = masks[i]
        vis[m] = (0.5 * vis[m] + 0.5 * color).astype(np.uint8)
        x0, y0, x1, y1 = (int(v) for v in boxes[i])
        cv2.rectangle(vis, (x0, y0), (x1, y1), tuple(int(c) for c in color), 1)
    return vis


def _to_bgr(image_tensor, image_format):
    image = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    if image_format != "BGR":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def dump_one(out_dir, idx, src_name, out, class_names, image_format, transforms,
             overlay=False):
    """Writes every layer of one mapped sample as its own file. Returns the info dict."""
    image = _to_bgr(out["image"], image_format)
    h, w = image.shape[:2]
    instances = out["instances"]
    n = len(instances)
    masks = instances.gt_masks.numpy().astype(bool) if n else np.zeros((0, h, w), bool)
    boxes = instances.gt_boxes.tensor.numpy() if n else np.zeros((0, 4))
    classes = instances.gt_classes.numpy() if n else np.zeros((0,), int)

    prefix = os.path.join(out_dir, f"{idx:03d}")
    cv2.imwrite(f"{prefix}_image.png", image)

    inst_map = np.zeros((h, w, 3), np.uint8)
    class_map = np.zeros((h, w, 3), np.uint8)
    box_map = np.zeros((h, w, 3), np.uint8)
    coverage = masks.sum(axis=0) if n else np.zeros((h, w), int)

    per_instance = []
    for i in range(n):
        mask = masks[i]
        color = _PALETTE[i % len(_PALETTE)]
        inst_map[mask] = color
        cid = int(classes[i])
        name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        class_map[mask] = _class_color(cid, len(class_names))

        x0, y0, x1, y1 = (float(v) for v in boxes[i])
        cv2.rectangle(box_map, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)

        cv2.imwrite(
            f"{prefix}_mask_{i:02d}_{name}.png", (mask.astype(np.uint8) * 255)
        )

        mask_area = int(mask.sum())
        box_area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        per_instance.append({
            "index": i,
            "class_name": name,
            "contiguous_class_id": cid,
            "mask_area_px": mask_area,
            "box_area_px": round(box_area, 1),
            "mask_over_box_ratio": round(mask_area / box_area, 2) if box_area else None,
            "box_xyxy": [round(v, 1) for v in (x0, y0, x1, y1)],
            # How much of this instance's mask is also claimed by another instance.
            # A healthy composited frame has a little (real occlusion is resolved to
            # modal masks upstream, so this should be ~0); a shared pad band shows up
            # here as a huge number repeated across every instance.
            "overlap_with_other_masks_px": int((mask & (coverage > 1)).sum()),
        })

    if overlay:
        cv2.imwrite(f"{prefix}_overlay.png",
                    _overlay(image, masks, boxes, classes, class_names))
    cv2.imwrite(f"{prefix}_instances.png", inst_map)
    cv2.imwrite(f"{prefix}_classes.png", class_map)
    cv2.imwrite(f"{prefix}_boxes.png", box_map)

    info = {
        "source_frame": src_name,
        "image_size": [h, w],
        "image_format": image_format,
        "num_instances": n,
        "geometric_transforms": transforms,
        "pixels_claimed_by_multiple_instances": int((coverage > 1).sum()),
        "instances": per_instance,
    }
    with open(f"{prefix}_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def run_variant(
    out_dir, cfg, mapper, pristine_gens, class_names, sample, seed, seg_pad_value,
    print_header, overlay=False,
):
    os.makedirs(out_dir, exist_ok=True)

    sink = []
    mapper.tfm_gens = _instrument_tfm_gens(
        pristine_gens, sink, seg_pad_value=seg_pad_value
    )

    # Both the copy-paste compositor (`random`) and detectron2's augmentations
    # (`np.random`) must be reseeded, or the two pad-fix variants would not be
    # comparing the same composited frames under the same augmentation draws.
    random.seed(seed)
    np.random.seed(seed)

    print(print_header)
    for i, d in enumerate(sample):
        sink.clear()
        out = mapper(d)
        if out is None:
            print(f"  [{i}] mapper returned None (unreadable frame), skipping")
            continue
        info = dump_one(
            out_dir, i, os.path.basename(d["file_name"]), out, class_names,
            cfg.INPUT.FORMAT, list(sink), overlay=overlay,
        )
        shared = info["pixels_claimed_by_multiple_instances"]
        areas = [inst["mask_area_px"] for inst in info["instances"]]
        boxes = [inst["box_area_px"] for inst in info["instances"]]
        print(
            f"  [{i}] {info['source_frame']}: {info['num_instances']} inst, "
            f"mask areas {areas}, box areas {boxes}, "
            f"shared-by->1-instance px {shared}"
        )
    print(f"  -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify.yaml",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", default="/tmp/dump_training_samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overlay", action="store_true",
        help="additionally write <nnn>_overlay.png: masks tinted onto the image "
             "with boxes drawn - the quick 'does this look plausible' view "
             "(absorbed from the former preview_copy_paste.py)",
    )
    parser.add_argument(
        "--compare-pad-fix", action="store_true",
        help="dump the same seeded samples twice, into <out>/fixed and <out>/broken, "
             "differing only in FixedSizeCrop's seg_pad_value (0 vs 255)",
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="trailing detectron2 KEY VALUE config overrides",
    )
    args = parser.parse_args()

    # Seed BEFORE building: CopyPasteCompositor seeds its private jitter/perspective
    # streams from the global RNG at construction time, so constructing first and
    # seeding second would give a different appearance stream on every invocation -
    # and two arms dumped in separate processes would not be comparable.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Built once: CopyPasteCompositor caches every eligible source instance up front,
    # so constructing a second mapper would re-read the whole training set.
    cfg, mapper, dataset_dicts, class_names = build_preview_mapper(
        args.config_file, args.opts
    )
    pristine_gens = list(mapper.tfm_gens)

    # Pick the frames from an isolated RNG, so the per-variant reseeding inside
    # run_variant cannot change *which* frames get dumped.
    n = min(args.num_samples, len(dataset_dicts))
    sample = random.Random(args.seed).sample(dataset_dicts, n)
    print(f"[dump] {len(dataset_dicts)} train frames, dumping {n}")
    print(f"[dump] tfm_gens: {pristine_gens}")

    variants = [(args.out_dir, None, "[dump] variant: as configured")]
    if args.compare_pad_fix:
        variants = [
            (os.path.join(args.out_dir, "fixed"), 0,
             "[dump] variant: fixed (seg_pad_value=0)"),
            (os.path.join(args.out_dir, "broken"), 255,
             "[dump] variant: broken (seg_pad_value=255, detectron2 default)"),
        ]
    for out_dir, seg_pad_value, header in variants:
        run_variant(
            out_dir, cfg, mapper, pristine_gens, class_names, sample, args.seed,
            seg_pad_value, header, overlay=args.overlay,
        )


if __name__ == "__main__":
    main()
