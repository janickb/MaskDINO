# Copy-paste compositing — reference

`maskdino/data/augmentations/copy_paste.py` (`CopyPasteCompositor`). Enabled per config via
`cfg.INPUT.COPY_PASTE.ENABLED`, wired in by `Hdf5CocoInstanceDatasetMapper` (reclassify phase
only). Combines MBOI's (Badilla-Solorzano et al., IJCARS 2022) cut/rotate/paste with Cut,
Paste and Learn's (Dwibedi et al., arXiv:1708.01642) per-instance randomized blend mode — see
the module's own header comment for the full rationale on what's deliberately *not* copied
from either paper (no flip/rescale, no retry-to-avoid-overlap).

## Pipeline

**Once, in `__init__`** (`_build_instance_cache`): every eligible annotation (`iscrowd == 0`)
across all `source_dicts` is cut out exactly once — decode its RLE mask, crop the source frame
to that mask's bounding box **plus a margin** (`_crop_margin_px`, derived from
`blur_kernel_range[1] // 2 + 1`, clamped to the source frame's bounds), and cache
`(crop, crop_mask, category_id)`. The margin exists so `gaussian_blur_edge`/`box_blur` later
have genuine background pixels to blur with instead of hitting the crop's own edge (see
Limitations).

**Per call** (`__call__`, once per training sample): draw `k ~ U(min_instances, max_instances)`,
sample `k` distinct instances from the cache via `random.sample(range(n), k)` (O(k), not O(n) —
deliberately not a `list(cache)` + `shuffle`, since the cache can be large but only `k` of it is
ever used). For each sampled instance: rotate it rigidly (`_transform_instance` — no flip, no
rescale), place it at one uniform-random position with no retry (`_place` — returns `None`,
skipping this instance, only if its rotated bounding box literally can't fit in the destination
canvas), composite it (`_blend`), then shrink every earlier layer's stored mask by whatever the
new paste's *hard* silhouette covers (`layer["visible"] &= ~full_mask`) so all returned
annotations stay modal (visible-region-only, never amodal).

## Functions

| function | does |
|---|---|
| `_read_colors` | reads an `.hdf5` frame's `colors` array, converts to `image_format`; returns `None` (logged) on a torn/missing read |
| `_build_instance_cache` | one-time cut-out of every eligible instance's crop + mask + category, margin-padded |
| `__call__` | per-sample entry point: sample instances, rotate/place/blend each, update occlusion, return `(composite, annotations)` |
| `_update_modal_annotation` | re-rasterizes one layer's segmentation/bbox/area/`visibility_fraction` from its current `visible` mask |
| `_transform_instance` | rotates crop + mask by `θ ~ U(-rotation_degrees, +rotation_degrees)`; same interpolation for both (avoids boundary mismatch — see `mask_interp`); also returns `valid_footprint` marking real content vs. the rotation canvas's synthetic black padding |
| `_place` | single uniform-random `(py, px)` so the whole rotated patch lands inside the canvas; `None` if it can't fit at all |
| `_blend` | picks a blend mode per instance, computes per-pixel `alpha`, composites `alpha*patch + (1-alpha)*region` |
| `_protect_core` | forces `alpha=1.0` more than `core_margin_px` inside the mask, so a wide blur/feather can't erode the object's own interior |
| `_blend_poisson` | `cv2.seamlessClone`-based blend; opt-in only (not in the default `blend_modes`) |

## Blend modes (`_blend`)

| mode | mechanism |
|---|---|
| `none` | hard 0/1 mask, no ramp |
| `gaussian_blur_edge` / `box_blur` | blur the hard mask itself with a random odd kernel from `blur_kernel_range`; naturally continuous since it's one linear filter pass |
| `alpha_feather` | two independent Euclidean-distance ramps (`dist_in` inside, `dist_out` outside), `width ~ U(*feather_width_range)`, stitched via `np.where(patch_mask, ...)`. **Not recommended** — qualitatively bad visual results, not just the boundary bug below |
| `poisson` | gradient-domain seamless cloning; falls back to `none` if it can't be solved or the patch would spill past the canvas edge |

## Known limitations

- **`alpha_feather` is not recommended** — qualitatively bad visual results in practice, beyond
  just the boundary bug below; that's why it's excluded from `BLEND_MODES`'s default (see
  Config). `dist_in`/`dist_out` each start counting from 1 (not 0) at the first pixel on their
  own side, so the two ramps don't join continuously — alpha dips right at the true edge
  instead of rising smoothly through it. Not thickness-dependent (it happens at every
  boundary), but its impact is negligible on thick objects (a 1px blemish swallowed by a large
  opaque interior) and much more visible on thin ones (no interior left to dilute it).
- **Thin structures can wash out under `gaussian_blur_edge`/`box_blur`.** `_protect_core`'s
  erosion can fully erase a thin object's "core" (nothing survives to force back to
  `alpha=1`), and a wide kernel then blurs across the *entire* object from both edges at once.
  The reference implementation (`debidatta/syndata-generation`, cited above) uses fixed 5×5
  (σ=2) / 3×3 kernels, well below this codebase's configurable range — keep
  `blur_kernel_range`'s upper bound modest for domains with thin instrument features (finger
  loops, blades).
- **Occlusion bookkeeping uses the hard silhouette, not the soft alpha actually painted.**
  `layer["visible"] &= ~full_mask` is a binary cut; at a feathered/blurred boundary pixel the
  underlying layer may still be mostly visible pixel-wise, but the GT mask treats it as fully
  occluded. Believed intentional (blending is photometric camouflage only; GT should stay
  geometrically exact) rather than a bug.

## Config (`cfg.INPUT.COPY_PASTE`)

| key | default | meaning |
|---|---|---|
| `ENABLED` | `False` | attach the compositor (force-disabled at eval time regardless) |
| `MIN_INSTANCES` / `MAX_INSTANCES` | `0` / `10` | range for `k`, the number of paste attempts per sample |
| `ROTATION_DEGREES` | `30.0` | `θ ~ U(-this, +this)`; MBOI itself uses `U(-90, 90)` |
| `MASK_INTERP` | `["nearest"]` | list of eligible modes, randomized per instance like `BLEND_MODES`: `"nearest"` (blocky, bit-exact mask), `"linear"`/`"bicubic"`/`"lanczos"` (anti-aliased, mask thresholded at 0.5). `"bicubic"`/`"lanczos"` genuinely ring (~15-20% brightness overshoot at a high-contrast edge in testing) unlike `"linear"`'s pure convex combination — saturates safely into `uint8`, doesn't jaggy up the mask boundary, but is a real visible artifact tradeoff |
| `BLEND_MODES` | `["none", "gaussian_blur_edge", "box_blur"]` | randomized per pasted instance; `"poisson"` is implemented but deliberately excluded by default (hurt in Cut-Paste-and-Learn's own ablation); `"alpha_feather"` is excluded too and **not recommended** — qualitatively bad visual results (see Limitations) |
| `BLUR_KERNEL_RANGE` | `[3, 7]` | odd kernel size range for `gaussian_blur_edge`/`box_blur`; also sets the cache's crop margin (`upper // 2 + 1`) |
| `FEATHER_WIDTH_RANGE` | `[2, 7]` | ramp width range for `alpha_feather` |
| `CORE_MARGIN_PX` | `2` | how far inward from any boundary `_protect_core` forces `alpha=1`; `0` disables it (restores the old fully-symmetric ramp) |

Not a config knob: `image_format` is always `cfg.INPUT.FORMAT` (kept in sync with the rest of
the mapper, not copy-paste-specific). `CopyPasteCompositor`'s constructor has no defaults of
its own — every kwarg above is always passed explicitly from this config block, so this table
is the single source of truth; don't expect (or add) a second default elsewhere.
