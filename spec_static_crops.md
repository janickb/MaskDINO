# Static crop folder for the classifier — spec

## 1. Ask — verbatim

Ground truth: [requirements_static_crop_dataset.md](requirements_static_crop_dataset.md).

> but the whole architecture seems a bit crooked, how about we generate crop images on the fly
> and train a classifer from it like we did with the segmenter?

> lets first test the version where we use the current train images to produce train images for
> the classifer and then feed classifier training from this dataset such that we can just apply
> augmentation rotation and batch normalization like in a static training

This is a quotation, not a worklist. Never reworded, never ticked.

## 2. Decision

A script writes the crops to a folder, one folder per class; training reads that folder
through an ordinary `DataLoader`. The detector costs ~600-900 ms/frame, so a 1024-frame pool
is ~15 minutes — cheap enough that changing a mask setting means re-running the script rather
than storing every variant. That is what keeps this small: no manifest of labels, no checks
that the manifest still matches reality, no load-time filtering.

Two things the format has to get right. Crops are stored as **RGBA at native resolution**: RGB
un-blacked, mask in alpha. The alpha channel is what keeps photometric jitter usable, since it
has to run on the raw crop *before* blackout — contrast blends toward the image mean and a
blacked crop's mean is near zero. And storing the native box rather than a finished 224 crop
keeps `mask_dilate` a real-pixel operation; dilating an already-resized mask by 8 px would mean
a different real radius for a large instrument than a small one.

## 3. Requirements

- [ ] **The class is the folder name.** `crops/<class_name>/<frame>_p<k>.png`. No sidecar
      mapping crops to labels, so no stored integer that a reordered class list could remap.
- [ ] **RGBA, native resolution, un-blacked.** Blackout, `mask_only`, `mask_dilate`,
      `mask_close`, letterbox to `crop_size`, rotation, photometric and `mask_channel` all happen
      at load time, so the loader emits `3 + int(mask_channel)` channels. Alpha already holds the
      mask the 4th channel needs, so `mask_channel` costs no re-extraction.
- [ ] **A crop from the folder equals the crop the live path builds**, bit for bit, for the
      same frame, instance and mask settings.
- [ ] **One detector forward per frame**, in the script. Training performs **none** — it does
      not construct the detector at all unless validation needs it.
- [ ] **Only detections passing the rule get a crop**: `best_iou >= 0.5` AND
      `runner_up_iou < 0.2` (`spec_cropcls.md:54`). Everything else gets a CSV row naming the
      verdict, which is what the assignment rate is read from.
- [ ] **`margin` is replaced by `pad_frac`.** AMENDED 2026-11-04, ground truth
      [requirements_crop_padding.md](requirements_crop_padding.md). The old item said `margin`
      only mattered for closing; true then, moot now. `margin: int` is deleted;
      `pad_frac: float = 0.10` grows the box per axis off the tight box — 10% of its width on
      each side and of its height on each side, aspect preserved, clipped to the frame.
      Relative, not pixels: the long side spans 12..505 px over `crops/train`, and the uniform
      letterbox would turn a fixed border into a different fraction of the canvas per instance.
- [ ] **The stored patch is a ceiling, not the crop.** The loader re-derives the box from
      `cfg.pad_frac` and clips it to the stored patch, so one folder serves every `pad_frac` at
      or below the stored one — including the tight control at `0.0`. `cfg.pad_frac` above the
      stored one **raises**; those pixels are not in the PNG. A `meta.json` without `pad_frac`
      also raises, rather than defaulting to 0 and training tight under a padded config.
- [ ] **Six crop call sites change together**, or training and scoring disagree about the field
      of view and the result is a silently worse number: `cropcls/data.py`,
      `cropcls/crop_folder.py`, `cropcls/export.py`, `tools/confidence_metrics.py`,
      `cropcls/viz.py`, `tools/viz_class_crops.py`. The growth belongs in one helper in
      `cropcls/crops.py`. `tools/viz_misclassified.py` goes through `data.py` and needs nothing.
      Two cases that look like oversights and are not: `confidence_metrics.py` reads pre-letterbox
      `(h, w, area)` for the scale diagnostic and that read stays on the **tight** box; and the
      docstrings in `crops.py`, `crop_folder.py` and `tests/test_cropcls_crops.py` that assert
      callers pass the tight box need updating, but the invariant they protect — a passed `box`
      overrides the dilated one, so dilation never widens the view — still holds.
- [ ] **`min_border_px` is the single thinnest detection, and that gates the folder.** The
      border is `round(side * pad_frac)`, so any detection under `1/(2*pad_frac)` px on its short
      axis - 5 px at 0.10 - stores a 0 px border and `mask_close` then raises for the whole
      folder. Such detections do occur and do pass the assignment rule: a 3-frame smoke
      extraction produced a 22x3 px one. Conservative on purpose, since a partial mismatch is
      still a silent one, and cheap while closing stays at +0.4.
      **Verify:** the `border_px` column of the padded folder's `crops.csv` and `min_border_px`
      in its `meta.json` - the smoke run above went to a scratch dir and is not an artifact.
- [ ] **The closing guard becomes measured, not declared.** `meta.json` records `min_border_px`,
      the minimum stored border over all crops counting only sides that were not frame-clipped,
      and `mask_dilate` is checked against that. The threshold is `border > mask_dilate`, not
      `>=`: with the crop region now reaching the patch edge, a border equal to the radius puts
      the `close_mask` leak inside the crop. Measured at 1053 differing mask pixels on a 204x160
      instance at `pad_frac=0.10`. The `margin` scalar it replaces was also optimistic -
      `patch_bbox` clips to the frame, so an instrument at the image border had less.
- [ ] **Retiring `margin` must not orphan the 53 finished runs.** `load_config` raises on keys
      absent from the dataclass and all 53 `runs/*/config.json` carry `"margin": 0`, so deleting
      the field would break `tools/confidence_metrics.py` and `tools/viz_misclassified.py` on
      every past run. `load_config` gains a retired-key set: those are dropped with a warning,
      other unknown keys still raise.
- [ ] **`cropcls-scale-feature-notes.md` §6 item 6 must be corrected in the same commit.** It
      claims the scale feature needs no re-extraction because `(h, w, area)` is readable from the
      PNG, which held only while the PNG was the tight box. Recover it from alpha instead.
- [ ] **Static batching**: `shuffle=True`, `drop_last=True`, fixed size in **crops**.
      `ims_per_batch` keeps its FRAMES meaning and is used only by validation.
- [ ] **One rotation sampled per draw**, from its own `Generator` — never all four of an
      instance in one batch, which is what correlates the BatchNorm statistics today.
- [ ] **No trunk BatchNorm change.** "batch normalization like in a static training" is met by
      giving BN a constant-size batch of globally shuffled crops, not by reconfiguring it.
- [ ] **`meta.json` records which data this is**: the resolved pool path, a digest of the frame
      names, the detector checkpoint, and the settings baked in. The folder name does not
      identify its data — image roots have been repointed between renders of the same scene
      set. `pool_realpath` is what names the root: the digest is over frame *names*, and the
      two roots holding a phase-1 pool are byte-identical copies as of 2026-11-03 (same 1024
      names and sizes, digest `0d90fc4eb1bf` for both), so the digest catches replenishment,
      not which root.

## 4. Done when

- [x] **A folder crop equals the live crop**, over 5 frames and every mask setting including
      `mask_close`, which is the only case where the stored border is load-bearing, and including
      `blackout: false` and `mask_channel: true` - the settings `spec_cropcls.md` Step 10 ships,
      and the ones the matrix did not reach when it was first written.
      **Verify:** `tests/test_cropcls_crop_folder.py::test_folder_crop_equals_live_crop`
- [x] **The settings actually do something.** Dilation, closing, blackout and photometric each
      change the crop. The earlier suite's fixture was a solid rectangle, which made every
      dilation assertion pass with dilation deleted.
      **Verify:** `test_dilating_changes_the_crop`, `test_closing_differs_from_plain_dilation`,
      `test_blackout_off_keeps_background`, `test_photometric_runs_before_blackout`
- [x] **Training runs no detector forward.** Shown by the startup ordering: the
      `loading frozen detector` line appears *after* the last iteration, because the detector
      is built lazily and only validation needs it.
      **Verify:** 12-iteration run, `crop_folder` set, `eval_period=0`.
- [x] **The script's output loads back.** 35 crops from 6 real frames, 100% assignment rate,
      865 ms/frame on `dbe-cia-sl19-01` with 7 of 8 GPUs at 94-100% (so not a clean timing).
      **Verify:** `test_written_folder_loads_back`, `test_write_frame_crops_writes_only_kept_detections`
- [x] **Extraction cost and folder size**, naming the machine **and its load**, with frame and
      crop counts. Reported, not capped. Both measured on `dbe-cia-sl19-01`:
      - tight, `crops/train`, extracted 2026-10-28: 186 MB, 4623 crops, 1024 frames, 769 s
        (751 ms/frame), load unrecorded.
      - padded, `crops/train_pad20`, `pad_frac: 0.20`, extracted 2026-11-04: **312 MB**, the
        same 4623 crops from the same 1024 frames (`frame_digest` 0d90fc4eb1bf matches), 1417 s
        at **1384 ms/frame with GPUs 5 and 6 at ~100% from another job** - so not a clean
        timing, and on the pure-PyTorch MultiScaleDeformableAttention fallback either way.
      312 MB against the 268 MB projected from 1.44x the pixels: PNG compresses a
      mostly-background border better than pixel count implies, so the projection was 15% high.
      `min_border_px` is 1 and no crop has a 0 px border. `crops/train` can no longer serve as
      the `pad_frac: 0.0` control - it predates `pad_frac` and `CropFolder` raises on it - so
      both arms come from this one extraction. Quota 3.5 GB free (61440M limit, 57887M used),
      not the ~14 GB this item claimed before.
      **Verify:** reported in `cropcls-results.md` section 8.1, both folders, with the machine
      and its load.
- [ ] **Static training matches live training.** `classaware_f1` from the baseline's own
      `HungarianInstanceEvaluator` (`score_mode=detector`) against a **fresh live arm on the
      same frames** — not the published r2 = 0.8758, which ran on another host with an
      unrecorded render set. Within the measured endpoint seed variance of 0.82 points
      (`cropcls-results.md:166-169`). Compare at matched **total crops seen**, not matched
      `max_iter`: the live arm emits 4 rotations per instance per forward and the static arm
      one per draw, so equal iterations is not equal training volume. `eval.json` records
      `total_crops_seen` for both arms. Crop top-1 is not the metric — it misled in both
      directions (`cropcls-results.md:149-156`).
      **Verify:** reported in `cropcls-results.md` with run ids, iterations, machine and load.

## 5. Out of scope (explicit)

- Validation stays on the live path, in training and under `--eval-only`. Caching it needs a
  second folder and an `evaluate()` taking a loader, and it would mean deleting the crops
  breaks re-scoring a finished run.
- `CropPipeline.iter_batches` (the live path) is not deleted — it is the comparison arm.
- `--viz-only` stays live: it needs whole frames and overlays, which the folder does not store.
  No second, folder-sourced contact sheet.
- Trunk BatchNorm configuration unchanged (`freeze_trunk`, `CropClassifier.train`).
- No scene_generator renders. "like we did with the segmenter" is read as *static dataset*; the
  second message pins it to "the current train images".
- Phase 2 / set B.
- **One deviation from `spec_cropcls.md`'s out-of-scope clause:** `maskdino/` *was* modified —
  the `training/images` symlink is no longer used, so nothing silently de-registers when it
  goes missing. The current form is `maskdino/data/datasets/images_root.py`: the root comes
  from `$SURGICAL_IMAGES_ROOT`, else from the single `~/training/images_*` match, and several
  matches **raise** instead of being picked silently. `dbe-cia-sl19-01` has three, so that
  variable must be exported there.

## Files

- `tools/extract_crops.py` — the script.
- `cropcls/crop_folder.py` — format helpers, `write_frame_crops`, `CropFolder`, `build_loader`.
- `tests/test_cropcls_crop_folder.py` — 31 test functions, 50 cases with parametrization.
- `cropcls/config.py` — `crop_folder`, `pad_frac`, `crops_per_batch`, `num_workers`.
- `train_cropcls.py` — static branch, lazy detector, `total_crops_seen`.
- Deleted: `cropcls/cache.py` and its two test files (commit `5b31900` if needed).
