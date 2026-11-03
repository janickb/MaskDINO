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
- [ ] **`margin` only matters for closing.** Measured: plain dilation is exact at `margin=0`,
      because every mask pixel is inside its own tight box. `mask_close` needs
      `margin >= mask_dilate` exactly — the closing window is truncated at the patch edge
      otherwise — so that combination **raises** rather than quietly differing.
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
- [ ] **Extraction cost and folder size**, naming the machine **and its load**, with frame and
      crop counts. Reported, not capped: the quota has ~14 GB free and the folder is ~40 KB per
      crop at `margin=0`, so size is provenance rather than a constraint.
      **Verify:** reported in `cropcls-results.md`.
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
- `tests/test_cropcls_crop_folder.py` — 26 tests.
- `cropcls/config.py` — `crop_folder`, `margin`, `crops_per_batch`, `num_workers`.
- `train_cropcls.py` — static branch, lazy detector, `total_crops_seen`.
- Deleted: `cropcls/cache.py` and its two test files (commit `5b31900` if needed).
