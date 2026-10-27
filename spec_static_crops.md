# Static crop dataset for the classifier — spec

## 1. Ask — verbatim

Ground truth: [requirements_static_crop_dataset.md](requirements_static_crop_dataset.md).

> but the whole architecture seems a bit crooked, how about we generate crop images on the fly
> and train a classifer from it like we did with the segmenter?

> lets first test the version where we use the current train images to produce train images for
> the classifer and then feed classifier training from this dataset such that we can just apply
> augmentation rotation and batch normalization like in a static training

This is a quotation, not a worklist. Never reworded, never ticked.

## 2. Decision

Extract crops to disk once per frame, then train from them through an ordinary `DataLoader`:
global shuffle, fixed batch size in **crops**, one rotation sampled per draw. The premise is in
the code — with `augment_frames: false`, `cropcls/data.py:154-162` makes the detector's masks
**deterministic per frame**, so today's per-batch forward recomputes identical output every
time a frame is drawn.

`train_seta_mm` is a **live pool**, not a static set: `register_hdf5_instance.py:267-280`
registers it through `register_hdf5_pool_instances` as a `LivePoolDataset`
(`_POOL_VIRTUAL_SIZE` 1024), continuously replenished by sgdata. So the invariant is **one
forward per frame per pool lifetime**, not zero forwards during training — and the cache must
mirror pool eviction or it leaks retired scenes and grows ~2.8 GB/day
(`spec_cropcls.md:136`).

Store the native-resolution patch plus its mask **pre-blackout**, not the finished letterbox:
letterbox scale is set by the crop box, so baking it would fix `mask_dilate` — the largest
effect measured, **+4.2** at 16 px (`cropcls-results.md:130`). Rotations are not stored; on a
square canvas they are lossless permutations, so copies add no information.

## 3. Requirements

Literal. Every threshold carries its number **and its direction**.

- [ ] **One forward per frame per pool lifetime:** extract a frame on first sight; never
      re-extract a frame still resident in the pool. Training itself performs no detector
      forward for an already-cached frame.
- [ ] **Cache keyed by frame filename**, as `spec_cropcls.md:135` already required.
- [ ] **Provenance, enforced on load:** the manifest records the detector checkpoint path and
      its `config.yaml`, `margin`, `crop_size`, and the dataset name. Loading a cache whose
      detector does not match the configured one **raises**, it does not warn. A cache built by
      a different detector that loads silently is the documented "silence looked like success"
      failure.
- [ ] **Mirror pool eviction:** when a frame leaves the pool, its records are deleted.
- [ ] **Store pre-blackout:** each record is the native-resolution RGB patch (mask bbox grown by
      `margin` px) plus its boolean mask. Blackout, `mask_only`, `mask_dilate`, `mask_close`,
      letterbox to `crop_size`, rotation and photometric all happen at **load** time.
- [ ] **Dilation headroom:** `margin` >= the largest `mask_dilate` to be swept. Loading with
      `mask_dilate > margin` **raises** rather than silently clipping, which would destroy the
      one sweep that matters.
- [ ] **Patches only for crops that will be trained on.** Write the pixel patch only for
      detections passing `best_iou >= 0.5` AND `runner_up_iou < 0.2` (`spec_cropcls.md:54` —
      *"If the second highest overlap is over 0.2, drop this sample from classifying."*).
      Nothing consumes a dropped patch: there is no reject class (`spec_cropcls.md:55`) and the
      step-4 contact sheets render live through `CropPipeline`, not the cache.
- [ ] **A metadata row for every detection**, kept or dropped: `best_iou`, `runner_up_iou`,
      `verdict`, `score`. Rows are bytes, not kilobytes. They preserve the assignment-rate
      monitor (98.2% measured, which cross-checks the detector's 2.1% class-agnostic FN,
      `spec_cropcls.md:87`) and record *why* a detection has no patch.
- [ ] **Thresholds tighten at load time, loosen only by re-extraction.** Raising `iou_floor`
      above 0.5 or lowering `runner_up_ceiling` below 0.2 filters the cache with no
      re-extraction. Going the other way needs a re-extraction, and must **raise** rather than
      silently return a short dataset. This is a deliberate narrowing of
      `spec_cropcls.md:141`, which wanted both directions load-time: at a 1.8% drop rate the
      patches that buys are ~2 crops in 110, and storing data with no consumer invites someone
      to train on it or report a metric over it by accident. Re-extraction is one pass.
- [ ] **Static batching:** a `DataLoader` with `shuffle=True`, `drop_last=True`, and a fixed
      batch size counted in **crops**. The existing `ims_per_batch` counts **frames**
      (`cropcls/config.py:105`) and keeps that meaning; the new key is named distinctly so the
      two units cannot be confused.
- [ ] **No change to trunk BatchNorm:** `freeze_trunk` stays `False` and `CropClassifier.train`
      is untouched. "batch normalization like in a static training" is satisfied by giving BN a
      constant-size batch of globally shuffled, uncorrelated crops — not by reconfiguring it.
- [ ] **One rotation per draw**, sampled from `cfg.rotations` — never all four of an instance in
      one batch, which is what correlates BN statistics today.
- [ ] **Epochs are not introduced.** The trainer is iteration-keyed
      (`train_cropcls.py:276`, with `max_iter`, `warmup_iters`, `eval_period` and the OneCycle
      schedule all in iterations). The loader is consumed as an infinite reshuffled stream, as
      `iter_batches` already is.
- [ ] **Bit-identical to the live path:** for the same frame, instance and mask settings, a crop
      built from the cache equals the crop the live pipeline builds.
- [ ] **Viewable sample:** finished blacked-out letterboxed crops are written for the first
      `viz_frames` frames only — a spot-check sheet, explicitly not the training source.
- [ ] **The live path stays:** `CropPipeline.iter_batches` is not deleted. It is the comparison
      arm.

## 4. Done when

Each item measurable, and naming the metric it is measured by.

- [ ] **Static training matches live training.** `classaware_f1` from the baseline's own
      `HungarianInstanceEvaluator` (`score_mode=detector`), against the live arm
      **r2 tight = 0.8758** (`cropcls-results.md:126`), within the measured endpoint seed
      variance of **0.82 points** (`cropcls-results.md:166-169`), reported with run id and
      iteration. Both arms apply the same IoU rule and the crops are bit-identical, so the crop
      population is **identical** and only the batching differs — any difference is therefore
      attributable to batching, which is the point of the test. Crop top-1 is still not the
      metric: it has misled in both directions (`cropcls-results.md:149-156`).
      **Verify:** reported in `cropcls-results.md` with run id, iteration, machine and load.
- [ ] **Batching is genuinely static.** Every batch has the configured crop count; batch
      composition changes between passes; a given instance's rotation varies across draws.
      **Verify:** `tests/test_cropcls_static_batching.py` — assert constant batch size over
      N batches; assert the multiset of crop ids differs between two passes; assert one
      instance receives >= 2 distinct rotations over N draws (fails for a sampler that pins a
      rotation, which the per-epoch formulation could not detect).
- [ ] **Mask handling and thresholds are load-time.** Changing `iou_floor`,
      `runner_up_ceiling`, `blackout`, `mask_only` or `mask_dilate` changes the training set or
      the crop with **no** re-extraction.
      **Verify:** `tests/test_cropcls_cache.py` — a loader at `iou_floor=0.7` over a cache
      extracted at `0.5` yields fewer crops than one at `0.5`; a loader at `iou_floor=0.3`
      **raises** rather than silently returning the 0.5 population
      (`test_loosening_a_threshold_raises_not_silently_short`); a crop at `mask_dilate=8`
      differs from the same crop at `0`; `mask_dilate > margin` raises
      (`test_mask_dilate_above_margin_raises_not_clips`).
- [ ] **The cache cannot be silently wrong.** A cache built by a different detector is refused;
      an evicted frame's records are gone; no record is degenerate.
      **Verify:** `tests/test_cropcls_cache.py` —
      `test_mismatched_detector_checkpoint_raises_not_warns`;
      `test_evicted_frame_records_are_deleted`;
      `test_all_black_patch_is_rejected_not_stored` and the empty-mask / NaN cases, in the house
      style of `test_empty_mask_is_rejected_not_nan` (`cropcls/README.md:30`);
      manifest row count equals files on disk.
- [ ] **Cost and size, with provenance.** Extraction wall-clock naming the machine **and its
      load** (629 vs 125 ms/frame contended vs idle, `cropcls/README.md:58-60`), frame count,
      crop count, and cache size in MB measured on disk. **Budget: <= 5 GB steady-state** for a
      1024-frame pool; exceeding it is a failure, not a note, given the ~60 GB `/home` quota.
      **Verify:** reported in `cropcls-results.md`; `du -sh` figure quoted.

## 5. Out of scope (explicit)

- `maskdino/` is not modified.
- The detector checkpoint and its dumped `config.yaml` are not touched.
- `CropPipeline.iter_batches` (the live path) is not deleted or disabled.
- The eval path is unchanged: `--hungarian` and `cropcls/export.py` keep running the live
  detector at inference. Only *training* reads the cache.
- Trunk BatchNorm configuration is unchanged (`freeze_trunk`, `CropClassifier.train`).
- Photometric augmentation is not dropped — storing pre-blackout is what keeps it runnable
  pre-blackout (`cropcls/README.md:27`).
- No scene_generator renders. "like we did with the segmenter" is read as *static dataset*, not
  *synthetic data*; the second message pins it to "the current train images".
- No epoch semantics added to the trainer.
- **Training on the dropped detections.** The cache stores them so the thresholds are
  load-time, but `spec_cropcls.md:54` says to drop contaminated samples and that stands.
  Whether the classifier would benefit from the crowded/occluded cases it is *evaluated* on by
  `classaware_f1` is a real question and a separate arm — not this change.
- Phase 2 / set B.

## Files

- `cropcls/cache.py` (new) — extraction, manifest, provenance check, eviction, loading.
- `tests/test_cropcls_cache.py` (new), `tests/test_cropcls_static_batching.py` (new).
- `cropcls/config.py`, `train_cropcls.py` — new keys and the static training path.

<!-- What the reviewers check:
  - §1 unticked and unedited, always.
  - Every §4 item has a **Verify:** clause naming a real file and a real assertion.
    A clause that cannot fail is a finding.
  - Rank by classaware_f1 (score_mode=detector), never crop top-1.
  - Every number names its iteration; every timing names the machine and its load. -->
