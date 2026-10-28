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
(`_POOL_VIRTUAL_SIZE` 1024), continuously replenished by sgdata. So extraction is a separate
**pre-pass** that each frame enters once per pool lifetime, and a training run reads a frozen
snapshot of the result; eviction is maintenance between runs, or the cache leaks retired scenes
and grows ~2.8 GB/day (`spec_cropcls.md:136`).

Store the native-resolution patch plus its mask **pre-blackout**, not the finished letterbox:
letterbox scale is set by the crop box, so baking it would fix `mask_dilate` — the largest
effect measured, **+4.2** at 16 px (`cropcls-results.md:130`). Rotations are not stored; on a
square canvas they are lossless permutations, so copies add no information.

## 3. Requirements

Literal. Every threshold carries its number **and its direction**.

- [ ] **Extraction is a separate pre-pass** — never inside the training loop and never inside
      a `DataLoader` worker. Workers fork (`live_pool_dataset.py:15-16`), so extracting there
      would mean a CUDA detector per worker and concurrent manifest writers. Training reads a
      finished cache, performs **zero** detector forwards, and a cache miss **raises** rather
      than falling back to a live forward. This is the ask read literally: *produce* train
      images, *then* feed training from this dataset.
- [ ] **One forward per frame, ever, per cache:** the pre-pass skips any frame whose records
      already exist.
- [ ] **Enumerate distinct frames, not virtual indices.** `LivePoolDataset.__len__` returns
      `virtual_size` (1024), and `__getitem__` maps `files[idx % len(files)]`
      (`live_pool_dataset.py:101-102,136`). The two coincide here — the pool holds exactly 1024
      frames — but if the count ever differs, indexing 0..1023 would either re-extract the same
      frame many times or leave frames unreachable. Extraction therefore iterates the
      **distinct `file_name` set** and keys on it.
- [ ] **Cache keyed by frame filename**, as `spec_cropcls.md:135` already required.
- [ ] **Provenance, enforced on load:** the manifest records the detector checkpoint path and
      its `config.yaml`, `margin`, `crop_size`, the dataset name, the full `thing_classes`
      list, and — critically — the **resolved realpath of the pool directory** plus a digest of
      its sorted frame basenames and the frame count. `training/images` is a per-host symlink,
      so the dataset *name* does not identify the data: this host offers two different 1024-frame
      renders of `train_phase1_pool_set_a_mm` (`images_dgx-a100` and
      `images_lambdablade-6000`, frame-name digests `abe2ae25810c` vs `d0cd49821f66`). No
      existing run records which it used. Loading a cache whose
      detector does not match the configured one **raises**, it does not warn. A cache built by
      a different detector that loads silently is the documented "silence looked like success"
      failure.
- [ ] **No eviction handling.** The pool is **static for the duration of a classifier run** —
      no frames are added or evicted (user, 2026-10-27). So there is nothing to mirror: extract
      once, then train. If that ever changes, re-extract; do **not** add mid-run eviction, which
      would leave drawn indices pointing at deleted records (a sampler snapshots
      `len(dataset)` when its iterator is created).
- [ ] **Store pre-blackout:** each record is the native-resolution RGB patch (mask bbox grown by
      `margin` px) plus its boolean mask. Blackout, `mask_only`, `mask_dilate`, `mask_close`,
      letterbox to `crop_size`, rotation and photometric all happen at **load** time.
- [ ] **Dilation headroom:** `margin` >= the largest `mask_dilate` to be swept. The check reads
      the **manifest's** recorded margin — the one the patches were actually grown by — not the
      live config's. A cache extracted at `margin=4`, loaded under a config claiming
      `margin=32` with `mask_dilate=16`, must raise; trusting the config would clip silently
      and destroy the +4.2 sweep (`cropcls-results.md:130`).
- [ ] **Patches only for crops that will be trained on.** Write the pixel patch only for
      detections passing `best_iou >= 0.5` AND `runner_up_iou < 0.2` (`spec_cropcls.md:54` —
      *"If the second highest overlap is over 0.2, drop this sample from classifying."*).
      Nothing consumes a dropped patch: there is no reject class (`spec_cropcls.md:55`) and the
      step-4 contact sheets render live through `CropPipeline`, not the cache.
- [ ] **The class is written as a name, not an index.** Each row carries `class_name` (the
      `thing_classes` string) as the **authoritative** label, and the manifest records the full
      `thing_classes` list it was extracted against. The integer `label` is *derived at load
      time* by indexing that list — never read back from disk as an int. A bare int is only
      meaningful relative to an ordering, so a reordered or different `thing_classes` would
      silently remap every cached crop to the wrong instrument. This follows the repo's
      existing rule for the class count: derived from `train_dataset`'s `thing_classes`, never
      stated (`cropcls/config.py:34-35`).
- [ ] **Loading against a different label space raises.** If the configured dataset's
      `thing_classes` differs from the manifest's — reordered, renamed, or a different length —
      the load fails loudly rather than remapping.
- [ ] **A metadata row for every detection**, kept or dropped. Fields are `CropRecord`'s
      (`cropcls/data.py:98-111`) minus `rotation`, which is sampled at load time:
      `frame_index`, `file_name`, `pred_index`, `label`, `class_name`, `best_iou`,
      `runner_up_iou`, `verdict`, `score`. Rows are bytes, not kilobytes. They preserve the
      assignment-rate monitor (98.2% measured, which cross-checks the detector's 2.1%
      class-agnostic FN, `spec_cropcls.md:87`) and record *why* a detection has no patch.
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

- [ ] **Static training matches live training.** The anchor is a **fresh live arm run on
      exactly the frames the cache holds** — not the published r2 = 0.8758, which ran on
      `dbe-cia-sl23-01` with an unrecorded render set. Metric: `classaware_f1` from the
      baseline's own `HungarianInstanceEvaluator` (`score_mode=detector`), within the measured
      endpoint seed variance of **0.82 points** (`cropcls-results.md:166-169`). Beating that
      band is a pass with a note, not a failure. Quote r2 alongside as a sanity reference only.
      The data is `train_seta_mm` resolving through `training/images` ->
      `images_lambdablade-6000` (1024 frames, frame-name digest `d0cd49821f66`).
      **This is indicative, not a clean attribution.** Three things differ from r2 and the
      write-up must say so: (a) batching — crops per step vs frames per step; (b) **crops seen
      per step** — the live arm emits 4 rotations per instance per forward
      (`spec_cropcls.md:234`) while the static arm emits one per draw, so equal `max_iter` is
      *not* equal training volume. Compare at matched **total crops seen**, not matched
      `max_iter`. (The pool is static during a run, so there is no snapshot-vs-replenished
      difference, and the fresh live arm removes the render-set one.) Crop top-1 is still not
      the metric: it misled in both directions (`cropcls-results.md:149-156`).
      **Verify:** reported in `cropcls-results.md` with run id, iteration, machine, load, and
      crops-seen-per-step plus total crops seen for **both** arms.
- [ ] **Batching is genuinely static.** Every batch has the configured crop count; successive
      passes draw in a different order; a given instance's rotation varies across draws.
      **Verify:** `tests/test_cropcls_static_batching.py` — constant batch size over N batches;
      the **ordered sequence** of crop ids differs between two passes (not the multiset: a full
      `shuffle=True` pass yields every crop exactly once, so a multiset assertion would fail on
      a *correct* implementation); one instance receives >= 2 distinct rotations over N draws.
- [ ] **The cache is used, not bypassed.** Training runs zero detector forwards against a warm
      cache, and a miss raises.
      **Verify:** `tests/test_cropcls_cache.py` —
      `test_training_step_runs_no_detector_forward_on_warm_cache` (patch the detector, assert
      call count 0 over N steps); `test_cache_miss_raises_not_silently_forwards`. Without this
      every other item stays green while the forward quietly still runs — the whole point of
      the change, untested.
- [ ] **A cached crop equals the live crop.** Bit-identical for the same frame, instance and
      mask settings. The comparison above rests on this.
      **Verify:** `tests/test_cropcls_cache.py` —
      `test_cached_crop_is_bit_identical_to_live_crop` over >= 5 frames.
- [ ] **Mask handling is load-time; thresholds tighten only.** `blackout`, `mask_only` and
      `mask_dilate` (up to the manifest's `margin`) change the crop with no re-extraction, and
      `iou_floor` / `runner_up_ceiling` can be **tightened** with none. Loosening either raises.
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
      `test_reordered_thing_classes_raises_not_remaps` — the label space is checked by name,
      not by length, so a permutation of the same 8 names is caught;
      `test_evicted_frame_records_are_deleted`;
      `test_all_black_patch_is_rejected_not_stored` and the empty-mask / NaN cases, in the house
      style of `test_empty_mask_is_rejected_not_nan` (`cropcls/README.md:30`);
      manifest row count equals files on disk.
- [ ] **Cost and size, with provenance.** Extraction wall-clock naming the machine **and its
      load** (629 vs 125 ms/frame contended vs idle, `cropcls/README.md:58-60`), frame count,
      crop count, and cache size in MB measured on disk. **Budget: <= 500 MB** for the
      1024-frame pool. Estimate to beat: ~5,500 kept crops at ~40 KB per native patch is
      ~0.2 GB, so 500 MB is a real ceiling rather than an unfalsifiable one. This matters:
      the quota is **61440 MB with 57680 MB used — 3.7 GB free** (measured 2026-10-27, `quota
      -s`), and the experiment needs two training runs on top of the cache. Check headroom
      before extracting, not after.
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
- **Training on the dropped detections.** The cache keeps a metadata row for them but **no
  patch**, and `spec_cropcls.md:54` says to drop contaminated samples, which stands.
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
