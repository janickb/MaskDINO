# Experiment log

One line per study, newest first. Every one of them is chasing the same thing: the
`forcep03_mm` → `forcep04_mm` collapse on `val_setab_mm`, and what (data, geometry,
appearance) actually moves it.

Run artifacts live in `runs/<timestamp>_<arm>/` and are **gitignored** — only these write-ups
are tracked, so a number that isn't in a document here is not recoverable from this repo.

| when | study | asked | verdict | status |
|---|---|---|---|---|
| 2026-10-26 → 10-28 | [scaling_experiment.md](scaling_experiment.md) | does more resolution (pixels per mm, via camera focal length 0.67x / 1.0x / 1.5x) help forcep03/04? | **helps find and segment them** (best-threshold 03+04 F1 0.803 → 0.853, all from fewer misses; AP75 0.3/0.4 → 45/64) **but not tell them apart**: the scores separate 03 from 04 at every level (AUC ≈ 0.98), the one-way 03→04 confusion is a **bias toward 04** that resolution doesn't touch. Replicated on a val set with every tool in frame (v2): all-tools F1 +0.019 for 1.5x, 03→04 bias unchanged | result of record |
| 2026-10-05 → 10-09 | [appearance_aug_sweep_phase2.md](appearance_aug_sweep_phase2.md) | which augmentation axis fixes forcep03? | **per-instance photometric jitter is the whole effect** (+0.025 class-aware f1 = +5.3σ); frame-level realism and perspective are null or negative | result of record |
| *not yet run* | [augmentation_experiments.md](augmentation_experiments.md) | control / low / medium / object-jitter-only dose wave on a true no-op baseline | — | **pending**, configs written |
| 2026-10-02 | [trainset_confusion_check_forcep03_results.md](trainset_confusion_check_forcep03_results.md) | is the 03/04 collapse data/separability or appearance coverage? | **near-identical twins with effectively 2 distinct training appearances**; density/occlusion and resolution ruled out | live |
| 2026-10-02/03 | [archive/scale_jitter_ab_phase2.md](archive/scale_jitter_ab_phase2.md) | does input scale jitter fix it? | scale coverage is not the explanation — **but arm B trained on corrupted mask targets** (`seg_pad_value=255`) | superseded |

Last training run of any kind: **2026-10-27 02:27**
(`runs/20261027011700_focal_1.5x_seed2`, the last arm of the focal sweep).

## Open items

Carried out of the documents above, none of them closed:

1. **`forcep11_mm` recall regresses under jitter** (0.70–0.85 → 0.55–0.68, cleanly separated
   across all 17 arms). Darkening pasted instances makes an already-dim thin tool harder to
   *find* — a detection loss, not confusion. Net arithmetic still favours jitter, but this
   deserves one targeted run (per-class jitter centres, or a brightness floor at 0.85) before
   `abl_jitteronly` becomes the default.
2. **Wave 0 never ran.** The scale A/B rerun on top of the `FixedSizeCrop seg_pad_value` fix was
   planned and skipped, so the archived doc's "keep scale jitter on" is still unconfirmed and
   **every arm in the appearance sweep is pinned at the `1.0/1.0` no-op**. All its numbers are
   without scale jitter.
3. **The decisive separability experiment never ran:** retrain with forcep03/forcep04 **side B
   withheld**, then evaluate on side B. Current evidence cannot distinguish "separable in a way
   that transfers" from "two memorized crops".
4. **The new dose wave is unrun, and the launcher can't reach it.**
   [configs/coco/instance-segmentation/augmentation/](../../configs/coco/instance-segmentation/augmentation/)
   holds its five arm configs, but
   [run_aug_sweep.sh](../../tools/augmentation/run_aug_sweep.sh) builds its config path as
   `configs/coco/instance-segmentation/..._aug_${ARM}.yaml` — the parent directory — so
   `run_aug_sweep.sh control` picks up the **old** control (default dose 0.7–1.3 / LIGHTING 0.9),
   not the new zero rung. Note also that this wave needs **its own noise floor** from repeated
   seeds: the 0.8783 ± 0.0046 floor does not apply to a true no-op control.
   `realistic_measured.yaml` is explicitly parked (placeholder values, not calibrated).
5. **Interim mitigation not applied:** `SCORE_THRESH` 0.3 nearly doubles correct forcep03
   (0.249 → 0.430) at modest precision cost.
6. **forcep03 → forcep04 confusion regressed with the new phase-1 base or the new renders**
   (0.028 → 0.131 at 1.0x, score 0.5, identical 03/04 finish in both). One run with the old
   base on the new 1.0x data separates the two; see
   [scaling_experiment.md](scaling_experiment.md), caveat 1.
7. **forcep03/04 masks cap at IoU ~0.68 at 1.0x** (AP75 ≈ 0). Before paying for camera
   resolution, test whether the mask path is the limit: evaluate the 1.0x data at input 1536.
8. **The 03→04 confusion is a score bias, not inseparability** (AUC ≈ 0.98 at every focal
   level; a fixed forcep03 offset lifts balanced accuracy 0.84 → 0.93 in-sample). Fit the
   offset on half the val drops, test on the other half; see
   [scaling_experiment.md](scaling_experiment.md), section 3. Related to item 5: the best
   score threshold for every focal level is ~0.40, not 0.5.

## Shared gotchas

- **Evaluation semantics changed on 2026-10-27: tools cut by the frame edge are now ignore
  regions.** `apply_truncated_instance_filter` marks them `iscrowd=1` instead of deleting
  them, COCO AP and the Hungarian eval both skip predictions that land on one, and the
  box-based prediction drop (`DropTruncatedPredictions`) is gone. Before, every detection of a
  cut tool counted as a false positive, unless its predicted box touched the edge. A correct
  detection of an *uncut* tool whose box touched the edge was silently dropped and counted as
  a miss. GT counts, matches and misses on kept objects are unchanged; false positives,
  precision, F1 and AP go up, and more so at higher zoom (cached predictions, seed 42:
  1.5x FP 70 → 18, AP +0.9; 1.0x FP 87 → 61, AP +0.26). **Precision, F1 and AP from earlier
  runs are not directly comparable with new ones** — re-run their eval
  (`EVAL_ONLY=1 run_focal_sweep.sh`, or `train_net.py --eval-only`).
- **Training logs contain NUL bytes**, so `grep` treats them as binary and silently matches
  nothing — pipe through `tr -d '\000'` first. Costs an hour every time it is rediscovered.
- **`IMS_PER_BATCH 2` per GPU hangs during eval on an 11 GB 2080 Ti.** Both ranks pin at
  ~10.5 GB, the parent exits 1 with no traceback and no CUDA OOM message, and the ranks must be
  killed by PID. 1 img/GPU peaks at 5.0 GB and completes cleanly, so the only safe route to a
  larger effective batch is more cards at 1 image each.
- Two more (`--eval-only` overwriting a live run's `inference/`, and `LivePoolDataset` as a test
  set looping forever) are at the end of
  [trainset_confusion_check_forcep03_results.md](trainset_confusion_check_forcep03_results.md).
- **`*.png` is gitignored repo-wide**, with an exception for `docs/**` only. Figures belong under
  `docs/`; a plot left anywhere else is silently untracked — which is how
  `scratch/forcep0304_sides.png`, the forcep03/04 comparison image, was lost.

## Tooling

- [tools/compare_runs.py](../../tools/compare_runs.py) — the three tables these write-ups are
  built from, over any set of run dirs (`--gt-class <name>` for a per-class confusion row).
- [tools/augmentation/run_aug_sweep.sh](../../tools/augmentation/run_aug_sweep.sh) — one arm per
  invocation; holds the shared protocol on the command line so no arm file can drift on anything
  but its own dose. See open item 4 before using it with the new configs.
- [tools/augmentation/dump_training_samples.py](../../tools/augmentation/dump_training_samples.py)
  — previews what an augmentation config actually does to training samples, into `debug_vis/`.
- [tools/scaling/launch_focal_grid.sh](../../tools/scaling/launch_focal_grid.sh) /
  [run_focal_sweep.sh](../../tools/scaling/run_focal_sweep.sh) — the focal-length sweep
  (levels x seeds queued per GPU; each run ends with a full-val eval into `<run>/eval_full/`).
- [tools/scaling/analyze_focal_sweep.sh](../../tools/scaling/analyze_focal_sweep.sh) —
  per-GT-instance replays of every focal run plus both analyses: paired miss / swap /
  correct across levels, and best-threshold F1 with 03-vs-04 separability. The replays
  ([replay_focal_instances.py](../../tools/scaling/replay_focal_instances.py), official
  matching; [sweep_focal_thresholds.py](../../tools/scaling/sweep_focal_thresholds.py), one
  label per mask, scores 0.10–0.90) work on any run that has an `eval_full/` directory
  (`EVAL_DIR` / `--eval-dir` for another one, e.g. `eval_full_v2/` from `VAL_TAG=v2`).
- `runs/aug_config_comparison.md` (gitignored, lives with the runs) — every swept
  `INPUT.COPY_PASTE` / `INPUT.PHOTOMETRIC` key side by side across all 21 sweep arms.
