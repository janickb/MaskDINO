# Experiment log

One line per study, newest first. Every one of them is chasing the same thing: the
`forcep03_mm` → `forcep04_mm` collapse on `val_setab_mm`, and what (data, geometry,
appearance) actually moves it.

Run artifacts live in `runs/<timestamp>_<arm>/` and are **gitignored** — only these write-ups
are tracked, so a number that isn't in a document here is not recoverable from this repo.

| when | study | asked | verdict | status |
|---|---|---|---|---|
| 2026-10-05 → 10-09 | [appearance_aug_sweep_phase2.md](appearance_aug_sweep_phase2.md) | which augmentation axis fixes forcep03? | **per-instance photometric jitter is the whole effect** (+0.025 class-aware f1 = +5.3σ); frame-level realism and perspective are null or negative | result of record |
| *not yet run* | [augmentation_experiments.md](augmentation_experiments.md) | control / low / medium / object-jitter-only dose wave on a true no-op baseline | — | **pending**, configs written |
| 2026-10-02 | [trainset_confusion_check_forcep03_results.md](trainset_confusion_check_forcep03_results.md) | is the 03/04 collapse data/separability or appearance coverage? | **near-identical twins with effectively 2 distinct training appearances**; density/occlusion and resolution ruled out | live |
| 2026-10-02/03 | [archive/scale_jitter_ab_phase2.md](archive/scale_jitter_ab_phase2.md) | does input scale jitter fix it? | scale coverage is not the explanation — **but arm B trained on corrupted mask targets** (`seg_pad_value=255`) | superseded |

Last training run of any kind: **2026-10-09 17:49**
(`runs/20261009160904_abl_jitter_persp12_seed42`). Nothing has run since.

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

## Shared gotchas

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
- `runs/aug_config_comparison.md` (gitignored, lives with the runs) — every swept
  `INPUT.COPY_PASTE` / `INPUT.PHOTOMETRIC` key side by side across all 21 sweep arms.
