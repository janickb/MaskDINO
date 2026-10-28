# Results: does the model confuse forcep03 / forcep04 on *train* data too?

Executed 2026-10-02 against **`runs/20260929114041_reclassify_unfreeze_encoder_decoder_v3/model_final.pth`**
(the plan this executed targeted the
`20261001204312_…_long` checkpoint at iter 49999). Branch `34-enhance-augmentation-to-validation-style-piles`.

## Answer

**Largely yes — the train-clean result is weaker evidence than it first appears, and this
section was overstated in the first version of this document.** See
[Correction](#correction-what-the-400-instances-actually-are) below.

What survives: **density and occlusion do not cause the 03/04 confusion** (a controlled
within-train comparison). What does *not* survive: the claim that the model "separates 03 from
04 cleanly". It separates two memorized crops cleanly, which is a much smaller claim.

The practical recommendation is unchanged and in fact reinforced: **more/different forcep03
source data** — specifically source data that varies the *discriminating* feature, not more
augmentation of the two views already present.

## The four matrices

forcep03_mm GT row, row-normalized, identical weights and thresholds throughout
(`SCORE_THRESH 0.5`, `IOU_THRESH 0.5`, `MIN_VISIBILITY 0.5`, `EXCLUDE_TRUNCATED_INSTANCES true`):

| matrix | GT n | → forcep03 | → forcep04 | false neg |
|---|---|---|---|---|
| raw train | 20 | **1.000** | 0.000 | 0.000 |
| composited train, uniform ← *the trained condition* | 400 | **0.900** | 0.013 | 0.088 |
| composited train, pile (`CLUSTER_PROB 1.0`) | 414 | **0.773** | 0.029 | 0.198 |
| matched val (`val_setab_mm`, stride 1) | 269 | **0.257** | 0.424 | 0.320 |

And the forcep04 row, which is the control:

| matrix | GT n | → forcep03 | → forcep04 | false neg |
|---|---|---|---|---|
| composited train, uniform | 409 | 0.015 | 0.900 | 0.086 |
| composited train, pile | 400 | 0.023 | 0.758 | 0.220 |
| matched val | 272 | **0.000** | 0.835 | 0.162 |

### What this rules out — and what it does not

- **Not density or occlusion.** *(survives)* The pile condition costs recall
  (FN 0.088 → 0.198) but **symmetrically** — forcep04 degrades the same way (0.086 → 0.220) —
  while cross-class leakage barely moves (0.013 → 0.029). Both arms paste the *same* source
  crops, so memorization cancels out and the comparison is valid. Clustering reproduces the
  *recall* cost of a pile but **not the asymmetric 03→04 collapse**. The pile direction of this
  branch will not close this gap on its own.
  *Narrower than first stated:* this isolates occlusion applied to **familiar** appearances. It
  cannot rule out an interaction between occlusion and **novel** appearance, which is what val
  actually presents.
- **~~Not a data/separability problem.~~** *(retracted — see Correction below)*

<a name="correction-what-the-400-instances-actually-are"></a>
### Correction: what the 400 composited instances actually are

The first version argued that compositing lifted forcep03 from 20 to 400 GT instances, giving
the power to detect a val-sized 42 % leak. **That reasoning is wrong.** The 400 instances are
re-pastes of the same 20 source frames, and those 20 are not 20 appearances:

```
train_phase2_set_ab_mm/forcep03_mm_{A,B}_{000..324}deg.hdf5   = 2 sides x 10 in-plane rotations
```

In-plane rotation carries almost no new information for a model already trained with 90°
rotation augmentation, so the **effective number of distinct forcep03 appearances in training is
2**, not 20 and certainly not 400. Compositing multiplies *placements and contexts*, not
*appearances*. The eval is therefore measuring train-set accuracy on memorized crops, and a
fine-tuned model scoring ~0.90 on those is close to the least surprising possible outcome.

**The two tools are also near-identical objects**, which is the deeper issue:

| | bbox (px) | mask area (px) |
|---|---|---|
| forcep03 side A / B | 192x90 | 1383 / 1384 |
| forcep04 side A / B | 195x90 | 1415 / 1415 |

A 1.5 % difference in length and 2.3 % in area. Both are ring-handled, ratcheted, curved-tip
forceps; the only visible discriminator is a subtle jaw/tip difference carried by a handful of
pixels out of ~1400 (see `scratch/forcep0304_sides.png`, no longer present). Sides A
and B are vertical mirrors of each other — genuinely distinct (a mirror is not reachable by
rotation), but only two of them.

**Resolution is *not* the train/val gap.** Val instances sit at the same physical pixel scale as
the training renders, by design (`sgdata.real_augment` sizes its crops to match):

| | n | p10 | median | p90 |
|---|---|---|---|---|
| val forcep03 | 277 | 1144 | **1416** | 1631 |
| val forcep04 | 283 | 1210 | **1471** | 1668 |
| train forcep03 / forcep04 | 20 each | — | **1383 / 1415** | — |

So the gap is pose, articulation, lighting and real-world finish — not scale.

### What the val failure actually looks like

- **Strictly one-way.** forcep04 → forcep03 is **0 of 272**. Only 03 collapses onto 04.
- **Confined to the pair.** forcep03's val predictions are *only ever* forcep03, forcep04, or
  missed — never any of the other 14 classes.
- **Dominant.** forcep03 → forcep04 is **114 of all 223** misclassifications in val (51 %).
- **Partly underconfidence, not only confusion.** Re-scoring the cached val predictions at lower
  gates recovers a large share:

  | SCORE_THRESH | → forcep03 | → forcep04 | false neg |
  |---|---|---|---|
  | 0.5 | 0.249 | 0.412 | 0.336 |
  | 0.3 | **0.430** | 0.444 | 0.119 |
  | 0.2 | **0.534** | 0.347 | 0.108 |

  *Caveat:* `tools/rerun_hungarian_from_cache.py` does not apply the truncated-instance filter
  (GT 277 vs 269), so these are not directly comparable to the table above — but all three rows
  share one basis, so the trend is sound. Roughly a third of forcep03's apparent loss at 0.5 is
  **correct-but-low-confidence** predictions, not misclassification.

## Suggested next steps

1. **The decisive experiment this study could not run: a held-out-appearance test.** Retrain with
   forcep03/forcep04 **side B withheld**, then evaluate on side B. That is the only clean way to
   ask whether 03 and 04 are separable *in a way that transfers*, rather than memorized. Current
   evidence cannot distinguish the two.
2. **Get forcep03 source renders that vary the discriminating feature** — jaw/tip geometry under
   many viewing angles, not more in-plane rotations of the same two resting poses. Two mirrored
   views of a near-identical twin object is the binding constraint.
3. **Check whether the pair is worth separating at all.** Given 1.5 % length and 2.3 % area
   difference, consider whether merging forcep03/forcep04 into one class, or accepting a
   coarse-then-fine two-stage head, serves the downstream task better than forcing a distinction
   the imagery may not support at this scale.
4. **Interim mitigation:** `SCORE_THRESH` 0.3 nearly doubles correct forcep03 (0.249 → 0.430) at
   a modest precision cost.
5. Do **not** expect pile clustering alone to fix this; it addresses recall under occlusion,
   a real but separate deficiency.

## What was run

```bash
# step 0 - sanitize (only MODEL.MaskDINO.RECLASSIFY_FINETUNE.ENCODER_LR_MULTIPLIER dropped)
./.venv/bin/python scratch/sanitize_config.py \
  runs/20260929114041_…/eval_fullval_excl_truncated/config.yaml scratch/v3final_config.yaml

# step 1 - raw train + matched val, one invocation
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python train_net.py --eval-only --num-gpus 1 \
  --config-file scratch/v3final_config.yaml \
  MODEL.WEIGHTS runs/20260929114041_…/model_final.pth \
  DATASETS.TEST '("reclass_setab_mm","val_setab_mm")' DATASETS.TEST_SAMPLE_STRIDE 1 \
  OUTPUT_DIR runs/_eval_trainset_confusion/v3final_raw

# step 2 - composited, both conditions
CUDA_VISIBLE_DEVICES=2 ./.venv/bin/python tools/eval_composited_confusion.py \
  --config-file scratch/v3final_config.yaml --weights runs/20260929114041_…/model_final.pth \
  --dataset reclass_setab_mm --repeats 3 --seed 0 \
  --output-dir runs/_eval_trainset_confusion/v3final_composited_uniform \
  INPUT.COPY_PASTE.CLUSTER_PROB 0.0
# …and the same with CLUSTER_PROB 1.0 CLUSTER_RADIUS_FRACTION 0.25 -> …_composited_pile
```

### Deviations from the plan

- **Base config.** Sanitized `eval_fullval_excl_truncated/config.yaml` rather than the run's
  top-level `config.yaml`: it already carries `TEST_SAMPLE_STRIDE 1`,
  `EXCLUDE_TRUNCATED_INSTANCES true`, and weights at `model_final.pth`. Confirmed faithful — the
  step-1 val matrix reproduces that eval's published numbers **exactly** (69/114/86 of 269).
- **`tools/preflight_copy_paste_density.py` does not exist on this branch.** Replaced that
  verification with `scratch/density_crosscheck.py` (no longer present), which runs the
  *real* `Hdf5CocoInstanceDatasetMapper` with copy-paste on. Valid here because
  `INPUT.MIN_SCALE == MAX_SCALE == 1.0` and rotations are 90° multiples, so the mapper's geometry
  is count-preserving. **Training mapper 6.08 instances/frame vs the tool's 6.10** — the tool
  composites like training.
- **`tools/rerun_hungarian_from_cache.py` has no `sys.path.insert`**, unlike every other tool in
  `tools/`, so it dies with `ModuleNotFoundError: No module named 'maskdino'` when run from the
  repo root. Worked around with `PYTHONPATH=.`; worth a one-line fix.
- The new tool deliberately skips train-time brightness/contrast jitter and the 90° rotation, so
  the composited-train eval is marginally *easier* than true training conditions. Conservative for
  the "is train clean?" question — it cannot manufacture the clean result.

### Verification (all pass)

- Row sums of both composited confusion CSVs match the per-class GT counts the tool prints, all
  16 classes, both conditions.
- `instance_matching_per_image.csv` has exactly 1140 rows (380 frames × 3 repeats) in both.
- Sample PNGs: scattered under (a), overlapping pile with reduced `visibility_fraction` under (b).
- `tests/test_copy_paste.py` (25), `test_hungarian_instance_evaluation.py` +
  `test_truncated_prediction_filter.py` (29) all pass. Added
  `test_returned_annotations_match_evaluator_gt_contract` to pin the GT contract the new tool
  depends on. (`tests/test_summarize_reclass.py` still fails at collection — pre-existing,
  Gotcha 4.)

### Artifacts

```
runs/_eval_trainset_confusion/
    v3final_raw/inference/{reclass_setab_mm,val_setab_mm}/   # step 1
    v3final_composited_uniform/{…, samples/}                 # step 2a
    v3final_composited_pile/{…, samples/}                    # step 2b
```

## Gotchas worth keeping

Carried over from the planning document this study executed (`trainset_confusion_check_forcep03.md`,
removed 2026-10-20 once superseded by these results). Both are still live hazards:

1. **`--eval-only` does not timestamp `OUTPUT_DIR`.** [train_net.py:627-645](../../train_net.py#L627-L645)
   skips the stamping under `--eval-only`, so without an explicit `OUTPUT_DIR` override the eval
   writes straight into the training run's own directory and **overwrites its `inference/`
   artifacts** — confusion matrices included, possibly while that run is still training. Always
   pass `OUTPUT_DIR runs/_eval_.../<something>`.
2. **Never point `DATASETS.TEST` at `train_seta_mm`** (or any other `LivePoolDataset`). It is not a
   list; `apply_truncated_instance_filter` iterates it and `LivePoolDataset.__getitem__` does
   `files[idx % len(files)]`, so it never raises `IndexError` → **infinite loop re-reading HDF5**.
   With `TEST_SAMPLE_STRIDE > 1` it instead dies in `slice % int`. Only the static sets
   (`reclass_setab_mm`, `reclass_set_home`, `val_*`) are safe as test sets.

Also still true: [tools/rerun_hungarian_from_cache.py](../../tools/rerun_hungarian_from_cache.py)
has no `sys.path.insert`, unlike every other tool in `tools/`, so it needs `PYTHONPATH=.` when run
from the repo root.
