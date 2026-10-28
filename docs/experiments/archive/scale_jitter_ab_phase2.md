# Does input scale jitter fix the forcep03/forcep04 collapse?

Phase-2 reclassification A/B, run 2026-10-02/03 on `dbe-cia-wl20-01`. Follow-up to
[trainset_confusion_check_forcep03_results.md](../trainset_confusion_check_forcep03_results.md), which asked whether
the forcep03 -> forcep04 collapse is a data/separability problem or a training-time appearance
coverage problem. This answers one half of it: **scale coverage is not the explanation.**

> **Superseded, 2026-10-05 — arm B trained on corrupted mask targets.**
> `FixedSizeCrop` was left at detectron2's default `seg_pad_value=255`, an *ignore*
> label that `BitMasks` casts to `True`. Whenever the sampled scale fell below 1.0
> (about half of arm B's samples under `0.8/1.2`, and never under arm A's `1.0/1.0`),
> every instance's `gt_mask` additionally claimed the whole padded band — e.g. a
> `forcep03` mask of 1295 px became 71750 px against an unchanged 20234 px box, and
> all 11 instances in one frame shared 220476 px of it. Fixed in
> `build_transform_gen` (`seg_pad_value=0`), pinned by
> `tests/test_transform_gen_padding.py`.
>
> The numbers below are left as recorded, but **the arm A vs arm B comparison is no
> longer clean**: arm B is handicapped relative to a fixed-pad rerun, so if anything
> it understates scale jitter. Wave 0 of the appearance-dose sweep re-runs both arms
> on top of the fix and supersedes this table.

## Setup

Two runs, identical except `INPUT.MIN_SCALE` / `INPUT.MAX_SCALE`:

| | config | scale | run dir |
|---|---|---|---|
| arm A | `..._scale_fixed.yaml` (removed) | 1.0 / 1.0 (no-op resize) | `runs/20261002225457_reclassify_enc_dec_sim_scale_fixed` |
| arm B | `..._scale_jitter.yaml` (removed) | 0.8 / 1.2 (LSJ) | `runs/20261003000355_reclassify_enc_dec_sim_scale_jitter` |

Both inherit
[maskdino_R50_surgical_tools_reclassify_unfreeze_encoder_decoder.yaml](../../../configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_unfreeze_encoder_decoder.yaml):
encoder + decoder + class head trainable (backbone frozen), `WEIGHTS` = the phase-1 set-A
checkpoint `runs/20260923203934_phase1_training_seta_v1-0/model_final.pth`, train
`reclass_setab_mm` -> eval `val_setab_mm` at `TEST_SAMPLE_STRIDE 4` (256 images, 1003 GT
instances), copy-paste on with `CLUSTER_PROB 1.0` and the class-balanced paste sampler,
`MIN_VISIBILITY 0.5`, `BASE_LR 1e-5` WarmupCosine, `MAX_ITER 6000`.

Launched sequentially on GPUs 1+2 by `scratch/run_scale_ab.sh`;
~51 min of training each (0.51 s/it) plus ~13 min of eval hooks.

## Headline (HungarianInstanceEvaluator, `val_setab_mm`)

| iter | arm | precision | recall | class-aware recall | f1 | mean IoU |
|---|---|---|---|---|---|---|
| 2000 | A no jitter | 0.984 | 0.180 | 0.171 | 0.305 | 0.810 |
| 2000 | B jitter | 0.992 | 0.239 | 0.228 | 0.386 | 0.825 |
| 4000 | A no jitter | 0.984 | 0.560 | 0.517 | 0.714 | 0.839 |
| 4000 | B jitter | 0.983 | 0.632 | 0.592 | 0.769 | 0.840 |
| 6000 | A no jitter | 0.981 | 0.630 | 0.585 | 0.768 | 0.844 |
| 6000 | **B jitter** | 0.984 | **0.682** | **0.641** | **0.806** | 0.844 |

Jitter wins at identical precision and identical mask quality, and gets there sooner: arm B's
iter-4000 recall (0.632) already equals arm A's final (0.630). 6000 iters was enough - both
curves are flattening by the last tick.

## The forcep03 row

Row-normalized confusion at iter 6000 (`inference/hungarian_confusion/iter_0006000/`):

| GT | arm | correct | -> forcep04 | false negative | **correct / detected** |
|---|---|---|---|---|---|
| `forcep03_mm` | A no jitter | 0.100 | 0.350 | 0.550 | **22%** |
| `forcep03_mm` | B jitter | 0.217 | 0.400 | 0.383 | **35%** |
| `forcep04_mm` | A no jitter | 0.757 | - | 0.243 | 76% |
| `forcep04_mm` | B jitter | 0.814 | - | 0.186 | 81% |

Read the last column, not the raw `-> forcep04` fraction: jitter raises forcep03's *detection*
rate (FN 0.55 -> 0.38), so more instances land in both the correct and the wrong column. Among
instances actually detected, jitter classifies 35% correctly vs 22% - better on both halves. The
raw misclassification fraction rising 0.35 -> 0.40 is a side effect of detecting more, not a
regression.

**But the pair is still not separated.** forcep03 ends at 0.217 correct against 0.400 read as
forcep04, while its three sibling forceps sit at 0.75-0.85. Jitter narrows the gap; it does not
close it.

## Per-class recall delta (B - A, iter 6000)

| class | A | B | delta |
|---|---|---|---|
| `clamp11_mm` | 0.024 | 0.333 | **+0.310** |
| `scissor11_mm` | 0.369 | 0.536 | +0.167 |
| `scalpel01_mm` | 0.714 | 0.841 | +0.127 |
| `forcep03_mm` | 0.100 | 0.217 | +0.117 |
| `hammer01_mm` | 0.800 | 0.906 | +0.106 |
| `scissor12_mm` | 0.375 | 0.464 | +0.089 |
| `hammer11_mm` | 0.570 | 0.628 | +0.058 |
| `forcep04_mm` | 0.757 | 0.814 | +0.057 |
| `tweezer11_mm` | 0.424 | 0.439 | +0.015 |
| `sharpspoon01_mm` | 0.961 | 0.961 | 0.000 |
| `scalpel11_mm` | 0.719 | 0.719 | 0.000 |
| `forcep02_mm` | 0.767 | 0.750 | -0.017 |
| `forcep11_mm` | 0.132 | 0.113 | -0.019 |
| `scarstick01_mm` | 0.905 | 0.881 | -0.024 |
| `forcep01_mm` | 0.849 | 0.822 | -0.027 |
| `adapter11_mm` | 0.764 | 0.673 | -0.091 |

10 classes up, 5 slightly down (all within what a single-seed difference plausibly covers, so
don't read the losses as real; `adapter11` is the only one worth re-checking). The big winners
are the classes that were *undetected*, not misclassified - `clamp11` goes from effectively
invisible (0.024) to 0.333.

Other confusions worth knowing: arm A has `tweezer11 -> scarstick01` at 0.106 and
`scissor11 -> scissor12` at 0.095; arm B trades those away but picks up a mild
`forcep01 <-> forcep02` mixup (0.055 / 0.100). False positives are negligible in both
(11-12 of ~690 predictions).

## Conclusion / next step

1. **Keep scale jitter on** for phase 2 - 0.8/1.2 is a free +0.05 recall / +0.04 f1 at equal
   precision, and it costs nothing in runtime.
2. **forcep03 is still the open problem.** It is not a scale-coverage issue, and its failure mode
   is now mostly "detected but called forcep04." With only 20 source renders each, the next
   lever is data (more/differently-posed forcep03 renders, or a hard-pair-aware sampler), not
   geometric augmentation.
3. `clamp11` / `scissor11` / `scissor12` / `forcep11` recall (0.11-0.54) is a localization
   problem, separate from the forcep03 story, and is the bigger contributor to overall recall
   capping at 0.68.

## Gotchas hit while running this

1. **`IMS_PER_BATCH 6` does not fit an 11 GB 2080 Ti in this condition.** `20260929114041` ran it
   at batch 6 / 14.1 GB per GPU on `sl19-01`. Measured here on 2 GPUs: 2 img/GPU peaks at 9.5 GB
   and then **hangs** once an eval pass is added - both ranks sit pinned at ~10.5 GB, the parent
   exits 1 with no traceback and no CUDA OOM message, and the ranks must be killed by PID.
   1 img/GPU peaks at 5.0 GB and completes eval (incl. the Hungarian evaluator) cleanly. Both arms
   therefore ran `IMS_PER_BATCH 2` with `BASE_LR` left at 1e-5 (not rescaled), so **absolute
   numbers here are not comparable to `20260929114041`** - only arm A vs arm B is.
2. **The phase-1 checkpoint's class head is skipped, by design.** It is 8-class set A; the reclass
   space is 16, so fvcore skips `class_embed.weight`/`.bias` and `label_enc.weight` on shape
   mismatch and all 16 rows start fresh - the same thing `20260929114041` did. The backbone and
   pixel encoder do load, which is the part that matters. Pre-adapting the head with
   `tools/adapt_classifier_checkpoint.py` was considered and dropped: it would also rewrite the
   phase-1 run's own `class_mapping.json` sidecar, since it writes next to its output.
3. **Do not freeze the encoder when starting from stock COCO weights.** A first attempt used the
   `unfreeze_decoder` config with `models/maskdino_r50_...pth` (the phase-1 checkpoint had not
   been copied over yet) and sat at **0.043** class-agnostic recall at iter 2000 vs 0.18 here.
   A decoder-only retrain cannot localize these tools off stock COCO features.
4. **Eval cadence is 2000, not 1000.** `TEST.EVAL_PERIOD 1000` drives the cheap evaluator;
   `HUNGARIAN_EVAL.PERIOD 2000` drives the confusion matrix, so the matrices exist only at
   iters 1999 / 3999 / 6000.
5. Training logs contain NUL bytes, so `grep` treats them as binary and silently matches nothing -
   pipe through `tr -d '\000'` first.

## Reproduce / re-read

> **Archived 2026-10-20.** `scratch/` no longer exists, and neither do the two arm configs, so
> the commands below are a record of what was run, not something that still executes. Kept for
> the arm A vs arm B tables and the gotchas, both cited elsewhere.

```bash
# both arms, sequential on GPUs 1+2, arm 2 only if arm 1 exits 0
bash scratch/run_scale_ab.sh

# side-by-side forcep03/04 confusion rows per eval tick, both arms
./.venv/bin/python scratch/scale_ab/compare.py
```
