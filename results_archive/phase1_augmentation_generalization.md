# Phase-1 augmentation vs. generalization to unseen instruments

What two sweeps (12 runs, 2026-11-08) say about how phase-1 augmentation affects a
**class-agnostic single-class** segmenter's ability to find instruments it never trained on.
The run directories were deleted to reclaim quota; raw `metrics.json` / `config.yaml` for all
12 runs are in `phase1_aug_sweeps_raw.tgz` beside this file.

## The question this measures

Train on set A only (`train_phase1_pool_set_a_mm_singleclass`, 8 instrument types), validate
on `val_set_ab_mm_singleclass`, where **51.6% of GT instances are 8 set-B types absent from
training** (`adapter11_mm, clamp11_mm, forcep11_mm, hammer11_mm, scalpel11_mm, scissor11_mm,
scissor12_mm, tweezer11_mm`). So `classaware_f1` here is a generalization score, not a fit
score. Metric is from `HungarianInstanceEvaluator`; single-class makes `classaware_* ==
classagnostic_*`.

## Answer in one table

`MAX_SCALE` is the only augmentation knob that moves generalization, and it saturates at ~2.5.

| `MAX_SCALE` | 1.2 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|
| `classaware_f1` | 0.749 | 0.785–0.791 | **0.812–0.835** | (0.770)* | 0.820 | 0.824–0.831 |

\* the single 0.25–3.0 arm degraded monotonically after iter 999 and had the largest spread
(sd 0.027); treated as an unstable run, not as a property of that setting.

**Recommended phase-1 setting: `MIN_SCALE: 0.3`, `MAX_SCALE: 2.5`, photometric at defaults.**

## Findings

1. **Scale range is worth up to +6.3 points; nothing else is.** Sweep-1 marginals (mean of
   last 5 of 10 evals, 20000 iters, averaged over the photometric axis):
   0.8–1.2 → **0.7489**, 0.5–2.0 → **0.7849** (+0.0360), 0.3–2.5 → **0.8119** (+0.0630).
   Per-arm run-to-run spread was 0.0053–0.0141, so the effect is 5–12x the noise.

2. **Photometric dose is inert at the doses tested.** Marginal effect of widening
   brightness/contrast/saturation from [0.7,1.3]/[0.7,1.3]/[0.8,1.2] to
   [0.5,1.5]/[0.5,1.5]/[0.6,1.4]: **−0.0011**. Within each scale level it was
   −0.0046 / +0.0073 / −0.0060 — it changes sign and never exceeds the per-arm spread. Do not
   read it as a small positive.

3. **The benefit comes from the high end of the range, not the low end.** Sweep 2, 6000 iters,
   mean of last 3 evals: `0.5–4.0` (high only) **0.8314** is statistically indistinguishable
   from the full `0.3–2.5` **0.8351**, while `0.1–2.0` (low only) is **0.7912** — 4 points
   worse. At `MAX 4.0`, `MIN 0.15` gives 0.8237 and `MIN 0.5` gives 0.8314, i.e. `MIN_SCALE`
   anywhere in 0.15–0.5 is immaterial. Mechanistically: upsampling (large apparent
   instruments) is what teaches instrument-ness; zooming out to small instruments does not.

4. **Scale saturates at 0.3–2.5.** Sweep 2's top four arms span 0.8202–0.8351 against per-arm
   sd 0.008–0.022 — not separable. 0.2–3.5 and 0.15–4.0 buy nothing over 0.3–2.5.

5. **Longer training costs generalization.** Every one of sweep 1's six arms peaked at iter
   1999–7999, never at the end; both 0.3–2.5 arms peaked at iter 3999 (0.8548, 0.8618) and
   lost ~4–5 points by 20000. `MAX_ITER 20000` fits set A rather than teaching
   instrument-ness. A shorter phase-1 schedule (~4000–6000) is better for this objective.

6. **`SCORE_THRESH 0.5` is the wrong operating point for phase-1 models.** All six sweep-1
   arms put `thresh_at_f1_max` at **0.80**, where f1 is 2.8–5.5 points higher. Uniform across
   arms, so it does not affect the ranking. (This does *not* carry over to the phase-2
   copy-paste model, which peaks at 0.60 and is within 0.0006 of its f1 at 0.5.)

7. **The ranking is not a calibration artifact.** `classaware_f1` (fixed `SCORE_THRESH 0.5`)
   and `sweep/f1_max` (best over 10 thresholds) give the identical ordering in sweep 1 and
   agree on the winner in sweep 2.

## Per-arm results

Sweep 1 — 3 scale levels x 2 photometric doses, `MAX_ITER 20000`, `EVAL_PERIOD 2000`:

| arm | scale | photometric | final | mean last 5 | best | @iter |
|---|---|---|---|---|---|---|
| `s_xwide_p_base` | 0.3–2.5 | default | 0.8040 | **0.8149** ± 0.0141 | 0.8548 | 3999 |
| `s_xwide_p_str` | 0.3–2.5 | strong | 0.7962 | 0.8089 ± 0.0121 | 0.8618 | 3999 |
| `s_wide_p_str` | 0.5–2.0 | strong | 0.7834 | 0.7886 ± 0.0057 | 0.8211 | 7999 |
| `s_wide_p_base` | 0.5–2.0 | default | 0.7811 | 0.7812 ± 0.0064 | 0.8253 | 5999 |
| `s_base_p_base` | 0.8–1.2 | default | 0.7477 | 0.7512 ± 0.0053 | 0.7961 | 1999 |
| `s_base_p_str` | 0.8–1.2 | strong | 0.7416 | 0.7466 ± 0.0065 | 0.7710 | 3999 |

Sweep 2 — scale axis only, default photometric, `MAX_ITER 6000`, `EVAL_PERIOD 1000`:

| arm | scale | final | mean last 3 | sd | sweep/f1_max |
|---|---|---|---|---|---|
| `ctl_030_250` | 0.3–2.5 (repeat of sweep-1 winner) | 0.8224 | **0.8351** | 0.0147 | 0.8392 |
| `hi050_400` | 0.5–4.0 high only | 0.8381 | 0.8314 | 0.0107 | 0.8338 |
| `s015_400` | 0.15–4.0 | 0.8314 | 0.8237 | 0.0083 | 0.8284 |
| `s020_350` | 0.2–3.5 | 0.8444 | 0.8202 | 0.0220 | 0.8249 |
| `lo010_200` | 0.1–2.0 low only | 0.7794 | 0.7912 | 0.0105 | 0.7968 |
| `s025_300` | 0.25–3.0 (unstable) | 0.7409 | 0.7703 | 0.0271 | 0.7733 |

## Methodology, including what was got wrong

- **Evals are a 256-frame subset, not 1024.** `DATASETS.TEST_SAMPLE_STRIDE: 4` is inherited
  from `Base-Surgical-HDF5.yaml:27`; `apply_test_sample_stride` keeps `full[::4]`, a
  systematic sample measured as representative (13.3% B-only frames vs the population's
  12.6%). An independent check on the phase-2 winner found the strided subset optimistic by
  ~1 point (0.9845 strided vs 0.9752 full), so treat strided figures as ranking values only.
- **Never quote frame composition from a contiguous sample of `val_set_ab_mm`.** Its first 120
  frames are 22.5% B-only against the population's 12.6% — dataset order front-loads B-only
  scenes. The 22.5% figure was quoted before being checked and is wrong.
- **Absolute values carry a pool confound; the within-sweep comparisons do not.** Sweep 1 ran
  while `train_phase1_pool_set_a_mm` grew 288 → 1024 frames; sweep 2 ran after it hit its cap
  and began FIFO-refreshing. All arms of a sweep were launched inside one minute and shared
  one pool state, which is why arms are comparable within a sweep and **not across sweeps** —
  hence `ctl_030_250`, which repeats sweep 1's winner as an in-sweep control and reproduced it
  at 0.8351 vs 0.8149.
- **Rotation was on in every arm** (`RANDOM_ROTATION` defaults `True`, `maskdino/config.py:235`),
  inherited from the committed recipe and held constant. Not swept, not a between-arm confound.
  It is the recipe's substitute for `RandomFlip`, which is excluded because instruments have
  fixed handedness.
- **`sweep/classaware_f1_max` does not exist for single-class runs** — `_sweep_summary` is
  called with `classaware=self._check_classes` and `_check_classes = _num_classes > 1`
  (`hungarian_instance_evaluation.py:239`); `hungarian_evaluator_metrics.md:134` marks the
  group "Multi-class only". Use `sweep/f1_max`.
- **`score_mode` does not exist on this branch**, so CLAUDE.md's `score_mode=detector`
  guidance is not applicable; the fixed operating point is `SCORE_THRESH 0.5`.
- **Do not rank on COCO AP.** It averages over the eval set's category count, sound here only
  because `_singleclass` collapses GT to one category.
- Throughput, for reproduction: `IMS_PER_BATCH 6` OOMs on one 24 GB Quadro RTX 6000
  (22695 MiB); 3 fits at 17191 MiB, 1.11 s/iter solo and 1.21–1.32 s/iter with six arms
  resident. Eval 0.421 s/frame at N=256. Host `dbe-cia-sl19-01` with two GPUs held by another
  user; all timings are on the deliberate pure-PyTorch deform-attn fallback.

## Relation to phase 2

Phase-2 copy-paste finetuning on `train_phase2_set_ab_mm_singleclass` reaches
`classaware_f1 = 0.9752` on the full 1024-frame val set, but **set B is in-domain there**, so
that number is not comparable to the 0.835 above. Two phase-2 results bear on this document:
scale jitter stops mattering once copy-paste is on (top four arms within noise, widest range
mildly penalized), and without copy-paste the sparse phase-2 renders collapse to f1 0.118 by
teaching the model to predict 0.79 instances/image against 3.95 in the GT.
