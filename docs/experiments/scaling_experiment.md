# Does more resolution help forcep03 / forcep04?

Phase-2 focal-length sweep, 9 runs (3 levels x 3 seeds) on `dbe-sl21-01`, one A100,
trained 2026-10-26 15:17 -> 10-27 02:27 (~11 GPU-hours including full-val evals); all
nine evals re-run on 2026-10-27 with frame-edge-cut tools as ignore regions (see
[README gotcha](README.md#shared-gotchas)), and on 2026-10-28 on a second val set that
keeps every tool inside the 1.5x frame (section 6). Asks whether more pixels per mm on the
instrument (a better camera, a closer mount, a zoom) helps the model **tell forcep03 from
forcep04** and **miss fewer of them**.

## Answer

**More resolution helps find and segment the forceps, but not tell them apart. The 03/04
confusion is a bias toward forcep04, not a lack of resolution.**

- **Missed detections:** strongly resolution-limited below the current 1.0x render (0.67x
  doubles them), only mildly above it (1.5x: -0.027 at score 0.3, against -0.014 for all
  other tools). At each level's best score threshold, forcep03+04 F1 rises from 0.803 to
  **0.853 (+0.049 [+0.021, +0.078])**, all of it from fewer misses. For all tools together
  the gain is small but real: **+0.019 [+0.012, +0.026]** on the v2 val set, where every
  level is scored on the same objects (section 6); the old set showed a tie (+0.007)
  because its 1.5x frames cut off the easy tools near the edge.
- **03 vs 04:** of the forceps the model detects, the share labelled correctly is flat from
  645 px to 3247 px median mask area, and the 03 <-> 04 swap rate does not move.
- **Why:** the scores *do* separate the two at every level (AUC ~0.98), but forcep04's score
  systematically beats forcep03's, even on real forcep03. The top label gets only ~70 % of
  forcep03 right; a fixed bonus for forcep03 lifts balanced accuracy from 0.84 to 0.93 at
  1.0x (chosen in-sample, so optimistic). **This is a calibration / training-data problem,
  not an optics problem.** The v2 val set confirms it: forcep03 -> 04 even rises with zoom
  there (0.210 -> 0.259 at score 0.4) while forcep04 -> 03 stays at 0.010.
- **Mask quality:** the largest effect. forcep03/04 masks top out around IoU 0.68 at 1.0x;
  at 1.5x, 74 % of them clear IoU 0.75. AP75 goes from **0.3 -> 45.1 (03)** and
  **0.4 -> 63.5 (04)**.

## Why a focal-length sweep, not scaling the objects

The first version of this question was: if forcep03/04 are rendered 1.5x larger, the
validation set has to change too, so are the metrics still comparable? A one-seed pilot
that did exactly that (below) showed that they are not, for three reasons:

1. **Relative size.** 03/04 became larger than every tool around them: less of them is
   occluded, and size alone separates them from the other forceps. Classes whose training
   data was byte-identical moved as much as 03/04 did.
2. **Different instance sets.** The frame-edge truncation filter removed 10.5 % / 14.1 % of
   03/04 at 1.5x against 0.8 % / 5.8 % at 1.0x.
3. **Unpaired scenes.** Object size changes the pile physics, and the generator is unseeded,
   so the two val sets were unrelated random scenes.

Changing the **camera's focal length** instead scales every tool, the pile and the
background together, leaves the physics and lighting untouched, and lets the same settled
pile be rendered at every level. Every val instance then exists at every level, and each
level can be scored on the same physical objects.

## Protocol

| | |
|---|---|
| levels | camera focal length 0.67x / 1.0x / 1.5x (`zoom_levels` in the scene_generator config); **1.0x is the control** |
| train | `reclass_setab_fixedsurf_mm-<level>`: 380 classification renders (2 sides x 10 rotations per instrument), same renders at every level |
| val | `val_setab_fixedsurf_mm-<level>`: 1000 frames (2 per drop), **same drops and identical filenames at every level**; replication on `val_setab_fixedsurf_mm_v2-<level>` (piles kept inside the 1.5x frame, section 6) |
| 03/04 finish | identical in train and val (`standard`, roughness 0.12) - geometry is the only cue |
| base model | `runs/20261025111522_phase1_training_seta_v1-0_1.5-scale` - phase 1 trained with scale jitter 0.67-1.5, so the frozen backbone has seen all three levels and favours none |
| recipe | [scaling/focal_sweep.yaml](../../configs/coco/instance-segmentation/scaling/focal_sweep.yaml): calibrated per-instance jitter ([calibrated_jitter.yaml](../../configs/coco/instance-segmentation/augmentation/calibrated_jitter.yaml)), encoder + decoder + head trainable, backbone frozen, `MAX_ITER 2000`, `IMS_PER_BATCH 6`, `BASE_LR 3e-5` cosine, scale jitter off, `model_final` only |
| seeds | 42, 1, 2 at every level |
| eval | `model_final` on the **full** val set (stride 1); tools cut by the frame edge are `iscrowd` ignore regions (neither a miss nor a false positive); Hungarian matching at IoU 0.5; other scores replayed from the cached predictions |
| wall clock | 70-86 min per run including the full-val eval |

**Paired analysis.** Unless a table says "all GT", numbers are restricted to the
**3262 instances (438 forcep03/04) that are uncut at all three levels**, across 437 drops.
Each object is identified by its index in `instance_segmaps`, so the same physical object
is compared across levels. Rates are the mean of the three seeds. 95 % CIs come from a
bootstrap over drops (both frames of a drop resampled together), applied jointly to every
level and seed, so each delta is paired. **The CIs cover val sampling only.** Training-seed
noise is about as large (forcep03+04 correct rate, 1.5x - 1.0x: standard error over seeds
0.013 at score 0.3, 0.020 at 0.5), so per-seed deltas are listed where they matter.

## Results

### 1. Missed detections: steep below 1.0x, flattening above

| forcep03 + forcep04 | 0.67x | 1.0x | 1.5x | 0.67x - 1.0x | 1.5x - 1.0x |
|---|---|---|---|---|---|
| median mask area | 645 px | 1444 px | 3247 px | | |
| missed @ 0.5 | 0.508 | 0.233 | 0.185 | +0.275 [+0.228, +0.323] | -0.048 [-0.077, -0.020] |
| missed @ 0.3 | 0.289 | 0.047 | 0.020 | +0.242 [+0.195, +0.291] | -0.027 [-0.046, -0.011] |
| *other 14 classes, missed @ 0.3* | 0.112 | 0.049 | 0.035 | +0.063 [+0.052, +0.072] | -0.014 [-0.021, -0.008] |

Per class at score 0.5, forcep03 goes 0.639 / 0.389 / 0.323 and forcep04 0.381 / 0.081 / 0.051.
The forcep03 1.5x - 1.0x delta at score 0.5 is **not seed-stable** (seeds 1 / 2 / 42:
+0.019 / -0.102 / -0.116); at score 0.3 it is (-0.042 / -0.060 / -0.037).

**Misses at 0.5 are low-confidence detections, not bad masks.** At 1.0x, 95 % of missed
03/04 have no prediction above the threshold that overlaps them at IoU >= 0.1, and lowering
the threshold to 0.3 recovers most of them (0.233 -> 0.047).

### 2. Best score threshold per level

Comparing levels at one fixed threshold mixes resolution with score calibration (section 5).
Each level is therefore compared at its own best threshold, with two deployment-style rules
([sweep_focal_thresholds.py](../../tools/scaling/sweep_focal_thresholds.py)): **one label per
mask** (MaskDINO emits a query's mask once per class; below ~0.5, 7-9 % of forcep03/04
objects otherwise carry both labels and the score-blind matcher picks one arbitrarily), and
cut tools as ignore regions. Class-aware F1 counts a swap as both a miss and a false
positive. Each level's own kept objects, so not paired.

| | 0.67x | 1.0x | 1.5x | 1.5x - 1.0x (best vs best) |
|---|---|---|---|---|
| all tools: best F1 | 0.827 @ 0.40 | 0.899 @ 0.40 | 0.907 @ 0.40 | +0.007 [+0.000, +0.014] |
| all tools: precision / recall | 0.859 / 0.797 | 0.922 / 0.878 | 0.936 / 0.879 | |
| all tools: false positives per image | 0.34 | 0.13 | 0.06 | |
| forcep03+04: best F1 | 0.645 @ 0.35 | 0.803 @ 0.45 | **0.853 @ 0.40** | **+0.049 [+0.021, +0.078]** |
| forcep03+04: missed at that threshold | 0.283 | 0.167 | 0.069 | |
| forcep03+04: swapped at that threshold | 0.092 | 0.089 | 0.102 | |

The CI re-chooses the threshold inside every bootstrap resample, so picking the best
threshold does not inflate the difference. Per seed, the forcep03+04 gain is +0.037 /
+0.062 / +0.048; the all-tools gain is -0.000 / +0.014 / +0.009, i.e. no practical
difference. All three levels peak at or next to **score 0.40**; the 0.5 used so far sits
past the peak.

**At fixed thresholds 0.5 and 0.4**, same rules, all GT, seed mean (0.3 is in
`threshold_analysis.txt` too):

| level | score | all tools: F1 | FP / image | 03+04 F1 | forcep03 right | forcep03 -> 04 | forcep03 missed | forcep04 -> 03 |
|---|---|---|---|---|---|---|---|---|
| 0.67x | 0.5 | 0.819 | 0.214 | 0.612 | 0.356 | 0.043 | 0.601 | 0.010 |
| 0.67x | 0.4 | 0.827 | 0.339 | 0.642 | 0.455 | 0.124 | 0.418 | 0.017 |
| 1.0x | 0.5 | 0.891 | 0.060 | 0.789 | 0.493 | 0.131 | 0.377 | 0.001 |
| 1.0x | 0.4 | 0.899 | 0.125 | 0.801 | 0.588 | 0.244 | 0.167 | 0.004 |
| 1.5x | 0.5 | 0.877 | 0.024 | 0.819 | 0.535 | 0.142 | 0.323 | 0.000 |
| 1.5x | 0.4 | 0.907 | 0.062 | 0.853 | 0.687 | 0.207 | 0.106 | 0.000 |

- **At 0.5, 1.5x trails 1.0x on all-tools F1 (0.877 vs 0.891); at 0.4 it leads (0.907 vs
  0.899) with half the false positives.** Fixed-0.5 counts understate the higher zoom (see
  also section 5).
- **About half of the forcep03 recovered by the lower threshold come back as forcep04.** At
  1.0x, going from 0.5 to 0.4 cuts forcep03 misses by 0.210, but only 0.095 turns into
  correct labels and 0.113 into forcep04. Many forcep03 "misses" at 0.5 are forcep04 labels
  just below the cut. A lower threshold finds forcep03; it does not fix the label
  (section 3). forcep04 -> 03 stays at or below 0.004 at 1.0x and 1.5x.
- These rates keep one label per mask, so they differ from the official-matching figures in
  section 3 (forcep03 -> 04 at 0.3, 1.0x: 0.299 here vs 0.219). When one mask carries both
  labels, the official, score-blind matcher picks one of the two copies arbitrarily, so it
  sometimes credits the 03 copy even though the 04 copy scores higher.

### 3. Telling 03 from 04: a bias, not a resolution limit

| forcep03 + forcep04 (paired) | 0.67x | 1.0x | 1.5x | 1.5x - 1.0x |
|---|---|---|---|---|
| 03 <-> 04 swap @ 0.5 | 0.021 | 0.062 | 0.070 | +0.008 [-0.018, +0.035] |
| 03 <-> 04 swap @ 0.3 | 0.084 | 0.107 | 0.105 | -0.002 [-0.031, +0.024] |
| correct of detected @ 0.5 | 0.950 | 0.918 | 0.913 | |
| correct of detected @ 0.3 | 0.862 | 0.883 | 0.891 | |

- The lower swap rate at 0.67x is a selection effect: fewer forceps are detected there, and
  the ones that are tend to be the easy ones.
- **The confusion is one-way** (all GT, score 0.3): forcep03 -> forcep04 is 0.121 / 0.219 / 0.204
  across the three levels, forcep04 -> forcep03 only 0.060 / 0.012 / 0.009.

**Threshold-free separability.** For each paired forcep03/04 object, take the best
03-labelled and the best 04-labelled score on its mask (IoU >= 0.5, scores >= 0.05) and
look at their difference:

| | 0.67x | 1.0x | 1.5x |
|---|---|---|---|
| objects with any 03/04-labelled mask | 0.725 | 0.960 | 0.993 |
| AUC 03 vs 04 (score difference) | 0.960 | 0.976 +/- 0.003 | 0.981 +/- 0.004 |
| forcep03 right with the top label | 0.735 | 0.681 | 0.731 |
| forcep04 right with the top label | 0.959 | 0.994 | 0.991 |
| balanced accuracy, top label | 0.847 | 0.837 | 0.861 |
| balanced accuracy, best fixed offset (in-sample) | 0.893 @ -0.19 | 0.929 @ -0.38 | 0.933 @ -0.46 |

The scores rank 03 against 04 well at every level, and resolution barely changes that
ranking (AUC 0.976 -> 0.981). What fails is the decision: forcep04's score beats
forcep03's on about 30 % of real forcep03. A single offset on the score difference, i.e.
"call it 03 unless 04 leads by more than ~0.4", recovers most of it. The offset was chosen
on the same val set, so 0.93 is an upper bound until it is fitted on one half and tested
on the other.

### 4. Mask quality: the large effect

Mask AP, each level's own GT set, mean +/- sd over seeds:

| | 0.67x AP / AP75 | 1.0x AP / AP75 | 1.5x AP / AP75 |
|---|---|---|---|
| all classes | 50.6 / 57.8 | 62.1 / 65.1 | 68.8 / 75.8 |
| forcep03 | 9.1 / 0.0 | 29.1 +/- 0.7 / 0.3 +/- 0.1 | 48.5 +/- 1.1 / **45.1 +/- 1.6** |
| forcep04 | 12.4 / 0.0 | 33.5 +/- 0.5 / 0.4 +/- 0.4 | 54.2 +/- 0.6 / **63.5 +/- 1.9** |
| forcep01 | 11.5 / 7.5 | 31.7 / 13.7 | 47.3 / 31.7 |
| tweezer11 | 24.3 / 4.6 | 34.6 / 12.3 | 39.8 / 33.2 |

| paired forcep03/04, correctly classified | 0.67x | 1.0x | 1.5x |
|---|---|---|---|
| median mask IoU | 0.572 | 0.678 | 0.785 |
| share with IoU >= 0.75 | 0 % | 2 % | 74 % |

The ~0.68 IoU cap at 1.0x matches the forcep-only pilot (0.682 / 0.685). Thin tools in
general gain the same way (forcep01, tweezer11). **Likely explanation, untested:** MaskDINO
predicts masks from its stride-4 feature map (256 x 256 for a 1024 input), so the thin jaws
and ring handles at 1.0x are only 1-2 mask cells wide. If that holds, a finer mask path or
an upsampled input would recover much of this without a new camera.

### 5. The score threshold is not level-invariant

At 1.5x several other tools are detected with lower scores. At 0.5 their misses rise
(paired, vs 1.0x): hammer01 +0.119, tweezer11 +0.112, clamp11 +0.108, forcep11 +0.097,
all 14 pooled +0.034 [+0.025, +0.045]. At 0.3 the pooled delta flips to -0.014, and their
mean AP rises 66.5 -> 71.3. So 1.5x is not worse at finding them; their scores just sit
lower. **Compare levels at their best threshold (section 2) or by AP**, not by fixed-0.5
Hungarian counts.

### 6. Replication: every tool inside the 1.5x frame (val v2)

The val set above cuts 18.7 % of tools at 1.5x (2 % at 0.67x), so AP and best-threshold F1
scored each level on a different set of objects (caveat 2). The same nine models, no
retraining, were therefore evaluated again on `val_setab_fixedsurf_mm_v2-<level>` on
2026-10-28: same generator and finishes, containment box half-extent 0.5 m -> 0.14 m, 500
drops x 2 frames, identical filenames across levels.

- Tools cut at 1.5x: **1.6 %** (was 18.7 %). Every object kept at 1.5x is kept at every
  level, so the paired set is the whole 1.5x GT: **4427 instances (505 forcep03/04) in 442
  drops** (116 frames are empty by design; the old set had 123).
- The piles are a little denser (4.5 vs 4.2 objects per frame at 1.0x) and every level
  scores 1-4 AP lower. **Compare the level deltas between the two sets, not the absolute
  numbers.**

Three seeds each, same rules as sections 1-3 (paired rates, simulation-bootstrap CIs):

| | old val | v2 val |
|---|---|---|
| best F1, all tools (0.67x / 1.0x / 1.5x) | 0.827 / 0.899 / 0.907 | 0.781 / 0.873 / 0.891 |
| ... 1.5x - 1.0x, best vs best | +0.007 [+0.000, +0.014] | **+0.019 [+0.012, +0.026]** (seeds +0.014 / +0.021 / +0.020) |
| best F1, forcep03+04 | 0.645 / 0.803 / 0.853 | 0.532 / 0.738 / 0.796 |
| ... 1.5x - 1.0x, best vs best | +0.049 [+0.021, +0.078] | +0.058 [+0.022, +0.093] (seeds +0.046 / +0.052 / +0.077) |
| best threshold, all tools | 0.40 / 0.40 / 0.40 | 0.45 / 0.40 / 0.35 |
| mask AP, all classes | 50.6 / 62.1 / 68.8 | 46.2 / 58.4 / 67.5 |
| AP75 forcep03 / forcep04 at 1.5x | 45.1 / 63.5 | 45.7 / 46.0 |
| paired 03+04 missed @ 0.3, 1.5x - 1.0x | -0.027 [-0.046, -0.011] | **-0.079 [-0.109, -0.052]** |
| paired 03+04 swap @ 0.3, 1.5x - 1.0x | -0.002 [-0.031, +0.024] | +0.020 [-0.010, +0.051] |
| paired other 14 missed @ 0.5 / @ 0.3, 1.5x - 1.0x | +0.034 / -0.014 | +0.038 / -0.017 |
| AUC 03 vs 04 | 0.960 / 0.976 / 0.981 | 0.940 / 0.962 / 0.967 |
| forcep03 / forcep04 right with the top label | 0.735 / 0.681 / 0.731 and 0.959 / 0.994 / 0.991 | 0.711 / 0.697 / 0.685 and 0.940 / 0.983 / 0.989 |
| forcep03 -> 04 / forcep04 -> 03, all GT @ 0.4 | 0.124 / 0.244 / 0.207 and 0.017 / 0.004 / 0.000 | 0.129 / 0.210 / 0.259 and 0.027 / 0.010 / 0.010 |
| median mask IoU, correct 03/04 | 0.572 / 0.678 / 0.785 | 0.575 / 0.673 / 0.778 |

- **Caveat 2 was real: on the same objects, 1.5x beats 1.0x for all tools.** Best-vs-best
  F1 +0.019, every seed positive, against a tie on the old set; the AP gap grows from +6.7
  to +9.1. The old set hid this because at 1.5x it lost the easy stray tools near the frame
  edge.
- **Finding forcep03/04 gains more.** At score 0.3, 1.5x misses 0.079 fewer than 1.0x
  (old set 0.027), because the denser v2 piles leave more to recover at 1.0x (0.116 missed
  vs 0.047).
- **The 03 -> 04 bias does not shrink; if anything it grows with zoom.** forcep03 -> 04
  rises 0.210 -> 0.259 at score 0.4, while forcep04 -> 03 stays at 0.010. Per class at 0.3,
  1.5x - 1.0x, forcep04 correct gains +0.070 [+0.038, +0.104] but forcep03 correct only
  +0.057 [-0.004, +0.122]: about half of the forcep03 that 1.5x newly finds come out
  labelled forcep04. The scores still rank 03 above 04 well (AUC 0.967), and a fixed
  offset of about -0.40 again lifts balanced accuracy (0.84 -> 0.91 at 1.5x, in-sample).
- **The best threshold now falls with zoom** (0.45 / 0.40 / 0.35), and the other tools'
  scores again sit lower at 1.5x (section 5): more misses at 0.5, fewer at 0.3.
- forcep04 AP75 at 1.5x is 46.0 here against 63.5 on the old set, while forcep03 is
  unchanged at ~45. Not investigated.

## Pilot: scaling only forcep03/04 (superseded)

Two runs on 2026-10-22, one seed each, identical except the forcep03/04 object scale
(0.001 -> 0.0015) in train and val, old phase-1 base
`runs/20261013204802_phase1_training_seta_v1-0`, same recipe otherwise:
`runs/20261022152351_augexp_calibrated_jitter_fixsurf_ident_on_forcep` (1.0x) and
`runs/20261022190024_augexp_calibrated_jitter_larger_forcep_fixsurf_ident_on_forcep` (1.5x).
Full-val (stride 1) re-evals in `runs/_eval_fullval/`, **with the old truncation handling**
(cut tools deleted from GT, box-based prediction drop).

| forcep03 + forcep04, all GT | 1.0x | 1.5x | delta [95 % CI] |
|---|---|---|---|
| missed @ 0.5 | 93/510 = 0.182 | 48/409 = 0.117 | -0.065 [-0.120, -0.006] |
| 03 <-> 04 swap @ 0.5 | 13/510 = 0.025 | 10/409 = 0.024 | -0.001 [-0.022, +0.020] |
| missed @ 0.3 | 38/510 = 0.075 | 15/409 = 0.037 | -0.038 [-0.078, -0.000] |
| 03 <-> 04 swap @ 0.3 | 32/510 = 0.063 | 21/409 = 0.051 | -0.011 [-0.044, +0.022] |
| AP75 forcep03 / forcep04 | 0.2 / 0.1 | 38.5 / 50.7 | unchanged classes +1.2 +/- 5.3 |

At score 0.5 the unchanged classes moved about as much as 03/04: forcep01 -0.099,
forcep02 -0.075, scalpel01 +0.074, all with CIs excluding zero. That run-to-run noise, plus
the confounds listed above, is why this design was replaced. Its direction agrees with the
sweep on all three questions.

## Caveats

1. **03 -> 04 confusion regressed relative to the pilot.** At 1.0x, score 0.5, all GT: 0.131
   here against 0.028 in the pilot's 1.0x arm, with identical 03/04 finish in both. The two
   differ in the phase-1 base (scale-jitter base here) and in the renders. Which of the two
   caused it is not known. The pilot's "the 03/04 confusion is already solved at 1.0x" does
   not carry over.
2. **AP and best-threshold F1 are not paired.** They are computed on each level's own uncut
   instances, which differ between levels: 1.5x loses the isolated stray tools near the
   frame edge, which are the easy ones (0.67x, score 0.40: 03/04 missed 0.113 among them vs
   0.377 in the paired set). Both comparisons are therefore conservative for 1.5x.
   **Resolved by the v2 val set (section 6):** there all levels share one object set, and
   the all-tools gain of 1.5x grows from +0.007 to +0.019.
3. **The CIs cover val sampling only;** training-seed noise is about as large. With three
   seeds, a difference is trusted only where all seeds agree in sign.
4. **Synthetic val only.** No real-photo check yet.
5. **One training budget** (2000 iterations, from `model_final`). Higher levels might profit
   from longer training differently; not tested.
6. **Earlier versions of this doc** used the old truncation handling. Re-evaluated, the
   paired miss / swap / correct rates move by at most 0.002, while 1.5x gains up to 1 AP
   and loses 50-77 false positives per run; the conclusions did not change.

## Open items

1. **Fix the forcep03 bias, cheaply first:** fit a score offset (or a separate forcep03
   threshold) on one half of the val drops and test it on the other half, per level. If it
   holds, retrain with class-balanced loss or more forcep03 appearances and compare.
2. **Separate base model from data for the 03 -> 04 regression:** one run with the old
   phase-1 base (`runs/20261013204802_phase1_training_seta_v1-0`) on
   `reclass_setab_fixedsurf_mm-1.0x`.
3. **Mask-path test:** evaluate (and if promising, train) the 1.0x data at input 1536. If
   the 03/04 IoU cap lifts, the limit is the mask head's resolution, fixable in software.
4. **Real-photo check:** real images at native, 3/4 and 1/2 resolution give the real-world
   curve below the camera's native resolution.
5. **forcep03 source data with more distinct appearances**, carried over from
   [trainset_confusion_check_forcep03_results.md](trainset_confusion_check_forcep03_results.md).

## What was run

```bash
# 3 levels x seeds 42 1 2 queued on GPU 0; each run = phase-2 finetune + full-val eval into <run>/eval_full/
bash tools/scaling/launch_focal_grid.sh "0"

# 2026-10-27: all nine evals re-run with cut tools as iscrowd ignore regions, 5 side by side on GPU 1
FORCE=1 EVAL_ONLY=1 bash tools/scaling/launch_focal_grid.sh "1 1 1" "1 2"
FORCE=1 EVAL_ONLY=1 LEVELS="1.0x 1.5x" bash tools/scaling/launch_focal_grid.sh "1" "42"

# per-instance replays of every run (CPU, ~2 min in parallel) + both analyses
OUT=runs/_analysis/focal_sweep_newval bash tools/scaling/analyze_focal_sweep.sh
#   -> runs/_analysis/focal_sweep_newval/focal_analysis.txt      sections 1, 3 (paired), 4, 5
#   -> runs/_analysis/focal_sweep_newval/threshold_analysis.txt  sections 2, 3 (separability)
#   (runs/_analysis/focal_sweep_oldval/ is the same analysis before the 10-27 eval re-run,
#    superseded; its eval_full/ predictions were overwritten by the re-run)

# 2026-10-28: the same nine models on the v2 val set (every tool inside the 1.5x frame),
# into <run>/eval_full_v2/ - seed 42 on GPU 0, seeds 1 and 2 six side by side on GPU 2
EVAL_ONLY=1 VAL_TAG=v2 bash tools/scaling/launch_focal_grid.sh "0 0 0" "42"
EVAL_ONLY=1 VAL_TAG=v2 bash tools/scaling/launch_focal_grid.sh "2 2 2 2 2 2" "1 2"
EVAL_DIR=eval_full_v2 OUT=runs/_analysis/focal_sweep_newval_dataset_v2 bash tools/scaling/analyze_focal_sweep.sh
#   -> runs/_analysis/focal_sweep_newval_dataset_v2/{focal,threshold}_analysis.txt   section 6
```

The focal train sets (`train_phase2_set_ab_fixedsurf_mm-<level>`) have since been deleted
from disk, so retraining an arm needs them re-rendered. The v2 evals did not need them,
but without the train set registered, train_net's train/test class-mapping check silently
passes; the v2 val sets' mapping was checked by hand against all nine `model_final.pth`.

| level | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| 0.67x | `20261026151720_focal_0.67x_seed42` | `20261026191052_focal_0.67x_seed1` | `20261026224806_focal_0.67x_seed2` |
| 1.0x | `20261026164331_focal_1.0x_seed42` | `20261026202438_focal_1.0x_seed1` | `20261027000238_focal_1.0x_seed2` |
| 1.5x | `20261026180004_focal_1.5x_seed42` | `20261026213742_focal_1.5x_seed1` | `20261027011700_focal_1.5x_seed2` |

- [replay_focal_instances.py](../../tools/scaling/replay_focal_instances.py) replays one run
  per GT instance with the official matching and stores its per-class COCO AP. At score 0.5
  it reproduces every run's official `instance_matching_confusion.csv` and false-positive
  count exactly (checked on all nine).
- [sweep_focal_thresholds.py](../../tools/scaling/sweep_focal_thresholds.py) sweeps the score
  threshold with one label per mask; at 0.5 it differs from the official outcome on at most
  one object per run.
- [analyze_focal_sweep.py](../../tools/scaling/analyze_focal_sweep.py) and
  [analyze_focal_thresholds.py](../../tools/scaling/analyze_focal_thresholds.py) compute the
  tables above.
