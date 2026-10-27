# Crop classifier: baseline numbers and the localization confound

Verified 2026-10-10 against the run artifacts in `/home/janick.bilang/dev/MaskDINO/runs/`.
All figures read from `metrics.json` / `config.yaml` / `inference/` of the named runs, not
from training-log scalars.

## 1. The two runs

| | 8-class baseline | single-class frontend |
|---|---|---|
| run | `20261006204751_phase1_training_seta_v1-1` | `20261007130733_single_class_v2` |
| `SEM_SEG_HEAD.NUM_CLASSES` | 8 | 1 (`class_embed` is `(1, 256)`) |
| train / test set | `train_seta_mm` / `val_seta_mm` | `train_seta_mm_singleclass` / `val_seta_mm_singleclass` |
| role | the thing the cropcls pipeline is compared against | supplies the masks the classifier classifies |

Shared eval protocol: stride 4, `EXCLUDE_TRUNCATED_INSTANCES: true`, 256 images, **1201 GT
instances**, Hungarian `IOU_THRESH 0.5` / `SCORE_THRESH 0.5`. The GT population is identical, so
the localization numbers below are directly comparable.

## 2. Baseline: 8-class MaskDINO (the comparison target)

`20261006204751_phase1_training_seta_v1-1`, iteration 20000:

```
classagnostic   precision 0.9781   recall 0.9276   f1 0.9521    <- localization only
classaware      precision 0.9508   recall 0.9017   f1 0.9256    <- localization + classification
misclassified   31 / 1114 matched = 2.78%                       <- classification error alone
mean_iou_matched 0.8361
gt 1201   pred 1139   matched 1114   correct 1083   fp 25
segm/AP 65.38
```

Per class:

| class | agnostic recall | aware recall | misclassified |
|---|---|---|---|
| forcep01_mm | 0.8968 | 0.8645 | 5 |
| forcep02_mm | 0.9456 | 0.8912 | 8 |
| forcep03_mm | 0.7815 | 0.7285 | 8 |
| forcep04_mm | 0.8310 | 0.7676 | 9 |
| scalpel01_mm | 0.9699 | 0.9639 | 1 |
| hammer01_mm | 1.0000 | 1.0000 | 0 |
| sharpspoon01_mm | 1.0000 | 1.0000 | 0 |
| scarstick01_mm | 0.9924 | 0.9924 | 0 |

**30 of the 31 misclassifications are forcep01-04.** This is the spec's opening premise
("problems with distinguishing similar objects") quantified, and it is confined to the forceps
family. forcep03 and forcep04 are within 4% of each other on median mask area (1421 vs 1469 px),
6% on aspect ratio (1.80 vs 1.69) and 3% on fill (0.072 vs 0.070) - they are near-identical in
every shape statistic.

**The classification bar for the classifier: 97.22% top-1 accuracy on matched masks** (1083/1114).

## 3. Frontend: single-class MaskDINO

`20261007130733_single_class_v2`, iteration 50000:

```
classagnostic   precision 0.9735   recall 0.9792   f1 0.9763
mean_iou_matched 0.8414
gt 1201   pred 1208   matched 1176   fp 32
bbox/AP 95.15   segm/AP 67.99
```

Confusion matrix collapses to one row under the singleclass registration, so there is **no
per-class breakdown available** for this run:

```
gt\pred,surgical-instruments,(false negative)
surgical-instruments,0.9792,0.0208
```

## 4. Where the localization difference comes from

The frontend localizes better - `classagnostic f1` 0.9763 vs 0.9521, and 1176 vs 1114 GT
instances matched out of the same 1201. **Most of that gap is recipe, not architecture.**

| | baseline `v1-1` | `single_class_v2` |
|---|---|---|
| `MODEL.BACKBONE.FREEZE_AT` | **5** (entire ResNet-50 frozen) | **0** (fully trainable) |
| `SOLVER.MAX_ITER` | 20,000 | 50,000 |
| `SOLVER.LR_SCHEDULER_NAME` | WarmupCosineLR | ReduceLROnPlateau |
| `SOLVER.BASE_LR` | 3e-4 | 3e-4 |
| `SOLVER.IMS_PER_BATCH` / `WARMUP_ITERS` | 6 / 250 | 6 / 250 |
| `MODEL.RESNETS.NORM` | FrozenBN | FrozenBN |
| training data | set A pool | set A pool (same frames, collapsed labels) |

Three contributions, largest first:

**(a) The unfrozen backbone - the dominant term.** `FREEZE_AT: 5` freezes stem + res2-res5, i.e.
all of ResNet-50, so `v1-1` adapts only the transformer on top of COCO-pretrained features.
`single_class_v2` adapts the whole feature extractor to this domain. Note `RESNETS.NORM` is
`FrozenBN` in both, so "unfrozen" here means weights receive gradients, not that BN statistics
drift.

**(b) The 2.5x longer schedule - not the driver.** The trajectories rule it out:

```
single_class_v2 @  9,999 : agnostic f1 0.9533    <- already beats v1-1's FINAL
single_class_v2 @ 19,999 : agnostic f1 0.9712
v1-1            @ 20,000 : agnostic f1 0.9521
```

At matched iteration count the frontend is still ~2 points ahead, and it reaches `v1-1`'s final
localization by iteration 10,000. The back 30k iterations add only 0.9712 -> 0.9763.

**(c) A genuine single-class effect, and it is about matching rather than mask quality:**

```
mean_iou_matched :  0.8414  vs  0.8361      <- nearly identical
matched instances:  1176    vs  1114        <- +62
```

The masks it draws are no better; it matches more GT. That is the signature of the 1-wide class
head. In `SetCriterion`'s Hungarian cost the classification term is effectively constant when
every query predicts the same class, so assignment is driven almost purely by mask/box geometry.
Consequences: a query with a good mask but the wrong label can no longer lose its match to a
worse-mask query with the right label, and all 100 queries specialize on "find a thing" instead
of splitting into class-specific roles. This is the decoupling effect the thesis argues for - it
is real, but smaller than (a).

## 5. Consequence for how results are reported

**Top-1 accuracy on matched masks is confound-free.** It measures classification given whatever
masks are supplied. Baseline: **97.22%**. Report this as the primary number.

**`classaware_f1` inherits the confound, and it is large.** The cropcls pipeline is handed 1176
masks where the baseline only ever localized 1114. A perfect classifier on the frontend's masks
would score `classaware recall = 1176/1201 = 0.9792` against the baseline's 0.9017 - **+7.75
points for free**, from masks rather than from classification.

Two ways to handle it:

- **Cheap and honest:** report `classaware_f1` with `classagnostic_f1` beside it and state the
  recipe difference, so a reader can see which half moved.
- **Airtight:** retrain the 8-class baseline with `FREEZE_AT: 0` and `MAX_ITER: 50000`. The only
  remaining difference is then 8-class vs 1-class, and the residual gap is the decoupling effect.
  ~17 h, and it would also yield a stronger `segm/AP` baseline than 65.38 for whole-pipeline
  comparisons.

The first suffices for the claim "the classifier classifies as well as MaskDINO's own head". The second
is required for the claim "decoupling localization from classification helps".

## 6. Side note: COCO AP is not the metric here

MaskDINO's `instance_inference` ([maskdino/maskdino.py:550-586](maskdino/maskdino.py#L550-L586))
flattens 100 queries x N classes and takes `topk(100)`, so one mask may be emitted under several
labels and COCO AP rewards that hedging. A classifier emitting one argmax label per instance
cannot hedge, so an AP comparison would penalize it for reasons unrelated to classification
quality. With f1 / top-1 accuracy as the metric, the spec's "output the most probable case" rule
is correct and the issue does not arise.
