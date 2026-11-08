# Crop classifier — results

Run 2026-10-10/11 on `dbe-cia-sl23-01` (4x RTX 2080 Ti, GPUs 1–3 idle). All numbers
from `val_seta_mm`; the baseline is `20261006204751_phase1_training_seta_v1-1` (8-class
MaskDINO, stride 4, excl-truncated, 1201 GT).

Six runs. Measured **endpoint** seed variance (r2 vs r4, identical but for the seed) is
**0.09 points** on crop top-1 and **0.82 points** on `classaware_f1`. An earlier 2.3-point
figure in this file came from iteration 500 and was wrong for final numbers.

## 1. What ran

Common to all six: 4000 iterations, `ims_per_batch 4`, `base_lr 3e-4`, crop 224, frozen
`single_class_v2` detector run on **unaugmented** frames, augmentation applied at crop level.

| run | trunk | augmentation | input | seed |
|---|---|---|---|---|
| r1 | resnet50 | rotations only | RGB, blacked | 0 |
| r2 | resnet50 | rotations + photometric + bbox jitter | RGB, blacked | 0 |
| r3 | resnet50 | full | **silhouette only** | 0 |
| r4 | resnet50 | full | RGB, blacked | **1** |
| r5 | resnet50 | full | RGB, **background kept** | 0 |
| r6 | **resnet18** | full | RGB, blacked | 0 |
| dil3 | resnet50 | full | RGB, **mask dilated 3 px** | 0 |
| dil8 | resnet50 | full | RGB, **mask dilated 8 px** | 0 |
| dil16 | resnet50 | full | RGB, **mask dilated 16 px** | 0 |

## 2. Headline

**All numbers are `classaware_f1` from the baseline's own `HungarianInstanceEvaluator`
(`score_mode=detector`). Crop top-1 is reported only as a training monitor — it has
misled in both directions (section 5) and should not be used to rank runs.**

| run | mask handling | classagnostic_f1 | **classaware_f1** | vs base | (crop top-1) |
|---|---|---|---|---|---|
| **8-class MaskDINO** | — | 0.9521 | **0.9256** | — | 0.9722 |
| **dil16** | dilate 16 px | 0.9812 | **0.9175** | **−0.0081** | 0.9195 |
| dil3 | dilate 3 px | 0.9812 | 0.9077 | −0.0179 | 0.9115 |
| dil8 | dilate 8 px | 0.9812 | 0.9003 | −0.0253 | 0.9044 |
| r1 | tight, rotations only | 0.9812 | 0.8824 | −0.0432 | 0.9000 |
| close16 | closing 16 px | 0.9812 | 0.8799 | −0.0457 | 0.9124 |
| r2 | tight, full aug | 0.9812 | 0.8758 | −0.0498 | 0.8947 |
| close3 | closing 3 px | 0.9812 | 0.8734 | −0.0522 | 0.8956 |
| r4 | tight, seed 1 | 0.9812 | 0.8676 | −0.0580 | 0.8956 |
| r6 | tight, resnet18 | 0.9812 | 0.8538 | −0.0718 | 0.8805 |
| r3 | tight, silhouette only | 0.9812 | 0.8235 | −0.1021 | 0.8965 |
| **dil32** | dilate 32 px | 0.9812 | **0.8105** | −0.1151 | 0.9195 |
| r5 | no blackout | 0.9812 | 0.7982 | −0.1274 | 0.8876 |

**Arm C reaches parity with end-to-end MaskDINO while localizing 2.9 points better:**

```
classagnostic_f1   0.9812  vs  0.9521    +2.91
classaware_f1      0.9175  vs  0.9256    -0.81
```

`classagnostic_f1` is 0.9812 in **all twelve** runs — every run shares the same frozen
detector, and the exported masks are the detector's own (dilation affects only the crop fed
to the classifier). Both required and a clean consistency check.

### The decisive variable: padding the predicted mask

```
dilate  0  (tight)      0.8734 - 0.8824      <- 8 configurations, nothing escapes this band
dilate  3               0.9077   +2.5
dilate  8               0.9003   +1.8
dilate 16               0.9175   +3.5        <- best
dilate 32               0.8105   -6.5        <- cliff
no blackout (box)       0.7982   -7.8        <- worst
closing 3 / 16          0.8734 / 0.8799      <- no benefit at either radius
```

Three things this says:

1. **Dilation helps in a window, roughly 3–16 px**, and is worth more than trunk size,
   augmentation recipe, input modality, seed and background handling *combined*.
2. **Closing never helps.** It fills gaps and merges limbs without moving the silhouette,
   and gains nothing at either radius. So the benefit is *growth* — recovering real
   instrument pixels the detector clipped — not repairing a fragmented mask.
3. **There is a cliff past ~16 px.** At 32 px, adjacent instruments in a pile dilate into
   one another and a crop contains its neighbour.

**Upstream implication.** The frontend's class-agnostic recall is 0.98, but recall counts an
instance found at `IoU >= 0.5` — it is blind to mask *quality*. On the forceps the masks
arrive at IoU 0.55–0.83, under-covering the instrument. Dilation is a workaround; the
principled fix is better masks from `single_class_v2`.

**Caveat: every dilation point is n=1** and the 3/8/16 ordering is non-monotonic. A seed
repeat at 16 px is running.

## 3. The finding: forcep03/forcep04 collapse into one class

| run | config | forcep03 | forcep04 | **pair sum** |
|---|---|---|---|---|
| r1 | rotations only | 0.6142 | 0.7926 | 1.407 |
| r2 | full aug, seed 0 | 0.9213 | 0.4148 | 1.336 |
| r3 | silhouette only | 0.9134 | 0.5259 | 1.439 |
| r4 | full aug, seed 1 | 0.9370 | 0.3704 | 1.307 |
| r5 | no blackout | 0.5984 | 0.8222 | 1.421 |
| r6 | resnet18 | 0.9291 | 0.3185 | 1.248 |
| **baseline** | — | 0.9322 | 0.9237 | **1.856** |

**Six configurations — varying trunk size, augmentation, input modality and background
handling — all produce a pair sum of 1.25–1.44. None approaches 1.856.** Whichever member
wins, the pair is worth ~0.45 less than MaskDINO extracts from it.

**Padding the predicted mask is the one thing that moves it.** dil3/dil8/dil16 reach
1.563 / 1.542 / 1.567 — clear of the entire six-run range, closing ~38% of the gap to the
baseline. The crops explain why: the predicted masks arrive **fragmented**, IoU 0.55–0.83,
with shafts rendered as dashed segments and holes through the middle
(`runs/forceps_crops.png`). The discriminating detail between two ring-handled forceps is
at the jaws — the thin end a mask drops first. That is the real,
config-independent deficit, and it accounts for essentially the whole gap: every non-forceps
class lands within 0.2–2.3 points of the baseline.

The model can classify forcep03 at 0.93, matching MaskDINO — it simply cannot do both at
once. The capability is there; the discriminating information is not reaching it.

**Which member wins is NOT seed noise.** Both seeds of the full-augmentation config favour
forcep03 (0.9213 / 0.9370) with only 1.6 points of wobble. The 31-point swing between r1 and
r2/r4 tracks the *configuration*, not the seed. (n=1 per config, so association, not cause.)

Consistent with their geometry: median area 1421 vs 1469 px, aspect ratio 1.80 vs 1.69, fill
0.072 vs 0.070 — all within 6%.

## 4. What mattered, ranked (`classaware_f1`, against r2 tight = 0.8758)

> **The plain-dilation rows below are confounded — see section 8.** Until 2026-11-05
> `cropcls/export.py` passed no `box=`, so the scoring path cropped the *dilated* tight box
> while training cropped the undilated one: wider by `2 * mask_dilate` px per axis. Measured
> over the 4623 crops in `crops/train`, the mean long-axis ratio is 1.04x at dilate 4, 1.08x at
> 8, 1.16x at 16 and 1.33x at 32. The model was therefore scored on a wider framing than it
> trained on, by an amount that grows with the radius, so the larger radii carried the bigger
> handicap. `dilate 32 = -6.5` is the row to distrust; `dilate 16 = +4.2` was achieved *despite*
> the mismatch and is if anything understated. The baseline, both closing rows and the
> blackout/RGB rows are unaffected: the baseline and those ablations run at `mask_dilate: 0`,
> and `close_mask` does not grow the outer silhouette, so its tight box is unchanged.

| change | effect | reading |
|---|---|---|
| **dilate the mask 16 px** | **+4.2** | largest effect found, by a wide margin |
| dilate 3 px | +3.2 | your first guess; already most of the benefit |
| dilate 8 px | +2.5 | — |
| rotations-only vs full aug (r1) | +0.7 | inside seed noise |
| closing 16 px | +0.4 | not a real gain |
| closing 3 px | −0.2 | nothing |
| seed (r2 vs r4) | ±0.8 | — |
| resnet50 → resnet18 | −2.2 | capacity is not the binding constraint |
| remove RGB | −5.2 | texture is load-bearing despite ~95%-zero crops |
| **dilate 32 px** | **−6.5** | over-dilation merges neighbours in piles |
| remove blackout | −7.8 | a rectangle of real background is the worst input |

**Overfitting is present but is not the binding constraint.** r1 ends at train loss 0.0095
against val loss 0.3492, val loss rising from iteration 1500, val accuracy flat from 2500.
Yet r2, which barely memorizes, lands in the same place, and reducing capacity (r6) makes
things worse. More regularization alone would not have closed the gap — padding the mask did.

## 5. Methodology notes, including what I got wrong

**Crop top-1 is unusable for ranking runs — it misleads in BOTH directions.** It scores only
crops that passed the assignment rule (`IoU >= 0.5` AND `runner_up < 0.2`), i.e. clean,
isolated instances; `classaware_f1` scores everything, including crowded and occluded cases,
and counts false positives. Three times it *understated* an effect ~10x (mask-only, no
blackout). Once it *flattered* one badly: dil32 is tied-best on crop top-1 at 0.9195 and
second-worst on `classaware_f1` at 0.8105 — because 32 px of padding only hurts in crowded
scenes, which crop top-1 filters out by construction. close16 likewise looked like a gain
(0.9124) and was not (0.8799). I ranked runs by it repeatedly and had to retract each time.

**Crop top-1 and `classaware_f1` are not interchangeable.** r1 and r3 differ by 0.35 points on
crop top-1 but **5.9 points** on `classaware_f1`. Crop top-1 only scores crops that passed the
assignment rule (`IoU >= 0.5` and `runner_up < 0.2`) — clean, isolated instances. The Hungarian
protocol scores everything, including crowded and occluded instances, and counts false
positives. An occluded *silhouette* is far more ambiguous than an occluded *textured* crop, so
the filtered metric flatters mask-only by excluding exactly the cases where it fails.
**Report `classaware_f1`; treat crop top-1 as a training monitor only.**

**Seed variance at convergence is small.** r2 vs r4 differ only in seed: 0.8947 vs 0.8956 on
crop top-1 (**0.09 points**) and 0.8758 vs 0.8676 on `classaware_f1` (**0.82 points**). The
gap was 2.3 points at iteration 500 and shrank monotonically — an early-training figure is
not a variance estimate for final numbers, and I quoted one as such for several hours.

**Per-class forceps numbers ARE reproducible across seeds** (1.6 points on forcep03, 4.4 on
forcep04) but swing 31+ points across configurations. I initially attributed the swing to
"initialization and data order"; the seed repeat disproved that.

**Two bugs worth keeping fixed.** (a) `--eval-only` shared the training run's output dir and
overwrote its `config.json` with the eval's defaults, destroying the provenance of all three
round-1 runs. (b) The eval then rebuilt the model from the base YAML rather than the run's
own config, so evaluating r6 built a resnet50 and died on a size mismatch against resnet18
weights — and `hungarian.sh` piped stderr through a grep filter, so the traceback matched
nothing and the log came out empty. Silence looked like success.

**The score formula couples localization to classifier confidence.**
`score = objectness * class_prob` is MaskDINO parity, but an undertrained classifier emits
softmax ~0.35, pushing good masks below `SCORE_THRESH 0.5` and depressing even the
*class-agnostic* metric by 21 points. At convergence both score modes agree to four decimals,
so this only bites on early checkpoints. `score_mode="detector"` gives the clean localization
readout.

**GPU contention dominated the first timing estimates.** The same detector forward is
629 ms/frame on a 95%-contended Quadro RTX 6000 and 125 ms/frame on an idle RTX 2080 Ti. My
original "18.6 hours, caching is a prerequisite" conclusion was mostly an artifact of the
shared card; the real figure is ~5.4 h uncached, and caching is an optimization.

**convnext_tiny beats resnet50 by 5.68 points at crop 448.** On `dbe-cia-sl19-01`,
2026-10-31. The run now at `runs/20261031121447_cxt_c448_dil0_lr1e4` was originally created as
`..._r50_c448_dil0_lr1e4` despite being configured `trunk: convnext_tiny`; the real control was
re-run as `..._REAL_r50_...`. **Both directories were renamed on 2026-10-31 to match their
configs** (`cxt` / `r50`, `REAL_` dropped) and their stored `output_dir` patched to match, so
the names below are the current ones. Both configs are otherwise identical (`base_lr 1e-4`,
crop 448, `dilate_px 0`, `freeze_trunk false`, 4000 iter, seed 0,
`train_seta_mm` / `val_seta_mm`):

| run dir | `trunk` | `classaware_f1` | `classagnostic_f1` |
|---|---|---|---|
| `20261031121447_cxt_c448_dil0_lr1e4` | `convnext_tiny` | 0.9391 | 0.9808 |
| `20261031140925_r50_c448_dil0_lr1e4` | `resnet50` | 0.8823 | 0.9808 |

Both `score_mode=detector`. The identical `classagnostic_f1` confirms the same frozen detector
and the same masks, so the whole gap is the trunk. n=1: this sits outside the six-run round
above and has not been seed-repeated.

**Which chain drove which run.** The sweeps were run as chained background commands that wrote
`TRAINED` / `SCORED` / `ALLDONE` progress markers to `runs/*.out`, while each step's real stdout
went to its own `runs/*.log`. Both sets were transient and were cleaned up on 2026-10-31; the
table below is transcribed from them and is now the only record of the grouping:

| chain | runs it drove |
|---|---|
| `sweep_dil0` / `dil8` / `dil32` / `dil64` | the four `20261030122636_c448_dil*` |
| `stem_r50_224` / `stem_r50d_224` / `stem_r50d_448` | the three `20261031115041_*` |
| `ctrl_chain` (the only one with an `ALLDONE`) | scored the convnext run above, then trained the r50 control |

`c448_dil4` and `c448_dil16` predate the sweep batch and had no marker file.

## 6. What to try next

1. **Seed-repeat the winning radius.** Every dilation point is n=1 and 3/8/16 is
   non-monotonic. One repeat at the best radius settles whether 16 is genuinely better than
   3 or they are the same within noise.
2. **Round 4, running:** `close3` / `close16` (morphological closing — repairs the mask
   without moving the silhouette, verified bbox-identical) against `dil32` (does the curve
   turn over?). close-vs-dilate at matched radius separates *repair* from *growth*, which
   decides whether the fix belongs in the mask pipeline (nearly free) or is a crop-design
   parameter to tune.
3. **Better masks at source.** If repair is the mechanism, the real fix is upstream: the
   frontend's masks are fragmented at IoU 0.55–0.83 on this pair even though its
   class-agnostic recall is 0.98. Recall counts an instance as found at IoU >= 0.5; it says
   nothing about mask *quality*, and quality is what the classifier needs.
4. **Look at `runs/forceps_crops.png`.** forcep03 and forcep04 are both ring-handled forceps
   with identical topology. Whether they are separable in principle, by someone who knows
   the instruments, bounds everything above.

## 7. Needs a decision

1. **Regenerate the scene pool at 5–10 instruments?** Your generator, and it would evict the
   current 1024 frames. I would now say **don't, yet**: six configurations produce the same
   forceps collapse, so this is an information problem and more frames of an indistinguishable
   pair will not fix it. Resolve the pair question first.
2. **Snapshot the current 1024 frames** to a static directory before any regeneration.
3. **Nothing is committed** — all `cropcls/` work is untracked on branch 42.

## 8. Crop padding (`pad_frac`), 2026-11-05

`pad_frac` grows the crop region by a fraction of the mask's tight box on each side, per axis.
Unlike `mask_dilate` it widens the **field of view** and leaves the mask alone, so under
`blackout: false` the border is real background — the pixels the detector's mask clipped.
Ground truth for the ask: `requirements_crop_padding.md`. Spec: `spec_static_crops.md` section 3.

### 8.1 The folder

`crops/train_pad20`, extracted at `pad_frac: 0.20`, which is a **ceiling**: the loader crops to
its own `cfg.pad_frac` within it, so one folder serves every fraction at or below 0.20,
including the tight control at 0.0. Both measured on `dbe-cia-sl19-01`:

| folder | `pad_frac` | size | crops | frames | time |
|---|---|---|---|---|---|
| `crops/train` (old, tight) | — (`margin: 0`) | 186 MB | 4623 | 1024 | 769 s, 751 ms/frame, load unrecorded |
| `crops/train_pad20` | 0.20 | **312 MB** | 4623 | 1024 | 1417 s, **1384 ms/frame, GPUs 5-6 at ~100% from another job** |

Same `frame_digest` (`0d90fc4eb1bf`) and the same 4623 crops, as expected from a frozen
detector and an unchanged assignment rule. 312 MB against 268 MB projected from 1.44x the
pixels — PNG compresses a mostly-background border better than pixel count implies, so scaling
bytes by area ran 15% high. `min_border_px` is 1; no crop has a 0 px border.

**`crops/train` is no longer loadable** — it predates `pad_frac` and `CropFolder` raises rather
than defaulting to 0 and training tight under a padded config. The `pad_frac: 0.0` control
therefore comes off `crops/train_pad20` too. Verified byte-identical first: 16 crops sampled
across classes, `crops/train`'s PNGs equal `crops/train_pad20` re-tightened to its own alpha,
and the filename sets match — which is what makes the two pre-existing tight runs valid
controls instead of needing a fresh pair.

### 8.2 Verdict: 0.10 is nothing; 0.05 on resnet50 is borderline and unresolved

`classaware_f1`, `score_mode=detector`, c224, `blackout: false`, `mask_channel: true`,
4000 iters, all off `crops/train_pad20`. The `pad_frac: 0.0` rows are controls; the two
convnext and two resnet50 controls at seeds 0-1 are pre-existing `mask4_*` runs, not re-run.

| trunk | `pad_frac` | seeds | runs | mean | vs 0.0 | arm spread |
|---|---|---|---|---|---|---|
| resnet50 | 0.0 | 0,1,2,3 | 0.9140 / 0.9182 / 0.9115 / 0.9224 | 0.9165 | — | **1.09 pt** |
| resnet50 | 0.05 | 0,1,2,3 | 0.9257 / 0.9215 / 0.9282 / 0.9281 | **0.9259** | **+0.94 pt** | 0.67 pt |
| resnet50 | 0.10 | 0,1 | 0.9115 / 0.9182 | 0.9149 | −0.17 pt | 0.67 pt |
| convnext_tiny | 0.0 | 0,1 | 0.9249 / 0.9240 | 0.9245 | — | 0.08 pt |
| convnext_tiny | 0.05 | 0,1 | 0.9249 / 0.9341 | 0.9295 | +0.50 pt | 0.92 pt |
| convnext_tiny | 0.10 | 0,1 | 0.9299 / 0.9124 | 0.9211 | −0.33 pt | 1.75 pt |

**0.10 buys nothing on either trunk. 0.05 on resnet50 is borderline: +0.94 pt at p = 0.057,
4 seeds against 4.** Everything else overlaps its control.

**This number moved twice, and the history matters more than the endpoint.** At 2-vs-2 the
control was 0.9140/0.9182 and the arm 0.9257/0.9215 — complete separation, +0.75 pt, recorded
here as "suggestive". Control seeds 2 and 3 then came in at 0.9115 and **0.9224**, widening the
control from 0.42 pt to 1.09 pt and putting 0.9224 above the arm's 0.9215; that was recorded as
a retraction, "the effect was a narrow control, not the padding". Arm seeds 2 and 3 then scored
0.9282 and 0.9281, lifting the mean to 0.9259 and the gap to +0.94 pt, exact two-sided
permutation **p = 0.0571** on 4 vs 4.

So the retraction was too strong. The honest reading is **borderline and unresolved**: short of
the 0.05 threshold, one overlapping pair (control max 0.9224 against arm min 0.9215), but a
consistent direction across all four seeds and the strongest candidate anywhere in this
investigation. Two more seeds per arm would settle it either way; nothing smaller will.
The lesson is the one section 5 already records — at 2 seeds per arm, neither a separation nor
an overlap is evidence.

**n=2 cannot estimate its own spread, and this is the demonstration.** The convnext controls
look reassuringly tight at 0.08 pt, but convnext `pad_frac=0.10` spans 1.75 pt on the same two
seeds. A 0.08 pt spread is not evidence of low variance; it is one draw. Treat anything under
~1.1 pt at c224 as noise, which is wider than the 0.82 pt this file records in section 5.

**Both trunks agree on the direction at 0.10**: −0.17 pt and −0.33 pt, consistent with the
resolution trade — the silhouette drops from 100% to 83% of the canvas edge — but inside noise
either way.

Best single number in the table is convnext `pad_frac=0.05` seed 1 at 0.9341. It is one seed,
and its sibling scored 0.9249, exactly the control. Do not quote it.

`classagnostic_f1` is 0.9808 on all 14 runs, identical to four decimals. The detector is frozen
and `score_mode=detector` keeps scores classifier-independent, so localization cannot move and
only class assignment can. It is the check that the eval isolates classification.

Crop top-1 ranked the resnet50 arms in a different order than `classaware_f1` did, as usual.

### 8.3 Per class: one candidate, not significant, and not the one that matters

The evaluator emits no per-class F1 or precision — `misclassified_per_class` is keyed by **GT**
class, so it cannot give precision. These numbers come from
`inference_detector/instance_matching_confusion.csv` (GT rows, predicted columns, plus a
false-negative column and false-positive row), which does: `TP = M[c][c]`,
`FN = rowsum - TP`, `FP = colsum - TP`.

Pooled across both trunks: 6 control runs against 8 padded (0.05 and 0.10 together), exact
two-sided permutation test on the difference of means.

| class | control | padded | delta | control seed spread | p |
|---|---|---|---|---|---|
| scarstick01_mm | 0.9452 | 0.9614 | **+1.62 pt** | 2.71 pt | **0.0147** |
| hammer01_mm | 0.9964 | 0.9908 | −0.56 pt | 0.92 pt | 0.0586 |
| forcep04_mm | 0.8269 | 0.8339 | +0.70 pt | 4.76 pt | 0.49 |
| sharpspoon01_mm | 0.9753 | 0.9795 | +0.42 pt | 1.80 pt | 0.34 |
| forcep03_mm | 0.8151 | 0.8190 | +0.39 pt | 5.83 pt | 0.66 |
| forcep01_mm | 0.9045 | 0.9078 | +0.33 pt | 4.13 pt | 0.73 |
| scalpel01_mm | 0.9658 | 0.9638 | −0.20 pt | 1.29 pt | 0.64 |
| forcep02_mm | 0.9119 | 0.9115 | −0.04 pt | 3.99 pt | 0.97 |

**Per-class noise dwarfs aggregate noise.** The control's own seed spread is 5.83 pt on
forcep03 and 4.76 on forcep04, against 1.09 pt on aggregate `classaware_f1`. Any per-class
claim under ~4 pt on the forceps is unreadable at these seed counts.

**scarstick01_mm is the only candidate, and it does not survive correction.** p = 0.0147 is
nominal; with 8 classes tested the Bonferroni threshold is 0.00625, and the chance of seeing
something this strong somewhere among 8 classes is about 11%. What makes it more than a lucky
p-value is that it is consistent in sign in all four trunk x fraction cells (+0.48 to +2.13 pt)
and 6 of 8 padded runs beat the best of 6 control runs. What makes it less than a result is
that nothing else moves and the aggregate is null.

**Its mechanism is specific and legible.** The gain is entirely precision — recall was already
0.9975, i.e. essentially every scarstick is found with or without padding:

| | control | padded | delta |
|---|---|---|---|
| precision | 0.8984 | 0.9283 | **+2.99 pt** |
| recall | 0.9975 | 0.9971 | −0.03 pt |
| things wrongly called scarstick | 14.8 | 10.1 | **−4.7** |

Broken down, the predictions that stop being called scarstick are `scalpel01` (5.8 -> 4.2),
`sharpspoon01` (5.8 -> 4.6) and spurious detections matching no GT (1.8 -> 0.2). So padding
helps the classifier stop calling *other elongated instruments* scarstick. That fits the
geometry: scarstick is by far the most elongated class (mean aspect **5.33**, next is scalpel at
3.15 and sharpspoon at 2.78), and surrounding context is what separates a bare thin shaft from
a blade or a spoon. Note the obvious alternative story is **not** supported — if padding worked
by recovering clipped instrument, scarstick's silhouette should sit on its tight-box edge more
than other classes, and it sits there *least* (0.025 of box perimeter, against 0.046-0.055 for
the forceps).

**The errors move, they do not disappear.** `false_positives_total` (20), `false_negatives_total`
(26) and `num_mask_matched_total` (1175) are identical to two decimals across all 14 runs, as
they must be with a frozen detector and `score_mode=detector`. `misclassified_total` goes
73.83 -> 70.12, a 5% relative reduction worth +0.32 pt of class-aware recall — and a permutation
test on it gives **p = 0.4376** (control 67-83, padded 56-83). Not a result.

**Padding does not touch the actual problem.** Section 3's finding is the forcep03/forcep04
collapse, and those two sit at 0.815 and 0.827 F1 — far the worst classes — with deltas of
+0.39 and +0.70 pt against seed spreads of 5.83 and 4.76. Whatever padding does, it does not
help the pair that is costing the most.

### 8.4 Pending and not done

- resnet50 `pad_frac=0.05` seeds 2 and 3 are training, to make that arm 4-vs-4 against its
  control. They can only narrow the conclusion above, not reverse the retraction: the control
  already reaches 0.9224.
- `pad_frac` untried at c448, where the resolution trade should be cheaper and where convnext
  scores highest (0.9474/0.9499 tight).
- The dilation sweep in section 4 should be re-run now that `export.py` frames crops the way
  training does, since `pad_frac` is a cleaner way to widen the view than dilation was.
- Nothing is committed.
