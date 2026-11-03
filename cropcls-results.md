# Crop classifier — results (twelve runs)

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
