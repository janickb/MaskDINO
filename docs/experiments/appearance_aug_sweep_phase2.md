# Which augmentation axis actually fixes the forcep03 collapse?

Phase-2 appearance/geometry sweep, 21 runs on `dbe-cia-sl23-01`, 2026-10-05 -> 10-09
(~101 GPU-hours). Follow-up to [scale_jitter_ab_phase2.md](archive/scale_jitter_ab_phase2.md), which
closed the scale question ("scale coverage is not the explanation") and left forcep03 open.
This answers the next one: **one axis - a per-instance photometric offset calibrated to real
tool pixels - is the whole effect. Everything else tested is null or negative.**

## Protocol

All arms launched by [tools/augmentation/run_aug_sweep.sh](../../tools/augmentation/run_aug_sweep.sh),
which sets the shared protocol on the command line so no arm file can drift on anything but
its own dose:

| | |
|---|---|
| base config | [..._reclassify_unfreeze_encoder_decoder.yaml](../../configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_unfreeze_encoder_decoder.yaml) (encoder+decoder+head trainable, backbone frozen) |
| train -> eval | `reclass_setab_mm` -> `val_setab_mm` @ `TEST_SAMPLE_STRIDE 4` = 256 images, 1003 GT instances |
| budget | `MAX_ITER 10000`, `IMS_PER_BATCH 3` (3 GPUs x 1 img - 2 img/GPU hangs during eval on an 11 GB 2080 Ti), `BASE_LR 1e-5`, seed 42 unless stated |
| scale | `MIN_SCALE/MAX_SCALE 1.0/1.0` in every arm (the no-op; no pad band) |
| score | `instance_matching/classaware_f1` averaged over the last 2 eval ticks (`compare_runs.objective_value`) |
| wall clock | ~1 h 40 per arm |

**Noise floor.** Four control runs (seeds 42/43/44 plus a seed-42 repeat) give
**0.8783 +/- 0.0046** (1 sigma). Every delta below is quoted in that sigma, and nothing under
~2 sigma is treated as real.

## All arms

| arm | score | delta | sigma | what it changes vs control |
|---|---|---|---|---|
| `abl_jitter_persp12` | 0.9059 | +0.0276 | +6.0 | jitter + per-instance tilt 12 deg |
| `abl_jitter_valcal` | 0.9037 | +0.0254 | +5.5 | jitter, centres solved against **val** renders (leaky, diagnostic only) |
| **`abl_jitteronly`** | **0.9030** | **+0.0247** | **+5.3** | **per-instance jitter alone - the shippable recipe** |
| `abl_jitter_persp5` | 0.9024 | +0.0241 | +5.2 | jitter + tilt 5 deg |
| `aug_realism` s42 | 0.9018 | +0.0234 | +5.1 | jitter + frame-level realism bundle |
| `aug_realism` s43 | 0.8996 | +0.0213 | +4.6 | same, seed 43 |
| `abl_jitter_persp30` | 0.8996 | +0.0212 | +4.6 | jitter + tilt 30 deg |
| `abl_jitter_persp60` | 0.8964 | +0.0181 | +3.9 | jitter + tilt 60 deg |
| `abl_jitter_symmetric` | 0.8870 | +0.0087 | +1.9 | jitter re-centred on 1.0, same widths |
| `aug_perspective_tilt30` | 0.8827 | +0.0044 | +0.9 | tilt 30 deg alone |
| control (mean of 4) | 0.8783 | - | - | today's pipeline |
| `aug_clusteroff` | 0.8681 | -0.0103 | -2.2 | `CLUSTER_PROB 1.0 -> 0.0` |
| `abl_nojitter` | 0.8671 | -0.0113 | -2.4 | frame-level realism bundle **without** jitter |
| `aug_perspective_tilt60` | 0.8580 | -0.0203 | -4.4 | tilt 60 deg alone |

The winning recipe is
[..._aug_abl_jitteronly.yaml](../../configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_unfreeze_encoder_decoder_aug_abl_jitteronly.yaml):
per pasted instance, brightness `[0.72, 0.88]`, contrast `[1.50, 1.90]`, saturation
`[0.9, 1.3]`, hue `4 deg`, blur sigma `[0.0, 0.5]` - centres 0.80/1.70 calibrated to land
synthetic tool-pixel mean/std on the real-photo target (130.6 / 53.7).

## What the gain is made of

It is a **classification** fix, not a detection fix. Final tick, seed 42:

| | misclassified | false negatives | false positives | correct class | mean IoU |
|---|---|---|---|---|---|
| control | 44 | 118 | 19 | 841 / 1003 | 0.851 |
| jitteronly | 24 | 117 | 18 | 862 / 1003 | 0.850 |

Misclassifications roughly halve; FN, FP and mask quality do not move. And it is almost
entirely one class pair - the one the whole line of work was chasing:

| GT `forcep03_mm` | correct | -> `forcep04_mm` | FN | correct / detected |
|---|---|---|---|---|
| control | 0.233 | 0.417 | 0.300 | 33% |
| jitteronly | 0.550 | 0.100 | 0.333 | **82%** |

forcep03 is the most seed-stable class in the set (control runs: 0.217 / 0.233 / 0.250 / 0.233),
and the split is perfect across all 17 ten-thousand-iter arms: **every** non-jitter arm sits at 0.100-0.250, **every**
jitter arm at 0.500-0.567. Not noise.

## The direction matters, not the variation

`abl_jitter_symmetric` keeps the exact widths but re-centres on 1.0: score +1.9 sigma (marginal)
and forcep03 dead flat at 0.233. So the mechanism is not "any per-paste variation breaks the
~2-cached-crops-per-class memorization" - it is the **systematic offset**: synthetic instruments
render 18% too bright (154.3 vs 130.6) and too flat (43.2 vs 53.7 internal std), and only an
off-centre range corrects a systematic error. Calibrating to val renders instead of real photos
(`valcal`, +5.5 sigma) is within noise of calibrating to real photos (+5.3 sigma), so the target
statistic barely matters - being dark and high-contrast at all is what counts. Report
`jitteronly`, never `valcal`: its centres came from the eval set.

## Null and negative axes

- **Frame-level realism** (narrowed per-image brightness/contrast, widened saturation, blur,
  sensor noise) is worth **nothing on its own** - `abl_nojitter` lands 2.4 sigma *below* control -
  and adds nothing on top of jitter (`realism` 0.9018 vs `jitteronly` 0.9030). Drop the bundle,
  keep the one axis.
- **Per-instance perspective** is null on control (tilt 30: +0.9 sigma) and harmful when pushed
  (tilt 60: -4.4 sigma, FN 118 -> 164). On top of jitter the whole 5/12/30/60 ladder lands
  0.8964-0.9059, i.e. inside +/-1.5 sigma of jitter alone, with FN rising monotonically with tilt.
  Missing camera-viewpoint diversity is apparently not what limits this model; leave
  `PERSPECTIVE.PROB` at 0.
- **Cluster paste** earns its keep, barely: turning it off costs -2.2 sigma. Keep `CLUSTER_PROB 1.0`.

## The one real cost

`forcep11_mm` recall drops from 0.698-0.849 (all 9 non-jitter arms) to 0.547-0.679 (all 8 jitter arms) -
cleanly separated in the wrong direction, so it is a genuine trade, not seed noise. Its confusion
row shows the loss is **detection** (FN 0.245 -> 0.32), not confusion: darkening pasted instances
makes an already-dim thin tool harder to find. Net arithmetic still favours jitter -
forcep03 +0.30 over 60 instances vs forcep11 -0.17 over 53 - which is the +21 correct-class
instances in the table above. Worth one targeted run (e.g. per-class jitter centres, or
brightness floor raised to 0.85) before shipping.

## Conclusion / next step

1. **Ship per-instance photometric jitter** (`abl_jitteronly` values) as the phase-2 default:
   +0.025 class-aware f1 = +5.3 sigma, free at runtime, and it is the first thing in this whole
   line of work to move forcep03 (33% -> 82% of detected instances classified correctly).
2. **Drop the frame-level realism bundle and the perspective axis.** Keep cluster paste.
3. **forcep03 is now a localization problem, not a confusion problem** - 0.55 correct against
   0.33 FN. The remaining lever there is data (more/differently-posed forcep03 renders), the same
   conclusion the scale A/B reached, but now with the appearance half genuinely closed.
4. Re-check `forcep11` under jitter before this becomes the default (see above).

## Superseded / open

- **The 2026-10-05 wave (3 runs, 6000 iters, batch 1) is not comparable to anything here** and is
  superseded by the 10-06 batch-3 controls. At batch 1 the seed spread alone is 0.553-0.649
  (sigma 0.049, **10x** the batch-3 floor) - no augmentation effect in this study would have been
  visible through it. `20261006064018` (12000 iters, batch 4, 0.8870) changes two protocol knobs
  at once and is likewise outside the comparison.
- **Wave 0, the scale A/B rerun on top of the `FixedSizeCrop seg_pad_value` fix, never ran.**
  Every arm here is pinned at the 1.0/1.0 no-op, so `scale_jitter_ab_phase2.md`'s "keep scale
  jitter on" recommendation is still unconfirmed post-fix, and all numbers in this report are
  *without* scale jitter.
- `MODEL.MaskDINO.TEST.VAL_LOSS.ENABLED` was on for the 10-05 wave and off from 10-06 onward
  (eval-only, no effect on training).

## Reproduce / re-read

```bash
# one arm per invocation; GPU arg may be a comma list ("1,2,3" -> batch 3)
bash tools/augmentation/run_aug_sweep.sh abl_jitteronly 1,2,3

# the three tables this write-up is built from, any set of arms
./.venv/bin/python tools/compare_runs.py runs/*_aug_control_seed4* runs/*_abl_jitteronly_*
./.venv/bin/python tools/compare_runs.py --gt-class forcep11_mm runs/...
```

Gotcha carried over from the scale A/B: training logs contain NUL bytes, so `grep` treats them
as binary and silently matches nothing - pipe through `tr -d '\000'` first.
