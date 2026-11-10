# Spec — phase-1 single-class domain randomization sweep

Verbatim ask: [requirements_phase1_domain_randomization.md](requirements_phase1_domain_randomization.md).

Status at the time of writing: the six arms are **already launched** (2026-11-08 ~01:05) and
are ~40 iterations in. Preflight findings can still be acted on — an arm can be killed,
reconfigured and relaunched at a cost of minutes, not hours.

## 1. Ask — verbatim

Quotation, not a worklist. Never read these as unchecked work items.

- [ ] "I need a better domain randomization for the phase 1 training with single-class detector / segmenter."
- [ ] "the idea would be that the segmenter can understand what a surgical instrumetn looks like, so trian on set a and validate on set a and b and see if this works well. or for instance just validate on set b."
- [ ] "I modified the singleclass.yaml training config a bit such that it does now a larger Augmentation of scaling."
- [ ] "Find a setting which does good ranodmization."
- [ ] "Use the gpus avaialbe report back to me your  findings."
- [ ] "validate only on val_set_ab_mm_singleclass, a setab is already on lambdablade"
- [ ] "accept the drift, the pool is capped at 1024 and deletes / feeds new images chronologicaly, so the datapool is going to be big, note this"
- [ ] "Scale range (Recommended), Photometric dose (Recommended)"
- [ ] "Calibrate first, then decide (Recommended)"
- [ ] "checkout that you can use --num-gpus" / "e.g. --num-gpus 2"
- [ ] "i go afk, just stop and ask questions if you hit a roadblock, use the agents available under .claude agent"

## 2. Decision

A 3x2 grid over the two axes the user picked, one arm per free GPU, all six launched
together.

| arm | `INPUT.MIN_SCALE`/`MAX_SCALE` | `PHOTOMETRIC` brightness / contrast / saturation |
|---|---|---|
| `aug_s_base_p_base` | 0.8 / 1.2 | [0.7,1.3] / [0.7,1.3] / [0.8,1.2] |
| `aug_s_wide_p_base` | 0.5 / 2.0 | [0.7,1.3] / [0.7,1.3] / [0.8,1.2] |
| `aug_s_xwide_p_base` | 0.3 / 2.5 | [0.7,1.3] / [0.7,1.3] / [0.8,1.2] |
| `aug_s_base_p_str` | 0.8 / 1.2 | [0.5,1.5] / [0.5,1.5] / [0.6,1.4] |
| `aug_s_wide_p_str` | 0.5 / 2.0 | [0.5,1.5] / [0.5,1.5] / [0.6,1.4] |
| `aug_s_xwide_p_str` | 0.3 / 2.5 | [0.5,1.5] / [0.5,1.5] / [0.6,1.4] |

`s_base_p_base` is the committed recipe and serves as the control; `s_wide_p_base` is exactly
the user's own edit to `maskdino_R50_surgical_tools_finetune_singleclass.yaml`.

Everything outside `INPUT` comes from `Base-Surgical-Phase1.yaml`, with four overrides applied
identically to all six arms: `SOLVER.IMS_PER_BATCH 3`, `SOLVER.MAX_ITER 20000`,
`SOLVER.CHECKPOINT_PERIOD 100000`, `TEST.EVAL_PERIOD 2000`,
`MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.PERIOD 2000` and `SWEEP_THRESHOLDS
[0.05..0.9]`.

Decisions that need their reasons on record:

- **One GPU per arm at `IMS_PER_BATCH 3`, not `--num-gpus 2` at the committed 6.** Measured:
  batch 6 on one 24 GB card OOMs at 22695 MiB; batch 3 peaks at 17191 MiB. Two GPUs per arm
  would fit batch 6 but only 3 arms fit on the 6 free cards, forcing two sequential waves —
  and the pool FIFOs between waves, so the photometric axis would be confounded by a data
  change. Six simultaneous arms share one pool state. The cost is batch 3 vs 6 at an unchanged
  `BASE_LR 0.0003`, which is uniform across arms and therefore affects ranking only through
  interaction, not through level.
- **`--num-gpus 2` is used for the winner's re-run**, at the committed `IMS_PER_BATCH 6`.
- **Validation is `val_set_ab_mm_singleclass` only**, per the user's answer. `val_set_ab_mm`
  carries 16 instrument classes: the 8 set-A types the pool trains on plus 8 never seen
  (`adapter11_mm, clamp11_mm, forcep11_mm, hammer11_mm, scalpel11_mm, scissor11_mm,
  scissor12_mm, tweezer11_mm`).
- **Periodic evals score 256 frames, not 1024.** `DATASETS.TEST_SAMPLE_STRIDE: 4` is set in
  `Base-Surgical-HDF5.yaml:27` and inherited by every arm (confirmed: `TEST_SAMPLE_STRIDE: 4`
  in each emitted `runs/*_aug_*/config.yaml`). `apply_test_sample_stride` keeps `full[::4]`,
  i.e. a systematic every-4th sample, so it is representative rather than contiguous -
  measured, the subset is 13.3% B-only / 65.2% mixed / 10.9% A-only / 10.5% empty frames
  against the full set's 12.6 / 66.4 / 11.0 / 10.0. Added 2026-11-08 after claim-verifier
  finding 3; the earlier wording claimed full-1024 periodic evals, which was wrong.
- **About half of all GT instances in the eval set are instruments never seen in training.**
  Measured on the eval subset: 521 set-A instances (48.4%) vs 555 set-B (51.6%) of 1076; on
  the full set 2175 vs 2166 of 4341. This is what makes `classaware_f1` here a
  generalization metric rather than a fit metric, and it is an instance-level count because
  the metric is instance-level. (A frame-level count over the *first* 120 frames gives 22.5%
  B-only, nearly double the population's 12.6% - dataset order front-loads B-only scenes, so
  contiguous samples of this set are biased and must not be quoted.)
- **The pool is live and drifting**, per the user's instruction to accept it. Recorded here as
  the user asked ("note this"): capped at 1024 frames with chronological FIFO eviction, so the
  effective training distribution is far larger than 1024 and keeps moving. It held 288 frames
  at 00:48 and 322 at 00:52 during calibration.

## 3. Requirements

- [x] Only `INPUT` differs between arms. **Verify:** `diff` any two of
  `configs/coco/instance-segmentation/augmentation/phase1/aug_*.yaml`; the only differing lines are
  `MIN_SCALE`, `MAX_SCALE`, `BRIGHTNESS`, `CONTRAST`, `SATURATION`, `OUTPUT_DIR` and the
  header comment.
- [x] Every arm trains on the pool and validates on the A+B set. **Verify:** each
  `aug_*.yaml` has `TRAIN: ("train_phase1_pool_set_a_mm_singleclass",)` and `TEST:
  ("val_set_ab_mm_singleclass",)`; confirmed in each run's emitted `runs/*_aug_*/config.yaml`.
- [x] Both axes the user selected are **swept**, and no other axis **varies between arms**.
  **Verify:** the six configs span exactly 3 scale levels x 2 photometric levels; and in the
  six emitted `runs/*_aug_*/config.yaml` the effective values of `RANDOM_ROTATION`,
  `ROTATION_ANGLES`, `ROTATION_EXPAND`, `PHOTOMETRIC.NOISE_PROB`,
  `PHOTOMETRIC.GRAYSCALE_PROB` and `COPY_PASTE.ENABLED` are identical across all six.
  - Amended 2026-11-08 after preflight BLOCK finding 1. The original wording said "covered,
    and only those" and verified only that no `aug_*.yaml` *sets* `ROTATION_*`. That clause
    could not fail: `cfg.INPUT.RANDOM_ROTATION` defaults to `True` (`maskdino/config.py:235`)
    and no config in the inheritance chain turns it off, so all six arms do apply a
    90/180/270 degree `T.RandomRotation` (confirmed: `RANDOM_ROTATION: true` at line 102 of
    each emitted `config.yaml`). Rotation is **inherited from the committed recipe, held
    constant across arms, and deliberately not swept** - it is the recipe's replacement for
    `RandomFlip`, which is excluded because instruments have fixed handedness. Constant
    across arms means it cannot confound the comparison; the fix is therefore to the wording
    and the Verify clause, not to the arms.
- [x] Batch size was calibrated before the sweep was sized, not guessed. **Verify:**
  `IMS_PER_BATCH 6` OOM at 22695 MiB, `3` at 17191 MiB / 1.11 s/iter, `2` at 11771 MiB /
  0.85 s/iter, all three on one Quadro RTX 6000 on `dbe-cia-sl19-01` with GPUs 5-6 loaded by
  another user, on the pure-PyTorch deform-attn fallback.
  - Eval cost, corrected 2026-11-08 after claim-verifier finding 1: the converged figure is
    **0.421 s/frame** (`Total inference time: 0:01:45.685464 (0.421058 s/iter)`, N=256), not
    the 0.44 I first quoted off a still-converging running average at frame ~133/256.
  - Sweep completion is gated by the **slowest** arm, not the median: `aug_s_xwide_p_str`
    holds 1.32 s/iter against the other five at 1.215-1.252 (claim-verifier finding 2), so
    20000 iters is ~7.3 h for it, plus ~10 x 256-frame evals. Any ETA must quote that arm.
  - The 323 MiB checkpoint size is corroborated **indirectly** (claim-verifier finding 4):
    the single-class calibration checkpoints were deleted for quota, so the figure comes from
    structurally identical 8-class phase-1 checkpoints at 338723069 B. Treat as ~323 MiB.
- [x] The six arms share one pool state. **Verify:** `runs/*_aug_*/log.txt` start timestamps
  all fall inside one minute.
- [ ] No arm writes a checkpoint that outlives it. **Verify:** after the driver exits,
  `find runs -name 'model_final.pth' -newermt '2026-11-08 01:00'` returns nothing.

## 4. Done when

- [ ] The six arms are ranked on `val_set_ab_mm_singleclass` by **`classaware_f1`** from
  `HungarianInstanceEvaluator` at a **common iteration**. **Verify:** for each of the six
  `runs/*_aug_*/metrics.json`, the `classaware_f1` value at the highest iteration all six
  reached; the ranking table cites that iteration explicitly.
  - Primary metric rationale: single-class collapses the class axis, so
    `classaware_* == classagnostic_*` (`hungarian_evaluator_metrics.md:112`) and the key
    measures localization alone. `score_mode` does **not** exist on this branch — it is absent
    from both `maskdino/evaluation/hungarian_instance_evaluation.py` and
    `tools/score_predictions.py` — so CLAUDE.md's `score_mode=detector` guidance cannot be
    applied here, and the fixed operating point is `SCORE_THRESH 0.5`.
- [ ] The ranking is cross-checked against **`sweep/f1_max`**, and any arm whose rank differs
  between the two is reported as calibration-driven rather than quality-driven.
  **Verify:** both columns appear in the results table for all six arms.
  - Amended 2026-11-08 after preflight BLOCK finding 2. This item originally named
    `sweep/classaware_f1_max`, which **can never be emitted here**: `_sweep_summary` is called
    with `classaware=self._check_classes` and `_check_classes = self._num_classes > 1`
    (`hungarian_instance_evaluation.py:239`), so with `NUM_CLASSES == 1` the whole
    `sweep/classaware_*` group is skipped - `hungarian_evaluator_metrics.md:134` labels it
    "Multi-class only". `sweep/f1_max` is the correct field and loses nothing, because under
    single-class the class-aware and class-agnostic values coincide anyway
    (`hungarian_evaluator_metrics.md:112`). The Verify clause as first written could not pass.
  - Confound and magnitude: a fixed-threshold F1 moves with score calibration alone.
    `hungarian_evaluator_metrics.md` states that a change which only shifts calibration
    ("learning rate, **augmentation**, loss weights") moves those numbers "without making the
    model better or worse at the threshold it would actually be deployed at"; CLAUDE.md
    records the coupling costing **21 points** on even the class-agnostic metric. The sweep
    re-matches at ten thresholds over one cached IoU matrix, so it is the confound-free read.
- [ ] COCO AP is **not** used to rank arms. **Verify:** no `segm/AP` value appears in the
  ranking table.
  - Confound: COCO AP averages over the **eval set's** category count. It is sound here only
    because the `_singleclass` view collapses GT to one category; the same model scored
    against the 16-category `val_set_ab_mm` would report AP/16. The failure is on record
    (single_class_v1 read 1.37/0.64 instead of 92.3/64.8).
- [ ] Every number reported to the user has been re-derived by `claim-verifier` from the run
  artifacts, not from this session's transcript. **Verify:** a `claim-verifier` report exists
  covering the calibration table and the ranking table, and names anything it could not
  re-derive.
- [ ] The winning setting is re-run at the committed recipe. **Verify:** a run exists with
  `--num-gpus 2`, `IMS_PER_BATCH 6` and the winner's `INPUT` block, and its emitted
  `config.yaml` shows `IMS_PER_BATCH: 6`.
- [ ] The winner is confirmed on the **full** val set, not only the strided subset.
  **Verify:** an eval of the winning arm with `DATASETS.TEST_SAMPLE_STRIDE 1` exists, its log
  reports `images=1024`, and its `classaware_f1` is quoted alongside the 256-frame figure the
  ranking used. Added 2026-11-08 after claim-verifier finding 3: the ranking is a fair
  between-arm comparison on a representative 256-frame systematic sample, but no number
  quoted as "performance on val_set_ab_mm" may come from the subset alone.

**Out of scope (explicit):**

- Rotation axes (`RANDOM_ROTATION`, `ROTATION_ANGLES`, `ROTATION_EXPAND`) — **not swept.** Note
  that rotation is **on** in every arm, inherited from the committed recipe
  (`maskdino/config.py:235` defaults `RANDOM_ROTATION = True`); it is held constant, not
  disabled. Turning it off is out of scope: it would change the committed recipe and remove
  the recipe's only orientation augmentation.
- Noise and grayscale (`PHOTOMETRIC.NOISE_PROB/NOISE_STD/GRAYSCALE_PROB`) — offered and not
  selected.
- Copy-paste compositing (`INPUT.COPY_PASTE.*`) — offered and not selected.
- `RandomFlip` — deliberately absent from `build_transform_gen` because instruments have fixed
  handedness. Not to be added.
- A B-only validation set, whether by new registration code or by a new render — explicitly
  narrowed by the user to "validate only on val_set_ab_mm_singleclass".
- Any change to `cropcls/`, to the reclassification line of work, or to the multiclass leaf
  config.
- Any change to `Base-Surgical-Phase1.yaml` or `Base-Surgical-HDF5.yaml`: the arms override
  from the leaf so the committed recipe stays untouched.
- Re-enabling the compiled `MultiScaleDeformableAttention` kernel.
