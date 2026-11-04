# cropcls — invariants and traps

The classifier half of Arm C: a frozen class-agnostic MaskDINO supplies instance masks,
and a CNN **trunk** + global pool + linear head assigns the class to each cropped,
blacked-out instance. See `../spec_cropcls.md`.

Nothing here registers a detectron2 `META_ARCHITECTURE`, config namespace or dataset.
`maskdino/` is **not modified**. The multi-arch config machinery (`build_base_cfg`,
`arch_ns`, `_ARCH_CONFIG_NS`) exists only on branch `41-train-on-mask2former` and is
unmerged, so registering an architecture would have blocked this work on that merge.

## What is reused from `maskdino/`

| what | why it is safe to reuse |
|---|---|
| `Hdf5CocoInstanceDatasetMapper` | `@configurable`, so plain-kwargs construction works |
| `mask_iou_matrix` | exact, box-prefiltered; returns float32 on CPU |
| `HungarianInstanceEvaluator` | `distributed=False` kills all collectives |
| `apply_test_sample_stride` | pure function over `DatasetCatalog` |
| dataset registrations | imported via `import maskdino` |

## Invariants, and the test that enforces each

| invariant | why | test |
|---|---|---|
| background is **exactly** 0 after blackout | the model must not learn a background cue | `test_background_is_exactly_zero` |
| photometric runs **pre**-blackout | `RandomLighting` adds a constant colour shift; post-blackout it lifts the background off zero. `RandomContrast` blends toward the crop mean, which on a ~95%-black crop is near zero | `test_photometric_runs_before_blackout_so_background_is_still_zero` |
| rotations are **lossless** | 0/90/180/270 on a square canvas are pixel permutations: no interpolation, so crop and mask stay aligned | `test_rotations_are_lossless_permutations` |
| aspect ratio preserved, never skewed | proportions are discriminative, and p90 bbox AR is 6.0 on set A | `test_aspect_ratio_preserved_not_skewed` |
| an empty mask is **dropped**, not returned | otherwise an all-black crop trains as a real class | `test_empty_mask_is_rejected_not_nan` |
| the eval cfg carries the **classifier's** class count | `HungarianInstanceEvaluator` sets `_check_classes = NUM_CLASSES > 1`; the detector's cfg says 1, so misclassification would be silently reported as 0.0 | `test_eval_cfg_clone_restores_the_class_count` |
| a zero-GT frame returns empty, never throws | such frames exist in this data (e.g. `012676FD8_01-008.hdf5`) and `(K,0).max(1)` raises | `test_zero_gt_frame_returns_empty_instead_of_throwing` |
| an 8-class checkpoint is **rejected** | nothing would crash: the detector would just emit 8 logits and the class-agnostic premise would break invisibly | `test_multiclass_checkpoint_is_rejected` |
| predictions are upsampled; GT is **never** downsampled | at fill 0.070 forcep03/04 are filigree and would thin away | `test_predictions_at_lower_resolution_are_upsampled_not_gt_downsampled` |
| the Grad-CAM input carries `requires_grad` | under `freeze_trunk` every trunk parameter is a non-grad leaf, so nothing in the activation's history requires grad and `autograd.grad` raises. An **unfrozen** trunk works either way, which is what hides it | `test_a_frozen_trunk_still_yields_a_map` |
| a Grad-CAM pass writes no parameter `.grad`, leaks no hook, and restores the train/eval mode | the model object is shared with the optimizer and the in-training val monitor: a stray `.grad` lands in the next step and a forward left in train mode moves the trunk's BatchNorm stats | `test_the_pass_leaves_no_grads_no_hook_and_the_original_mode` |
| the Grad-CAM is normalized per crop | the raw peak is of order 1e-3 (1.6e-03 at resnet50 / crop 224), so an unnormalized map renders black and reads as "the model looks nowhere" rather than as a bug. Only the order is stable - at crop 64 the peak is 1.6e-02 - so no fixed constant can stand in for the per-crop max | `test_the_map_covers_the_crop_and_is_normalized` |

## Traps that cost real time

**`apply_test_sample_stride` is not idempotent.** It removes the catalog entry and
re-registers a plain lambda, so a second call strides the already-strided list (4 -> 16)
and you silently evaluate a quarter of the frames you think you do. `export.py` guards
with a module-level `STRIDE_APPLIED` set.

**The mapper drops `annotations` entirely when `is_train=False`.** The assignment step
needs GT on val as much as on train, so the pipeline always builds it with
`is_train=True` and controls augmentation separately via `cfg.augment_frames`. Using
`is_train=False` would have produced an empty val metric with no error.

**`score = objectness * class_prob` couples localization to classifier confidence.**
It is MaskDINO parity (`sigmoid(class_logit) * mask_score`), but an undertrained
classifier emits softmax ~0.35, so `0.9 * 0.35 = 0.32` falls below `SCORE_THRESH 0.5`
and good masks vanish — depressing even the **class-agnostic** metric by 21 points
(0.9758 -> 0.7619, measured). Use `score_mode="detector"` to read out localization alone.

**The single-class checkpoint was trained by branch-41 code.** Its dumped `config.yaml`
carries `MODEL.MASK_FORMER` keys that this branch does not register, so the merge needs
`set_new_allowed(True)`. The keys are inert for building a MaskDINO.

**GPU contention dominates the timing.** The same detector forward is 629 ms/frame on a
95%-contended Quadro RTX 6000 and 125 ms/frame on an idle RTX 2080 Ti — a nominally
slower card. Any timing claim must name the machine and its load.

**The dataset names in `configs/cropcls/*.yaml` are not registered on every host.** Datasets are
registered under their *folder* name beneath `~/training/images_*`
(`register_all_hdf5_instances`), so `train_seta_mm` / `val_seta_mm` exist only where the folders
are named that. On `dbe-cia-sl19-01` the folders are `train_phase1_pool_set_a_mm` and
`val_set_a_mm` - same 8 classes, same order - so a run there needs
`train_dataset=... val_dataset=...` overrides. The failure is
`ValueError: train_seta_mm has no thing_classes`, whose message blames the missing `training/images`
symlink; with three roots under `~/training/` the real fix is usually
`export SURGICAL_IMAGES_ROOT=<one of them>` **and** the dataset overrides, not the symlink.

**Grad-CAM is a 7x7 map at `crop_size 224`.** The last feature stage is stride 32, so the heat is
blobby by construction and a blob reaching past the instrument is upsampling, not evidence that
the model reads the background. `--gradcam-layer layer3` (`stages.2` on convnext) gives the 14x14
read.
