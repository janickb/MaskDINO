# Specification: Two-Stage Instance Classification (frozen segmenter + crop classifier)

Named `cropcls` throughout: the role, not the architecture. The trunk is a pluggable timm
backbone (`resnet50` by default), so neither "unet" nor "resnet" belongs in the name.

## 1. Context & Goals
* **Problem:** The current MaskDINO Model which does Classification and Segmentation shows problems with distinguishing similar objects, further it is not know how it will handle huge amount of classes (e.g. 1000).
* **Objective:** A separate classifier should now do the classification part, it is to be analyzed if it performs better than MaskDINO on its own.
* **User Story:** MaskDINO "single_class" model provides the segmentation Mask from a given image during inference. The Network then has to analyze the image with segmentation mask to find the classification for the segmented parts.

## 2. Technical Architecture & Data Flow
MaskDINO provides a instance Level Segmentation Mask, it always predicts the same class "surgical_instrument". It should be fed to the Classifier cropped along the bbox and also segmented, such that only the segmented pixel are valid and the background is black. The CLassifier then classifies this image.

### Components impacted:
- New top-level package `cropcls/` (answering "find a suitable location"): a plain Python
  package, sibling to `maskdino/`, which is **not modified**. The classifier network itself
  lives in `cropcls/modeling/classifier.py` as `CropClassifier`. Full layout in section 6.

### Data Models / Schema:
- use this phase 1 model to get the segmentation masks: /home/janick.bilang/dev/MaskDINO/runs/20261007130733_single_class_v2/model_final.pth
- use these images to train: /home/janick.bilang/training/images/train_phase1_pool_set_a_mm/
- use these images to validate: /home/janick.bilang/training/images/val_set_a_mm/

**Classes:** 8, from `train_seta_mm` / `val_seta_mm` - `forcep01_mm`, `forcep02_mm`,
`forcep03_mm`, `forcep04_mm`, `scalpel01_mm`, `hammer01_mm`, `sharpspoon01_mm`,
`scarstick01_mm`. (Not the 1-class `*_singleclass` registrations - those are the detector's
label space, not the classifier's.)

**Crop / classifier input contract:**

- crop size **224 x 224**, letterboxed: uniform scale on the longest axis, zero padding, no skew
- classifier input tensor **`(B, 3, 224, 224)`**, float
- channels **3 (RGB)**. The mask is implicit in the blackout - non-zero pixels are the instrument.
  A 4th explicit mask channel is an ablation, not the default.
- normalization: reuse **MaskDINO's own pixel statistics**, which are ImageNet's scaled to 0-255 -
  `PIXEL_MEAN [123.675, 116.28, 103.53]`, `PIXEL_STD [58.395, 57.12, 57.375]`, `FORMAT: RGB`.
  Divide by 255 first if the trunk expects 0-1 inputs.

**Why 224:** ImageNet classifiers are pretrained at 224, so a pretrained trunk operates at the object
scale / receptive-field ratio it was trained for. It also sits at native scale for this data -
median long side is 201 px over 4802 `val_seta_mm` instances, so crops are only mildly resampled
either way, and just 1.8% of instances fall under 16 px on the short axis (at 128 it would be 6.1%,
which is where forcep03/forcep04 discrimination dies).

Two structural facts, not tunable via crop size: pad fraction is ~0.51 (median bbox aspect ratio
2.03) and median mask fill is 0.103, so the typical input tensor is **~95% zeros**. Run the
**mask-only ablation** (binary mask, no RGB) early: if it scores near the RGB model the task is
shape matching and forcep03/04 may be near-indistinguishable by construction - their median area
(1421 vs 1469 px), aspect ratio (1.80 vs 1.69) and fill (0.072 vs 0.070) agree within 6%.

## 3. Core Requirements & Logic
- [ ] **BBOX Cropping:** The segmented instances from MaskDINO have to be cropped to its bounding box, scale it along its largest axis, and pad the image. Do not skew it, only uniform scale is acceptable.
- [ ] **Segmenting of Image:** The segmented instances are used to segment the rgb pixels, everything outside the segmentation mask stays black.
- [ ] **Class-Type** Use really the segmentation of MaskDINO and use the class type from the groundtruth. Assign the Classtype according to IoU so if the segmentation occurs over two classes in the groundtruth take the class which segmentation map has the higher IoU. The IoU with the underlying groundtruth has to be >0.5 at least though. Check second highest overlap to be lower than < 0.2 in order to have a distinct Groundtruth for this segmented mask. If the second highest overlap is over 0.2, drop this sample from classifying. For doing these calculations, upsample the segmap, do not downsample the image to predict!
- [ ] **Edge Cases:** If the classifier is unsure, it should output the most probable case. To compare it to the current MaskDINO implementation which also has no "uncertanty state"
- [ ] **Training Image View**Output to me the Images which are used for Training in the classifier, I want to see them with and without segmentation maps and also overlapped segmentation map from the original 
- [ ] **Phase 1 Train** analogous to the MaskDINO training, lets train this Classifier in a first step on Phase 1 Training images. 
- [ ] **Phase 2 Train** this is not part of this features, leave it out
- [ ] **Augmentation Pipeline** Use the same augmentation *content* as MaskDINO phase-1, but
      applied **after** the detector, not before it. The detector is frozen, so augmenting its
      input regularizes nothing - it only perturbs the masks, and the bbox crop + letterbox
      discards the scale and position variation anyway. Running it on clean frames also matches
      its inference condition (where its 97.9% class-agnostic recall was measured) and makes its
      output deterministic per frame, hence cacheable. Crop-level equivalents:
      rotation 0/90/180/270 (lossless on a square crop, and 4x the classifier samples per
      detector forward); `RandomBrightness/Contrast/Saturation/Lighting` at the same ranges as
      `build_transform_gen`, applied pre-blackout; optional mask dilate/erode in place of the mask-shape
      variation frame jitter used to produce incidentally. No flipping - instruments have fixed
      handedness. Controlled by `augment_frames` (default `false`); set it `true` to A/B against
      the original recipe.

## 4. API Endpoints / Interface Definitions
- The Training Data will be also Pool based as in MaskDINO training, adapte the Dataloader accordingly

## 5. Definition of Done (DoD) & Testing

- [x] **Pipeline works correctly.** Measured order, which differs from the original sketch -
      the detector runs on the **clean** frame and augmentation happens after cropping
      (see section 3 and step 5 for why):

      Image -> SingleClassV2 (instance segmentation map) -> assign GT class via IoU
            -> bbox crop + blackout + letterbox -> crop-level augmentation -> Train

      Verified on 20 real frames: 108 assigned, 1 dropped contaminated, 1 dropped low-IoU
      (98.2% assignment rate, consistent with the detector's 2.1% class-agnostic FN).

- [x] **Unit tests** cover bbox cropping, segmentation and class assignment - 32 passing:
      `tests/test_cropcls_crops.py` (square output, aspect ratio preserved 10:1, background
      exactly 0, empty mask rejected not NaN, rotations lossless),
      `tests/test_cropcls_assign.py` (assigned / low-IoU / contaminated, **zero-GT frame**,
      single-GT, predictions upsampled rather than GT downsampled),
      `tests/test_cropcls_config.py`, `tests/test_cropcls_detector.py`
      (`class_embed (1,256)`, frozen, eval-mode, rejects an 8-class checkpoint).

- [x] **Short run: val loss reduces.** 250 iterations, `ims_per_batch 2`, 8,856 crops:
      train loss 2.0820 -> 0.3396 (chance is ln(8) = 2.079), val_loss 0.4791,
      val_top1 0.7711 over 201 crops. Plumbing only - not a result.

- [ ] **Comparison to `20261006204751_phase1_training_seta_v1-1`** (8-class MaskDINO,
      `val_seta_mm`, stride 4, excl-truncated, 1201 GT). Needs a full run.
      - primary, confound-free: **top-1 accuracy on matched masks**, baseline **97.22%**
        (1083/1114). Both sides condition on an instance having been localized, so the two
        detectors' differing recall cannot flatter either.
      - secondary: `classaware_f1`, baseline **0.9256**; report `classagnostic_f1`
        (baseline 0.9521) beside it, since the frontend supplies 1176 matched masks against
        the baseline's 1114 - worth up to +7.75 points of classaware recall on its own.
      - per class, because 30 of the baseline's 31 errors are forcep01-04. The 250-iteration
        smoke run already reproduces that shape: every non-forceps class 0.88-1.00,
        forcep02 0.62, forcep04 0.42, forcep03 0.35.

- [x] **Per-stage timings reported - and yes, the detector forward is too slow.**
      Measured (`ims_per_batch 4`, Quadro RTX 6000):

      | stage | ms/frame | % |
      |---|---|---|
      | detector | 602.0 | 72.0% |
      | augment (frame-level, now removed) | 189.0 | 22.6% |
      | assign | 24.9 | 3.0% |
      | crop | 14.4 | 1.7% |
      | collate + mask_to_cpu | 5.7 | 0.7% |

      Batching does not help: flat at ~630 ms/frame from batch 2 to 12. Within the detector,
      backbone is 23.8 ms and deformable attention is ~740 ms (pixel decoder 211.6, transformer
      decoder 531.2) - the MSDeformAttn CUDA op is not compiled and there is no `nvcc` on this
      machine, so it runs the pure-PyTorch fallback. At 20,000 iterations that is ~18.6 h of
      data generation alone.

- [ ] **The adaptation** (follows from the timing result, implemented in part):
      - [x] drop frame-level augmentation so the detector sees clean frames -> its output is
            deterministic per frame (`augment_frames: false`; removes the 189 ms/frame stage)
      - [ ] cache crops keyed by frame filename: one detector forward per frame, ever.
            ~12 min for the static 1024 frames. If the live pool is running, the cache must
            **mirror the pool's FIFO eviction** - delete a frame's crops when the frame is
            evicted - or retired scenes leak back into training and the cache grows ~2.8 GB/day.
      - [ ] cache the crop **before** blackout, thresholding and rotation, so photometric
            augmentation can run pre-blackout and `iou_floor` / `runner_up_ceiling` become
            load-time filters rather than precompute decisions.

## 6. Implementation Checklist (The TODO)
*The step-by-step roadmap for the AI.*

Sequenced so that **everything is verifiable before GPU hours are spent**: steps 1-4 need no
training, step 4 is a human inspection gate, and step 5 answers the timing question in the DoD
before any schedule is chosen.

No new `META_ARCHITECTURE` is registered - the multi-arch config machinery (`build_base_cfg`,
`arch_ns`, `_ARCH_CONFIG_NS`) exists only on branch `41-train-on-mask2former` and is unmerged, so
registering an arch would block this feature on that merge. `maskdino/` is not modified at all.

**Target file layout**

```
cropcls/
  __init__.py
  config.py               dataclass + yaml.safe_load (no yacs, no registry)
  detector.py             frozen single_class_v2 wrapper -> masks + scores
  crops.py                letterbox crop + mask blackout        (step 2)
  assign.py               IoU -> GT class, runner-up guard      (step 3)
  viz.py                  the "Training Image View" deliverable (step 4)
  data.py                 loader: detect -> assign -> crop -> augment (step 5)
  modeling/classifier.py  CNN trunk + global pool + linear head    (step 6)
  metrics.py              top-1 on matched masks, timing aggregation
train_cropcls.py      own training loop, --eval-only, --viz-only
configs/cropcls/phase1_seta.yaml
tests/test_cropcls_{crops,assign,detector,timing}.py
```

- [x] **Step 0 - scaffolding.** Package + config dataclass + `configs/cropcls/phase1_seta.yaml`.
      Config carries: detector checkpoint path, train/val dataset names, `CROP_SIZE: 224`, `NUM_CLASSES: 8`,
      trunk name, LR, iters, IoU floor (0.5) and runner-up ceiling (0.2). The detector path goes
      in the YAML, not in code. **Verify:** config loads, paths assert-exist at startup.

- [x] **Step 1 - frozen detector wrapper** (`detector.py`). Bootstrap via `get_cfg()` +
      `add_deeplab_config` + `add_maskdino_config` + merge the run's own `config.yaml`, then
      `build_model` -> `.eval()` -> `DetectionCheckpointer(model).load(...)` - the pattern at
      `tools/benchmark_inference.py:60-70`. Wrap the forward in `no_grad` + `autocast`, pin to
      `.eval()` permanently. Returns per-frame `(masks, scores)` - the learned `pred_boxes`
      are deliberately not carried, since step 2 crops from each mask's tight bbox instead.
      **Verify:** load the checkpoint, run one `val_seta_mm` frame, assert `class_embed` is `(1, 256)`,
      print mask count and shapes, print the forward time.

- [x] **Step 2 - crops** (`crops.py`). Pure function, no model, no detectron2. bbox crop ->
      zero every pixel outside the mask -> uniform scale on the longest axis -> pad to square.
      No skew. **Verify:** `tests/test_cropcls_crops.py` - output is square and the requested size;
      aspect ratio preserved within a pixel; background is exactly 0; a 40x400 bbox is not stretched;
      an empty mask is rejected rather than producing a NaN or all-black crop.

- [x] **Step 3 - class assignment** (`assign.py`). Reuse
      `maskdino.evaluation.hungarian_instance_evaluation.mask_iou_matrix` (exact, box-prefiltered).
      Upsample the predicted segmap to GT resolution - never downsample the GT. Rule:
      `best >= 0.5` AND `runner_up < 0.2` -> assign `gt_classes[argmax]`; otherwise drop.
      **Verify:** `tests/test_cropcls_assign.py` - clean single-GT mask assigned correctly;
      a mask straddling two GTs at 0.55/0.38 dropped; `best < 0.5` dropped; a frame with **zero GT**
      returns empty instead of throwing (such frames exist in this data, e.g.
      `012676FD8_01-008.hdf5`); correct behaviour when only one GT is present.

- [x] **Step 4 - training image view** (`viz.py`, `--viz-only`). GATE: no training before this is
      inspected. For N sampled frames write a contact sheet per instance: raw crop, crop with
      background blacked, and the predicted mask overlaid on the original full frame. Label each
      tile with assigned class, best IoU, runner-up IoU, and verdict (assigned / dropped-low-IoU /
      dropped-contaminated). Also write the dropped bucket, so the thresholds can be calibrated by
      eye. **Verify:** human review - the dropped pile should look genuinely ambiguous. Adjust 0.5 /
      0.2 here, not later.

- [x] **Step 5 - pool dataloader + timing** (`data.py`). Order is **detect -> assign -> crop ->
      augment**: the detector runs on the clean frame, so its masks and `instances.gt_masks` share
      a coordinate frame trivially and its output is deterministic per frame. Reuse
      `Hdf5CocoInstanceDatasetMapper` with `is_train=True` and `tfm_gens=[]` - `is_train=True` is
      required because the mapper **drops `annotations` entirely** when False, and the assignment
      step needs GT on val as much as on train; `tfm_gens=[]` makes it an identity transform. It
      returns `instances.gt_masks` as an `(N,H,W)` bitmask tensor and `gt_classes`, so no mapper
      change is needed. Index the pool with `d[i]`, not `for r in d`. Instrument every stage.
      **Verify:** `tests/test_cropcls_timing.py` smoke + a printed per-stage table
      (scene gen / augment / detector forward / upsample+IoU / crop / collate). **Report the
      timings before step 7 picks `MAX_ITER`.** If the detector forward dominates, the likely cause
      is MSDeformAttn running its pure-PyTorch fallback (`CUDA_HOME` is empty here, there is no
      `nvcc` on this machine, so the CUDA op cannot be built).

      **Measured, and it is too slow**: 602 ms/frame of a 836 ms/frame pipeline (72%), flat under
      batching from batch 2 to 12, with deformable attention accounting for ~740 ms of the 767 ms
      detector forward (backbone is only 23.8 ms). See section 5 for the full table and the
      adaptation. Dropping frame-level augmentation removes a further 189 ms/frame and makes the
      detector output cacheable, which is what actually resolves it.

      **All augmentation moves to crop level** (`augment_frames: false`). Rotation: emit each crop
      at 0/90/180/270 after cropping. On a square letterboxed crop
      those four are lossless pixel permutations and the axis-aligned bbox is symmetric under them,
      so the result is identical to rotating the frame - but one detector forward then yields 4x the
      classifier samples instead of one rotation's worth. Since the detector forward is the
      expensive stage, this is where the throughput is. Keep the frame-level photometric augs
      (`RandomBrightness/Contrast/Saturation/Lighting`) as they are - they survive the crop
      untouched. Do **not** add flipping: instruments have fixed handedness, so a mirrored scalpel
      is an invalid shape (see the comment in `build_transform_gen`). Note the four rotations of one
      instance are correlated, so effective diversity grows well under 4x - this buys throughput,
      not a substitute for more frames.

- [x] **Step 6 - trunk + head** (`modeling/classifier.py`). A plain image-classification CNN -
      **no decoder of any kind**, since the masks come from MaskDINO. Trunk -> global average
      pool -> `nn.Linear(C, 8)`. Trunk
      pluggable (`detectron2_resnet` / `timm`); `timm 1.0.28` is present. Stock PyTorch ops only.
      If the trunk is frozen and uses real `BatchNorm2d`, pin it to `.eval()`.
      **Verify:** forward shape test; parameter count printed.

- [x] **Step 7 - training loop** (`train_cropcls.py`). Own loop: AMP, checkpoint save/load, LR via
      `maskdino/solver`, detectron2 `EventStorage` + writers, `--eval-only`. Cross-entropy over 8
      classes, argmax at inference (no reject state, per spec). Log train loss, **val loss**, and
      **top-1 accuracy on matched masks**. **Verify:** 20 iterations at batch 2 on 1 GPU - loss
      decreases, checkpoint is small, memory sane.

- [ ] **Step 8 - short run + comparison.** Train, then report against
      `runs/20261006204751_phase1_training_seta_v1-1` (8-class, `val_seta_mm`, stride 4,
      excl-truncated, 1201 GT):
      - primary, confound-free: **top-1 accuracy on matched masks**, baseline **97.22%** (1083/1114)
      - secondary: `classaware_f1`, baseline **0.9256** - report `classagnostic_f1` beside it, since
        the frontend supplies 1176 matched masks vs the baseline's 1114 and that is worth up to
        +7.75 points of classaware recall on its own (see `cropcls-baseline-comparison.md`)
      - per-class breakdown, because 30 of the baseline's 31 errors are forcep01-04
      **Verify:** DoD items 3 and 4 satisfied.

- [ ] **Step 9 - full run** at the `MAX_ITER` chosen from the step-5 timings.

**Out of scope (explicit):** phase 2 / set B, COCO AP as a headline metric (argmax cannot hedge
the way `instance_inference`'s `topk(100)` does), any modification to `maskdino/` or `train_net.py`.
