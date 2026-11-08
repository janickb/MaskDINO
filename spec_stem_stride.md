# Stem stride for the crop classifier — spec

## 1. Ask — verbatim

Ground truth: [requirements_stem_stride.md](requirements_stem_stride.md).

> but img reduction still happens the same in the first few layers compared when doing
> outputstride 16, so the fine detail in early layer is also still lost

> alright lets try the stem idea

> only test it with convnext-tiny

This is a quotation, not a worklist. Never reworded, never ticked.

## 2. Decision

One config field, `stem_stride`, overriding the stride of the trunk's first conv. On
convnext_tiny that conv is a 4x4/stride-4 patchify, so `stem_stride: 2` turns it into
overlapping 4x4/stride-2 — **kernel shape and pretrained weights unchanged**, parameter count
identical, only the sampling density changes. Measured: stem 56 -> 112 and the four stages
56/28/14/7 -> 112/56/28/14, i.e. exactly c448's spatial resolution at c224 input and with no
upsampling.

Padding is raised in step (`4x4/s2/p0` yields 111 and odd sizes all the way down; `p1` yields
112 and stays even). The field therefore sets stride **and** the padding that keeps sizes even,
rather than exposing two knobs that are only ever set together.

Why this and not `output_stride`: `output_stride=16` changes only `layer4`, so it cannot test
the early-detail hypothesis the ask names. Why this and not c896: the thinnest classes are the
ones that collapse (forcep03/04 at a 0.40-cell half-width after the stem at c224), and halving
the early stride fixes that at 1/4 the pixel cost of doubling the input twice, without
interpolating — no crop in `crops/train_pad20` is 896 px (max 707).

## 3. Requirements

- [ ] **`stem_stride: int = 0`, 0 meaning "leave the trunk's own stride alone".** Not
      `None`: every other field in `CropClsConfig` is a plain scalar and `load_config` round-
      trips through JSON.
- [ ] **The conv is located by `pretrained_cfg["first_conv"]`**, the same mechanism
      `add_mask_input_channel` uses (`cropcls/modeling/classifier.py:42`), never a per-trunk
      table. A trunk that declares no `first_conv` must **raise**, not silently no-op.
- [ ] **Stem surgery runs BEFORE `add_mask_input_channel`.** That function rebuilds the conv
      copying `stride=old.stride` and `padding=old.padding`
      (`cropcls/modeling/classifier.py:52-64`), so it inherits the new values. The reverse
      order would silently discard them, and nothing would raise.
- [ ] **Padding is set so the stem output stays even**, `padding = ceil((kernel - stride) / 2)`,
      i.e. `(k - stride + 1) // 2`: 4x4/s2 -> p1 -> 112 at c224, and p0 gives 111/55/27/13.
      AMENDED 2026-11-05 — this item first stated the FLOOR form `(k - stride) // 2`, which is
      correct only for an even kernel. It agrees for convnext_tiny's 4x4 so the arm was
      unaffected, but it turns resnet50's native 7x7/s2/p3 into p2 and resnet50d's 3x3/s2/p1
      into p0, breaking the no-op guarantee two items below. The code implements the ceil form.
- [ ] **Pretrained weights must be bit-identical to the unmodified trunk.** Only `.stride` and
      `.padding` change; the kernel keeps its shape, so nothing is resampled or re-tiled. A
      test pins this.
- [ ] **`stem_stride` equal to the trunk's existing stride is a no-op, not an error** — so the
      control arm can state the field explicitly rather than omitting it.
- [ ] **Rejected: patching `m.stem[0]` by name.** convnext-only, and goes stale the first time
      a different `trunk=` is passed. The ask says convnext only; the *code* must not.
- [ ] **A stem-stride mismatch on checkpoint load must RAISE.** `.stride` and `.padding` are
      Conv2d attributes, not parameters: verified that a stride-2 and a stride-4 convnext_tiny
      have identical `state_dict` keys and identical tensor shapes, so
      `model.load_state_dict` (`train_cropcls.py:252`) accepts a stride-2 checkpoint into a
      stride-4 model **without error** and then runs a 7x7 feature map where the weights expect
      14x14. `mask_channel` is safe from the analogous mistake only because its conv changes
      *shape* and torch raises; here nothing would. So the checkpoint records the **effective**
      first-conv stride and the loader compares it, raising on disagreement. Scoped to the
      manual `weights=<other run>/model_final.pth` override, since `--eval-only` reads the
      run's own `config.json`.
- [ ] **Compare the EFFECTIVE stride, not the config field.** `stem_stride: 0` and
      `stem_stride: 4` are the same model on convnext_tiny, so a field-equality check would
      reject a legitimate pairing. A checkpoint with no recorded stride predates this field and
      is read as the trunk's native stride, which is what every existing run trained at.
- [ ] **Out of scope:** `output_stride` / dilation, resnet or resnet50d arms, c448 or c896
      arms, and any change to the crop pipeline. The ask is convnext_tiny at one input size.

## 4. Done when

- [ ] **The stem change does what it claims, measured not asserted.** Feature-map sizes
      112/56/28/14 against 56/28/14/7, and parameter count unchanged.
      **Verify:** `tests/test_cropcls_stem_stride.py::test_stem_stride_halves_every_stage` and
      `::test_parameter_count_is_unchanged`.
- [ ] **Step-0 equivalence.** With `stem_stride` set to the trunk's own stride, the model is
      bit-identical to one built without the field.
      **Verify:** `tests/test_cropcls_stem_stride.py::test_native_stride_is_a_no_op`.
- [ ] **Pretrained kernels untouched.**
      **Verify:** `tests/test_cropcls_stem_stride.py::test_stem_weights_are_not_resampled`.
- [ ] **A mismatched checkpoint raises instead of running at the wrong stride.**
      **Verify:** `tests/test_cropcls_stem_stride.py::test_loading_a_mismatched_stem_stride_raises`.
- [ ] **Ranked by `classaware_f1`** from `HungarianInstanceEvaluator` with
      `score_mode=detector`, convnext_tiny, c224, `blackout: false`, `mask_channel: true`,
      4000 iters, against the existing `pad_frac`-matched convnext control. Crop top-1 is a
      monitor only.
      **Verify:** reported in `cropcls-results.md` with the seed count and the noise floor.
- [ ] **At least two seeds per arm.** The measured per-arm spread at c224 reaches 1.75 points
      (`cropcls-results.md` section 8.2), so a one-seed gap is not rankable.
      **Verify:** two run directories per arm, their `config.json` seeds differing, both
      listed in `cropcls-results.md` with the control's spread quoted alongside.
- [ ] **Cost reported**, naming the machine and its load: s/iter and peak memory against the
      stride-4 control, since this is a pure-compute change at identical parameter count.
      **Verify:** both numbers in `cropcls-results.md`, with the machine and its GPU load, read
      from the runs' own logs rather than estimated.
