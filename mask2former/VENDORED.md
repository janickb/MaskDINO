# Vendored Mask2Former

Source: [facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)
@ `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a` (2022-05-20).

This is a **deliberately partial** copy. Everything that is *harness* rather than
*architecture* — dataset registration, dataset mappers, evaluation, TTA, the Swin
backbone and the MSDeformAttn CUDA ops — comes from the `maskdino/` package instead, so
both architectures run through one `train_net.py` with one augmentation pipeline, one
Hungarian evaluator and one set of AP reporting. That is what makes a
MaskDINO-vs-Mask2Former AP delta attributable to the architecture rather than to the
pipeline around it.

`tests/test_mask2former_harness_parity.py` and `tests/test_mask2former_registry.py`
enforce the invariants described here.

## Kept

| File | State |
|---|---|
| `maskformer_model.py` | **patched** — see Patches A–D |
| `config.py` | **rewritten** — `add_mask2former_config`, see below |
| `__init__.py`, `modeling/__init__.py` | **rewritten** — see below |
| `modeling/criterion.py` | patched import only |
| `modeling/matcher.py` | patched import only |
| `modeling/pixel_decoder/msdeformattn.py` | patched imports only |
| `modeling/pixel_decoder/fpn.py` | verbatim — holds `build_pixel_decoder` |
| `modeling/meta_arch/mask_former_head.py` | verbatim |
| `modeling/transformer_decoder/mask2former_transformer_decoder.py` | verbatim |
| `modeling/transformer_decoder/maskformer_transformer_decoder.py` | verbatim — holds `TRANSFORMER_DECODER_REGISTRY` + `build_transformer_decoder` |
| `modeling/transformer_decoder/position_encoding.py` | verbatim |
| `modeling/transformer_decoder/transformer.py` | verbatim — `_get_clones` / `_get_activation_fn` |
| `modeling/{meta_arch,pixel_decoder,transformer_decoder}/__init__.py` | verbatim |

`fpn.py` and `maskformer_transformer_decoder.py` are kept whole even though their
baseline classes (`TransformerEncoderPixelDecoder`, `StandardTransformerDecoder`) are
never built here — they hold the registries and builder functions, and trimming them
invites a re-sync conflict.

## Deleted, and what replaces each

| Deleted | Replacement / reason |
|---|---|
| `modeling/pixel_decoder/ops/**` | `maskdino/modeling/pixel_decoder/ops` — **blocker-class if duplicated.** That copy's `MSDeformAttn.forward` is pinned to the pure-PyTorch `ms_deform_attn_core_pytorch` kernel because the compiled CUDA extension corrupts the CUDA stream on this torch/CUDA build (it surfaces as `invalid resource handle` on the next, unrelated CUDA call, so a try/except at the call site does not catch it). A second `ops/` tree would let someone run its `make.sh` and reintroduce that. |
| `modeling/backbone/**` | `maskdino/modeling/backbone/swin.py`. `maskdino/modeling/__init__.py` already registers `D2SwinTransformer` in detectron2's global `BACKBONE_REGISTRY`; a second registration of the same key raises at import. |
| `data/**` | `maskdino/data/datasets/register_hdf5_instance.py`, `register_hdf5_pool_instance.py`, `live_pool_dataset.py`, `class_mapping.py`, and `dataset_mappers/hdf5_coco_instance_dataset_mapper.py`. Using the same mapper is what makes "same data, same augmentation" true by construction. |
| `evaluation/**` | `maskdino/evaluation/instance_evaluation.py` (the same upstream file), plus this fork's `hungarian_instance_evaluation.py` and `truncated_prediction_filter.py`. |
| `utils/**` | `maskdino/utils/misc.py` already has `nested_tensor_from_tensor_list` and `is_dist_avail_and_initialized`. |
| `test_time_augmentation.py` | `maskdino/test_time_augmentation.py`. |
| `modeling/meta_arch/per_pixel_baseline.py` | Semantic-only baselines, never built; would add two dead `SEM_SEG_HEADS_REGISTRY` entries. |

## Patches to `maskformer_model.py`

**A — mask-derived `pred_boxes`** (in `instance_inference`). Upstream sets
`pred_boxes = Boxes(torch.zeros(N, 4))` because `COCOEvaluator`'s segm task never reads
it. Two consumers here do:

1. `DropTruncatedPredictions` reads `instances.pred_boxes.tensor` and runs
   `touches_frame_edge()` on it. A `[0,0,0,0]` box touches the left *and* top edge, so
   with `INPUT.EXCLUDE_TRUNCATED_INSTANCES` **True (the default)** it would drop *every*
   prediction → segm AP 0.0 with nothing in the log to explain it. That reads as "the
   architecture cannot do this task" rather than as a plumbing bug.
2. `COCOEvaluator`'s bbox task needs real boxes to report bbox AP beside segm AP.

The boxes are `.to(result.pred_masks.device)`: `BitMasks.get_bounding_boxes()`
allocates with `torch.zeros(N, 4)` and no device, so it returns CPU boxes for CUDA
masks, and the Patch-B slice would then index a CPU tensor with a CUDA keep-mask.

**These are not a learned box prediction** — Mask2Former has no box branch. This arm's
bbox AP is a re-parameterization of its own segm AP and must not be compared head-to-head
with MaskDINO's learned-box bbox AP. Report segm AP.

**B — `class_aware_mask_nms`.** Imported from `maskdino/modeling/postprocess.py` (one
shared definition) and gated by `MODEL.MASK_FORMER.TEST.NMS_IOU`. The `0.0` default
reproduces upstream exactly, since upstream ships no dedup step; the surgical configs
set `0.5` to match the MaskDINO arm. **Both arms must use the same value** — the
queries × classes flattening can emit one mask under several labels, and duplicates
count as false positives in the Hungarian evaluator.

**C — reclassify-finetune freezing.** Mirrors `maskdino/maskdino.py`. Module names
differ: `MultiScaleMaskedTransformerDecoder` has no `bbox_embed` and exposes three
parallel layer stacks, so `UNFREEZE_DECODER` extends the prefix tuple with
`transformer_self_attention_layers`, `transformer_cross_attention_layers`,
`transformer_ffn_layers`, `decoder_norm` and `mask_embed`. `UNFREEZE_ENCODER` adds
`sem_seg_head.pixel_decoder`, which is the same module path in both architectures.

The `[reclassify-finetune]` log line prints the trainable **parameter** total as well as
the tensor count, because the tensor counts are not comparable across architectures.
Measured (16-class reclass head):

| variant | Mask2Former | MaskDINO |
|---|---|---|
| `class_embed` only | 2 tensors / 4,369 params | 2 tensors / 4,112 params |
| `+UNFREEZE_DECODER` | 172 tensors / 14,411,025 (32.79%) | 218 tensors / 14,222,388 (32.48%) |
| `+UNFREEZE_ENCODER` | 289 tensors / 20,446,929 (46.52%) | 335 tensors / 20,258,292 (46.27%) |

So despite the differing tensor counts the parameter budgets match within ~1.3%. The
class-head difference (4,369 vs 4,112) is exactly the extra no-object row: Mask2Former's
`class_embed` is `nn.Linear(hidden, num_classes + 1)` with softmax CE, MaskDINO's is
`nn.Linear(hidden, num_classes)` with sigmoid focal loss.

**D — `losses = ["labels"]` gating.** Same condition as MaskDINO's `from_config`: when
`RECLASSIFY_FINETUNE.ENABLED` and neither the decoder nor the encoder is unfrozen, only
`class_embed` trains and the mask/dice losses have no gradient path to it. Leaving
`loss_mask`/`loss_dice` in `weight_dict` is harmless — `forward()` only weights keys the
criterion actually returned.

## `config.py`

`add_mask2former_config` defines **only** `cfg.MODEL.MASK_FORMER`, then calls
`maskdino.config.add_surgical_arch_config` to inject this fork's shared keys
(`TEST.NMS_IOU`, `TEST.HUNGARIAN_EVAL.*`, `TEST.VAL_LOSS.ENABLED`,
`RECLASSIFY_FINETUNE.*`) so `maskdino.config.arch_ns(cfg)` resolves equivalently for
either architecture.

It is **not** upstream's `add_maskformer2_config`, which would also reset
`INPUT.DATASET_MAPPER_NAME`, `MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME`,
`INPUT.{IMAGE_SIZE,MIN_SCALE,MAX_SCALE,...}`, `SOLVER.{OPTIMIZER,BACKBONE_MULTIPLIER,...}`
and re-create `MODEL.SWIN` from scratch — changing behaviour for every MaskDINO config
that relies on a default. `tests/test_arch_ns.py` pins that nothing outside
`MODEL.MASK_FORMER` moves.

## Compat patches for Python 3.13 / torch 2.13

- `from torch.cuda.amp import autocast` → `from torch.amp import autocast`, and
  `autocast(enabled=False)` → `autocast("cuda", enabled=False)`, in `modeling/matcher.py`
  and `modeling/pixel_decoder/msdeformattn.py`. The decorator form in `msdeformattn.py`
  constructs at *import* time, so it would fire a `FutureWarning` on every import.
  Precedent: `maskdino/modeling/pixel_decoder/maskdino_encoder.py:16,362`.
- `modeling/criterion.py` imports its two DETR helpers from `maskdino.utils.misc`.
- `modeling/pixel_decoder/msdeformattn.py` imports `MSDeformAttn` from
  `maskdino.modeling.pixel_decoder.ops.modules` — **absolute**, because that `ops/`
  directory is a namespace package (no top-level `__init__.py`; only `ops/functions/`
  and `ops/modules/` have one).

`@torch.jit.unused`, the `// self.sem_seg_head.num_classes` floor-division and
`nn.MultiheadAttention`'s `(L, N, E)` layout are all fine on this stack and are left
alone — each has working precedent in `maskdino/`.

## Re-syncing with upstream

1. Re-clone upstream at the new SHA and diff the **Kept** files above.
2. Re-apply Patches A–D and the compat patches.
3. Do not restore `data/`, `evaluation/`, `utils/`, `modeling/backbone/`,
   `modeling/pixel_decoder/ops/`, `test_time_augmentation.py` or
   `per_pixel_baseline.py`.
4. Run `./.venv/bin/python -m pytest tests/test_mask2former_registry.py
   tests/test_mask2former_harness_parity.py tests/test_arch_ns.py` — these catch a
   restored `ops/` tree, a re-registered `D2SwinTransformer`, a reverted Patch A, and
   config drift between the two arms.
5. Update the SHA at the top of this file.
