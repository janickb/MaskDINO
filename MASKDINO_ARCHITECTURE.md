# MaskDINO Architecture Layout

A navigable map of this codebase tying each component to the corresponding section of the MaskDINO paper (Li et al., 2022, *"Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation"*, arXiv:2206.02777).

---

## 1. Backbone — paper §3.1

Extracts multi-scale features from the input image.

- **ResNet**: not reimplemented here — used directly from detectron2 (`build_resnet_backbone`), selected via `cfg.MODEL.BACKBONE.NAME` in configs (e.g. `configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml:4`). Outputs `res2..res5` at strides 4/8/16/32.
- **Swin Transformer**: [swin.py](maskdino/modeling/backbone/swin.py)
  - `SwinTransformer` — core backbone (line 498)
  - `D2SwinTransformer` — detectron2-compatible wrapper (lines 686–762), `forward()` at 743–758 returns `{res2: ..., res3: ..., res4: ..., res5: ...}`
- Wired in: [maskdino.py](maskdino/maskdino.py) `from_config`, line 118 — `backbone = build_backbone(cfg)`, `backbone.output_shape()` (line 119) tells the pixel decoder what channels/strides to expect.

## 2. Pixel Decoder / Multi-scale Feature Enhancer — paper §3.2

The paper calls this the "multi-scale feature enhancer" — in code it's named `MaskDINOEncoder` and doubles as the pixel decoder.

File: [maskdino_encoder.py](maskdino/modeling/pixel_decoder/maskdino_encoder.py)

- `MSDeformAttnTransformerEncoderOnly` (lines 43–187) — a deformable-attention self-attention encoder over concatenated multi-level tokens (Deformable DETR style), using `MSDeformAttn` from `modeling/pixel_decoder/ops/modules`.
- `MaskDINOEncoder` (lines 190–428) — docstring literally says *"this is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models"* (192–194).
  - `forward_features(features, masks)` (362–428): projects backbone features → runs the deformable encoder → produces enhanced multi-scale memory → fuses the highest-res level with an FPN top-down path using untouched high-res backbone features.
  - **Returns**: `(mask_features, transformer_encoder_features, multi_scale_features)` — `mask_features` is the 1/4-resolution per-pixel embedding map later dot-producted with mask queries; `multi_scale_features` feeds the transformer decoder's deformable cross-attention.
- Wired in: [maskdino_head.py](maskdino/modeling/meta_arch/maskdino_head.py) `layers()` (77–82) — calls `pixel_decoder.forward_features(...)`, passes outputs into the `predictor` (the transformer decoder).

## 3. Transformer Decoder — paper §3.3–3.5

File: [maskdino_decoder.py](maskdino/modeling/transformer_decoder/maskdino_decoder.py), `MaskDINODecoder.forward`.

### 3a. Unified Query Selection — paper §3.3

Selects top-K encoder tokens as both content and positional (box anchor) initialization for decoder queries — the "unified" part is that the same selection drives detection boxes *and* mask embeddings.

Lines 400–422 (`if self.two_stage:`):
- `gen_encoder_output_proposals` (line 401, defined in [utils.py:33](maskdino/utils/utils.py)) turns flattened encoder memory into per-token box anchors (grid-based, size `0.05 * 2**level`).
- Lines 402–405: per-token class logits (`enc_outputs_class_unselected`) and box deltas (`enc_outputs_coord_unselected`) computed via the shared `class_embed` / `_bbox_embed` heads.
- Line 406–407: `topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]` — top-K tokens by best class score (max-over-classes then topk-over-tokens; note this is a *different* pattern from the `//num_classes` trick at `maskdino.py:462`, see §6 below).
- Lines 408–414: gather box coords (`refpoint_embed`) and content features (`tgt_undetach`) at those top-K token indices.
- Lines 415–422: run `forward_prediction_heads` on the selected content immediately → `interm_outputs` (the paper's encoder-level/"two-stage" auxiliary prediction, supervised separately in the loss — see §5).

### 3b. Mask-Enhanced Anchor Box Initialization — paper §3.4 (MaskDINO's specific contribution over DINO)

Lines 424–438 (`if self.initialize_box_type != 'no':`):
- Instead of trusting the regressed box (`refpoint_embed` from 3a), derive the box directly from the **predicted mask's bounding box** — more spatially precise than box regression at this stage.
- `'bitmask'` mode (430): exact bbox from binarized mask via detectron2 `BitMasks` (slower, more accurate).
- `'mask2box'` mode (432): faster approximate mask→box conversion (`box_ops.masks_to_boxes`).
- Result overwrites `refpoint_embed`, converted to cxcywh, normalized, and inverse-sigmoided (438) to match the space decoder layers expect.

### 3c. Denoising (DN) Training — paper §3.5

`prepare_for_dn` (lines 191–313) + `dn_post_process` (315–332); consumed at 445–458, 482–487.

- **Query construction**: replicate GT labels/boxes `scalar` times (`dn_num // max_GT_per_image`), noise labels with random-class substitution (238–242) and boxes with uniform shift+scale jitter (243–249) — classic DN-DETR noising.
- Noised labels → `label_enc` embedding (line 140, 251) become DN query *content*; noised boxes → inverse-sigmoid (253) become DN query *position*.
- **Attention mask** (276–288): built so matching queries can't see DN queries, and DN queries from different noise groups can't see each other — prevents GT leakage.
- DN queries are prepended to the real queries (line 450: `tgt = cat([input_query_label, tgt])`) and pushed through the *same* decoder stack with `tgt_mask=attn_mask` enforcing isolation.
- After decoding, `dn_post_process` splits the first `pad_size` outputs back out as the DN branch's own predictions (stored for the DN loss); the rest are the real matching-based predictions.

### 3d. Decoder Layer Stack — paper §3.3 (iterative refinement)

File: [dino_decoder.py](maskdino/modeling/transformer_decoder/dino_decoder.py)

- `TransformerDecoder.forward` (94–168): per-layer loop over `DeformableTransformerDecoderLayer`s.
  - `gen_sineembed_for_position` + `ref_point_head` MLP (line 131) turn the current reference box into a positional query embedding (DAB/DINO-style dynamic anchor).
  - **Iterative box refinement** (153–161): each layer predicts a box delta on top of the *previous* layer's reference (via a shared `bbox_embed` MLP list, wired from `maskdino_decoder.py:161`), producing a refined reference for the next layer.
  - All per-layer hidden states are collected (163) → feeds deep supervision (aux losses).
- `DeformableTransformerDecoderLayer.forward` (220–270):
  - **Self-attention** among queries (246–250), masked by the DN `self_attn_mask`.
  - **Deformable cross-attention** (260–263) via `MSDeformAttn` — each query samples a small set of learned points around its current reference box in the multi-scale memory (efficient vs. dense attention).
  - Standard FFN (214–218) with residual + LayerNorm throughout.

### 3e. Prediction Heads — paper §3.3

Defined in `MaskDINODecoder.__init__`:
- `class_embed` (137/139) — `Linear(hidden_dim, num_classes[+1])`.
- `mask_embed` (141) — 3-layer `MLP(hidden_dim, hidden_dim, mask_dim)`.
- `_bbox_embed`/`bbox_embed` (156–161) — 3-layer `MLP(hidden_dim, hidden_dim, 4)`, zero-initialized final layer, **shared across all decoder layers**.

`forward_prediction_heads` (503–512):
```python
decoder_output = self.decoder_norm(output).transpose(0, 1)      # (bs, nq, hidden_dim)
outputs_class = self.class_embed(decoder_output)                # (bs, nq, num_classes)
mask_embed = self.mask_embed(decoder_output)                    # (bs, nq, mask_dim)
outputs_mask = einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (bs, nq, H, W)
```
This is the core "dot product of mask query embedding with per-pixel mask features" step — called once for `interm_outputs` (3a), once for the pre-decoder `initial_pred` (454), and once per decoder layer (471–474) for deep supervision.

`pred_box` (343–361): mirrors the in-decoder box refinement (3d) to produce the final stacked box tensor across all layers.

**Final output assembly** (491–501): `pred_logits`/`pred_masks`/`pred_boxes` = last layer; `aux_outputs` = all other layers (deep supervision); `interm_outputs` = the two-stage/query-selection prediction (3a).

## 4. Top-Level Model & Inference — paper §3.6 / Fig. 2

File: [maskdino.py](maskdino/maskdino.py), `class MaskDINO` (25–114).

- `forward()` (222–332): preprocess (248–250) → `backbone(images)` (252) → training: `sem_seg_head(features, targets)` runs pixel decoder + transformer decoder + DN in one call (265), then `criterion(outputs, targets, mask_dict)` (267) computes losses. Inference: `sem_seg_head(features)` (no targets, DN disabled), then dispatch to `semantic_inference` (374–391), `panoptic_inference` (393–453), or `instance_inference` (455–489) per the configured task.
- `prepare_targets`/`prepare_targets_detr` (334–372): converts GT masks/boxes into the padded-mask + normalized-cxcywh-box format the criterion expects.

### The `topk_indices // num_classes` line (`maskdino.py:462`)

This lives in `instance_inference` (455–489) — a **test-time top-k detection decoding** step, unrelated to the training-time "unified query selection" in §3a (different mechanism, same "flatten and topk" idea reused at a different stage):

1. `scores = mask_cls.sigmoid()` — shape `[num_queries, num_classes]`.
2. `scores.flatten(0,1).topk(k)` — flattens to a 1D vector of length `Q*C` and takes the global top-k over **every (query, class) pair** — so one query can yield multiple final detections if it scores high on several classes.
3. The flat index encodes `query_idx * num_classes + class_idx`, so `topk_indices // num_classes` recovers **which query** each selected entry came from (used to gather `mask_pred[topk_indices]`), while `labels[topk_indices]` (via the precomputed repeated-arange `labels` tensor) recovers the class.

This is the standard Deformable-DETR/DINO top-k inference trick, applied here to turn per-query multi-class scores into a ranked list of final instance detections.

## 5. Matcher & Losses — paper §3.3 (loss) / Eq. for Hungarian matching

- **Matcher**: [matcher.py](maskdino/modeling/matcher.py) `HungarianMatcher` (76–230) — cost = weighted sum of classification cost (sigmoid focal-style), box L1 + GIoU cost, and point-sampled mask dice + BCE cost; solved via `scipy.optimize.linear_sum_assignment` (192).
- **Criterion**: [criterion.py](maskdino/modeling/criterion.py) `SetCriterion` (125–443):
  - `loss_labels`/`loss_labels_ce` (163–202), `loss_boxes`/`loss_boxes_panoptic` (204–248), `loss_masks` (250–300, point-sampled BCE + dice via detectron2 PointRend utilities).
  - `forward()` (334–427): matches + computes losses for the final layer, **DN losses** (`_dn` suffix, 372–386, using a direct known-index mapping instead of Hungarian matching since DN queries are pre-assigned), **deep supervision** losses per decoder layer (`_i` suffix, 389–417), and **interm/two-stage** losses (`_interm` suffix, 419–425) supervising the unified query selection output from §3a.

---

## Component-to-Paper Cross-Reference Table

| Paper concept | Code location |
|---|---|
| Backbone | `maskdino.py:118`; `modeling/backbone/swin.py:498,686` |
| Multi-scale feature enhancer / pixel decoder | `modeling/pixel_decoder/maskdino_encoder.py:190-428` |
| Unified query selection | `modeling/transformer_decoder/maskdino_decoder.py:400-422` |
| Mask-enhanced box initialization | `modeling/transformer_decoder/maskdino_decoder.py:424-438` |
| Denoising (DN) training | `modeling/transformer_decoder/maskdino_decoder.py:191-332`; losses in `criterion.py:344-386` |
| Decoder layer stack (iterative refinement) | `modeling/transformer_decoder/dino_decoder.py:18-270` |
| Prediction heads (class/box/mask) | `modeling/transformer_decoder/maskdino_decoder.py:503-512, 343-361` |
| Hungarian matching + losses | `modeling/matcher.py:76-230`; `modeling/criterion.py:125-443` |
| Test-time top-k decoding (`//num_classes` trick) | `maskdino.py:455-489`, line 462 specifically |
| Inference post-processing (sem/pan/instance) | `maskdino.py:374-489` |
