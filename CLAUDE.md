# MaskDINO — surgical instrument instance segmentation

A MaskDINO fork for instance segmentation of surgical instruments, trained on
synthetic HDF5 renders. For the code↔paper map (backbone, pixel decoder,
transformer decoder, heads, losses), read **[MASKDINO_ARCHITECTURE.md](MASKDINO_ARCHITECTURE.md)** —
do not re-derive it.

## Environment

- `uv`-managed; the interpreter is `./.venv/bin/python`. Use it explicitly —
  there is no activated venv in a fresh shell.
- `detectron2` is installed from git; `scene-generator` (`sgdata`) is a local
  editable dependency at `../scene_generator`.
- Tests: `./.venv/bin/python -m pytest` (`testpaths = ["tests"]`). Lint: `ruff`.
- Training entry point: `train_net.py`. Runs land in
  `runs/<YYYYMMDDHHMMSS>_<name>/`.

## Two training phases

**Phase 1** — full training on the common-instrument set (`phase1_training_seta_*`).

**Phase 2 "reclassification"** — few-shot adaptation to new/exchanged
instruments from 10-20 images per class.

### Reclassification unfreezes the encoder AND decoder

This is the current and only live configuration. **Do not propose or reason
about classifier-only retraining as the active setup.**

- Current config:
  [`maskdino_R50_surgical_tools_reclassify_v1_real.yaml`](configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_v1_real.yaml)
  → `UNFREEZE_DECODER: True`, `UNFREEZE_ENCODER: True`,
  `ENCODER_LR_MULTIPLIER: 1.0`. Only the ResNet backbone stays frozen.
- **Abandoned:** training only `sem_seg_head.predictor.class_embed`. It gave bad
  results — set-A instruments reached class-agnostic recall of only ~0.0-0.5 in
  the frozen-decoder baseline (`runs/reclassify_v2`), because the frozen decoder
  did not even *localize* them, so no class-head retrain could recover them.
- [`maskdino_R50_surgical_tools_reclassify.yaml`](configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify.yaml)
  is the classifier-only **base** config that the live variants inherit from.
  Its own `RECLASSIFY_FINETUNE` block leaves the unfreeze flags at their `False`
  defaults — reading it alone will mislead you about what is actually run.
  Always check the leaf config.
- Consequence in code: `RECLASSIFY_FINETUNE` forces a labels-only loss **only**
  when both unfreeze flags are `False` ([maskdino.py:249-253](maskdino/maskdino.py#L249-L253)).
  With the decoder trainable, mask + box losses and deep supervision are back,
  anchoring the decoder on segmentation quality.

### Reclassification plumbing

- `NUM_CLASSES: -1` means "derive from dataset metadata" via
  `set_num_classes_from_metadata` and the dataset's compact class mapping.
- When the phase-2 label space is wider than phase 1's, `MODEL.WEIGHTS` must
  point at a checkpoint adapted by
  [`tools/adapt_classifier_checkpoint.py`](tools/adapt_classifier_checkpoint.py),
  which re-indexes `class_embed.{weight,bias}` rows into their new slots and
  fresh-inits the rest.
- `DN: "no"` in the reclass configs, so the `num_classes`-shaped `label_enc` is
  inert.
- `TEST.HUNGARIAN_EVAL` writes `inference/instance_matching_confusion*.{csv,png}`
  and costs roughly 2x per-image eval — keep its `PERIOD` a multiple of
  `EVAL_PERIOD`.

## Architecture facts that are easy to get wrong

Verified in code, and consequential:

- `TWO_STAGE=True` and `LEARN_TGT=False` are the defaults
  ([config.py:58-91](maskdino/config.py#L58-L91)) and no config overrides them.
  So decoder queries are **not** learned embeddings — they are gathered encoder
  memory tokens (`tgt = tgt_undetach.detach()`,
  [maskdino_decoder.py:416](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L416)).
- `class_embed` is a single shared `nn.Linear` called `3 + DEC_LAYERS` times per
  forward: once densely on **every** encoder token
  ([line 403](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L403)) —
  whose output is also the top-k query *selector*
  ([line 406](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L406)) —
  then on the selected tokens (`interm_outputs`), at `initial_pred`, and once per
  decoder layer. Inference uses only the last. `DEC_LAYERS: 9` in every real
  config (inherited from `maskdino_R50_bs16_50ep_3s.yaml`, **not** the `config.py`
  default of 6), so the measured count is 12.
- Scores are **sigmoid**, not softmax: the instance path builds `class_embed` with
  no background row (`semantic_ce_loss=False`) and `instance_inference` reads
  `mask_cls.sigmoid()` ([maskdino.py:553](maskdino/maskdino.py#L553)). Classes are
  independent and do not sum to 1.
- `interm_outputs` and `aux_outputs` carry per-stage logits/boxes/masks and are
  discarded at inference (`del outputs`, [maskdino.py:375](maskdino/maskdino.py#L375)).
  They are the cheap way to see how a class decision evolves across stages.
- `MSDeformAttn` is **forced to the pure-PyTorch fallback**: the compiled CUDA
  kernel corrupts the CUDA stream under this torch/CUDA build (surfaces as
  "invalid resource handle" on the next unrelated CUDA call), and a try/except
  around it does not catch that. See
  [ms_deform_attn.py](maskdino/modeling/pixel_decoder/ops/modules/ms_deform_attn.py).

## Visualization extension

[`maskdino/visualization/`](maskdino/visualization/) holds inspection tooling
that is not part of train or eval. The panels live here rather than in the
external `hdf5_viewer` repo so they cannot drift from the capture code beside
them; the viewer runs on this repo's venv and imports them directly.

> Decoder cross-attention capture ("where did the model look?") lived here too
> and is archived on the `43-attention-visualization-export` branch. It was
> retired because it could not discriminate the failure mode under
> investigation: on a frame pair where one detection succeeded and the other
> collapsed, attention landed in the same places in both (91.5% vs 94.3% of
> mass in-box at layer 0). Restore that branch before re-deriving it.

[`class_map.py`](maskdino/visualization/class_map.py) captures the **dense
encoder-level class map** — "what class does the model think is at each
location?" — the `class_embed(output_memory)` tensor at
[maskdino_decoder.py:403](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L403)
that MaskDINO computes over every encoder token and then discards.

- Needs **no model edit**: `class_embed` is an `nn.Module`, so plain forward
  hooks see every call. The dense call is identified by its sequence length
  matching `sum(h*w)` over the feature levels; all others are at `num_queries`.
- `_topk_tokens` reproduces the two-stage query gate exactly (verified against
  the real `topk_proposals`: 300/300, same order). So `selected_points()` shows
  which peaks of the score field actually received a query.
- Read it as the classifier on **encoder** tokens — the decision gating which
  queries exist — **not** as an explanation of a final detection's label.
- `class_map_panel.py` owns the **visual language** (the sequential ramp, the
  capped categorical palette, the argmax drawing) and `tools/export_class_map.py`
  imports it, so the offline PNG and the live viewer panel cannot diverge.
  Per-class **small multiples**, not a 16-colour argmax mosaic: an argmax map is
  an all-pairs categorical case where only ~3 hues stay distinguishable, so
  `--argmax` caps at 3 classes + "other" and fades by score.
- Score overlays use `overlay_cmap()` — a perceptually-uniform matplotlib map
  (default `inferno`) with an **alpha knee** so
  near-zero scores stay transparent. A single-hue ramp was tried and removed: on
  a photographic frame the background already occupies the lightness channel, and
  a shallow alpha knee hazed the whole image instead of marking it. Avoid
  `viridis` on the synthetic renders — its green/yellow collides with the colours
  the instruments are drawn in. Switchable via `--cmap` and in the panel.
- Wired into `hdf5_viewer`: `run_inference` wraps the one forward pass in the
  capture handle, and `render_class_map` feeds the panel. When rendering over
  a source frame, pass `image_size` as that frame's shape — the default infers
  the *padded model frame*, which crops the overlay to its top-left corner.

[`stage_trace.py`](maskdino/visualization/stage_trace.py) steps through the
**per-query** `class_embed` calls — "where in the decoder does the label stop
agreeing with the mask?" — the diagnostic that `class_map` cannot give, because
it answers for one detection rather than for the frame.

- At inference the model computes a mask **only** at `interm` and the last layer
  (`pred_mask` is gated on `self.training`,
  [maskdino_decoder.py:454](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L454)
  and [:472](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L472)), so
  the intermediate masks needed for a drift comparison do not exist. They are
  reconstructed by running `mask_embed` on the `decoder_output` captured from a
  forward **pre**-hook on `class_embed` — exactly what `forward_prediction_heads`
  would have done. Reconstructed masks are at mask-feature resolution, so
  per-stage scores are self-consistent but sit ~0.002 off the reported score
  (`final_score_check()` reports the gap).
- Stage naming reads the predictor's `two_stage` / `initial_pred` flags. Both
  leading stages are independently optional, so the stage *count* alone cannot
  say which one a single leading stage is.
- **`encoder` and `interm` are the same classifier on the same token and still
  disagree**, which is the one thing guaranteed to confuse anyone comparing
  this panel against the class map:
  `encoder` = `class_embed(output_memory)` ([line 403](maskdino/modeling/transformer_decoder/maskdino_decoder.py#L403)),
  the tensor top-k query selection actually gates on and what `class_map` draws;
  `interm` = the same thing behind the extra `decoder_norm` LayerNorm that
  `forward_prediction_heads` applies. Measured on
  `20260929114041_reclassify_unfreeze_encoder_decoder_v3`: the LayerNorm scales
  the top class by **0.62x** and flips the winning class in **8.3%** of
  detections. Both are carried as separate stages for exactly this reason, and
  `encoder_stage()` prefers the dense one because that is what selection uses.
- Needs `Instances.query_indices` ([maskdino.py](maskdino/maskdino.py),
  `instance_inference`) — the only detection → query mapping. This is the one
  model edit the visualization layer relies on.
- Observed signature on a real failure: the mask track is **flat** at ~0.98
  across all 11 stages while the class track peaks at `layer2` and flips at
  `layer4`. The segmentation was right the whole time; the classifier talked
  itself out of the label. Nothing in the loss penalises that contradiction.

## Conventions

- Prediction filtering is the *caller's* job: `instance_inference()` always
  returns `TEST.DETECTIONS_PER_IMAGE` predictions regardless of confidence.
- DDP: collective-running hooks must be ordered identically on every rank, and
  validation-loss hooks must keep collectives matched. See `maskdino/solver/val_loss.py`
  and `tests/test_hook_order_ddp.py`.
- `/home` quota is tight; checkpoints are ~500 MB. Keep `CHECKPOINT_PERIOD`
  coarse.
