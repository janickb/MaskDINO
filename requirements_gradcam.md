# Grad-CAM for the crop classifier — raw ask

Asked 2026-11-02, in conversation. Pasted unedited: typos, tone and all.
This file is the ground truth the reviewers check against. Do not reword it.

---

> checkout the different models i am using for instance convnext-tiny or resnet50 and resnet50d,
> i want to visualize gradcam for them. start with resnet50 implement gradcam during inference.
> for instance, when i --eval-only or --viz-only i want to see also where the model focuses on.

After the plan was presented:

> good plan, ill copy the crops from the other server to here

---

## Decisions (2026-11-02/03, from the user)

The verbatim section above is unchanged and remains the ground truth. This section records
only what was decided when asked, so a later reader does not think the unbuilt parts were
forgotten.

Asked because "where the model focuses on" has three free parameters the ask does not pin: which
crops get a map, which class the map explains, and whether the existing error-sheet tool grows
one too.

- **`--eval-only` writes one sheet per GT class, errors first.** Up to `--gradcam-per-class`
  crops each (default 8): the most-confident mistakes first, then the most-confident correct
  ones, so every class appears regardless of which frames the val pass happened to start with
  and the forcep01-04 confusions land on the first page. "The first N crops in pass order" and
  "misclassified only" were both offered and declined — the latter because without correct crops
  there is no reference for what a healthy CAM looks like.
- **The CAM explains the predicted class only.** One backward per crop, the standard Grad-CAM
  reading: where the model looked to say what it said. Rendering a second map for the GT class
  on the crops it got wrong ("where was the evidence you missed") was offered and declined, as
  was both maps on every crop.
- **`tools/viz_misclassified.py` gets a CAM panel too.** It is the tool already run to inspect
  mistakes, and it reuses the same `cropcls/gradcam.py`. Keeping the change to
  `train_cropcls.py` alone was offered and declined.
- **No new config field.** Grad-CAM is a CLI concern (`--gradcam-per-class`, `--gradcam-layer`).
  A `CropClsConfig` field would be replayed from a finished run's own `config.json` by the
  `--eval-only` probe at `train_cropcls.py:160`, which would make a visualization switch part of
  the architecture record.
- **`--viz-only` keeps working without a checkpoint.** The CAM column appears only when one
  resolves (`weights=` override, else `<output_dir>/model_final.pth`); otherwise the existing
  three-column sheet renders and one log line says why. The step-4 gate runs before any
  checkpoint exists.

**Out of scope (explicit):** `--hungarian` and every metric path (`cropcls/export.py`) - no
number this produces may be ranked; the detector's own attention, which is a different artifact
for a different model and already exists on branch `49-attention-visualization-export`
(`maskdino/visualization/attention.py`); `tools/viz_class_crops.py`, `tools/class_report.py`,
`tools/loss_breakdown.py`; a `pytorch-grad-cam` or `captum` dependency; any change to
`maskdino/` or `train_net.py`.
