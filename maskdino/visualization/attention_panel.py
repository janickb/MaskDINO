"""Streamlit panel rendering one frame's decoder attention, for hdf5_viewer.

Lives here rather than in the hdf5_viewer repo so it cannot drift from the
capture and rasterization code next door in attention.py - the viewer already
runs on MaskDINO's venv and already imports MaskDINO for its prediction
overlay.

It renders from an `AttentionRecord` in memory, not from a file, because the
viewer runs the model live: `model.run_inference` captures attention during the
same forward pass it already does, so there is nothing to export first. Reading
a `.attn.hdf5` sidecar (the batch workflow from tools/export_attention.py) is a
one-liner on top:

    from maskdino.visualization.attention import read_attention
    render_attention_panel(read_attention(path), rgb)

streamlit and matplotlib are imported lazily inside the render functions, so
importing this module costs nothing in a non-viewer process.
"""

from __future__ import annotations

import numpy as np

from .attention import heatmap


def _upscale(heat, shape):
    """Nearest-neighbour upscale of a downscaled heatmap back to frame size."""
    if heat.shape == tuple(shape):
        return heat
    ys = (np.arange(shape[0]) * heat.shape[0] / shape[0]).astype(int).clip(0, heat.shape[0] - 1)
    xs = (np.arange(shape[1]) * heat.shape[1] / shape[1]).astype(int).clip(0, heat.shape[1] - 1)
    return heat[np.ix_(ys, xs)]


def render_attention_panel(record, rgb, score_threshold=0.0, masks=None, key_prefix="attn"):
    """Render the attention controls and overlay for one frame.

    `record` is an AttentionRecord covering every decoder query (the viewer
    captures them all, like it keeps all 100 raw detections, so moving the
    score slider never re-runs inference); `score_threshold` picks which ones
    are offered in the detection list.

    `masks` is an optional (D, H, W) array aligned with `record`, for callers
    that captured without `include_masks` because they already hold their own
    copy - index it with `record.detection_index` before passing it in.
    """
    import streamlit as st

    def k(name):  # streamlit widgets need stable, unique keys per panel instance
        return f"{key_prefix}_{name}"

    kept = np.nonzero(np.asarray(record.scores) > score_threshold)[0]
    if len(kept) == 0:
        st.caption(f"No detection above the score threshold ({score_threshold:.2f}).")
        return
    # highest score first, so the detection picker opens on the best one
    kept = kept[np.argsort(-np.asarray(record.scores)[kept], kind="stable")]

    det = st.selectbox(
        "Detection", list(kept), format_func=record.label, key=k("det"),
        help="Each entry is one decoder query, identified by the detection it produced.",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        layer_options = ["Σ all layers"] + [f"layer {i}" for i in range(record.num_layers)]
        layer_choice = st.select_slider("Decoder layer", layer_options, key=k("layer"))
    with col2:
        mode = st.radio("Mode", ["heatmap", "points"], horizontal=True, key=k("mode"))

    heads = st.multiselect(
        "Attention heads", list(range(record.num_heads)),
        default=list(range(record.num_heads)), key=k("heads"),
        help="Each head attends independently; its weights sum to 1 over levels x points.",
    )
    if not heads:
        st.warning("Select at least one head.")
        return

    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Overlay alpha", 0.0, 1.0, 0.6, 0.05, key=k("alpha"))
    show_mask = c2.checkbox("Mask outline", value=True, key=k("mask"))
    show_refs = c3.checkbox("Reference box walk", value=False, key=k("refs"))

    layers = (
        list(range(record.num_layers)) if layer_choice.startswith("Σ")
        else [int(layer_choice.split()[-1])]
    )

    base = np.asarray(rgb)[:, :, :3]
    h, w = record.image_size
    if base.shape[:2] != (h, w):
        # the record is in the padded model frame; with SIZE_DIVISIBILITY=32 and
        # divisible renders these match, but crop rather than silently misalign
        st.caption(
            f"Frame is {base.shape[1]}x{base.shape[0]}, model frame {w}x{h} - "
            "showing the overlapping region."
        )
    hh, ww = min(h, base.shape[0]), min(w, base.shape[1])
    base = base[:hh, :ww]

    _draw(st, record, det, layers, heads, base, (hh, ww), mode, alpha,
          show_mask, show_refs, masks)

    st.caption(
        f"query #{int(record.query_index[det])} · "
        f"{record.num_layers} decoder layers × {record.num_heads} heads × "
        f"{record.sampling_locations.shape[3]} levels × "
        f"{record.sampling_locations.shape[4]} points per layer"
    )
    with st.expander("What am I looking at?"):
        st.markdown(_EXPLAINER)


def _draw(st, rec, det, layers, heads, base, shape, mode, alpha, show_mask,
          show_refs, masks=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hh, ww = shape
    fig, ax = plt.subplots(figsize=(7, 7 * hh / max(ww, 1)))
    ax.imshow(base)
    ax.set_axis_off()

    if mode == "heatmap":
        heat = _upscale(heatmap(rec, det, layers=layers, heads=heads, downscale=2), (hh, ww))
        peak = float(heat.max())
        ax.imshow(heat / peak if peak > 0 else heat, cmap="inferno", alpha=alpha,
                  interpolation="bilinear")
    else:
        locs = rec.sampling_locations[det][np.ix_(layers, heads)].reshape(-1, 2)
        weights = rec.attention_weights[det][np.ix_(layers, heads)].reshape(-1)
        peak = float(weights.max()) or 1.0
        ax.scatter(locs[:, 0], locs[:, 1], s=300 * weights / peak, c=weights,
                   cmap="inferno", alpha=min(1.0, alpha + 0.2), edgecolors="none")

    mask_src = rec.pred_masks if rec.pred_masks is not None else masks
    if show_mask and mask_src is not None:
        ax.contour(np.asarray(mask_src[det])[:hh, :ww].astype(float), levels=[0.5],
                   colors="white", linewidths=1.2)

    if show_refs:
        # normalized cxcywh -> pixels; early layers cool, late layers warm
        cmap = plt.get_cmap("cool")
        for i, (cx, cy, bw, bh) in enumerate(rec.reference_boxes[det]):
            ax.add_patch(Rectangle(
                ((cx - bw / 2) * ww, (cy - bh / 2) * hh), bw * ww, bh * hh,
                fill=False, edgecolor=cmap(i / max(rec.num_layers - 1, 1)), linewidth=1.0,
            ))
        x0, y0, x1, y1 = rec.pred_boxes[det]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="lime", linewidth=1.6, linestyle="--"))

    fig.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


_EXPLAINER = """
MaskDINO's decoder cross-attention is **deformable**: instead of attending to
every pixel, each query samples a handful of points per head per feature level,
around a reference box it refines layer by layer. So this is a sparse set of
weighted sample points, rasterized into a heatmap for display - not a dense
attention map.

- **Σ all layers** is the practical "where did this query look". A single layer
  is the better debugging view: self-attention mixes queries between layers, so
  the last layer's points are not a detection's full provenance.
- **points** mode is the truthful rendering (marker area ∝ attention weight);
  **heatmap** blurs each point by its feature level's stride, which is honest
  about receptive field but implies more coverage than was actually sampled.
- **Reference box walk** shows the box each layer attended around, cool → warm,
  with the final predicted box dashed in green. With two-stage query selection
  the layer-0 box is already close, so expect small refinements, not a big walk.
- Attention mass concentrates *around* an object, not strictly inside its mask -
  context near the boundary is informative. For a thin instrument, expect the
  hotspots at the tips.
"""
