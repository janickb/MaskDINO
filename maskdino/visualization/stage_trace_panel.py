"""Streamlit panel for stage_trace: where does the label stop agreeing with
the mask?

One chart, one y-axis. Class confidence and mask confidence are both
probabilities, so they share a scale honestly - no second axis. Mask *area*
is a different unit and therefore never shares the axis; it lives in the table
view below instead.

Colour follows the data-viz method: at most three class series, drawn from the
validated categorical slots in fixed order (blue, orange, aqua - the three that
clear the all-pairs gate in both modes). The mask track is deliberately NOT a
categorical hue - it is a reference line in neutral ink, dashed, so it is
separable by shape as well as colour. Series are direct-labelled at the right
edge in text ink, and a table view carries every number, which is what the
relief rule requires for the low-contrast slots.
"""

from __future__ import annotations

import numpy as np

# Validated categorical slots 1-3 (see the data-viz reference palette), plus
# the ink and surface roles, for both modes. Light first, dark second.
_THEME = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d8d7d2",
        "surface": "#fcfcfb",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "surface": "#1a1a19",
    },
}

MAX_CLASS_SERIES = 3


def _theme(st):
    try:
        base = st.get_option("theme.base") or "light"
    except Exception:
        base = "light"
    return _THEME["dark" if str(base).lower() == "dark" else "light"]


def render_stage_trace_panel(trace, score_threshold=0.5, rgb=None, key_prefix="trace"):
    """Render the per-stage class-vs-mask chart for one detection."""
    import streamlit as st

    def k(name):
        return f"{key_prefix}_{name}"

    if len(trace) == 0:
        st.caption("No detections to trace.")
        return

    det = st.selectbox(
        "Detection", list(range(len(trace))), format_func=trace.label, key=k("det"),
        help="Each entry is one decoder query, identified by the detection it produced.",
    )

    _agreement(st, trace, det, score_threshold)

    probs = trace.class_probs(det)
    classes = trace.top_classes(det, k=2)[:MAX_CLASS_SERIES]
    theme = _theme(st)

    _chart(st, trace, det, probs, classes, theme, score_threshold)
    _readout(st, trace, det, probs, score_threshold)

    with st.expander("All stages, as numbers"):
        import pandas as pd

        table = {"stage": trace.stage_names}
        for i, c in enumerate(classes):
            table[trace.class_name(c)] = probs[:, c].round(3)
        table["mask score"] = trace.mask_scores[det].round(3)
        table["mask px"] = trace.mask_areas[det]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    _mask_clip(st, trace, det, rgb, k)

    with st.expander("What am I looking at?"):
        st.markdown(_EXPLAINER)


def _agreement(st, trace, det, score_threshold):
    """How often the encoder's label survives the decoder, over this frame."""
    n_agree, n_total = trace.agreement_rate(min_score=score_threshold)
    if n_total == 0:
        st.caption(
            "No encoder-level prediction to compare against "
            "(TWO_STAGE is off, so there is no `interm` stage)."
            if trace.encoder_stage() is None else
            f"No detection above the score threshold ({score_threshold:.2f})."
        )
        return

    this = trace.agrees_with_final(det)
    pct = 100.0 * n_agree / n_total
    c1, c2 = st.columns([1, 2])
    c1.metric("encoder label kept", f"{pct:.0f}%", help=(
        "Share of this frame's detections whose leading class at `interm` - the "
        "encoder's own verdict on the query it selected - is still leading at the "
        "last decoder layer."))
    c2.caption(
        f"{n_agree} of {n_total} detections above {score_threshold:.2f} keep the "
        f"encoder's label through all {trace.num_stages} stages. "
        + ("**This detection keeps it.**" if this else
           "**This detection does not** - the decoder overturned the encoder.")
    )


def _chart(st, trace, det, probs, classes, theme, score_threshold):
    import matplotlib.pyplot as plt

    x = np.arange(trace.num_stages)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    # recessive grid and spines - the data should be the only strong ink
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])

    # the score threshold the viewer is filtering on - context, not data
    ax.axhline(score_threshold, color=theme["muted"], linewidth=1.0, linestyle=":",
               alpha=0.8)
    # right-aligned: the left end is where the encoder->interm drop lands
    ax.annotate(f"score threshold {score_threshold:.2f}",
                xy=(trace.num_stages - 1, score_threshold), xytext=(0, 5),
                textcoords="offset points", ha="right",
                color=theme["muted"], fontsize=8)

    # where the leading class changes hands
    div = trace.divergence(det)
    if div is not None:
        ax.axvline(div[0], color=theme["muted"], linewidth=1.0, alpha=0.5)
        ax.annotate(f"lead changes at {div[1]}", xy=(div[0], 0.0),
                    xytext=(4, 6), textcoords="offset points",
                    color=theme["muted"], fontsize=8)

    for i, c in enumerate(classes):
        ax.plot(x, probs[:, c], color=theme["series"][i], linewidth=2.0,
                marker="o", markersize=5, label=trace.class_name(c))

    # the mask track: neutral ink + dashed, so it never competes with a class
    ax.plot(x, trace.mask_scores[det], color=theme["muted"], linewidth=2.0,
            linestyle="--", marker="s", markersize=4, label="mask score")

    # direct labels at the right edge, in text ink (identity comes from the
    # adjacent mark, not from coloured text)
    for i, c in enumerate(classes):
        ax.annotate(trace.class_name(c), xy=(x[-1], probs[-1, c]),
                    xytext=(6, 0), textcoords="offset points",
                    color=theme["ink"], fontsize=8, va="center")
    ax.annotate("mask", xy=(x[-1], trace.mask_scores[det, -1]),
                xytext=(6, 0), textcoords="offset points",
                color=theme["muted"], fontsize=8, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(trace.stage_names, rotation=45, ha="right", fontsize=8,
                       color=theme["muted"])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("confidence", color=theme["muted"], fontsize=9)
    ax.tick_params(axis="y", colors=theme["muted"], labelsize=8)
    ax.set_xlim(-0.3, trace.num_stages + 1.4)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False,
                       fontsize=8, ncol=4)
    for text in legend.get_texts():
        text.set_color(theme["ink"])

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _mask_clip(st, trace, det, rgb, k):
    """The mask's own evolution across stages, as a clip.

    Led by whether it moves at all: for most detections the mask is settled
    before the first decoder layer (median first-stage IoU ~0.98), so the clip
    would be a still image and saying so is more useful than playing it.
    """
    first_iou, moved = trace.mask_moves(det)
    if not np.isfinite(first_iou):
        return

    with st.expander(
        f"Mask evolution - IoU {first_iou:.2f} from first stage to final"
        + ("  (it moves)" if moved else "  (essentially static)")
    ):
        if not moved:
            st.caption(
                f"This mask is {first_iou:.1%} of its final shape before any decoder "
                "layer runs, so the clip below is near-still. The decoder refined the "
                "*label*, not the segmentation - which is the usual case here."
            )
        stored = 0 if trace.stage_masks is None else len(trace.stage_masks)
        if det >= stored:
            st.caption(
                f"No per-stage masks stored for #{det}: only the {stored} "
                "highest-scoring detections keep them, to bound the cache entry. "
                f"Clips are available for #0-#{stored - 1}." if stored else
                "Per-stage masks were not captured for this frame."
            )
            return
        gif = _build_clip(trace, det, rgb)
        if gif is None:
            st.caption("Need the frame image to render the clip.")
            return
        st.image(gif, caption=f"{trace.label(det)} - mask probability per stage")
        st.caption(
            "Cropped to the detection's box. The `encoder` stage is absent: the "
            "model builds no mask there."
        )


def _build_clip(trace, det, rgb, pad=0.25, size=320, ms=420):
    """Animated GIF of the per-stage mask over the frame, cropped to the box."""
    import io

    from PIL import Image, ImageDraw

    if rgb is None:
        return None
    base = np.asarray(rgb)[:, :, :3]
    H, W = base.shape[:2]
    mh, mw = trace.stage_masks.shape[-2:]

    x0, y0, x1, y1 = trace.pred_boxes[det]
    m = pad * max(x1 - x0, y1 - y0)
    cx0, cy0 = max(int(x0 - m), 0), max(int(y0 - m), 0)
    cx1, cy1 = min(int(x1 + m), W), min(int(y1 + m), H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = base[cy0:cy1, cx0:cx1]

    frames = []
    # stage_masks has no encoder stage; align names to the tail of the list
    names = trace.stage_names[trace.num_stages - trace.stage_masks.shape[1]:]
    for i, name in enumerate(names):
        prob = trace.stage_masks[det, i].astype(np.float32) / 255.0
        # mask resolution -> frame resolution by nearest-neighbour, then crop
        ys = (np.arange(H) * mh // H).clip(0, mh - 1)
        xs = (np.arange(W) * mw // W).clip(0, mw - 1)
        full = prob[np.ix_(ys, xs)][cy0:cy1, cx0:cx1]
        over = crop.astype(np.float32).copy()
        # tint the foreground; alpha follows the probability so a fuzzy mask
        # looks fuzzy rather than being thresholded into false crispness
        alpha = np.clip((full - 0.5) * 2.0, 0, 1)[..., None]
        over = over * (1 - 0.55 * alpha) + np.array([255.0, 90.0, 40.0]) * 0.55 * alpha
        img = Image.fromarray(over.clip(0, 255).astype(np.uint8))
        img = img.resize((size, int(size * img.height / max(img.width, 1))))
        ImageDraw.Draw(img).text((6, 4), name, fill=(255, 255, 255))
        frames.append(img.convert("P", palette=Image.ADAPTIVE))

    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=ms, loop=0)
    return buf.getvalue()


def _readout(st, trace, det, probs, score_threshold):
    cls = int(trace.pred_classes[det])
    final_cls = float(probs[-1, cls])
    final_mask = float(trace.mask_scores[det, -1])
    rebuilt, reported = trace.final_score_check(det)

    c1, c2, c3 = st.columns(3)
    c1.metric("class confidence", f"{final_cls:.3f}")
    c2.metric("mask score", f"{final_mask:.3f}")
    c3.metric("final = class x mask", f"{reported:.3f}",
              delta=f"{reported - score_threshold:+.3f} vs threshold")

    div = trace.divergence(det)
    if div is not None:
        _, stage, was, now = div
        peak = int(np.argmax(probs[:, trace.class_names.index(was)])) if was in trace.class_names else None
        st.warning(
            f"The leading class changes at **{stage}**: {was} -> {now}."
            + (f" {was} peaked at {trace.stage_names[peak]}."
               if peak is not None else "")
            + "  The mask is unchanged across that switch if its track is flat -"
            " which means the segmentation never agreed with the relabel."
        )
    else:
        st.caption(f"One class leads at every stage - no identity flip. "
                   f"Reconstructed final score {rebuilt:.3f} vs reported {reported:.3f}.")


_EXPLAINER = """
MaskDINO reads **both** heads off the same 256-d query vector at every stage,
but they are siblings, not a pipeline - `class_embed` never sees the mask. This
chart puts the two readouts on one axis (both are probabilities) so the moment
they stop agreeing is visible.

- Stages run `encoder` -> `interm` -> `initial_pred` -> one per decoder layer.
- **`encoder` and `interm` are the same classifier on the same token and still
  disagree.** `encoder` is `class_embed(output_memory)` - the tensor two-stage
  query selection runs its top-k on, and what the Encoder class map panel
  shows. `interm` is the same thing behind an extra `decoder_norm` LayerNorm
  that `forward_prediction_heads` applies. On this checkpoint that scales the
  top class by ~0.62x and flips the winner outright in ~8% of detections, so a
  value read off the class map will not match the second point on this chart.
  The drop from the first to the second point is that LayerNorm, not a decision.
- The `encoder` stage has no mask score: the model never builds a mask there,
  so the mask track starts at `interm`.
- **A flat mask track with a wandering class track is the signature case**: the
  segmentation was right the whole time and the classifier talked itself out of
  the label. Nothing in the loss penalises that contradiction.
- Mask score here is the mean sigmoid probability inside the stage's own
  binarised mask - the quantity `instance_inference` multiplies the class score
  by. It is computed at mask-feature resolution (1/4 of the frame), so it is
  consistent *across stages* but sits a little off the reported score.
- Intermediate-layer masks do not exist at inference: the model only computes a
  mask at `interm` and the last layer. The rest are reconstructed by running
  `mask_embed` on the captured `decoder_output`, which is exactly what
  `forward_prediction_heads` would have done.
"""
