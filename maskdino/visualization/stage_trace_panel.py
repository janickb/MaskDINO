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


def render_stage_trace_panel(trace, score_threshold=0.5, key_prefix="trace"):
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

    with st.expander("What am I looking at?"):
        st.markdown(_EXPLAINER)


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
    ax.annotate(f"score threshold {score_threshold:.2f}",
                xy=(0, score_threshold), xytext=(0, 4), textcoords="offset points",
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

- Stages run `interm` (two-stage query selection) -> `initial_pred` -> one per
  decoder layer. `interm` and `initial_pred` read the same vector before any
  decoder layer runs, so they are usually identical.
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
