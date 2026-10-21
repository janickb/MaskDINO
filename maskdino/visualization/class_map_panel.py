"""Streamlit panel rendering one frame's dense encoder class map, for hdf5_viewer.

Lives here rather than in the hdf5_viewer repo for the same reason
attention_panel.py does: it must not drift from the capture and rasterization
code next door in class_map.py, and the viewer already runs on MaskDINO's venv.

It renders from an `EncoderClassMap` in memory, not from a file - the viewer
runs the model live, so `model.run_inference` captures the map during the same
forward pass it already does.

This module also owns the *visual language* for class maps (the sequential
ramp, the capped categorical palette), and tools/export_class_map.py imports it
from here, so the offline PNG and the live panel look identical.

streamlit and matplotlib are imported lazily inside the render functions, so
importing this module costs nothing in a non-viewer process - which is what
lets the CLI import the palette helpers without pulling in streamlit.
"""

from __future__ import annotations

import numpy as np

# Sequential magnitude ramps for a score drawn OVER a photographic frame.
#
# The usual chart rule - one hue, light to dark - assumes a flat controlled
# surface. Here the background is a render full of grey metal and reddish
# tissue, so it already occupies the lightness channel: a single-hue ramp has
# only saturation left to carry magnitude, and the mid-range turns to mush.
# A perceptually-uniform map that moves through lightness AND hue (inferno,
# viridis, magma) keeps the whole range separable against arbitrary content.
# These are not rainbows - lightness is monotonic and no hue cycles back, so
# they create no false boundaries.
#
# Default is inferno, which also matches attention_panel.py so the two overlays
# in the viewer read the same way. Checked against real renders: inferno's
# orange/yellow is the only family that collides with nothing in frame. viridis
# is the trap - its green/yellow hotspots sit right on top of the green and cyan
# the instruments themselves are rendered in, so a hotspot and an instrument
# become hard to tell apart. magma and plasma are near-equivalent to inferno;
# cividis is the muted option and the safest for a red/pink-dominated frame.
OVERLAY_CMAPS = ["inferno", "magma", "plasma", "cividis", "viridis"]
DEFAULT_CMAP = "inferno"

# Categorical slots 1-3 of the reference palette - the documented subset that
# validates under *all-pairs* separation (worst CVD dE 9.2 light / 9.4 dark).
# An argmax mosaic is an all-pairs case: any class can border any other, so no
# ordering of the full eight clears the floors. Hence the hard cap at three
# named classes + a neutral "other"; past three the honest move is faceting,
# which is exactly what the per-class small multiples are.
ARGMAX_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
OTHER_GREY = "#9aa0a6"
INK = "#202124"


def overlay_cmap(name=DEFAULT_CMAP, max_alpha=0.92, knee=0.25):
    """`name` with an alpha ramp, for drawing a score on top of a frame.

    The colours are matplotlib's unmodified (so the map stays perceptually
    uniform); only opacity is added, rising from fully transparent at 0 to
    `max_alpha`. That keeps the frame readable where the classifier says
    nothing, which a flat-alpha overlay destroys.

    `knee` is the fraction of the range spent fading in. Below it the overlay
    is mostly transparent, so near-zero scores do not tint the whole image -
    the defect that made the first single-hue ramp unusable was a knee too
    shallow to clear, which hazed the entire frame rather than marking it.
    """
    import matplotlib
    import numpy as _np
    from matplotlib.colors import ListedColormap

    base = matplotlib.colormaps[name].resampled(256)
    colours = base(_np.linspace(0.0, 1.0, 256))
    t = _np.linspace(0.0, 1.0, 256)
    # smooth ramp, steeper than linear below the knee so the low end clears
    ramp = _np.clip((t - knee) / max(1.0 - knee, 1e-6), 0.0, 1.0)
    colours[:, 3] = max_alpha * ramp ** 0.75
    return ListedColormap(colours, name=f"maskdino_{name}")


# Back-compat alias: the first cut of this module exported this name.
def sequential_cmap():
    return overlay_cmap()


def draw_argmax(ax, cmap, base, conf, named, levels=None):
    """Argmax mosaic over `base`, capped at the three classes in `named`.

    Faded by `conf`: an argmax over scores that are all near zero is noise, and
    painting it at full opacity would read as a confident decision the model
    never made.
    """
    import matplotlib.colors
    from matplotlib.patches import Patch

    hh, ww = base.shape[:2]
    arg = cmap.argmax_map(levels=levels)[:hh, :ww]
    coded = np.full(arg.shape, len(named), np.int32)  # default slot: "other"
    for slot, cls in enumerate(named):
        coded[arg == int(cls)] = slot
    palette = np.array([
        matplotlib.colors.to_rgb(c)
        for c in ARGMAX_COLORS[: len(named)] + [OTHER_GREY]
    ])
    strength = conf / max(float(conf.max()), 1e-6)
    ax.imshow(base)
    ax.imshow(np.concatenate([palette[coded], (strength ** 2)[:, :, None]], axis=2),
              interpolation="nearest")
    # identity is never colour-alone: the legend always ships
    ax.legend(
        handles=[Patch(facecolor=ARGMAX_COLORS[i], label=cmap.label(int(c)))
                 for i, c in enumerate(named)] + [Patch(facecolor=OTHER_GREY, label="other")],
        loc="lower right", fontsize=7, framealpha=0.9,
    )


def render_class_map_panel(cmap, rgb, key_prefix="clsmap"):
    """Render the class-map controls and figure for one frame.

    `cmap` is an `EncoderClassMap` covering every encoder token; nothing here
    re-runs the model, so moving a control is cheap.
    """
    import streamlit as st

    def k(name):  # streamlit widgets need stable, unique keys per panel instance
        return f"{key_prefix}_{name}"

    c1, c2, c3 = st.columns([1, 1, 1])
    reduce = c1.radio(
        "Combine levels", ["max", "mean"], horizontal=True, key=k("reduce"),
        help="max = 'any level fires here'; mean is steadier but dilutes a "
             "response that only one scale resolves.",
    )
    level_opts = [
        f"{i} ({int(h)}x{int(w)})" for i, (h, w) in enumerate(cmap.level_spatial_shapes)
    ]
    chosen_levels = c2.multiselect(
        "Feature levels", level_opts, default=level_opts, key=k("levels"),
        help="Level 0 is the finest (stride 8). A thin instrument can wash out "
             "in the coarse levels.",
    )
    if not chosen_levels:
        st.warning("Select at least one feature level.")
        return
    levels = [int(s.split()[0]) for s in chosen_levels]

    top_n = c3.slider("Classes shown", 1, min(12, cmap.num_classes),
                      min(6, cmap.num_classes), key=k("topn"))

    order, peaks = cmap.rank_classes(levels=levels, reduce=reduce)
    default = [cmap.label(int(c)) for c in order[:top_n]]
    labels = [cmap.label(c) for c in range(cmap.num_classes)]
    picked = st.multiselect(
        "Classes", labels, default=default, key=k("classes"),
        help="Ranked by peak score on this frame, not by how much area they claim.",
    )
    if not picked:
        st.warning("Select at least one class.")
        return
    classes = [labels.index(p) for p in picked]

    d1, d2, d3 = st.columns([1, 1, 1])
    show_gate = d1.checkbox("Gate panel (score field + queries placed)", value=True,
                            key=k("gate"))
    show_argmax = d2.checkbox("Argmax mosaic (top 3 + other)", value=False, key=k("argmax"))
    cmap_name = d3.selectbox(
        "Colour ramp", OVERLAY_CMAPS, key=k("cmap"),
        help="Perceptually-uniform maps over a photographic frame. viridis "
             "separates best from red tissue; inferno matches the attention panel.",
    )

    _draw(st, cmap, rgb, classes, peaks, levels, reduce, show_gate, show_argmax,
          cmap_name)

    st.caption(
        f"{cmap.logits.shape[0]} encoder tokens x {cmap.num_classes} classes, "
        f"levels {[tuple(int(v) for v in s) for s in cmap.level_spatial_shapes]} - "
        f"the dense class_embed call the model discards after top-k."
    )
    with st.expander("What am I looking at?"):
        st.markdown(_EXPLAINER)


def _draw(st, cmap, rgb, classes, peaks, levels, reduce, show_gate, show_argmax,
          cmap_name=DEFAULT_CMAP):
    import matplotlib.pyplot as plt

    h, w = cmap.image_size
    base = np.asarray(rgb)[:, :, :3]
    if base.shape[:2] != (h, w):
        st.caption(
            f"Frame is {base.shape[1]}x{base.shape[0]}, model frame {w}x{h} - "
            "showing the overlapping region."
        )
    hh, ww = min(h, base.shape[0]), min(w, base.shape[1])
    base = base[:hh, :ww]

    conf = cmap.confidence_map(levels=levels, reduce=reduce)[:hh, :ww]
    panels = len(classes) + (1 if show_gate else 0) + (1 if show_argmax else 0)
    cols = min(3, panels)
    rows = int(np.ceil(panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.4 * rows * hh / max(ww, 1)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()

    seq = overlay_cmap(cmap_name)
    # One shared score scale across every class panel. Per-panel autoscaling
    # would make a class that never fires look as confident as one that does.
    vmax = float(max(peaks[classes].max(), 1e-6))
    slot = 0

    if show_gate:
        axes[slot].imshow(base)
        axes[slot].imshow(conf, cmap=seq, vmin=0.0, vmax=max(vmax, float(conf.max())))
        pts = cmap.selected_points()
        inside = (pts[:, 0] < ww) & (pts[:, 1] < hh)
        axes[slot].scatter(pts[inside, 0], pts[inside, 1], s=7, facecolors="none",
                           edgecolors="#ffffff", linewidths=0.7)
        axes[slot].set_title(f"gate: max-over-classes + {int(inside.sum())} queries",
                             fontsize=10, color=INK)
        slot += 1

    for cls in classes:
        ax = axes[slot]
        heat = cmap.class_heatmap(int(cls), levels=levels, reduce=reduce)[:hh, :ww]
        ax.imshow(base)
        ax.imshow(heat, cmap=seq, vmin=0.0, vmax=vmax)
        ax.set_title(f"{cmap.label(int(cls))}  peak {peaks[cls]:.2f}", fontsize=10, color=INK)
        slot += 1

    if show_argmax:
        draw_argmax(axes[slot], cmap, base, conf, [int(c) for c in classes[:3]], levels)
        axes[slot].set_title("argmax (top 3 + other, faded by score)", fontsize=10, color=INK)

    mappable = plt.cm.ScalarMappable(cmap=seq)
    mappable.set_clim(0.0, vmax)
    bar = fig.colorbar(mappable, ax=axes.tolist(), fraction=0.02, pad=0.01)
    bar.set_label("class score (sigmoid), shared scale", fontsize=9, color=INK)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


_EXPLAINER = """
MaskDINO applies its one shared classifier (`class_embed`) to **every encoder
token** before any query exists, takes the max over classes to pick the top-k
queries, and then throws the dense tensor away. This panel is that tensor.

- It is the classifier on **encoder** tokens, not on the final refined query
  embedding. Inference reports a different (and better) decision. This is the
  one that **gates which queries exist at all**: a location scored low for every
  class never gets a query, and no decoder layer can recover it.
- So it answers "where does the classifier respond, and to what" - it is **not**
  an explanation of any individual detection's final label.
- The **gate panel** is the max over classes - literally the quantity the top-k
  ranks - with a ring on each token that actually became a query. If an
  instrument is visibly unlit here, the model never even considered it.
- Scores are **sigmoid**, not softmax: classes are independent and need not sum
  to 1 at a location. Several classes firing on the same pixels is normal, and
  tells you the encoder localizes better than it discriminates.
- All class panels share one colour scale, so they are comparable; the per-panel
  number is that class's own peak.
- The **argmax mosaic** is capped at three classes because that is as many as
  stay distinguishable for colour-vision-deficient readers when any class can
  border any other. It fades where nothing fires, since an argmax over near-zero
  scores is noise.
"""
