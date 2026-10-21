"""Unit tests for maskdino/visualization/attention_panel.py.

streamlit is not installed in this repo's venv (it belongs to hdf5_viewer, which
imports this panel), so the tests inject a stub streamlit module and drive the
panel through it. That still exercises the real logic - widget wiring, layer and
head selection, the coordinate maths and the matplotlib drawing path, which is
the part most likely to break - without the viewer being present.

matplotlib IS installed, so the figures are genuinely rendered, not faked.
"""
import sys

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

# imported as real package modules (not by file path like the other tests): the
# panel uses a relative import, and streamlit stays stubbable because the panel
# imports it lazily, inside the render functions.
from maskdino.visualization import attention as attn
from maskdino.visualization import attention_panel as panel_mod

# ------------------------------------------------------------- streamlit stub


class _Col:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


class _FakeStreamlit:
    """Minimal streamlit surface: every widget returns a deterministic choice."""

    def __init__(self, layer_choice=0, mode="heatmap", heads=None, show_refs=True):
        self.layer_choice = layer_choice  # index into the layer option list
        self.mode = mode
        self.heads = heads
        self.show_refs = show_refs
        self.figures = []
        self.offered = []
        self.infos = []
        self.warnings = []
        self.captions = []

    # caching is a passthrough here; the real one is keyed on sidecar mtime
    def cache_data(self, *a, **kw):
        def deco(fn):
            return fn
        return deco if not a else a[0]

    def selectbox(self, label, options, format_func=None, **kw):
        opts = list(options)
        self.offered.append(opts)
        if format_func:  # exercise the label formatter too
            format_func(opts[0])
        return opts[0]

    def select_slider(self, label, options, **kw):
        return list(options)[self.layer_choice]

    def radio(self, label, options, **kw):
        return self.mode

    def multiselect(self, label, options, default=None, **kw):
        return self.heads if self.heads is not None else list(default)

    def slider(self, label, lo, hi, default, step=None, **kw):
        return default

    def checkbox(self, label, value=False, **kw):
        return self.show_refs if "Reference" in label else value

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Col(self) for _ in range(n)]

    def expander(self, *a, **kw):
        return _Col(self)

    def pyplot(self, fig, **kw):
        self.figures.append(fig)

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def caption(self, msg):
        self.captions.append(msg)

    def markdown(self, msg):
        pass


@pytest.fixture
def panel(monkeypatch):
    """The panel with a stub streamlit in place of the real one."""
    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    return panel_mod, fake_st


RGB = None  # built per test from FRAME_H/FRAME_W


FRAME_H, FRAME_W = 96, 128
LAYERS, HEADS, LEVELS, POINTS = 4, 8, 3, 4


def _record(num_det=2):
    rng = np.random.default_rng(0)
    locs = rng.uniform(10, 80, (num_det, LAYERS, HEADS, LEVELS, POINTS, 2)).astype(np.float32)
    weights = rng.random((num_det, LAYERS, HEADS, LEVELS, POINTS)).astype(np.float32)
    if num_det:  # softmax-like normalization per (det, layer, head)
        weights /= weights.reshape(num_det, LAYERS, HEADS, -1).sum(-1)[..., None, None]
    masks = np.zeros((num_det, FRAME_H, FRAME_W), bool)
    masks[:, 20:40, 30:60] = True
    return attn.AttentionRecord(
        sampling_locations=locs,
        attention_weights=weights,
        reference_boxes=np.tile(np.array([0.5, 0.4, 0.2, 0.3], np.float32), (num_det, LAYERS, 1)),
        query_index=np.arange(num_det, dtype=np.int32),
        detection_index=np.arange(num_det, dtype=np.int32),
        scores=np.linspace(0.9, 0.6, num_det).astype(np.float32),
        pred_classes=np.zeros(num_det, np.int32),
        pred_boxes=np.tile(np.array([30, 20, 60, 40], np.float32), (num_det, 1)),
        level_spatial_shapes=np.array([[24, 32], [12, 16], [6, 8]], np.int32),
        image_size=(FRAME_H, FRAME_W),
        original_size=(FRAME_H, FRAME_W),
        pred_masks=masks,
        class_names=["forcep"],
        checkpoint="/runs/x/model_final.pth",
    )


def _rgb():
    return np.zeros((FRAME_H, FRAME_W, 3), np.uint8)


# ------------------------------------------------------------------- the panel


def test_renders_a_figure_for_a_detection(panel):
    mod, st = panel
    mod.render_attention_panel(_record(), _rgb())
    assert len(st.figures) == 1
    ax = st.figures[0].axes[0]
    assert ax.images, "no image drawn"
    assert st.captions and "query #" in st.captions[0]


def test_summed_and_single_layer_both_render(panel):
    mod, st = panel
    rec, rgb = _record(), _rgb()
    st.layer_choice = 0  # the "sum all layers" entry
    mod.render_attention_panel(rec, rgb)
    st.layer_choice = 2  # "layer 1"
    mod.render_attention_panel(rec, rgb)
    assert len(st.figures) == 2
    # summed and single-layer heatmaps must differ
    a, b = (f.axes[0].images[1].get_array() for f in st.figures)
    assert not np.allclose(a, b)


def test_points_mode_plots_one_marker_per_sample(panel):
    mod, st = panel
    st.mode = "points"
    mod.render_attention_panel(_record(), _rgb())
    ax = st.figures[0].axes[0]
    assert len(ax.collections) >= 1
    assert len(ax.collections[0].get_offsets()) == LAYERS * HEADS * LEVELS * POINTS


def test_reference_box_walk_adds_a_patch_per_layer_plus_the_prediction(panel):
    mod, st = panel
    st.show_refs = True
    mod.render_attention_panel(_record(), _rgb())
    from matplotlib.patches import Rectangle

    rects = [p for p in st.figures[0].axes[0].patches if isinstance(p, Rectangle)]
    assert len(rects) == LAYERS + 1  # one per layer, plus the dashed final box


def test_empty_head_selection_warns_instead_of_drawing(panel):
    mod, st = panel
    st.heads = []
    mod.render_attention_panel(_record(), _rgb())
    assert st.figures == []
    assert any("at least one head" in w for w in st.warnings)


def test_score_threshold_filters_the_detection_list(panel):
    """The viewer captures every query and filters here, so moving its score
    slider must never need another inference run."""
    mod, st = panel
    rec = _record(num_det=3)  # scores 0.9, 0.75, 0.6
    mod.render_attention_panel(rec, _rgb(), score_threshold=0.7)
    assert len(st.offered) == 1 and st.offered[0] == [0, 1]

    mod.render_attention_panel(rec, _rgb(), score_threshold=0.95)
    assert st.figures and len(st.figures) == 1  # nothing new drawn
    assert any("No detection above" in c for c in st.captions)


def test_externally_held_masks_are_used_when_the_record_has_none(panel):
    """The viewer captures with include_masks=False and passes its own copy."""
    mod, st = panel
    rec = _record()
    rec.pred_masks = None
    masks = np.zeros((len(rec), FRAME_H, FRAME_W), bool)
    masks[:, 20:40, 30:60] = True
    mod.render_attention_panel(rec, _rgb(), masks=masks)
    # the contour is drawn from the externally supplied mask
    assert st.figures[0].axes[0].collections, "no mask contour drawn"


def test_rgb_larger_than_the_model_frame_is_cropped(panel):
    mod, st = panel
    big = np.zeros((FRAME_H + 40, FRAME_W + 40, 3), np.uint8)
    mod.render_attention_panel(_record(), big)
    assert st.figures[0].axes[0].images[0].get_array().shape[:2] == (FRAME_H, FRAME_W)
    assert any("overlapping region" in c for c in st.captions)


# --------------------------------------------------------------------- helpers


def test_upscale_restores_shape_without_inventing_mass():
    mod = panel_mod
    small = np.zeros((4, 4), np.float32)
    small[1, 1] = 1.0
    big = mod._upscale(small, (8, 8))
    assert big.shape == (8, 8)
    assert big.max() == 1.0
    assert (big > 0).sum() == 4  # one source pixel -> one 2x2 block
    same = mod._upscale(small, (4, 4))
    assert same is small
