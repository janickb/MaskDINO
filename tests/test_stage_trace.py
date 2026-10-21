"""Unit tests for maskdino/visualization/stage_trace.py and its panel.

The capture is pure forward hooks on `class_embed` / the predictor, so a toy
module that merely *calls* those the way MaskDINODecoder.forward does exercises
the real code. streamlit is stubbed for the panel half; matplotlib is real, so
the figures are genuinely rendered.
"""
import sys
import types

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
torch = pytest.importorskip("torch")

from maskdino.visualization import stage_trace as st_mod

NQ, NCLS, DIM, MASK_H, MASK_W = 12, 5, 16, 24, 32
LAYERS = 4
DENSE_TOKENS = 97  # deliberately != NQ, like the real encoder-token call


class _ToyPredictor(torch.nn.Module):
    """Calls class_embed the way MaskDINODecoder.forward does: once densely on
    encoder tokens, then once per per-query stage (interm, initial, layers)."""

    def __init__(self, num_layers=LAYERS, with_initial=True):
        super().__init__()
        self.class_embed = torch.nn.Linear(DIM, NCLS)
        self.mask_embed = torch.nn.Linear(DIM, DIM)
        self.num_layers = num_layers
        self.num_queries = NQ
        self.with_initial = with_initial
        # the real MaskDINODecoder stores both, and stage naming reads them
        self.two_stage = True
        self.initial_pred = with_initial

    def forward(self, x, mask_features, masks=None, targets=None):
        self.class_embed(torch.randn(1, DENSE_TOKENS, DIM))       # dense call
        n_lead = 2 if self.with_initial else 1
        for _ in range(n_lead + self.num_layers):
            self.class_embed(torch.randn(1, NQ, DIM))
        return {"pred_logits": torch.randn(1, NQ, NCLS)}, None


def _toy_model(**kw):
    predictor = _ToyPredictor(**kw)
    return types.SimpleNamespace(
        sem_seg_head=types.SimpleNamespace(predictor=predictor)
    ), predictor


def _run(model, predictor):
    feats = torch.randn(1, DIM, MASK_H, MASK_W)
    with st_mod.capture_stage_trace(model) as cap:
        with torch.no_grad():
            predictor(torch.randn(1, 3, 8, 8), feats)
    return cap


def _instances(queries, scores, classes=None):
    from detectron2.structures import Boxes, Instances

    inst = Instances((128, 160))
    inst.query_indices = torch.as_tensor(queries, dtype=torch.int64)
    inst.scores = torch.as_tensor(scores, dtype=torch.float32)
    inst.pred_classes = torch.as_tensor(
        classes if classes is not None else [0] * len(queries), dtype=torch.int64)
    inst.pred_boxes = Boxes(torch.zeros(len(queries), 4))
    return inst


# ------------------------------------------------------------------ capture


def test_capture_is_pure_hooks_and_leaves_nothing_behind():
    model, predictor = _toy_model()
    before_ce = dict(predictor.class_embed._forward_hooks)
    _run(model, predictor)
    assert predictor.class_embed._forward_hooks == before_ce
    assert predictor.class_embed._forward_pre_hooks == {}
    assert predictor._forward_pre_hooks == {}


def test_dense_encoder_call_is_excluded_from_the_stages():
    model, predictor = _toy_model()
    cap = _run(model, predictor)
    trace = cap.record(_instances([0, 1], [0.9, 0.8]))
    # 2 leading stages + LAYERS; the DENSE_TOKENS call must not appear
    assert trace.num_stages == 2 + LAYERS
    assert trace.class_logits.shape == (2, 2 + LAYERS, NCLS)


def test_stage_names_follow_the_decoder_call_order():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0], [0.9]))
    assert trace.stage_names == ["interm", "initial"] + [f"layer{i}" for i in range(LAYERS)]


def test_stage_names_without_initial_pred():
    """interm and initial_pred are independently optional, so the stage count
    alone cannot say which of the two a single leading stage is - the naming
    reads the predictor's two_stage / initial_pred flags instead."""
    model, predictor = _toy_model(with_initial=False)
    trace = _run(model, predictor).record(_instances([0], [0.9]))
    assert trace.stage_names == ["interm"] + [f"layer{i}" for i in range(LAYERS)]


def test_mask_features_are_captured_from_the_predictor_call():
    model, predictor = _toy_model()
    cap = _run(model, predictor)
    assert cap.mask_features is not None
    assert cap.mask_features.shape[-2:] == (MASK_H, MASK_W)


def test_missing_forward_pass_is_a_clear_error():
    model, _predictor = _toy_model()
    with st_mod.capture_stage_trace(model) as cap:
        pass
    with pytest.raises((RuntimeError, IndexError)):
        cap.record(_instances([0], [0.9]))


def test_missing_query_indices_is_a_clear_error():
    model, predictor = _toy_model()
    cap = _run(model, predictor)
    inst = _instances([0], [0.9])
    inst.remove("query_indices")
    with pytest.raises(ValueError, match="no `query_indices` field"):
        cap.record(inst)


def test_non_maskdino_model_is_rejected():
    with pytest.raises(TypeError, match="no MaskDINO prediction heads"):
        with st_mod.capture_stage_trace(types.SimpleNamespace()):
            pass


# ------------------------------------------------------------------- record


def test_record_dedupes_queries_and_honours_max_detections():
    model, predictor = _toy_model()
    cap = _run(model, predictor)
    inst = _instances([3, 3, 7, 2], [0.4, 0.95, 0.8, 0.7], classes=[0, 1, 0, 0])
    trace = cap.record(inst, max_detections=2)
    np.testing.assert_array_equal(trace.query_index, [3, 7])
    np.testing.assert_allclose(trace.scores, [0.95, 0.8])
    # the kept row for query 3 is the higher-scoring one, so its class came with it
    assert int(trace.pred_classes[0]) == 1


def test_mask_scores_and_areas_are_per_stage_and_finite():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0, 5], [0.9, 0.8]))
    assert trace.mask_scores.shape == (2, 2 + LAYERS)
    assert trace.mask_areas.shape == (2, 2 + LAYERS)
    assert np.isfinite(trace.mask_scores).all()
    # a mask score is a mean sigmoid over the foreground, so it lives in (0.5, 1]
    assert ((trace.mask_scores > 0.5) & (trace.mask_scores <= 1.0)).all()


def test_mask_score_matches_instance_inference_arithmetic():
    """instance_inference computes (sigmoid(m) * (m > 0)).sum() / (m > 0).sum();
    the reconstruction must use that same definition, or the per-stage numbers
    are not comparable to the reported score."""
    model, predictor = _toy_model()
    cap = _run(model, predictor)
    trace = cap.record(_instances([2], [0.9]))

    feats = cap.mask_features[0]
    dec = cap._query_stages(NQ)[-1][0][0, 2]
    with torch.no_grad():
        m = torch.einsum("c,chw->hw", predictor.mask_embed(dec), feats)
    binary = m > 0
    expected = float((m.sigmoid() * binary).sum() / binary.sum())
    assert trace.mask_scores[0, -1] == pytest.approx(expected, abs=1e-5)
    assert int(trace.mask_areas[0, -1]) == int(binary.sum())


def test_divergence_reports_the_stage_where_the_lead_changes():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0], [0.9]))
    # hand-build a trajectory: class 0 leads, class 1 takes over at stage 3
    logits = np.full((1, trace.num_stages, NCLS), -5.0, np.float32)
    logits[0, :, 0] = 2.0
    logits[0, 3:, 1] = 4.0
    trace.class_logits = logits
    trace.class_names = [f"c{i}" for i in range(NCLS)]

    assert trace.divergence(0) == (3, trace.stage_names[3], "c0", "c1")
    np.testing.assert_array_equal(trace.winner_per_stage(0), [0, 0, 0, 1, 1, 1])


def test_divergence_is_none_when_one_class_leads_throughout():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0], [0.9]))
    logits = np.full((1, trace.num_stages, NCLS), -5.0, np.float32)
    logits[0, :, 2] = 3.0
    trace.class_logits = logits
    assert trace.divergence(0) is None


def test_top_classes_keeps_a_class_that_leads_early_and_loses():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0], [0.9]))
    logits = np.full((1, trace.num_stages, NCLS), -5.0, np.float32)
    logits[0, :2, 0] = 5.0     # leads only at the start
    logits[0, :, 1] = 1.0      # mid all along
    logits[0, 2:, 3] = 4.0     # leads at the end
    trace.class_logits = logits
    top = trace.top_classes(0, k=1)
    assert 0 in top and 3 in top, "an early leader must survive into the chart"


def test_final_score_check_reconstructs_class_times_mask():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([1], [0.42], classes=[3]))
    rebuilt, reported = trace.final_score_check(0)
    expected = float(trace.class_probs(0)[-1, 3] * trace.mask_scores[0, -1])
    assert rebuilt == pytest.approx(expected)
    assert reported == pytest.approx(0.42)


# ------------------------------------------------------------------ sidecar


def test_sidecar_round_trips(tmp_path):
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(
        _instances([0, 4], [0.9, 0.6], classes=[1, 2]),
        class_names=[f"c{i}" for i in range(NCLS)],
        source_frame="/data/f.hdf5", checkpoint="m.pth", config_file="c.yaml")
    back = st_mod.read_stage_trace(
        st_mod.write_stage_trace(tmp_path / "f.trace.hdf5", trace))

    for name in ("class_logits", "mask_scores", "mask_areas", "query_index",
                 "detection_index", "scores", "pred_classes", "pred_boxes"):
        np.testing.assert_allclose(getattr(back, name), getattr(trace, name),
                                   err_msg=f"{name} did not round-trip")
    assert back.stage_names == trace.stage_names
    assert back.class_names == trace.class_names
    assert back.mask_size == (MASK_H, MASK_W)
    assert back.source_frame == "/data/f.hdf5"


# -------------------------------------------------------------------- panel


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
    def __init__(self):
        self.figures, self.captions, self.warnings, self.metrics, self.tables = \
            [], [], [], [], []

    def get_option(self, name):
        return "light"

    def selectbox(self, label, options, format_func=None, **kw):
        opts = list(options)
        if format_func:
            format_func(opts[0])
        return opts[0]

    def columns(self, spec):
        return [_Col(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, *a, **kw):
        return _Col(self)

    def pyplot(self, fig, **kw):
        self.figures.append(fig)

    def metric(self, label, value, delta=None):
        self.metrics.append((label, value))

    def dataframe(self, df, **kw):
        self.tables.append(df)

    def caption(self, m):
        self.captions.append(m)

    def warning(self, m):
        self.warnings.append(m)

    def markdown(self, m):
        pass


@pytest.fixture
def panel(monkeypatch):
    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    from maskdino.visualization import stage_trace_panel

    return stage_trace_panel, fake


def _panel_trace():
    model, predictor = _toy_model()
    trace = _run(model, predictor).record(_instances([0], [0.44], classes=[1]))
    logits = np.full((1, trace.num_stages, NCLS), -5.0, np.float32)
    logits[0, :, 0] = 1.0
    logits[0, 3:, 1] = 2.0
    trace.class_logits = logits
    trace.class_names = [f"c{i}" for i in range(NCLS)]
    return trace


def test_panel_renders_a_chart_with_legend_and_table(panel):
    mod, st = panel
    mod.render_stage_trace_panel(_panel_trace(), score_threshold=0.5)
    assert len(st.figures) == 1
    ax = st.figures[0].axes[0]
    # 3 class series are capped, plus the mask track
    assert len(ax.lines) >= 2
    assert ax.get_legend() is not None, "a legend is required for >= 2 series"
    assert st.tables, "the table view is the relief for low-contrast slots"
    assert [m[0] for m in st.metrics] == ["class confidence", "mask score",
                                          "final = class x mask"]


def test_panel_caps_class_series_at_three(panel):
    mod, st = panel
    trace = _panel_trace()
    # make all five classes plausible so top_classes would return more than 3
    trace.class_logits = np.random.default_rng(0).normal(
        0, 2, (1, trace.num_stages, NCLS)).astype(np.float32)
    mod.render_stage_trace_panel(trace)
    ax = st.figures[0].axes[0]
    labels = [ln.get_label() for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len(labels) <= mod.MAX_CLASS_SERIES + 1  # + the mask track


def test_panel_warns_on_an_identity_flip(panel):
    mod, st = panel
    mod.render_stage_trace_panel(_panel_trace())
    assert any("leading class changes" in w for w in st.warnings)


def test_panel_is_quiet_when_no_flip(panel):
    mod, st = panel
    trace = _panel_trace()
    logits = np.full((1, trace.num_stages, NCLS), -5.0, np.float32)
    logits[0, :, 1] = 3.0
    trace.class_logits = logits
    mod.render_stage_trace_panel(trace)
    assert not st.warnings
    assert any("One class leads at every stage" in c for c in st.captions)


def test_panel_handles_an_empty_trace(panel):
    mod, st = panel
    trace = _panel_trace()
    trace.class_logits = trace.class_logits[:0]
    mod.render_stage_trace_panel(trace)
    assert st.figures == []
    assert any("No detections" in c for c in st.captions)
