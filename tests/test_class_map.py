"""Unit tests for maskdino/visualization/class_map.py.

The pure-numpy half (EncoderClassMap rendering + the top-k gate reproduction)
loads class_map.py by file path, like tests/test_attention_capture.py does, so
it runs without importing the maskdino package at all. The capture half needs
forward hooks, so it builds a toy module tree that mimics the two call sites
class_map.py relies on: a shared `class_embed` Linear called once densely and
once per prediction stage, and a `decoder` taking `spatial_shapes` as a KEYWORD
argument (which is how MaskDINODecoder.forward passes it).
"""
import importlib.util
import itertools
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "visualization", "class_map.py")
_spec = importlib.util.spec_from_file_location("mdclassmap", _MOD_PATH)
cm = importlib.util.module_from_spec(_spec)
sys.modules["mdclassmap"] = cm
_spec.loader.exec_module(cm)

# A toy 3-level pyramid in the same stride relationship as the real 3s config
# (res3/res4/res5 = /8, /16, /32 of a 1024px frame), just 8x smaller.
LEVEL_SHAPES = [(16, 20), (8, 10), (4, 5)]
TOKENS = sum(h * w for h, w in LEVEL_SHAPES)  # 320 + 80 + 20 = 420
PAD_H, PAD_W = 128, 160
NUM_CLASSES = 6
NUM_QUERIES = 10
CLASS_NAMES = ["grasper", "needle_holder", "scissors", "hook", "clipper", "irrigator"]


def _make_map(logits=None, selected=None):
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((TOKENS, NUM_CLASSES), dtype=np.float32) if logits is None else logits
    selected = cm._topk_tokens(logits, NUM_QUERIES) if selected is None else selected
    return cm.EncoderClassMap(
        logits=np.asarray(logits, np.float32),
        level_spatial_shapes=np.asarray(LEVEL_SHAPES, np.int32),
        selected_tokens=np.asarray(selected, np.int32),
        image_size=(PAD_H, PAD_W),
        original_size=(PAD_H, PAD_W),
        class_names=list(CLASS_NAMES),
    )


# ----------------------------------------------------------- shape bookkeeping


def test_level_slices_tile_the_token_axis():
    slices = _make_map().level_slices()
    assert slices[0] == (0, 320)
    assert slices[1] == (320, 400)
    assert slices[2] == (400, 420)
    # contiguous, no gaps, covering exactly T
    assert slices[0][0] == 0 and slices[-1][1] == TOKENS
    assert all(a[1] == b[0] for a, b in itertools.pairwise(slices))


def test_level_maps_recover_each_level_resolution():
    maps = _make_map().level_maps()
    assert [m.shape for m in maps] == [(16, 20, NUM_CLASSES), (8, 10, NUM_CLASSES), (4, 5, NUM_CLASSES)]
    one = _make_map().level_maps(cls="needle_holder")
    assert [m.shape for m in one] == [(16, 20), (8, 10), (4, 5)]


def test_level_maps_are_sigmoid_of_the_raw_logits():
    cmap = _make_map()
    raw = cmap.level_maps(cls=0, apply_sigmoid=False)[0]
    act = cmap.level_maps(cls=0, apply_sigmoid=True)[0]
    assert np.allclose(act, 1.0 / (1.0 + np.exp(-raw)), atol=1e-6)
    # the first level's class-0 column, read straight out of the flat logits
    assert np.allclose(raw.reshape(-1), cmap.logits[:320, 0], atol=1e-6)


def test_class_index_accepts_names_and_ids():
    cmap = _make_map()
    assert cmap.class_index("scissors") == 2
    assert cmap.class_index(2) == 2
    assert cmap.label("scissors") == "2 - scissors"
    with pytest.raises(KeyError):
        cmap.class_index("not_an_instrument")


# --------------------------------------------------------- frame-space renders


def test_class_heatmap_lands_in_the_padded_frame_and_is_a_probability():
    heat = _make_map().class_heatmap("needle_holder")
    assert heat.shape == (PAD_H, PAD_W)
    assert heat.dtype == np.float32
    assert 0.0 <= heat.min() and heat.max() <= 1.0


def test_class_heatmap_reduce_max_dominates_mean():
    cmap = _make_map()
    mean = cmap.class_heatmap(1, reduce="mean")
    mx = cmap.class_heatmap(1, reduce="max")
    assert np.all(mx >= mean - 1e-6)
    with pytest.raises(ValueError):
        cmap.class_heatmap(1, reduce="median")


def test_single_level_heatmap_is_that_level_upsampled():
    cmap = _make_map()
    heat = cmap.class_heatmap(0, levels=[2])
    # level 2 is 4x5 over a 128x160 frame -> 32x32 px blocks of constant value
    assert heat.shape == (PAD_H, PAD_W)
    assert len(np.unique(heat)) <= 4 * 5
    assert np.allclose(heat[:32, :32], heat[0, 0])


def test_confidence_map_is_the_gate_score_field():
    cmap = _make_map()
    conf = cmap.confidence_map(levels=[0])
    best = 1.0 / (1.0 + np.exp(-cmap.logits[:320].max(1)))
    assert np.isclose(conf.max(), best.max(), atol=1e-6)


def test_argmax_map_returns_class_ids():
    arg = _make_map().argmax_map()
    assert arg.shape == (PAD_H, PAD_W)
    assert arg.dtype == np.int32
    assert set(np.unique(arg)).issubset(set(range(NUM_CLASSES)))


def test_empty_level_selection_is_an_error():
    with pytest.raises(ValueError):
        _make_map().class_heatmap(0, levels=[])


# ------------------------------------------------------------ the top-k gate


def test_topk_tokens_matches_torch_topk_as_a_set():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    logits = rng.standard_normal((TOKENS, NUM_CLASSES)).astype(np.float32)
    mine = cm._topk_tokens(logits, NUM_QUERIES)
    theirs = torch.topk(
        torch.from_numpy(logits).max(-1)[0], NUM_QUERIES, dim=0
    )[1].numpy()
    assert set(mine.tolist()) == set(theirs.tolist())
    assert mine.shape == (NUM_QUERIES,)


def test_topk_tokens_is_sorted_best_first():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((TOKENS, NUM_CLASSES)).astype(np.float32)
    idx = cm._topk_tokens(logits, NUM_QUERIES)
    best = logits.max(1)[idx]
    assert np.all(np.diff(best) <= 1e-6)


def test_topk_tokens_caps_at_the_token_count():
    logits = np.zeros((4, NUM_CLASSES), np.float32)
    assert cm._topk_tokens(logits, NUM_QUERIES).shape == (4,)


def test_selected_points_put_a_known_token_at_the_right_pixel():
    # make one token at level 1, row 3, col 4 the clear winner
    logits = np.full((TOKENS, NUM_CLASSES), -10.0, np.float32)
    token = 320 + 3 * 10 + 4
    logits[token, 1] = 10.0
    cmap = _make_map(logits=logits, selected=[token])
    (x, y, level), = cmap.selected_points()
    assert int(level) == 1
    # level 1 is 8x10 over 128x160 -> 16px cells, centre of (row 3, col 4)
    assert x == pytest.approx((4 + 0.5) * 160 / 10)
    assert y == pytest.approx((3 + 0.5) * 128 / 8)


def test_selected_points_stay_inside_the_frame():
    pts = _make_map().selected_points()
    assert pts.shape == (NUM_QUERIES, 3)
    assert np.all((pts[:, 0] >= 0) & (pts[:, 0] <= PAD_W))
    assert np.all((pts[:, 1] >= 0) & (pts[:, 1] <= PAD_H))
    assert set(np.unique(pts[:, 2]).astype(int)).issubset({0, 1, 2})


def test_selected_classes_is_the_argmax_of_each_kept_token():
    cmap = _make_map()
    got = cmap.selected_classes()
    assert got.shape == (NUM_QUERIES,)
    expected = cmap.logits[cmap.selected_tokens].argmax(1)
    assert np.array_equal(got, expected)


# ------------------------------------------------------------------- resize


def test_resize_nearest_is_a_noop_at_matching_shape():
    plane = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert cm.resize_nearest(plane, (3, 4)) is plane or np.array_equal(
        cm.resize_nearest(plane, (3, 4)), plane
    )


def test_resize_nearest_preserves_the_value_set():
    plane = np.arange(20, dtype=np.float32).reshape(4, 5)
    big = cm.resize_nearest(plane, (40, 50))
    assert big.shape == (40, 50)
    assert set(np.unique(big).tolist()) == set(plane.reshape(-1).tolist())


def test_resize_nearest_handles_the_channel_axis():
    plane = np.zeros((4, 5, 3), np.float32)
    assert cm.resize_nearest(plane, (8, 10)).shape == (8, 10, 3)


# ---------------------------------------------------- round trip through dict


def test_to_dict_from_dict_round_trip():
    cmap = _make_map()
    back = cm.EncoderClassMap.from_dict(cmap.to_dict())
    assert np.array_equal(back.logits, cmap.logits)
    assert back.image_size == cmap.image_size
    assert back.class_names == cmap.class_names


# ------------------------------------------------------------------ capture
#
# The toy tree mimics only what class_map.py hooks: a shared class_embed Linear
# and a decoder whose forward takes spatial_shapes as a keyword. Crucially it
# reproduces the real CALL ORDER - the dense call happens before the decoder
# runs (maskdino_decoder.py:403 vs 460), so the spatial shapes needed to
# identify it arrive only afterwards.

torch = pytest.importorskip("torch")
DIM = 16


class _ToyDecoder(torch.nn.Module):
    def forward(self, tgt, spatial_shapes=None, valid_ratios=None):
        return [tgt] * 3, None


class _ToyPredictor(torch.nn.Module):
    """Mimics MaskDINODecoder's use of a single shared class_embed."""

    def __init__(self, two_stage=True, num_queries=NUM_QUERIES, stages=8):
        super().__init__()
        self.class_embed = torch.nn.Linear(DIM, NUM_CLASSES)
        self.decoder = _ToyDecoder()
        self.num_queries = num_queries
        self.two_stage = two_stage
        self.stages = stages

    def forward(self, memory, shapes):
        if self.two_stage:
            # line 403: the dense call, over every encoder token
            self.class_embed(memory)
        tgt = memory[:, : self.num_queries]
        self.decoder(tgt, spatial_shapes=shapes, valid_ratios=None)
        # interm + initial_pred + one per decoder layer, all at num_queries
        for _ in range(self.stages):
            self.class_embed(tgt)


class _ToyModel(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.sem_seg_head = torch.nn.Module()
        self.sem_seg_head.predictor = _ToyPredictor(**kw)

    def forward(self, memory, shapes):
        return self.sem_seg_head.predictor(memory, shapes)


def _run(model):
    memory = torch.randn(1, TOKENS, DIM)
    shapes = torch.as_tensor(LEVEL_SHAPES, dtype=torch.long)
    with cm.capture_encoder_class_logits(model) as cap:
        model(memory, shapes)
    return cap


def test_capture_picks_the_dense_call_not_a_query_stage():
    cap = _run(_ToyModel())
    cmap = cap.record(image_size=(PAD_H, PAD_W), class_names=CLASS_NAMES)
    # 1 dense + 8 per-stage calls were recorded; only the dense one is kept
    assert len(cap._calls) == 9
    assert cmap.logits.shape == (TOKENS, NUM_CLASSES)
    assert cmap.num_levels == 3
    assert cmap.num_classes == NUM_CLASSES


def test_capture_matches_a_direct_class_embed_evaluation():
    model = _ToyModel()
    memory = torch.randn(1, TOKENS, DIM)
    shapes = torch.as_tensor(LEVEL_SHAPES, dtype=torch.long)
    with cm.capture_encoder_class_logits(model) as cap:
        model(memory, shapes)
    cmap = cap.record(image_size=(PAD_H, PAD_W))
    with torch.no_grad():
        expected = model.sem_seg_head.predictor.class_embed(memory)[0].numpy()
    assert np.allclose(cmap.logits, expected, atol=1e-5)


def test_capture_leaves_no_hooks_behind():
    model = _ToyModel()
    ce = model.sem_seg_head.predictor.class_embed
    dec = model.sem_seg_head.predictor.decoder
    before = (len(ce._forward_hooks), len(dec._forward_pre_hooks))
    _run(model)
    assert (len(ce._forward_hooks), len(dec._forward_pre_hooks)) == before


def test_hooks_are_removed_even_when_the_forward_raises():
    model = _ToyModel()
    ce = model.sem_seg_head.predictor.class_embed
    with pytest.raises(RuntimeError, match="boom"), \
            cm.capture_encoder_class_logits(model):
        raise RuntimeError("boom")
    assert len(ce._forward_hooks) == 0


def test_selected_tokens_come_from_the_captured_logits():
    cap = _run(_ToyModel())
    cmap = cap.record(image_size=(PAD_H, PAD_W))
    assert cmap.selected_tokens.shape == (NUM_QUERIES,)
    expected = cm._topk_tokens(cmap.logits, NUM_QUERIES)
    assert set(cmap.selected_tokens.tolist()) == set(expected.tolist())


def test_record_without_two_stage_explains_itself():
    cap = _run(_ToyModel(two_stage=False))
    with pytest.raises(RuntimeError, match="TWO_STAGE"):
        cap.record(image_size=(PAD_H, PAD_W))


def test_record_before_any_forward_explains_itself():
    model = _ToyModel()
    with cm.capture_encoder_class_logits(model) as cap:
        pass
    with pytest.raises(RuntimeError, match="recorded no calls"):
        cap.record()


def test_image_size_defaults_to_level_zero_times_its_stride():
    cap = _run(_ToyModel())
    cmap = cap.record()
    assert cmap.image_size == (LEVEL_SHAPES[0][0] * 8, LEVEL_SHAPES[0][1] * 8)


def test_rejects_a_model_without_a_maskdino_class_head():
    with pytest.raises(TypeError, match="class_embed"):
        cm.capture_encoder_class_logits(torch.nn.Linear(4, 4)).__enter__()
