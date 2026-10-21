"""Unit tests for maskdino/visualization/attention.py and the two capture edits
it depends on (MSDeformAttn's `capture_attn` hook and instance_inference's
`query_indices` field).

The pure-numpy half (splat/heatmap/sidecar I/O) loads attention.py by file path,
like tests/test_class_mapping.py does, so it runs without importing the maskdino
package at all. The capture half needs the real MSDeformAttn and the real
MaskDINO.instance_inference, so it imports those directly - as
tests/test_hungarian_instance_evaluation.py already does.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "visualization", "attention.py")
_spec = importlib.util.spec_from_file_location("mdattention", _MOD_PATH)
attn = importlib.util.module_from_spec(_spec)
sys.modules["mdattention"] = attn
_spec.loader.exec_module(attn)

torch = pytest.importorskip("torch")

HEADS, LEVELS, POINTS, QUERIES, LAYERS, DIM = 8, 4, 4, 12, 6, 32
PAD_H, PAD_W = 128, 160
# 1/2, 1/4, 1/8, 1/16 of the frame, so the per-level splat sigmas (2/4/8/16 px)
# stay small next to it - the same sigma-to-frame ratio real MaskDINO levels have
# (1/8..1/64 of a ~1024px input). A coarser toy pyramid would put most of the
# widest Gaussian off-frame and make mass-conservation assertions meaningless.
LEVEL_SHAPES = [(64, 80), (32, 40), (16, 20), (8, 10)]


# ---------------------------------------------------------------- toy model


class _ToyDecoder(torch.nn.Module):
    """Mimics the DINO TransformerDecoder's call shape closely enough for the
    capture plumbing: a `.layers` list whose entries own a `.cross_attn`, and a
    forward() taking `spatial_shapes`/`valid_ratios` as KEYWORD arguments (which
    is what the forward-pre-hook relies on - MaskDINODecoder.forward passes them
    that way)."""

    def __init__(self, cross_attns):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            types.SimpleNamespace() and _ToyLayer(ca) for ca in cross_attns
        )

    def forward(self, tgt, memory, spatial_shapes=None, level_start_index=None,
                valid_ratios=None, reference_points=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, reference_points, spatial_shapes, level_start_index)
        return out


class _ToyLayer(torch.nn.Module):
    def __init__(self, cross_attn):
        super().__init__()
        self.cross_attn = cross_attn

    def forward(self, tgt, memory, reference_points, spatial_shapes, level_start_index):
        return self.cross_attn(tgt, reference_points, memory, spatial_shapes,
                               level_start_index)


class _ToyPredictor(torch.nn.Module):
    def __init__(self, decoder, mask_hw=(32, 40)):
        super().__init__()
        self.decoder = decoder
        self.mask_hw = mask_hw

    def forward(self, tgt, memory, **kwargs):
        self.decoder(tgt, memory, **kwargs)
        masks = torch.randn(1, QUERIES, *self.mask_hw)
        return {"pred_masks": masks, "pred_logits": torch.randn(1, QUERIES, 3)}, None


def _toy_model(num_layers=LAYERS):
    from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn

    attns = [MSDeformAttn(DIM, LEVELS, HEADS, POINTS) for _ in range(num_layers)]
    predictor = _ToyPredictor(_ToyDecoder(attns))
    # also give it an unrelated MSDeformAttn standing in for the pixel decoder's,
    # to prove capture does not reach it
    encoder_attn = MSDeformAttn(DIM, LEVELS, HEADS, POINTS)
    model = types.SimpleNamespace(
        sem_seg_head=types.SimpleNamespace(predictor=predictor, encoder_attn=encoder_attn)
    )
    return model, predictor, encoder_attn


def _toy_inputs():
    spatial_shapes = torch.as_tensor(LEVEL_SHAPES, dtype=torch.long)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    n_keys = int(spatial_shapes.prod(1).sum())
    return {
        "tgt": torch.randn(1, QUERIES, DIM),
        "memory": torch.randn(1, n_keys, DIM),
        "spatial_shapes": spatial_shapes,
        "level_start_index": level_start_index,
        "valid_ratios": torch.ones(1, LEVELS, 2),
        "reference_points": torch.rand(1, QUERIES, LEVELS, 4) * 0.5 + 0.25,
    }


def _run_capture(dense=False, num_layers=LAYERS, valid_ratios=None):
    model, predictor, encoder_attn = _toy_model(num_layers)
    kwargs = _toy_inputs()
    if valid_ratios is not None:
        kwargs["valid_ratios"] = valid_ratios
    with attn.capture_decoder_attention(model, dense=dense) as cap:
        with torch.no_grad():
            predictor(**kwargs)
    return cap, model, predictor, encoder_attn


# ------------------------------------------------------------------- capture


def test_capture_is_opt_in_and_leaves_no_residue():
    from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn

    # a bare forward with no flag records nothing
    bare = MSDeformAttn(DIM, LEVELS, HEADS, POINTS)
    kwargs = _toy_inputs()
    with torch.no_grad():
        bare(kwargs["tgt"], kwargs["reference_points"], kwargs["memory"],
             kwargs["spatial_shapes"], kwargs["level_start_index"])
    assert getattr(bare, "last_attn", None) is None
    assert getattr(bare, "capture_attn", False) is False

    cap, _model, predictor, encoder_attn = _run_capture()

    # the pixel decoder's attention is never armed
    assert getattr(encoder_attn, "capture_attn", False) is False
    assert getattr(encoder_attn, "last_attn", None) is None

    # and the model is left exactly as found: no flags, no caches, no hooks
    for layer in predictor.decoder.layers:
        assert layer.cross_attn.capture_attn is False
        assert layer.cross_attn.last_attn is None
    assert predictor.decoder._forward_pre_hooks == {}
    assert predictor._forward_hooks == {}
    # but the snapshot survives the block
    assert cap.sampling_locations.shape == (LAYERS, QUERIES, HEADS, LEVELS, POINTS, 2)
    assert cap.attention_weights.shape == (LAYERS, QUERIES, HEADS, LEVELS, POINTS)


def test_captured_weights_are_a_distribution_per_head():
    cap, *_ = _run_capture()
    # MSDeformAttn softmaxes over levels*points for each (query, head)
    sums = cap.attention_weights.reshape(LAYERS, QUERIES, HEADS, -1).sum(-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_spatial_shapes_and_valid_ratios_come_from_the_pre_hook():
    cap, *_ = _run_capture()
    np.testing.assert_array_equal(cap.spatial_shapes, np.asarray(LEVEL_SHAPES))
    np.testing.assert_allclose(cap._valid_ratio_xy(), [1.0, 1.0])


def test_dense_flag_controls_mask_logit_capture():
    cap_off, *_ = _run_capture(dense=False)
    assert cap_off.mask_logits is None
    cap_on, *_ = _run_capture(dense=True)
    assert cap_on.mask_logits.shape == (QUERIES, 32, 40)


def test_missing_forward_pass_is_a_clear_error():
    model, predictor, _ = _toy_model()
    with pytest.raises(RuntimeError, match="recorded no attention"):
        with attn.capture_decoder_attention(model):
            pass  # no forward
    # still disarmed despite the failure
    for layer in predictor.decoder.layers:
        assert layer.cross_attn.capture_attn is False


def test_non_maskdino_model_is_rejected():
    with pytest.raises(TypeError, match="no MaskDINO decoder stack"):
        with attn.capture_decoder_attention(types.SimpleNamespace()):
            pass


# -------------------------------------------------------------------- record


def _fake_instances(queries, scores, classes=None, with_masks=True):
    from detectron2.structures import Boxes, Instances

    inst = Instances((PAD_H, PAD_W))
    inst.query_indices = torch.as_tensor(queries, dtype=torch.int64)
    inst.scores = torch.as_tensor(scores, dtype=torch.float32)
    inst.pred_classes = torch.as_tensor(
        classes if classes is not None else [0] * len(queries), dtype=torch.int64
    )
    inst.pred_boxes = Boxes(torch.zeros(len(queries), 4))
    if with_masks:
        inst.pred_masks = torch.zeros(len(queries), PAD_H, PAD_W)
    return inst


def test_record_maps_detections_to_queries_and_rescales_to_pixels():
    cap, *_ = _run_capture()
    inst = _fake_instances([3, 7], [0.9, 0.8])
    rec = cap.record(inst, class_names=["forcep", "clamp"])

    assert len(rec) == 2
    np.testing.assert_array_equal(rec.query_index, [3, 7])
    assert rec.sampling_locations.shape == (2, LAYERS, HEADS, LEVELS, POINTS, 2)
    assert rec.reference_boxes.shape == (2, LAYERS, 4)
    assert rec.image_size == (PAD_H, PAD_W)

    # detection 0 is query 3, converted from normalized to padded-frame pixels
    expected = cap.sampling_locations[:, 3] * np.array([PAD_W, PAD_H], np.float32)
    np.testing.assert_allclose(rec.sampling_locations[0], expected, rtol=1e-5)


def test_record_undoes_the_valid_ratio_scaling():
    vr = torch.full((1, LEVELS, 2), 0.5)
    cap, *_ = _run_capture(valid_ratios=vr)
    rec = cap.record(_fake_instances([1], [0.9]))
    # locations were scaled by 0.5 inside the decoder; dividing it back out and
    # going to pixels must give twice the raw normalized value
    expected = cap.sampling_locations[:, 1] / 0.5 * np.array([PAD_W, PAD_H], np.float32)
    np.testing.assert_allclose(rec.sampling_locations[0], expected, rtol=1e-5)


def test_record_dedupes_repeated_queries_keeping_the_best_score():
    cap, *_ = _run_capture()
    # same query selected for two classes, plus an unrelated query
    inst = _fake_instances([5, 5, 2], [0.4, 0.95, 0.7], classes=[0, 1, 0])
    rec = cap.record(inst)
    np.testing.assert_array_equal(rec.query_index, [5, 2])
    np.testing.assert_allclose(rec.scores, [0.95, 0.7])
    np.testing.assert_array_equal(rec.pred_classes, [1, 0])


def test_record_honours_max_detections_by_score():
    cap, *_ = _run_capture()
    inst = _fake_instances([1, 2, 3], [0.6, 0.99, 0.75])
    rec = cap.record(inst, max_detections=2)
    np.testing.assert_array_equal(rec.query_index, [2, 3])


def test_record_without_query_indices_is_a_clear_error():
    cap, *_ = _run_capture()
    inst = _fake_instances([1], [0.9])
    inst.remove("query_indices")
    with pytest.raises(ValueError, match="no `query_indices` field"):
        cap.record(inst)


def test_layer_count_invariants():
    """reference_boxes carries one row per layer with cross-attention - the
    *inputs* each layer sampled around. There is deliberately no row for
    initial_pred's layer-0 prediction, which never attended to anything."""
    cap, *_ = _run_capture(num_layers=3)
    rec = cap.record(_fake_instances([0], [0.9]))
    assert rec.num_layers == 3
    assert rec.reference_boxes.shape[1] == 3
    assert rec.sampling_locations.shape[1] == 3


# ------------------------------------------------- the instance_inference edit


def test_instance_inference_exposes_query_indices():
    """The real MaskDINO.instance_inference, driven with a stub `self`."""
    from maskdino.maskdino import MaskDINO

    num_classes, num_queries, topk = 3, 5, 4
    stub = types.SimpleNamespace(
        sem_seg_head=types.SimpleNamespace(num_classes=num_classes),
        device="cpu",
        num_queries=num_queries,
        test_topk_per_image=topk,
        panoptic_on=False,
        focus_on_box=False,
        nms_iou=0.0,
    )
    # query 2 wins on class 1, query 4 on class 0, then queries 0 and 3
    logits = torch.full((num_queries, num_classes), -10.0)
    logits[2, 1], logits[4, 0], logits[0, 2], logits[3, 1] = 9.0, 8.0, 7.0, 6.0
    # constant and positive: every query gets the same mask score, so the final
    # score (class score x mask score) ranks by the logits we spiked above
    mask_pred = torch.full((num_queries, 8, 8), 5.0)
    boxes = torch.rand(num_queries, 4)

    result = MaskDINO.instance_inference(stub, logits, mask_pred, boxes)

    assert result.has("query_indices")
    assert len(result.query_indices) == len(result) == topk
    # ranked by score, the queries are exactly the ones we spiked
    order = torch.argsort(result.scores, descending=True)
    assert result.query_indices[order].tolist() == [2, 4, 0, 3]
    # and the field survives an index slice, as the NMS path does
    assert result[order[:2]].query_indices.tolist() == [2, 4]


# ------------------------------------------------------------ rasterization


def test_splat_integrates_to_the_weight_and_peaks_at_the_point():
    heat = attn.splat(np.array([[40.0, 25.0]]), np.array([1.0]), (64, 80), 4.0)
    assert heat.sum() == pytest.approx(1.0, abs=0.02)
    assert np.unravel_index(heat.argmax(), heat.shape) == (25, 40)


def test_splat_tolerates_off_frame_and_zero_weight_points():
    locs = np.array([[-500.0, -500.0], [10.0, 10.0], [1e9, 3.0]])
    heat = attn.splat(locs, np.array([1.0, 0.0, 1.0]), (32, 32), 2.0)
    assert heat.sum() == pytest.approx(0.0, abs=1e-6)  # all three contribute nothing

    nan = attn.splat(np.array([[np.nan, 1.0]]), np.array([1.0]), (16, 16), 2.0)
    assert nan.sum() == 0.0


def test_splat_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="disagree on point count"):
        attn.splat(np.zeros((3, 2)), np.ones(2), (8, 8), 1.0)


def test_level_sigmas_follow_the_feature_stride():
    sigmas = attn.level_sigmas(np.asarray(LEVEL_SHAPES), (PAD_H, PAD_W))
    # level 0 is 64 rows for a 128-row frame -> stride 2; the coarsest is 8 -> 16
    np.testing.assert_allclose(sigmas, [2.0, 4.0, 8.0, 16.0])
    assert attn.level_sigmas(np.zeros((0, 2)), (PAD_H, PAD_W))[0] == 8.0


def test_heatmap_mass_is_normalized_and_selection_independent():
    cap, *_ = _run_capture()
    rec = cap.record(_fake_instances([4], [0.9]))

    full = attn.heatmap(rec, 0)
    one_layer = attn.heatmap(rec, 0, layers=[2])
    one_head = attn.heatmap(rec, 0, layers=[2], heads=[0])
    # weights sum to 1 per (layer, head), and heatmap averages over the selection,
    # so every selection carries ~unit mass and stays comparable
    for heat in (full, one_layer, one_head):
        assert heat.shape == (PAD_H, PAD_W)
        # <= 1 always (clipping at the frame edge can only lose mass), and close
        # to 1 for this geometry, where the widest sigma is H/8
        assert heat.sum() <= 1.0 + 1e-5
        assert heat.sum() == pytest.approx(1.0, abs=0.1)

    with pytest.raises(ValueError, match="at least one layer and one head"):
        attn.heatmap(rec, 0, layers=[])


def test_heatmap_downscale_shrinks_the_map_but_keeps_the_mass():
    cap, *_ = _run_capture()
    rec = cap.record(_fake_instances([4], [0.9]))
    small = attn.heatmap(rec, 0, downscale=2)
    assert small.shape == (PAD_H // 2, PAD_W // 2)
    assert small.sum() == pytest.approx(attn.heatmap(rec, 0).sum(), abs=0.15)


# ----------------------------------------------------------------- sidecar


def test_sidecar_round_trips(tmp_path):
    cap, *_ = _run_capture(dense=True)
    inst = _fake_instances([3, 9], [0.9, 0.7], classes=[1, 0])
    rec = cap.record(
        inst, class_names=["forcep", "clamp"], source_frame="/data/f.hdf5",
        checkpoint="model_final.pth", config_file="cfg.yaml", original_size=(100, 120),
    )
    path = attn.write_attention(tmp_path / "f.attn.hdf5", rec)
    back = attn.read_attention(path)

    for name in ("sampling_locations", "attention_weights", "reference_boxes",
                 "query_index", "scores", "pred_classes", "pred_boxes",
                 "level_spatial_shapes", "pred_masks", "mask_logits"):
        np.testing.assert_array_equal(getattr(back, name), getattr(rec, name),
                                      err_msg=f"{name} did not round-trip")
    assert back.image_size == (PAD_H, PAD_W)
    assert back.original_size == (100, 120)
    assert back.class_names == ["forcep", "clamp"]
    assert back.source_frame == "/data/f.hdf5"
    assert back.checkpoint == "model_final.pth"
    assert back.config_file == "cfg.yaml"
    assert back.label(0) == "#0 - clamp - 0.90"


def test_sidecar_omits_absent_optional_arrays(tmp_path):
    cap, *_ = _run_capture(dense=False)
    rec = cap.record(_fake_instances([1], [0.9], with_masks=False))
    back = attn.read_attention(attn.write_attention(tmp_path / "f.attn.hdf5", rec))
    assert back.pred_masks is None
    assert back.mask_logits is None


def test_sidecar_path_naming(tmp_path):
    assert attn.sidecar_path("/a/b/frame_01-3.hdf5").endswith("/a/b/frame_01-3.attn.hdf5")
    assert attn.sidecar_path("/a/b/frame.hdf5", str(tmp_path)) == str(tmp_path / "frame.attn.hdf5")
