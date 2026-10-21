"""Decoder cross-attention capture, storage and rasterization - "where did the
model look when it detected this instrument?".

MaskDINO has no dense query->pixel attention to export. Its decoder
cross-attention is MSDeformAttn (see modeling/transformer_decoder/dino_decoder.py),
which samples `n_points` locations per head per feature level around each
query's reference box, so the real artifact is a *sparse* set of weighted
sample points - n_heads * n_levels * n_points (typically 8*4*4 = 128) per query
per decoder layer, plus the reference box each layer attended around. That is
~9 KB per detection for the whole 6-layer stack, so this module stores it
losslessly and rasterizes to a heatmap only at display time, rather than
writing dense per-query maps (100 queries x 7 layers x 1024^2 fp16 would be
1.4 GB per frame).

Capture is opt-in and leaves no trace: `capture_decoder_attention` flips a
`capture_attn` flag that MSDeformAttn.forward checks via getattr, and removes
it again on exit. Nothing in the train/eval path is touched, and no forward
signature changes - the 6 decoder layers are distinct modules
(`dec_layer_share=False` -> `_get_clones` deepcopies), so each one caches its
own last call and is read back per layer afterwards.

Typical use (see tools/export_attention.py):

    with capture_decoder_attention(predictor.model) as cap:
        instances = predictor(image_bgr)["instances"].to("cpu")
    record = cap.record(instances[instances.scores > 0.5])
    write_attention("frame.attn.hdf5", record)

and on the viewer side:

    record = read_attention("frame.attn.hdf5")
    heat = heatmap(record, det=0)            # (H, W) float32, sums to ~1
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

import h5py
import numpy as np

# HDF5 key names, kept as constants for the same reason sgdata.schema does it:
# the writer here and the reader in hdf5_viewer must not drift.
GROUP = "attention"

SAMPLING_LOCATIONS = f"{GROUP}/sampling_locations"
ATTENTION_WEIGHTS = f"{GROUP}/attention_weights"
REFERENCE_BOXES = f"{GROUP}/reference_boxes"
QUERY_INDEX = f"{GROUP}/query_index"
DETECTION_INDEX = f"{GROUP}/detection_index"
SCORES = f"{GROUP}/scores"
PRED_CLASSES = f"{GROUP}/pred_classes"
PRED_BOXES = f"{GROUP}/pred_boxes"
PRED_MASKS = f"{GROUP}/pred_masks"
MASK_LOGITS = f"{GROUP}/mask_logits"
LEVEL_SPATIAL_SHAPES = f"{GROUP}/level_spatial_shapes"

SOURCE_FRAME_ATTR = "source_frame"
CHECKPOINT_ATTR = "checkpoint"
CONFIG_FILE_ATTR = "config_file"
CLASS_NAMES_ATTR = "class_names"
IMAGE_SIZE_ATTR = "image_size"
ORIGINAL_SIZE_ATTR = "original_size"
NUM_DECODER_LAYERS_ATTR = "num_decoder_layers"

# Sidecar suffix. Not written into the frame itself: the live pool evicts and
# overwrites frames (sgdata.pool.evict_if_over_cap), sgdata.schema owns that
# key namespace, and attention is a property of a *checkpoint* - you will want
# several per frame.
SIDECAR_SUFFIX = ".attn.hdf5"


def sidecar_path(frame_path, out_dir=None):
    """`/x/frame.hdf5` -> `<out_dir or /x>/frame.attn.hdf5`."""
    import os

    stem = os.path.splitext(os.path.basename(frame_path))[0]
    return os.path.join(out_dir or os.path.dirname(frame_path), stem + SIDECAR_SUFFIX)


@dataclass
class AttentionRecord:
    """One frame's captured attention, for `D` kept detections.

    Shapes, with L decoder layers / Hd heads / Lv feature levels / P points:

    - sampling_locations  (D, L, Hd, Lv, P, 2) float32 - xy in PIXELS of the
      padded model frame (`image_size`), i.e. the same coordinate frame as
      `pred_masks` and `pred_boxes`, so an overlay needs no further transform.
    - attention_weights   (D, L, Hd, Lv, P)    float32 - softmax over Lv*P per
      (query, head), so each head's weights sum to 1.
    - reference_boxes     (D, L, 4)            float32 - normalized cxcywh; row
      i is the box layer i sampled around. Note L rows, not L+1: these are the
      *inputs* to each layer, and the final refined box is `pred_boxes`.
    - mask_logits         (D, h, w)            float16 - optional dense
      per-query response (pre-threshold, 1/4 resolution).
    - detection_index     (D,)                 int32 - row this came from in the
      source `Instances`, so a caller holding those masks/boxes already (the
      viewer does) can index them in step instead of storing a second copy.
    """

    sampling_locations: np.ndarray
    attention_weights: np.ndarray
    reference_boxes: np.ndarray
    query_index: np.ndarray
    detection_index: np.ndarray
    scores: np.ndarray
    pred_classes: np.ndarray
    pred_boxes: np.ndarray
    level_spatial_shapes: np.ndarray
    image_size: tuple = (0, 0)
    original_size: tuple = (0, 0)
    pred_masks: np.ndarray | None = None
    mask_logits: np.ndarray | None = None
    class_names: list = field(default_factory=list)
    source_frame: str = ""
    checkpoint: str = ""
    config_file: str = ""

    def __len__(self):
        return int(self.sampling_locations.shape[0])

    @property
    def num_layers(self):
        return int(self.sampling_locations.shape[1])

    @property
    def num_heads(self):
        return int(self.sampling_locations.shape[2])

    def label(self, det):
        """"#3 - needle_holder - 0.92", for a detection picker."""
        cls = int(self.pred_classes[det])
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        return f"#{det} - {name} - {self.scores[det]:.2f}"

    def to_dict(self):
        """Plain numpy/str/list dict - no tensors, no custom types.

        hdf5_viewer caches inference results through st.cache_data and wants the
        entry to hold nothing exotic; this keeps that contract while still
        letting the panel work from a live capture instead of a sidecar file.
        """
        out = {fld.name: getattr(self, fld.name) for fld in fields(self)}
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, data):
        known = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def filter(self, keep):
        """A new record holding only the detections in `keep` (an index array
        or boolean mask), for applying a score threshold after capture."""
        keep = np.asarray(keep)
        per_det = ("sampling_locations", "attention_weights", "reference_boxes",
                   "query_index", "detection_index", "scores", "pred_classes",
                   "pred_boxes", "pred_masks", "mask_logits")
        data = self.to_dict()
        for name in per_det:
            if data.get(name) is not None:
                data[name] = data[name][keep]
        return AttentionRecord.from_dict(data)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _decoder_and_cross_attns(model):
    """(DINO TransformerDecoder, its per-layer MSDeformAttn modules)."""
    try:
        decoder = model.sem_seg_head.predictor.decoder
        attns = [layer.cross_attn for layer in decoder.layers]
    except AttributeError as e:
        raise TypeError(
            f"{type(model).__name__} has no MaskDINO decoder stack at "
            "sem_seg_head.predictor.decoder.layers[*].cross_attn - attention capture "
            "is MaskDINO-specific (Mask2Former's dense cross-attention would need a "
            "different reader)."
        ) from e
    if not attns:
        raise ValueError("decoder has no layers; nothing to capture")
    return decoder, attns


class _Capture:
    """Handle yielded by `capture_decoder_attention`.

    Snapshots the per-layer caches when the `with` block exits, so `.record()`
    still works afterwards while the model itself is left clean.
    """

    def __init__(self, model, dense=False):
        self.model = model
        self.decoder, self._attns = _decoder_and_cross_attns(model)
        self.predictor = model.sem_seg_head.predictor
        self._dense = dense
        self._hooks = []
        # filled by _snapshot(), all with a leading layer axis
        self.sampling_locations = None
        self.attention_weights = None
        self.reference_points = None
        # filled by the decoder forward-pre-hook
        self.spatial_shapes = None
        self.valid_ratios = None
        # filled by the predictor forward-hook, when dense=True
        self.mask_logits = None

    # -- plumbing ----------------------------------------------------------

    def _pre_hook(self, _module, _args, kwargs):
        # Both arrive as kwargs from MaskDINODecoder.forward; grabbing them here
        # avoids plumbing them out through any return signature.
        if "spatial_shapes" in kwargs:
            self.spatial_shapes = kwargs["spatial_shapes"].detach().cpu().numpy()
        if "valid_ratios" in kwargs:
            self.valid_ratios = kwargs["valid_ratios"].detach().cpu().float().numpy()

    def _out_hook(self, _module, _args, output):
        # MaskDINODecoder.forward returns (out, mask_dict); out["pred_masks"] is the
        # dense per-query response map at 1/4 resolution, straight from the
        # mask_embed x mask_features einsum. Inference drops it (MaskDINO.forward
        # deletes `outputs`), so grab it here rather than re-running the head.
        out = output[0] if isinstance(output, tuple) else output
        masks = out.get("pred_masks") if isinstance(out, dict) else None
        if masks is not None:
            self.mask_logits = masks[0].detach().cpu().float().numpy()

    def _arm(self):
        for m in self._attns:
            m.capture_attn = True
            m.last_attn = None
        self._hooks.append(
            self.decoder.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        )
        if self._dense:
            self._hooks.append(self.predictor.register_forward_hook(self._out_hook))

    def _snapshot(self):
        missing = [i for i, m in enumerate(self._attns) if getattr(m, "last_attn", None) is None]
        if missing:
            raise RuntimeError(
                f"decoder layers {missing} recorded no attention - did the forward pass "
                "run inside the `with capture_decoder_attention(...)` block?"
            )
        # each (N, Lq, Hd, Lv, P, ...) -> (L, Lq, Hd, Lv, P, ...) for batch item 0
        triples = [tuple(t[0].numpy() for t in m.last_attn) for m in self._attns]
        locs, weights, refs = zip(*triples)
        self.sampling_locations = np.stack(locs)
        self.attention_weights = np.stack(weights)
        self.reference_points = np.stack(refs)

    def _disarm(self):
        for m in self._attns:
            m.capture_attn = False
            m.last_attn = None
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    # -- public ------------------------------------------------------------

    def record(self, instances, image_size=None, original_size=None, mask_logits=None,
               max_detections=None, class_names=None, source_frame="", checkpoint="",
               config_file="", include_masks=True):
        """Join the captured attention to a kept-detection `Instances`.

        `instances` must carry `query_indices` (added by
        MaskDINO.instance_inference) - that is the only link from a drawn
        detection back to a decoder query. Queries selected twice (one query,
        two classes both above threshold) are deduped, keeping the
        higher-scoring occurrence.

        `include_masks=False` leaves `pred_masks` out: for 100 queries at
        1024x1024 that is ~100 MB, and a caller that already holds the masks
        (hdf5_viewer caches them for its own overlay) can index its own copy
        through `detection_index` instead.
        """
        if self.sampling_locations is None:
            raise RuntimeError("nothing captured yet; call .record() after the with-block")
        if not instances.has("query_indices"):
            raise ValueError(
                "Instances has no `query_indices` field - MaskDINO.instance_inference "
                "must expose it for attention to be attributable to a detection."
            )

        queries = instances.query_indices.cpu().numpy().astype(np.int64)
        scores = instances.scores.cpu().numpy().astype(np.float32)

        # dedupe by query, highest score first, then cap
        order = np.argsort(-scores, kind="stable")
        _, first = np.unique(queries[order], return_index=True)
        keep = order[np.sort(first)]
        keep = keep[np.argsort(-scores[keep], kind="stable")]
        if max_detections is not None:
            keep = keep[:max_detections]

        q = queries[keep]
        image_size = tuple(image_size or instances.image_size)
        h_pad, w_pad = image_size

        # Undo the valid-ratio scaling MaskDINODecoder.forward applies
        # (reference_points_input = reference_points * cat([vr, vr])), then
        # normalized [0,1] -> pixels of the padded frame. vr is 1.0 whenever the
        # padded levels are all divisible by 32 (the all-zero-mask path), which
        # SIZE_DIVISIBILITY=32 guarantees, but undo it anyway so a config change
        # cannot silently shift every point.
        vr = self._valid_ratio_xy()
        locs = self.sampling_locations[:, q] / vr  # (L, D, Hd, Lv, P, 2)
        locs = locs * np.array([w_pad, h_pad], dtype=np.float32)
        refs = self.reference_points[:, q, 0, :].copy()  # (L, D, 4), level 0
        refs[..., 0::2] /= vr[0]
        refs[..., 1::2] /= vr[1]

        # dense per-query response, from the predictor hook (dense=True) unless the
        # caller passed its own; subset to the kept queries and halve the precision.
        dense = mask_logits if mask_logits is not None else self.mask_logits
        dense = None if dense is None else dense[q].astype(np.float16)

        return AttentionRecord(
            sampling_locations=np.ascontiguousarray(locs.transpose(1, 0, 2, 3, 4, 5), np.float32),
            attention_weights=np.ascontiguousarray(
                self.attention_weights[:, q].transpose(1, 0, 2, 3, 4), np.float32
            ),
            reference_boxes=np.ascontiguousarray(refs.transpose(1, 0, 2), np.float32),
            query_index=q.astype(np.int32),
            detection_index=keep.astype(np.int32),
            scores=scores[keep],
            pred_classes=instances.pred_classes.cpu().numpy()[keep].astype(np.int32),
            pred_boxes=instances.pred_boxes.tensor.cpu().numpy()[keep].astype(np.float32),
            pred_masks=(
                instances.pred_masks.cpu().numpy()[keep].astype(bool)
                if include_masks and instances.has("pred_masks") else None
            ),
            mask_logits=dense,
            level_spatial_shapes=(
                self.spatial_shapes.astype(np.int32)
                if self.spatial_shapes is not None else np.zeros((0, 2), np.int32)
            ),
            image_size=image_size,
            original_size=tuple(original_size or image_size),
            class_names=list(class_names or []),
            source_frame=source_frame,
            checkpoint=checkpoint,
            config_file=config_file,
        )

    def _valid_ratio_xy(self):
        """(x, y) valid ratio of feature level 0, as a broadcastable float32 pair."""
        if self.valid_ratios is None:
            return np.ones(2, np.float32)
        # (bs, nlevel, 2), stored as (ratio_w, ratio_h) by get_valid_ratio
        return self.valid_ratios[0, 0].astype(np.float32)


@contextmanager
def capture_decoder_attention(model, dense=False):
    """Record the decoder's deformable cross-attention for one forward pass.

    Arms only `sem_seg_head.predictor.decoder.layers[*].cross_attn` - NOT the
    pixel decoder's MSDeformAttn, whose pixel->pixel attention is a different
    and far larger thing. Yields a `_Capture` whose `.record(instances)` is
    valid after the block; the model is left exactly as it was found.

    `dense=True` additionally captures each query's mask-logit response map from
    the predictor's output, which inference otherwise discards.
    """
    cap = _Capture(model, dense=dense)
    cap._arm()
    try:
        yield cap
        cap._snapshot()
    finally:
        cap._disarm()


# --------------------------------------------------------------------------
# rasterization
# --------------------------------------------------------------------------


def level_sigmas(level_spatial_shapes, image_size, fallback=8.0):
    """Per-level splat radius = that level's stride in padded-frame pixels.

    A point sampled from a coarse level stands for a correspondingly coarse
    patch, so drawing it with the level's own stride is visually honest about
    receptive field instead of implying pixel precision.
    """
    shapes = np.asarray(level_spatial_shapes, dtype=np.float32)
    if shapes.size == 0:
        return np.full(1, fallback, np.float32)
    h_pad = float(image_size[0])
    return np.maximum(h_pad / np.maximum(shapes[:, 0], 1.0), 1.0).astype(np.float32)


def splat(locs, weights, shape, sigma):
    """Rasterize weighted sample points into a (H, W) float32 map.

    `locs` is (..., 2) in pixels, `weights` (...) matching, `sigma` a scalar or
    anything broadcastable to `weights`. Each point contributes a Gaussian
    whose integral is its weight, so the map's total mass equals
    `weights.sum()` (clipped at the frame edges). Points are drawn in a
    +-3 sigma window, so cost scales with sigma, not with frame area.
    """
    locs = np.asarray(locs, dtype=np.float32).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if locs.shape[0] != w.shape[0]:
        raise ValueError(f"locs {locs.shape} and weights {w.shape} disagree on point count")
    s = np.broadcast_to(np.asarray(sigma, np.float32), np.asarray(weights).shape).reshape(-1)

    h, w_px = shape[0], shape[1]
    heat = np.zeros((h, w_px), np.float32)
    for (x, y), wt, sg in zip(locs, w, s):
        if wt == 0.0 or not (np.isfinite(x) and np.isfinite(y)):
            continue
        sg = max(float(sg), 0.5)
        r = int(np.ceil(3.0 * sg))
        x0, x1 = max(0, int(x) - r), min(w_px, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue  # entirely off-frame
        dx = np.arange(x0, x1, dtype=np.float32) - x
        dy = np.arange(y0, y1, dtype=np.float32) - y
        g = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2.0 * sg * sg))
        heat[y0:y1, x0:x1] += g * (wt / (2.0 * np.pi * sg * sg))
    return heat


def heatmap(record, det, layers=None, heads=None, shape=None, downscale=1):
    """Attention heatmap for one detection, as a (H, W) float32 map.

    `layers` / `heads` are index sequences (None = all). Summing over all
    layers is the practical "where did this query look"; a single layer is the
    better debugging view, since decoder self-attention mixes queries between
    layers and the last layer's points are not a detection's full provenance.

    Normalized by the number of heads *and* layers averaged, so the map's mass
    is ~1 regardless of selection and is comparable across detections.
    """
    layers = range(record.num_layers) if layers is None else list(layers)
    heads = range(record.num_heads) if heads is None else list(heads)
    layers, heads = list(layers), list(heads)
    if not layers or not heads:
        raise ValueError("need at least one layer and one head")

    locs = record.sampling_locations[det][np.ix_(layers, heads)]  # (l, h, Lv, P, 2)
    weights = record.attention_weights[det][np.ix_(layers, heads)]  # (l, h, Lv, P)

    h_pad, w_pad = shape or record.image_size
    sigmas = level_sigmas(record.level_spatial_shapes, (h_pad, w_pad))
    # broadcast per-level sigma over (l, h, Lv, P)
    sigmas = np.broadcast_to(sigmas[None, None, :, None], weights.shape)

    if downscale > 1:
        locs = locs / downscale
        sigmas = sigmas / downscale
        h_pad, w_pad = round(h_pad / downscale), round(w_pad / downscale)

    heat = splat(locs, weights, (h_pad, w_pad), sigmas)
    return heat / (len(heads) * len(layers))


# --------------------------------------------------------------------------
# sidecar I/O
# --------------------------------------------------------------------------

_ARRAY_KEYS = {
    "sampling_locations": (SAMPLING_LOCATIONS, np.float32),
    "attention_weights": (ATTENTION_WEIGHTS, np.float32),
    "reference_boxes": (REFERENCE_BOXES, np.float32),
    "query_index": (QUERY_INDEX, np.int32),
    "detection_index": (DETECTION_INDEX, np.int32),
    "scores": (SCORES, np.float32),
    "pred_classes": (PRED_CLASSES, np.int32),
    "pred_boxes": (PRED_BOXES, np.float32),
    "level_spatial_shapes": (LEVEL_SPATIAL_SHAPES, np.int32),
}
_OPTIONAL_ARRAY_KEYS = {
    "pred_masks": (PRED_MASKS, bool),
    "mask_logits": (MASK_LOGITS, np.float16),
}


def write_attention(path, record):
    """Write one `AttentionRecord` to a `<frame>.attn.hdf5` sidecar."""
    with h5py.File(path, "w") as f:
        for name, (key, dtype) in _ARRAY_KEYS.items():
            f.create_dataset(key, data=np.asarray(getattr(record, name), dtype))
        for name, (key, dtype) in _OPTIONAL_ARRAY_KEYS.items():
            value = getattr(record, name)
            if value is None:
                continue
            # masks and dense logits are the only bulky members; everything else
            # is a few KB and not worth the compression round-trip.
            f.create_dataset(key, data=np.asarray(value, dtype), compression="gzip",
                             compression_opts=4)
        f.attrs[SOURCE_FRAME_ATTR] = record.source_frame
        f.attrs[CHECKPOINT_ATTR] = record.checkpoint
        f.attrs[CONFIG_FILE_ATTR] = record.config_file
        f.attrs[CLASS_NAMES_ATTR] = json.dumps(list(record.class_names))
        f.attrs[IMAGE_SIZE_ATTR] = np.asarray(record.image_size, np.int32)
        f.attrs[ORIGINAL_SIZE_ATTR] = np.asarray(record.original_size, np.int32)
        f.attrs[NUM_DECODER_LAYERS_ATTR] = int(record.num_layers)
    return path


def read_attention(path):
    """Read a `<frame>.attn.hdf5` sidecar back into an `AttentionRecord`."""
    with h5py.File(path, "r") as f:
        kwargs = {name: f[key][()] for name, (key, _) in _ARRAY_KEYS.items()}
        for name, (key, _) in _OPTIONAL_ARRAY_KEYS.items():
            kwargs[name] = f[key][()] if key in f else None
        kwargs["source_frame"] = str(f.attrs.get(SOURCE_FRAME_ATTR, ""))
        kwargs["checkpoint"] = str(f.attrs.get(CHECKPOINT_ATTR, ""))
        kwargs["config_file"] = str(f.attrs.get(CONFIG_FILE_ATTR, ""))
        kwargs["class_names"] = json.loads(f.attrs.get(CLASS_NAMES_ATTR, "[]"))
        kwargs["image_size"] = tuple(int(v) for v in f.attrs[IMAGE_SIZE_ATTR])
        kwargs["original_size"] = tuple(int(v) for v in f.attrs[ORIGINAL_SIZE_ATTR])
    known = {fld.name for fld in fields(AttentionRecord)}
    return AttentionRecord(**{k: v for k, v in kwargs.items() if k in known})
