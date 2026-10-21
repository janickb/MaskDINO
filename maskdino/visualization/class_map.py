"""Dense encoder-level classification capture - "what class does the model think
is at each location?".

MaskDINO's classifier is a single shared `nn.Linear(hidden_dim, num_classes)` at
`sem_seg_head.predictor.class_embed`, and with TWO_STAGE=True (the default, and
what every surgical config runs) it is applied to *every* encoder memory token
before any query exists:

    enc_outputs_class_unselected = self.class_embed(output_memory)   # (bs, T, N)
    topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, 1)[1]

(maskdino_decoder.py:403-407). That tensor is a dense multi-resolution class
score map over the whole frame - the real classifier, evaluated at every spatial
location - and it is thrown away as soon as the top-k is taken:
`interm_outputs['pred_logits']` keeps only the `num_queries` selected rows. On
the 3s config (res3/res4/res5, strides 8/16/32) at 1024x1024 it is
128^2 + 64^2 + 32^2 = 21504 tokens x N classes, i.e. ~1.4 MB at fp32 - small
enough to store losslessly, unlike the decoder's dense attention.

Unlike attention.py this needs *no* edit anywhere in the model: `class_embed` is
an `nn.Module`, so a plain forward hook sees every call. It is called
`3 + DEC_LAYERS` times per forward - once densely, then once per prediction
stage (interm, initial_pred, one per decoder layer) - which with the real
configs' DEC_LAYERS=9 is 12 calls, at two distinct sequence lengths. The dense
call is the one whose length matches sum(h*w) over the feature levels; every
other call is at num_queries.

Two things to keep straight about what this shows:

- It is the classifier on ENCODER tokens, not on the final refined query
  embedding. Inference reports `class_embed(decoder_norm(hs[-1]))`, a different
  and better decision. This is the one that gates which queries exist at all: a
  location the encoder scores low for every class never gets a query, and no
  decoder layer can recover it.
- So it answers "where does the classifier respond, and to what" - it is NOT an
  explanation of any individual detection's final label. For that you need
  attribution back through the decoder.

Scores are sigmoid, not softmax: the instance-segmentation path builds
`class_embed` with no background row (`semantic_ce_loss=False`,
maskdino_decoder.py:137-139) and `instance_inference` reads it as
`mask_cls.sigmoid()` (maskdino.py:553). Classes are therefore independent and
need not sum to 1 at a location.

Typical use:

    with capture_encoder_class_logits(predictor.model) as cap:
        predictor(image_bgr)
    cmap = cap.record(class_names=meta.thing_classes)

    heat = cmap.class_heatmap("needle_holder")   # (H, W) sigmoid, frame pixels
    gate = cmap.selected_points()                # the top-k the gate kept
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields

import numpy as np


def _sigmoid(x):
    # explicit rather than scipy.special.expit: this module is imported by the
    # viewer, which has numpy but need not have scipy.
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))


@dataclass
class EncoderClassMap:
    """One frame's dense encoder-level class logits.

    - logits               (T, N) float32 - T = sum(h*w) over feature levels,
      N = num_classes. Raw, pre-sigmoid, exactly as `class_embed` produced them.
    - level_spatial_shapes (Lv, 2) int32 - (h, w) per level, in level order, so
      `logits` can be split back into per-level maps. sum(h*w) must equal T.
    - selected_tokens      (Q,) int32 - the token indices the two-stage gate kept
      as queries, reproduced from `logits` (see `_topk_tokens`).
    - image_size           (h, w) of the padded model frame, so a map can be
      rendered in the same coordinate frame as pred_masks/pred_boxes.
    """

    logits: np.ndarray
    level_spatial_shapes: np.ndarray
    selected_tokens: np.ndarray
    image_size: tuple = (0, 0)
    original_size: tuple = (0, 0)
    class_names: list = field(default_factory=list)
    source_frame: str = ""
    checkpoint: str = ""
    config_file: str = ""

    # -- shape bookkeeping -------------------------------------------------

    @property
    def num_classes(self):
        return int(self.logits.shape[1])

    @property
    def num_levels(self):
        return int(np.asarray(self.level_spatial_shapes).shape[0])

    def level_slices(self):
        """[(start, stop), ...] into the token axis, one per feature level."""
        sizes = np.asarray(self.level_spatial_shapes, np.int64).prod(1)
        stops = np.cumsum(sizes)
        return [(int(stop - size), int(stop)) for size, stop in zip(sizes, stops)]

    def class_index(self, cls):
        """Accepts a class id or a class name; returns the id."""
        if isinstance(cls, str):
            try:
                return self.class_names.index(cls)
            except ValueError:
                raise KeyError(
                    f"{cls!r} is not in class_names ({list(self.class_names)})"
                ) from None
        return int(cls)

    def label(self, cls):
        """"3 - needle_holder", for a class picker."""
        i = self.class_index(cls)
        name = self.class_names[i] if i < len(self.class_names) else str(i)
        return f"{i} - {name}"

    # -- per-level maps ----------------------------------------------------

    def level_maps(self, cls=None, apply_sigmoid=True):
        """Per-level score maps as a list of (h, w) arrays - or (h, w, N) when
        `cls` is None.

        This is the lossless view: one array per feature level, at that level's
        own resolution, no resampling. Everything else in this module is a
        convenience built on top.
        """
        out = []
        for (start, stop), (h, w) in zip(self.level_slices(), self.level_spatial_shapes):
            block = self.logits[start:stop]
            block = _sigmoid(block) if apply_sigmoid else np.asarray(block, np.float32)
            block = block.reshape(int(h), int(w), self.num_classes)
            out.append(block[:, :, self.class_index(cls)] if cls is not None else block)
        return out

    # -- frame-space renderings -------------------------------------------

    def class_heatmap(self, cls, levels=None, reduce="mean", shape=None,
                      apply_sigmoid=True):
        """One class's score map, upsampled to the padded frame as (H, W) float32.

        `levels` selects feature levels (None = all). `reduce` combines them:
        "mean" is the stable default, "max" shows "any level fires here" and is
        the better choice when an instrument is only resolved at one scale.
        """
        maps = self.level_maps(cls=cls, apply_sigmoid=apply_sigmoid)
        chosen = range(len(maps)) if levels is None else list(levels)
        if not list(chosen):
            raise ValueError("need at least one feature level")
        shape = tuple(shape or self.image_size)
        stack = np.stack([resize_nearest(maps[i], shape) for i in chosen])
        if reduce == "mean":
            return stack.mean(0)
        if reduce == "max":
            return stack.max(0)
        raise ValueError(f"reduce must be 'mean' or 'max', got {reduce!r}")

    def confidence_map(self, levels=None, reduce="mean", shape=None):
        """max-over-classes score per location - the quantity the two-stage gate
        ranks tokens by (`enc_outputs_class_unselected.max(-1)[0]`), so this is
        literally "how likely is the gate to put a query here"."""
        best = self.logits.max(1)
        return self._render_token_scalar(_sigmoid(best), levels, reduce, shape)

    def argmax_map(self, levels=None, shape=None):
        """Winning class id per location, as (H, W) int32.

        Nearest-neighbour only - averaging class *ids* across levels would be
        meaningless, so with several levels this takes the class with the
        highest score at each location after upsampling.
        """
        shape = tuple(shape or self.image_size)
        chosen = range(self.num_levels) if levels is None else list(levels)
        scored = np.stack([
            resize_nearest(m, shape) for i, m in enumerate(self.level_maps())
            if i in set(chosen)
        ])  # (lv, H, W, N)
        return scored.max(0).argmax(-1).astype(np.int32)

    def _render_token_scalar(self, values, levels, reduce, shape):
        """Shared path for a per-token scalar (T,) -> (H, W)."""
        shape = tuple(shape or self.image_size)
        chosen = range(self.num_levels) if levels is None else list(levels)
        chosen = set(chosen)
        planes = []
        for i, ((start, stop), (h, w)) in enumerate(
            zip(self.level_slices(), self.level_spatial_shapes)
        ):
            if i not in chosen:
                continue
            planes.append(resize_nearest(values[start:stop].reshape(int(h), int(w)), shape))
        if not planes:
            raise ValueError("need at least one feature level")
        stack = np.stack(planes)
        return stack.mean(0) if reduce == "mean" else stack.max(0)

    def rank_classes(self, levels=None, reduce="max"):
        """(order, peaks): class ids sorted by peak score, best first.

        Peak rather than mean: an instrument covers a small part of the frame,
        so ranking by the mean over all tokens ranks by how much of the image a
        class claims, which is the wrong question.
        """
        peaks = np.array([
            self.class_heatmap(c, levels=levels, reduce=reduce).max()
            for c in range(self.num_classes)
        ], np.float32)
        return np.argsort(-peaks), peaks

    # -- the two-stage gate ------------------------------------------------

    def selected_points(self):
        """Where the gate put its queries, as a (Q, 3) float32 array of
        (x, y, level) in padded-frame pixels.

        These are the `num_queries` tokens that become decoder queries. Drawn
        over `confidence_map()` this is the whole two-stage selection story in
        one picture: the score field, and which peaks of it got a query.
        """
        shapes = np.asarray(self.level_spatial_shapes, np.int64)
        slices = self.level_slices()
        h_pad, w_pad = self.image_size
        out = np.zeros((len(self.selected_tokens), 3), np.float32)
        for row, token in enumerate(np.asarray(self.selected_tokens, np.int64)):
            level = next(
                i for i, (start, stop) in enumerate(slices) if start <= token < stop
            )
            start, _ = slices[level]
            h, w = int(shapes[level, 0]), int(shapes[level, 1])
            local = int(token - start)
            y, x = divmod(local, w)
            # token centre, in padded-frame pixels
            out[row] = ((x + 0.5) * w_pad / w, (y + 0.5) * h_pad / h, level)
        return out

    def selected_classes(self):
        """(Q,) int32 - the argmax class of each selected token, i.e. what the
        gate thought it was selecting."""
        return self.logits[np.asarray(self.selected_tokens, np.int64)].argmax(1).astype(np.int32)

    # -- plain-container round trip ---------------------------------------

    def to_dict(self):
        out = {fld.name: getattr(self, fld.name) for fld in fields(self)}
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, data):
        known = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def resize_nearest(plane, shape):
    """Nearest-neighbour resize of a 2-D or 3-D (H, W, C) array to `shape`.

    Nearest rather than bilinear on purpose: a level-2 token covers a 32x32
    patch of the frame, and smoothing it implies spatial precision the encoder
    never had. The blocky rendering is the honest one.
    """
    plane = np.asarray(plane, np.float32)
    if plane.shape[:2] == tuple(shape):
        return plane
    ys = (np.arange(shape[0]) * plane.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * plane.shape[1] / shape[1]).astype(int)
    ys = ys.clip(0, plane.shape[0] - 1)
    xs = xs.clip(0, plane.shape[1] - 1)
    return plane[np.ix_(ys, xs)] if plane.ndim == 2 else plane[np.ix_(ys, xs, range(plane.shape[2]))]


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _predictor_and_class_embed(model):
    """(MaskDINODecoder, its shared class_embed Linear)."""
    try:
        predictor = model.sem_seg_head.predictor
        class_embed = predictor.class_embed
    except AttributeError as e:
        raise TypeError(
            f"{type(model).__name__} has no MaskDINO class head at "
            "sem_seg_head.predictor.class_embed - dense class-map capture is "
            "MaskDINO-specific."
        ) from e
    return predictor, class_embed


class _ClassMapCapture:
    """Handle yielded by `capture_encoder_class_logits`.

    Records every `class_embed` call of one forward pass and picks out the dense
    one afterwards, so it does not depend on call order - which matters because
    the dense call happens *before* the decoder runs, while the spatial shapes
    needed to identify it arrive with the decoder's own invocation.
    """

    def __init__(self, model):
        self.model = model
        self.predictor, self.class_embed = _predictor_and_class_embed(model)
        self._hooks = []
        self._calls = []          # every class_embed output, (L, N) for batch item 0
        self.spatial_shapes = None

    # -- plumbing ----------------------------------------------------------

    def _class_hook(self, _module, _args, output):
        # (bs, L, N) for both the dense call and every per-stage call; keep
        # batch item 0 and let _dense_call() tell them apart by L.
        if output.dim() == 3:
            self._calls.append(output[0].detach().cpu().float().numpy())

    def _decoder_pre_hook(self, _module, _args, kwargs):
        if "spatial_shapes" in kwargs:
            self.spatial_shapes = kwargs["spatial_shapes"].detach().cpu().numpy()

    def _arm(self):
        self._calls = []
        self.spatial_shapes = None
        self._hooks.append(self.class_embed.register_forward_hook(self._class_hook))
        self._hooks.append(
            self.predictor.decoder.register_forward_pre_hook(
                self._decoder_pre_hook, with_kwargs=True
            )
        )

    def _disarm(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    # -- public ------------------------------------------------------------

    def _dense_call(self):
        """The (T, N) dense encoder-token logits out of the recorded calls."""
        if not self._calls:
            raise RuntimeError(
                "class_embed recorded no calls - did the forward pass run inside "
                "the `with capture_encoder_class_logits(...)` block?"
            )
        if self.spatial_shapes is None:
            raise RuntimeError(
                "the decoder was never called, so the feature-level shapes needed "
                "to identify the dense class_embed call are unknown"
            )
        tokens = int(np.asarray(self.spatial_shapes, np.int64).prod(1).sum())
        matches = [c for c in self._calls if c.shape[0] == tokens]
        if not matches:
            seen = sorted({c.shape[0] for c in self._calls})
            raise RuntimeError(
                f"no class_embed call covered all {tokens} encoder tokens (saw "
                f"sequence lengths {seen}). Dense class logits only exist when "
                "MODEL.MaskDINO.TWO_STAGE is True - check the config."
            )
        # One dense call per forward; if a future config somehow produced several,
        # the last is the one the top-k gate consumed.
        return matches[-1]

    def record(self, image_size=None, original_size=None, class_names=None,
               source_frame="", checkpoint="", config_file=""):
        """Build an `EncoderClassMap` from the captured forward pass.

        `image_size` is the padded model frame (h, w); pass the one the rest of
        your overlay uses (`instances.image_size`) so the maps line up with
        pred_masks. It defaults to the level-0 shape times its stride, which is
        correct whenever SIZE_DIVISIBILITY makes the levels exact divisions.
        """
        logits = self._dense_call()
        shapes = np.asarray(self.spatial_shapes, np.int32)
        if image_size is None:
            # level 0 is the finest enc level (stride 8 on the 3s configs)
            stride = 8
            image_size = (int(shapes[0, 0]) * stride, int(shapes[0, 1]) * stride)
        return EncoderClassMap(
            logits=np.ascontiguousarray(logits, np.float32),
            level_spatial_shapes=shapes,
            selected_tokens=_topk_tokens(logits, self.predictor.num_queries),
            image_size=tuple(image_size),
            original_size=tuple(original_size or image_size),
            class_names=list(class_names or []),
            source_frame=source_frame,
            checkpoint=checkpoint,
            config_file=config_file,
        )


def _topk_tokens(logits, num_queries):
    """Reproduce the two-stage gate: the `num_queries` tokens with the highest
    max-over-classes logit (maskdino_decoder.py:406-407).

    Exact, not approximate: the gate ranks raw logits and sigmoid is monotonic,
    so ordering by either agrees. np.argpartition matches torch.topk's choice of
    *set*; the order within it differs, which does not matter - the decoder
    treats its queries as a set.
    """
    best = np.asarray(logits, np.float32).max(1)
    k = int(min(num_queries, best.shape[0]))
    idx = np.argpartition(-best, k - 1)[:k]
    return idx[np.argsort(-best[idx], kind="stable")].astype(np.int32)


@contextmanager
def capture_encoder_class_logits(model):
    """Record `class_embed`'s dense encoder-token call for one forward pass.

    Pure forward hooks on `class_embed` and the decoder - no model edit, no
    config change, nothing left behind. Yields a `_ClassMapCapture` whose
    `.record()` is valid after the block.

    Composes with `capture_decoder_attention`: they hook different modules, so
    nesting both around one forward pass gives you the dense encoder class map
    and the per-detection decoder attention from the same inference call.
    """
    cap = _ClassMapCapture(model)
    cap._arm()
    try:
        yield cap
    finally:
        cap._disarm()
