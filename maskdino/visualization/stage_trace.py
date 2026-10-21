"""Per-stage class and mask confidence for one query - "where in the decoder
does the mask stop agreeing with the label?".

MaskDINO reads out its two heads from the *same* 256-d query vector at every
stage (`forward_prediction_heads`), but they are siblings, not a pipeline: the
classifier never sees the mask. So a query can hold a near-perfect mask while
its label drifts, and nothing in the architecture objects. This module makes
that drift visible stage by stage.

    interm -> initial_pred -> layer 0 ... layer N-1

The catch: at inference `pred_mask` is False for every intermediate layer
(maskdino_decoder.py:454 and :472 gate it on `self.training`), so the model
only ever computes a mask at `interm` and at the last layer. The intermediate
masks - exactly the ones needed to see a drift - do not exist.

They can be reconstructed exactly, because `class_embed` IS called at every
stage, on the very `decoder_output` the mask head would have used. A forward
PRE-hook on class_embed captures that tensor; running `mask_embed` and the
einsum on it reproduces `forward_prediction_heads`' own arithmetic. Nothing in
the model changes, and the only edit this relies on is
`Instances.query_indices` (maskdino.py), the detection -> query mapping.

One caveat on scale: the reconstructed masks live at `mask_features`
resolution (1/4 of the padded frame), while `instance_inference` scores the
upsampled mask. The per-stage mask scores here are therefore self-consistent
with each other - which is what a drift comparison needs - but sit slightly
off the reported detection score. `final_score_check()` reports the gap.

Typical use:

    with capture_stage_trace(predictor.model) as cap:
        instances = predictor(image_bgr)["instances"].to("cpu")
    trace = cap.record(instances[instances.scores > 0.3])
    trace.divergence(det=0)      # (stage index, what happened)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

import numpy as np

GROUP = "stage_trace"

STAGE_NAMES_ATTR = "stage_names"
CLASS_NAMES_ATTR = "class_names"
IMAGE_SIZE_ATTR = "image_size"
MASK_SIZE_ATTR = "mask_size"
SOURCE_FRAME_ATTR = "source_frame"
CHECKPOINT_ATTR = "checkpoint"
CONFIG_FILE_ATTR = "config_file"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float32)))


@dataclass
class StageTrace:
    """One frame's per-stage readout, for `D` detections over `S` stages.

    - class_logits (D, S, N) float32 - raw, pre-sigmoid. Scores are sigmoid and
      NOT softmax: the instance path builds class_embed with no background row,
      so classes are independent and do not sum to 1.
    - mask_scores  (D, S)    float32 - mean sigmoid probability inside the
      stage's own binarized mask, the same quantity instance_inference
      multiplies the class score by.
    - mask_areas   (D, S)    int32   - foreground pixels at mask resolution,
      so a mask collapsing to nothing is distinguishable from one that stays
      put but loses confidence.
    """

    stage_names: list
    class_logits: np.ndarray
    mask_scores: np.ndarray
    mask_areas: np.ndarray
    query_index: np.ndarray
    detection_index: np.ndarray
    scores: np.ndarray
    pred_classes: np.ndarray
    pred_boxes: np.ndarray
    image_size: tuple = (0, 0)
    mask_size: tuple = (0, 0)
    class_names: list = field(default_factory=list)
    source_frame: str = ""
    checkpoint: str = ""
    config_file: str = ""

    def __len__(self):
        return int(self.class_logits.shape[0])

    @property
    def num_stages(self):
        return int(self.class_logits.shape[1])

    def class_probs(self, det):
        """(S, N) sigmoid scores for every stage."""
        return _sigmoid(self.class_logits[det])

    def label(self, det):
        cls = int(self.pred_classes[det])
        name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
        return f"#{det} - {name} - {self.scores[det]:.2f}"

    def class_name(self, idx):
        idx = int(idx)
        return self.class_names[idx] if idx < len(self.class_names) else str(idx)

    def top_classes(self, det, k=3):
        """Class indices worth plotting: the top-k at any stage, union'd, so a
        class that leads early and loses is not dropped from the chart."""
        probs = self.class_probs(det)
        keep = {int(i) for stage in probs for i in np.argsort(-stage)[:k]}
        keep.add(int(self.pred_classes[det]))
        return sorted(keep, key=lambda i: -probs[:, i].max())

    def winner_per_stage(self, det):
        """(S,) index of the highest-scoring class at each stage."""
        return self.class_probs(det).argmax(-1)

    def divergence(self, det):
        """First stage where the leading class changes, as (index, name, from, to).

        Returns None when one class leads throughout - in which case any drift
        is in confidence, not identity, and `mask_scores` vs the winner's
        probability is the thing to read.
        """
        win = self.winner_per_stage(det)
        for s in range(1, len(win)):
            if win[s] != win[s - 1]:
                return (s, self.stage_names[s],
                        self.class_name(win[s - 1]), self.class_name(win[s]))
        return None

    def encoder_stage(self):
        """Index of the encoder-level readout.

        `interm` is `class_embed` on the gathered encoder memory tokens, before
        any decoder layer runs - the encoder's own verdict on the query it just
        selected. Returns None when TWO_STAGE is off, in which case there is no
        encoder-level per-query prediction to compare against.
        """
        return self.stage_names.index("interm") if "interm" in self.stage_names else None

    def agrees_with_final(self, det):
        """Does the encoder's leading class survive to the last stage?

        None when there is no encoder stage to compare (see `encoder_stage`).
        """
        enc = self.encoder_stage()
        if enc is None:
            return None
        win = self.winner_per_stage(det)
        return bool(win[enc] == win[-1])

    def agreement_rate(self, min_score=None):
        """(n_agree, n_total) over traced detections scoring above `min_score`.

        Measures how often the decoder *keeps* the encoder's label rather than
        overturning it. A low rate is not itself a fault - the decoder exists to
        refine - but paired with a flat mask track it says the refinement is
        changing the label while the segmentation stands still.
        """
        if self.encoder_stage() is None or len(self) == 0:
            return 0, 0
        dets = [d for d in range(len(self))
                if min_score is None or float(self.scores[d]) > min_score]
        return sum(1 for d in dets if self.agrees_with_final(d)), len(dets)

    def stage_agreement(self, min_score=None):
        """(S,) fraction of traced detections whose winner at each stage equals
        their own final winner - i.e. where the decision settles.

        Takes the same `min_score` filter as `agreement_rate` so the two
        describe the same population; without it the low-confidence tail of the
        traced set dominates and the numbers are not comparable.
        """
        dets = [d for d in range(len(self))
                if min_score is None or float(self.scores[d]) > min_score]
        if not dets:
            return np.zeros(self.num_stages, np.float32)
        wins = np.stack([self.winner_per_stage(d) for d in dets])
        return (wins == wins[:, -1:]).mean(0).astype(np.float32)

    def final_score_check(self, det):
        """(reconstructed, reported) final score.

        Reconstructed = last stage's class sigmoid x its mask score, i.e. what
        instance_inference computes. The gap is the 1/4-resolution mask
        reconstruction, not a logic error - see the module docstring.
        """
        cls = int(self.pred_classes[det])
        rebuilt = float(self.class_probs(det)[-1, cls] * self.mask_scores[det, -1])
        return rebuilt, float(self.scores[det])

    def to_dict(self):
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, data):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _predictor_of(model):
    try:
        predictor = model.sem_seg_head.predictor
        predictor.class_embed, predictor.mask_embed  # noqa: B018 - presence check
    except AttributeError as e:
        raise TypeError(
            f"{type(model).__name__} has no MaskDINO prediction heads at "
            "sem_seg_head.predictor.{class_embed,mask_embed}"
        ) from e
    return predictor


class _StageCapture:
    def __init__(self, model):
        self.model = model
        self.predictor = _predictor_of(model)
        self._hooks = []
        self._inputs = []      # decoder_output per class_embed call
        self._outputs = []     # logits per class_embed call
        self.mask_features = None

    # -- plumbing ----------------------------------------------------------

    def _predictor_pre_hook(self, _m, args, kwargs):
        # MaskDINODecoder.forward(x, mask_features, masks, targets=None)
        feats = kwargs.get("mask_features")
        if feats is None and len(args) > 1:
            feats = args[1]
        if feats is not None:
            self.mask_features = feats.detach()
        return None

    def _class_pre_hook(self, _m, args):
        self._inputs.append(args[0].detach())

    def _class_hook(self, _m, _args, output):
        self._outputs.append(output.detach())

    def _arm(self):
        self._hooks = [
            self.predictor.register_forward_pre_hook(self._predictor_pre_hook,
                                                     with_kwargs=True),
            self.predictor.class_embed.register_forward_pre_hook(self._class_pre_hook),
            self.predictor.class_embed.register_forward_hook(self._class_hook),
        ]

    def _disarm(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # -- public ------------------------------------------------------------

    def _query_stages(self, num_queries):
        """The per-query class_embed calls, in call order, dropping the dense
        encoder-token call (identified by its sequence length)."""
        pairs = [(i, o) for i, o in zip(self._inputs, self._outputs)
                 if i.shape[-2] == num_queries]
        if not pairs:
            seen = sorted({int(i.shape[-2]) for i in self._inputs})
            raise RuntimeError(
                f"no class_embed call at num_queries={num_queries}; saw lengths {seen}. "
                "Did the forward pass run inside the `with capture_stage_trace(...)` block?"
            )
        return pairs

    def stage_names(self, n_stages):
        """Name the per-query calls by MaskDINODecoder.forward's call order:
        interm (two-stage query selection), initial_pred, then one per decoder
        layer.

        Both leading stages are optional and independently so - `interm` needs
        TWO_STAGE, `initial` needs INITIAL_PRED - which the stage *count* alone
        cannot disambiguate. Read the predictor's own flags rather than guessing,
        and fall back to positional names if the arithmetic does not add up.
        """
        n_layers = getattr(self.predictor, "num_layers", None) or 0
        lead = []
        if getattr(self.predictor, "two_stage", False):
            lead.append("interm")
        if getattr(self.predictor, "initial_pred", False):
            lead.append("initial")
        names = lead + [f"layer{i}" for i in range(n_layers)]
        if len(names) != n_stages:  # unexpected arrangement - stay honest
            names = [f"stage{i}" for i in range(n_stages)]
        return names

    def record(self, instances, image_size=None, max_detections=None, class_names=None,
               source_frame="", checkpoint="", config_file=""):
        """Join the per-stage readouts to a kept-detection `Instances`."""
        import torch

        if not instances.has("query_indices"):
            raise ValueError(
                "Instances has no `query_indices` field - MaskDINO.instance_inference "
                "must expose it to map a detection back to its decoder query."
            )
        if self.mask_features is None:
            raise RuntimeError("no mask_features captured; did the forward pass run?")

        num_queries = int(self._inputs[-1].shape[-2])
        pairs = self._query_stages(num_queries)
        names = self.stage_names(len(pairs))

        queries = instances.query_indices.cpu().numpy().astype(np.int64)
        scores = instances.scores.cpu().numpy().astype(np.float32)
        order = np.argsort(-scores, kind="stable")
        _, first = np.unique(queries[order], return_index=True)
        keep = order[np.sort(first)]
        keep = keep[np.argsort(-scores[keep], kind="stable")]
        if max_detections is not None:
            keep = keep[:max_detections]
        q = queries[keep]

        # (S, D, C) decoder outputs for the kept queries only
        dec = torch.stack([i[0, q] if i.dim() == 3 else i[q] for i, _ in pairs])
        logits = torch.stack([o[0, q] if o.dim() == 3 else o[q] for _, o in pairs])

        feats = self.mask_features[0] if self.mask_features.dim() == 4 else self.mask_features
        s, d, c = dec.shape
        # Reproduce forward_prediction_heads' mask branch for the stages that
        # skipped it. Batched over stages x detections so this is one einsum,
        # not S*D of them; only the kept queries are materialised.
        with torch.no_grad():
            embed = self.predictor.mask_embed(dec.reshape(s * d, c))
            masks = torch.einsum("nc,chw->nhw", embed, feats.to(embed.dtype))
            prob = masks.sigmoid()
            binary = masks > 0
            area = binary.flatten(1).sum(1)
            score = (prob.flatten(1) * binary.flatten(1)).sum(1) / area.clamp_min(1)

        mask_scores = score.reshape(s, d).t().cpu().numpy().astype(np.float32)
        mask_areas = area.reshape(s, d).t().cpu().numpy().astype(np.int32)

        return StageTrace(
            stage_names=names,
            class_logits=np.ascontiguousarray(
                logits.permute(1, 0, 2).cpu().numpy(), np.float32),
            mask_scores=mask_scores,
            mask_areas=mask_areas,
            query_index=q.astype(np.int32),
            detection_index=keep.astype(np.int32),
            scores=scores[keep],
            pred_classes=instances.pred_classes.cpu().numpy()[keep].astype(np.int32),
            pred_boxes=instances.pred_boxes.tensor.cpu().numpy()[keep].astype(np.float32),
            image_size=tuple(image_size or instances.image_size),
            mask_size=tuple(feats.shape[-2:]),
            class_names=list(class_names or []),
            source_frame=source_frame,
            checkpoint=checkpoint,
            config_file=config_file,
        )


@contextmanager
def capture_stage_trace(model):
    """Record every per-query `class_embed` call of one forward pass, plus the
    `mask_features` needed to reconstruct the masks the model skipped.

    Pure hooks - no model edit, nothing left behind on exit.
    """
    cap = _StageCapture(model)
    cap._arm()
    try:
        yield cap
    finally:
        cap._disarm()


# --------------------------------------------------------------------------
# sidecar I/O
# --------------------------------------------------------------------------

_ARRAYS = {
    "class_logits": np.float32,
    "mask_scores": np.float32,
    "mask_areas": np.int32,
    "query_index": np.int32,
    "detection_index": np.int32,
    "scores": np.float32,
    "pred_classes": np.int32,
    "pred_boxes": np.float32,
}


def write_stage_trace(path, trace):
    import h5py

    with h5py.File(path, "w") as f:
        for name, dtype in _ARRAYS.items():
            f.create_dataset(f"{GROUP}/{name}", data=np.asarray(getattr(trace, name), dtype))
        f.attrs[STAGE_NAMES_ATTR] = json.dumps(list(trace.stage_names))
        f.attrs[CLASS_NAMES_ATTR] = json.dumps(list(trace.class_names))
        f.attrs[IMAGE_SIZE_ATTR] = np.asarray(trace.image_size, np.int32)
        f.attrs[MASK_SIZE_ATTR] = np.asarray(trace.mask_size, np.int32)
        f.attrs[SOURCE_FRAME_ATTR] = trace.source_frame
        f.attrs[CHECKPOINT_ATTR] = trace.checkpoint
        f.attrs[CONFIG_FILE_ATTR] = trace.config_file
    return path


def read_stage_trace(path):
    import h5py

    with h5py.File(path, "r") as f:
        kwargs = {name: f[f"{GROUP}/{name}"][()] for name in _ARRAYS}
        kwargs["stage_names"] = json.loads(f.attrs[STAGE_NAMES_ATTR])
        kwargs["class_names"] = json.loads(f.attrs.get(CLASS_NAMES_ATTR, "[]"))
        kwargs["image_size"] = tuple(int(v) for v in f.attrs[IMAGE_SIZE_ATTR])
        kwargs["mask_size"] = tuple(int(v) for v in f.attrs[MASK_SIZE_ATTR])
        kwargs["source_frame"] = str(f.attrs.get(SOURCE_FRAME_ATTR, ""))
        kwargs["checkpoint"] = str(f.attrs.get(CHECKPOINT_ATTR, ""))
        kwargs["config_file"] = str(f.attrs.get(CONFIG_FILE_ATTR, ""))
    return StageTrace.from_dict(kwargs)
