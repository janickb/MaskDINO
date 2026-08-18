# ------------------------------------------------------------------------
# Runtime-only add-on: captures MaskDINO's per-query decoder embedding
# (the 256-d vector feeding class_embed/mask_embed) for each final detected
# instance, without any change to how the detector itself is trained or run.
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F

from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances


class EmbeddingPredictor:
    """Wraps DefaultPredictor and attaches `pred_embeddings` (L2-normalized) and
    `raw_embedding` (unnormalized) fields, each shape (N, hidden_dim), to the
    returned Instances, one row per kept detection.

    The embedding for instance i is the decoder's per-query feature vector for
    whichever of the num_queries object queries produced that detection
    (Instances.query_idx, set in MaskDINO.instance_inference).
    """

    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)
        self._captured = None
        class_embed = self.predictor.model.sem_seg_head.predictor.class_embed
        class_embed.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # class_embed is called several times per forward pass (two-stage encoder
        # proposals, optional initial prediction, once per decoder layer); the
        # last call within a forward pass is always the final decoder layer's
        # output, which is what produces the final pred_logits/pred_masks/pred_boxes.
        # Overwriting on every call and reading after predictor() returns yields
        # exactly that final-layer tensor, shape (bs, num_queries, hidden_dim).
        self._captured = inputs[0].detach()

    def __call__(self, image_bgr) -> Instances:
        self._captured = None
        outputs = self.predictor(image_bgr)
        instances = outputs["instances"]

        if self._captured is None:
            raise RuntimeError(
                "class_embed forward hook did not fire; check that "
                "model.sem_seg_head.predictor.class_embed still exists at this path."
            )
        query_embeddings = self._captured[0]  # (num_queries, hidden_dim), still on model device

        if len(instances) == 0:
            instances.pred_embeddings = torch.zeros(
                (0, query_embeddings.shape[-1]), dtype=query_embeddings.dtype
            )
            instances.raw_embedding = instances.pred_embeddings
            return instances

        query_idx = instances.query_idx.to(query_embeddings.device)
        embeddings = query_embeddings[query_idx]
        # left on the model device, same as every other Instances field here (pred_masks,
        # pred_boxes, ...); callers already do `.to("cpu")` on the whole Instances object
        # (see demo/run_inference.py) before consuming predictions.
        instances.raw_embedding = embeddings
        instances.pred_embeddings = F.normalize(embeddings, dim=-1)
        return instances
