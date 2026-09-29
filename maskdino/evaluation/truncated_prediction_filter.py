# Copyright (c) Facebook, Inc. and its affiliates.
"""Prediction-side counterpart of apply_truncated_instance_filter
(maskdino/data/datasets/register_hdf5_instance.py). Once frame-border-
truncated GT instances are dropped from DATASETS.TEST
(cfg.INPUT.EXCLUDE_TRUNCATED_INSTANCES), a still-correct model prediction for
that same truncated instrument has no GT left to match against, and gets
scored as a false positive instead of a true positive. DropTruncatedPredictions
wraps a DatasetEvaluators list and drops predicted instances whose box also
touches the frame edge before delegating process() - the identical
edge-touching bbox test (touches_frame_edge) as the GT-side filter, so both
sides of the comparison treat "partially outside the frame" the same way.
"""
import torch
from detectron2.evaluation import DatasetEvaluators

from ..data.datasets.register_hdf5_instance import touches_frame_edge


class DropTruncatedPredictions(DatasetEvaluators):
    def process(self, inputs, outputs):
        super().process(
            inputs, [self._drop_truncated(i, o) for i, o in zip(inputs, outputs)]
        )

    @staticmethod
    def _drop_truncated(inp, out):
        if "instances" not in out or len(out["instances"]) == 0:
            return out
        instances = out["instances"]
        height, width = inp["height"], inp["width"]
        boxes = instances.pred_boxes.tensor
        keep = torch.tensor(
            [
                not touches_frame_edge(x1, y1, x2, y2, height, width)
                for x1, y1, x2, y2 in boxes.tolist()
            ],
            dtype=torch.bool,
            device=boxes.device,
        )
        return {**out, "instances": instances[keep]}
