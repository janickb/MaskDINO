# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskdino_config

# compact ("effective") class label space + embedded checkpoint mapping
from .data.class_mapping import (
    ClassMapping,
    derive_class_mapping,
    load_class_mapping,
    set_num_classes_from_metadata,
    write_class_mapping_sidecar,
)

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.detr_dataset_mapper import DetrDatasetMapper
from .data.dataset_mappers.hdf5_coco_instance_dataset_mapper import Hdf5CocoInstanceDatasetMapper

from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskdino import MaskDINO
# from .data.datasets_detr import coco
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
# solver
from .solver import build_warmup_cosine_restarts_lr_scheduler
# util
from .utils import box_ops, misc, utils
