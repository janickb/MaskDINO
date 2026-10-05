# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

from maskdino.config import add_surgical_arch_config


def add_mask2former_config(cfg):
    """Add Mask2Former's own MODEL.MASK_FORMER namespace - and nothing else.

    Deliberately NOT upstream's add_maskformer2_config. Every *other* key that function
    touches is already added by add_maskdino_config with this fork's values, and
    re-adding them would silently change behaviour for every MaskDINO config that
    relies on a default. Concretely, upstream would:

      cfg.INPUT.DATASET_MAPPER_NAME             -> "mask_former_semantic"  (fork: "MaskDINO_semantic")
      cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME -> "BasePixelDecoder"      (fork: "MaskDINOEncoder")
      cfg.MODEL.SWIN = CN()                     -> WIPES the fork's whole SWIN block
      cfg.INPUT.{IMAGE_SIZE,MIN_SCALE,MAX_SCALE,COLOR_AUG_SSD,SIZE_DIVISIBILITY}
      cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA
      cfg.MODEL.SEM_SEG_HEAD.{MASK_DIM,TRANSFORMER_ENC_LAYERS,DEFORMABLE_TRANSFORMER_ENCODER_*}
      cfg.SOLVER.{WEIGHT_DECAY_EMBED,OPTIMIZER,BACKBONE_MULTIPLIER}

    all back to upstream Mask2Former's defaults. Those keys are shared harness, not
    architecture, so they stay owned by add_maskdino_config - which is also what makes
    the two arms' resolved configs differ only inside their own MODEL.<ARCH> namespace.

    Every SEM_SEG_HEAD key the vendored code reads is already provided: detectron2
    supplies COMMON_STRIDE / CONVS_DIM / IGNORE_VALUE / IN_FEATURES / LOSS_WEIGHT /
    NORM / NUM_CLASSES, and add_maskdino_config supplies MASK_DIM /
    TRANSFORMER_ENC_LAYERS / PIXEL_DECODER_NAME /
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES.

    Call order (see maskdino.config.build_base_cfg) is add_maskdino_config ->
    add_mask2former_config; tests/test_arch_ns.py pins that nothing outside
    MODEL.MASK_FORMER moves.
    """
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    # NOTE: MultiScaleMaskedTransformerDecoder builds DEC_LAYERS - 1 layers, so the
    # published value 10 means 9 real decoder layers. MaskDINO's DEC_LAYERS: 9 describes
    # those same 9 layers. The two numbers are NOT directly comparable - do not
    # "align" them.
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # point loss configs (PointRend sampling; detectron2.projects.point_rend)
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k`
    # in the original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training.
    # Parameter `beta` in the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # This fork's shared knobs - the same keys, with the same defaults, as
    # MODEL.MaskDINO.*, so maskdino.config.arch_ns(cfg) resolves to an equivalent node
    # for either architecture and one train_net.py / HungarianInstanceEvaluator drives
    # both. Must come after MODEL.MASK_FORMER.TEST exists.
    add_surgical_arch_config(cfg.MODEL.MASK_FORMER)
