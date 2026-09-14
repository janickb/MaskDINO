# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from detectron2.config import CfgNode as CN


def add_maskdino_config(cfg):
    """
    Add config for MaskDINO.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "MaskDINO_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    # SGDR cosine annealing with warm restarts (maskdino/solver/lr_scheduler.py). Only
    # consulted when SOLVER.LR_SCHEDULER_NAME == "WarmupCosineRestartsLR".
    cfg.SOLVER.COSINE_RESTARTS = CN()
    cfg.SOLVER.COSINE_RESTARTS.T_0 = 1000  # iters; length of the first cycle
    cfg.SOLVER.COSINE_RESTARTS.T_MULT = 2.0  # growth factor per restart (2.0 = classic SGDR)
    # Soften every restart after cycle 0 with its own short ramp instead of snapping
    # straight to BASE_LR (plain SGDR). Both default to 0.0 = warm restarts, unchanged.
    cfg.SOLVER.COSINE_RESTARTS.RESTART_WARMUP_FACTOR = 0.0  # start-of-restart LR, as a fraction of BASE_LR
    cfg.SOLVER.COSINE_RESTARTS.RESTART_WARMUP_FRACTION = 0.0  # fraction of each restart cycle spent ramping

    # Adaptive alternative to the fixed schedules above (maskdino/solver/plateau.py).
    # Only consulted when SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau".
    cfg.SOLVER.PLATEAU = CN()
    cfg.SOLVER.PLATEAU.METRIC = "total_loss"  # EventStorage key to monitor
    cfg.SOLVER.PLATEAU.MODE = "min"  # "min" for a loss-like metric, "max" for accuracy-like
    cfg.SOLVER.PLATEAU.FACTOR = 0.5  # multiply LR by this on each reduction
    cfg.SOLVER.PLATEAU.PATIENCE = 3  # checks with no improvement before reducing
    cfg.SOLVER.PLATEAU.THRESHOLD = 1e-4  # min relative improvement to count as "improved"
    cfg.SOLVER.PLATEAU.COOLDOWN = 0  # checks to wait after a reduction before resuming patience
    # Floor for BASE_LR specifically; every param group (e.g. the backbone at
    # BACKBONE_MULTIPLIER x BASE_LR) is floored at the same fraction of its own
    # base, not this absolute number - see PlateauLRScheduler's docstring.
    cfg.SOLVER.PLATEAU.MIN_LR = 0.0
    cfg.SOLVER.PLATEAU.CHECK_PERIOD = 0  # iters between checks; 0 = disabled even if selected

    # MaskDINO model config
    cfg.MODEL.MaskDINO = CN()
    cfg.MODEL.MaskDINO.LEARN_TGT = False

    # loss
    cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
    cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.1
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.DEC_LAYERS = 6
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.BOX_LOSS = True
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.MaskDINO.DN="seg"
    cfg.MODEL.MaskDINO.DN_NOISE_SCALE=0.4
    cfg.MODEL.MaskDINO.DN_NUM=100
    cfg.MODEL.MaskDINO.PRED_CONV=False

    cfg.MODEL.MaskDINO.EVAL_FLAG = 1

    # Classifier-only few-shot retraining
    cfg.MODEL.MaskDINO.CLASSIFIER_RETRAIN = CN()
    cfg.MODEL.MaskDINO.CLASSIFIER_RETRAIN.ENABLED = False
    cfg.MODEL.MaskDINO.CLASSIFIER_RETRAIN.TRAINABLE_PARAM_PREFIXES = [
        "sem_seg_head.predictor.class_embed"
    ]
    # Also unfreeze the transformer decoder (DINO decoder layers + the mask/box
    # prediction heads it drives), leaving the backbone and the MSDeformAttn pixel
    # encoder frozen. Lets the query features adapt to a few-shot set instead of
    # only the linear class head. When True, the loss set is NOT forced to
    # labels-only - mask/box losses + deep supervision come back so the decoder
    # is anchored on segmentation quality.
    cfg.MODEL.MaskDINO.CLASSIFIER_RETRAIN.UNFREEZE_DECODER = False
    # Decoder params train at BASE_LR * this factor (the linear class head keeps
    # the full BASE_LR); only applied when UNFREEZE_DECODER is True.
    cfg.MODEL.MaskDINO.CLASSIFIER_RETRAIN.DECODER_LR_MULTIPLIER = 0.1

    # -1 = "derive from the dataset": for datasets registered with a compact class
    # mapping (the surgical HDF5 loaders), train_net.setup() fills this in from the
    # dataset's effective class count via set_num_classes_from_metadata(). Stock
    # datasets (COCO/ADE/panoptic) keep whatever their YAML pins.
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = -1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # MaskDINO inference config
    cfg.MODEL.MaskDINO.TEST = CN()
    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
    # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

    # Per-class mask-IoU NMS in instance_inference(): among predictions sharing the
    # SAME predicted label, greedily drop the lower-scoring one whenever mask IoU
    # exceeds this. Predictions with different labels are never compared, so two
    # genuinely distinct overlapping instruments are unaffected. 0 disables (default,
    # matches upstream - no NMS).
    cfg.MODEL.MaskDINO.TEST.NMS_IOU = 0.0

    # Hungarian mask-IoU instance evaluator (maskdino/evaluation/hungarian_instance_evaluation.py).
    # These knobs are read only by that evaluator; the model's own inference-time score
    # gating stays OBJECT_MASK_THRESHOLD / TEST.DETECTIONS_PER_IMAGE.
    cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL = CN()
    cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.ENABLED = False        # default off -> existing runs unchanged
    cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.SCORE_THRESH = 0.5     # drop preds below this confidence
    cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.IOU_THRESH = 0.5       # min mask IoU for a match to be accepted
    cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.BOX_PREFILTER = True   # box-IoU prune before exact mask IoU
    # GT visibility filtering is NOT a separate knob: the evaluator always reuses
    # INPUT.MIN_VISIBILITY so "recall" is measured against the same GT the model trained on.

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

    # transformer module
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    cfg.INPUT.MIN_VISIBILITY = 0.0

    cfg.INPUT.RANDOM_ROTATION = True
    cfg.INPUT.ROTATION_ANGLES = [-180.0, -90.0, 0.0, 90.0, ]
    # if False, keep the image size and let the corners rotate out of frame
    cfg.INPUT.ROTATION_EXPAND = False

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false
