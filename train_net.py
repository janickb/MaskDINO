# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskDINO Training Script based on Mask2Former.
"""

try:
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import random
import time
import weakref
from collections import OrderedDict
from typing import Any

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    ClassMapping,
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    DetrDatasetMapper,
    Hdf5CocoInstanceDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    PlateauLRHook,
    PlateauLRScheduler,
    SemanticSegmentorWithTTA,
    ValidationLossHook,
    add_maskdino_config,
    apply_test_sample_stride,
    assert_train_test_class_mapping_consistent,
    build_warmup_cosine_restarts_lr_scheduler,
    set_num_classes_from_metadata,
    write_class_mapping_sidecar,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # add model EMA
        kwargs = {
            "trainer": weakref.proxy(self),
        }
        # kwargs.update(model_ema.may_get_ema_checkpointer(cfg, model)) TODO: release ema training for large models
        
        # Ride the compact class-id -> {category_id, name} map inside every
        # model_*.pth (readable as torch.load(p)["class_mapping"]) so a checkpoint
        # is self-describing.
        _train_md = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        _class_mapping = (
            ClassMapping.from_metadata(_train_md)
            if _train_md.get("class_mapping_entries", None)
            else None
        )
        if _class_mapping is not None:
            kwargs["class_mapping"] = _class_mapping
            if comm.is_main_process():
                write_class_mapping_sidecar(cfg.OUTPUT_DIR, _class_mapping)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(
        cls, cfg, dataset_name, output_folder=None, include_coco=True, include_hungarian=True
    ):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.

        `include_coco`/`include_hungarian` let build_hooks() split the "coco" evaluator
        type's two evaluators (COCOEvaluator + the optional HungarianInstanceEvaluator)
        across two separately-scheduled EvalHooks when MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
        .PERIOD decouples the confusion-matrix cadence from TEST.EVAL_PERIOD - see
        build_hooks(). Both default True so every other call site (--eval-only,
        test_with_TTA) is unaffected.
        """
        if output_folder is None:
            if len(cfg.DATASETS.TEST) > 1:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            if include_coco:
                evaluator_list.append(
                    COCOEvaluator(
                        dataset_name, output_dir=output_folder, allow_cached_coco=False
                    )
                )
            if include_hungarian and cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.ENABLED:
                from maskdino.evaluation.hungarian_instance_evaluation import (
                    HungarianInstanceEvaluator,
                )

                evaluator_list.append(
                    HungarianInstanceEvaluator(
                        dataset_name, cfg, distributed=True, output_dir=output_folder
                    )
                )

        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON:
                evaluator_list.append(
                    COCOPanopticEvaluator(dataset_name, output_folder)
                )
        # COCO
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if (
            evaluator_type == "coco_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Mapillary Vistas
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        if (
            evaluator_type == "mapillary_vistas_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
        ):
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name, distributed=True, output_dir=output_folder
                )
            )
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert torch.cuda.device_count() > comm.get_rank(), (
                "CityscapesEvaluator currently do not work with multiple machines."
            )
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert torch.cuda.device_count() > comm.get_rank(), (
                "CityscapesEvaluator currently do not work with multiple machines."
            )
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON:
                assert torch.cuda.device_count() > comm.get_rank(), (
                    "CityscapesEvaluator currently do not work with multiple machines."
                )
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MaskDINO.TEST.INSTANCE_ON:
                assert torch.cuda.device_count() > comm.get_rank(), (
                    "CityscapesEvaluator currently do not work with multiple machines."
                )
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if (
            evaluator_type == "ade20k_panoptic_seg"
            and cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
        ):
            evaluator_list.append(
                InstanceSegEvaluator(dataset_name, output_dir=output_folder)
            )
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # coco instance segmentation lsj new baseline
        if cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_detr":
            mapper = DetrDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Semantic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # instance segmentation read directly from .hdf5 frames
        elif cfg.INPUT.DATASET_MAPPER_NAME == "hdf5_coco_instance":
            logger = logging.getLogger("detectron2.trainer")
            mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
            t0 = time.time()
            dataset = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            )
            n = len(dataset)
            logger.debug(
                "get_detection_dataset_dicts returned %d dicts in %.2fs", n, time.time() - t0
            )
            mapper.set_copy_paste_sources(dataset)
            t0 = time.time()
            sampler = TrainingSampler(n, seed=cfg.SEED)
            logger.debug("TrainingSampler built in %.2fs", time.time() - t0)

            t0 = time.time()
            loader = build_detection_train_loader(
                cfg, dataset=dataset, mapper=mapper, sampler=sampler
            )
            logger.debug(
                "build_detection_train_loader returned in %.2fs", time.time() - t0
            )
            return loader
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # instance segmentation read directly from .hdf5 frames
        if cfg.INPUT.DATASET_MAPPER_NAME == "hdf5_coco_instance":
            mapper = Hdf5CocoInstanceDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_val_loss_loader(cls, cfg, dataset_name):
        """Like build_test_loader(), but with an is_train=True mapper so GT
        "instances" are attached - required by ValidationLossHook, since the
        model only takes the loss branch (vs. inference) when self.training is
        True, and that branch reads batched_inputs[0]["instances"] (see
        maskdino/solver/val_loss.py). Only wired up for hdf5_coco_instance - the
        mapper the reclassify configs actually use.
        """
        assert cfg.INPUT.DATASET_MAPPER_NAME == "hdf5_coco_instance", (
            f"VAL_LOSS only supports DATASET_MAPPER_NAME=hdf5_coco_instance, "
            f"got {cfg.INPUT.DATASET_MAPPER_NAME!r}"
        )
        mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`, except for two
        names that aren't stock detectron2 schedulers (both would otherwise be
        rejected with ValueError by detectron2's own builder):
          - "WarmupCosineRestartsLR": SGDR, see maskdino/solver/lr_scheduler.py.
          - "ReduceLROnPlateau": adaptive, metric-driven decay - see
            maskdino/solver/plateau.py. Drives the usual SOLVER.WARMUP_ITERS /
            WARMUP_FACTOR ramp itself (ReduceLROnPlateau has no such concept), then
            hands off to a PlateauLRHook registered in build_hooks() below, which
            drives everything after that at SOLVER.PLATEAU.CHECK_PERIOD cadence.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "WarmupCosineRestartsLR":
            return build_warmup_cosine_restarts_lr_scheduler(cfg, optimizer)
        if cfg.SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau":
            p = cfg.SOLVER.PLATEAU
            return PlateauLRScheduler(
                optimizer,
                mode=p.MODE,
                factor=p.FACTOR,
                patience=p.PATIENCE,
                threshold=p.THRESHOLD,
                cooldown=p.COOLDOWN,
                min_lr_fraction=p.MIN_LR / cfg.SOLVER.BASE_LR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            )
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        """
        Adds PlateauLRHook on top of the stock hook list when
        SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau" - it's the hook that
        actually drives self.scheduler (a PlateauLRScheduler)'s LR changes, since
        that scheduler's own per-iteration step() (called by detectron2's stock
        hooks.LRScheduler, already in the list from super()) is a no-op by design.

        Also splits the single stock EvalHook in two when
        MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.PERIOD requests a cadence different from
        TEST.EVAL_PERIOD: HungarianInstanceEvaluator roughly doubles per-image eval cost
        (GT mask decode + mask-IoU matrix + Hungarian match, all serial CPU) on top of
        COCOEvaluator's own RLE encoding, but only feeds a diagnostic confusion matrix -
        not the bbox/segm AP that SOLVER.PLATEAU/TensorBoard actually track - so it
        doesn't need to run as often. See build_evaluator()'s include_coco/
        include_hungarian params.

        Also adds ValidationLossHook when MODEL.MaskDINO.TEST.VAL_LOSS.ENABLED -
        logs "validation_loss" (+ "val_<component>") to EventStorage at
        TEST.EVAL_PERIOD cadence, using the exact same loss function training
        does, just evaluated on DATASETS.TEST[0] - see maskdino/solver/val_loss.py.
        """
        ret = super().build_hooks()
        if self.cfg.SOLVER.LR_SCHEDULER_NAME == "ReduceLROnPlateau":
            ret.append(
                PlateauLRHook(
                    self.scheduler,
                    self.cfg.SOLVER.PLATEAU.METRIC,
                    self.cfg.SOLVER.PLATEAU.CHECK_PERIOD,
                )
            )

        he = self.cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
        if he.ENABLED and he.PERIOD > 0 and he.PERIOD != self.cfg.TEST.EVAL_PERIOD:

            def cheap_test_and_save_results():
                evaluators = [
                    self.build_evaluator(self.cfg, name, include_hungarian=False)
                    for name in self.cfg.DATASETS.TEST
                ]
                self._last_eval_results = self.test(
                    self.cfg, self.model, evaluators=evaluators
                )
                return self._last_eval_results

            def hungarian_test_and_save_results():
                evaluators = [
                    self.build_evaluator(self.cfg, name, include_coco=False)
                    for name in self.cfg.DATASETS.TEST
                ]
                return self.test(self.cfg, self.model, evaluators=evaluators)

            # Stock DefaultTrainer.build_hooks() always installs exactly one
            # hooks.EvalHook (running both evaluators together at TEST.EVAL_PERIOD) -
            # replace it in place with two separately-scheduled ones.
            idx = next(i for i, h in enumerate(ret) if isinstance(h, hooks.EvalHook))
            ret[idx : idx + 1] = [
                hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, cheap_test_and_save_results),
                hooks.EvalHook(he.PERIOD, hungarian_test_and_save_results),
            ]

        if self.cfg.MODEL.MaskDINO.TEST.VAL_LOSS.ENABLED:
            val_loader = self.build_val_loss_loader(self.cfg, self.cfg.DATASETS.TEST[0])
            ret.append(ValidationLossHook(self.cfg.TEST.EVAL_PERIOD, val_loader))

        return ret

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        # reclassify-finetune with the encoder unfrozen: the linear class head
        # keeps BASE_LR, the (pretrained) encoder trains gentler.
        rf = cfg.MODEL.MaskDINO.RECLASSIFY_FINETUNE
        encoder_lr_mult = (
            rf.ENCODER_LR_MULTIPLIER
            if rf.ENABLED and rf.UNFREEZE_ENCODER
            else 1.0
        )
        encoder_lr_prefixes = ("sem_seg_head.pixel_decoder",)

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: list[dict[str, Any]] = []
        memo: set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if encoder_lr_mult != 1.0 and module_name.startswith(encoder_lr_prefixes):
                    hyperparams["lr"] = hyperparams["lr"] * encoder_lr_mult
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Fail fast if train/test still disagree on the class id space (wrong dataset
    # pairing, a future reclass variant this doesn't know about, ...) instead of
    # silently scoring every prediction against the wrong class index.
    assert_train_test_class_mapping_consistent(cfg)
    # NUM_CLASSES == -1 means "size the class head to the dataset": fill it in from
    # the registered dataset's effective class count. No-op for stock datasets.
    _ds_for_classes = (
        cfg.DATASETS.TEST[0] if args.eval_only else cfg.DATASETS.TRAIN[0]
    )
    set_num_classes_from_metadata(cfg, _ds_for_classes)
    apply_test_sample_stride(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino"
    )
    return cfg


def main(args):
    cfg = setup(args)
    # print("Command cfg:", cfg)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    print("[DEBUG] before Trainer(cfg)", flush=True)
    trainer = Trainer(cfg)
    print("[DEBUG] after Trainer(cfg), before resume_or_load", flush=True)
    trainer.resume_or_load(resume=args.resume)
    print("[DEBUG] after resume_or_load, before train()", flush=True)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()
    # random port
    port = random.randint(1000, 20000)
    args.dist_url = "tcp://127.0.0.1:" + str(port)

    if not args.eval_only and not args.resume:
        # Fresh training run: stamp a yyyymmddhhmmss_ prefix onto the run folder's own
        # name so starting the same config twice doesn't overwrite the previous run's
        # checkpoints/logs. Computed once here (before launch() forks one process per
        # GPU) and passed down via args.opts so every rank resolves the identical
        # OUTPUT_DIR - computing it independently per-rank inside setup() could let two
        # ranks land on different seconds and disagree. --resume/--eval-only skip this:
        # they target an existing run's folder as configured, not a new one.
        _cfg = get_cfg()
        add_deeplab_config(_cfg)
        add_maskdino_config(_cfg)
        _cfg.merge_from_file(args.config_file)
        _cfg.merge_from_list(args.opts)
        _run_dir = _cfg.OUTPUT_DIR.rstrip("/")
        _stamped = os.path.join(
            os.path.dirname(_run_dir),
            f"{time.strftime('%Y%m%d%H%M%S')}_{os.path.basename(_run_dir)}",
        )
        args.opts += ["OUTPUT_DIR", _stamped]

    print("Command Line Args:", args)
    print("pwd:", os.getcwd())
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
