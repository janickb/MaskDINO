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
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_argument_parser,
    default_setup,
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
    SemanticSegmentorWithTTA,
    add_maskdino_config,
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
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
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
            evaluator_list.append(
                COCOEvaluator(
                    dataset_name, output_dir=output_folder, allow_cached_coco=False
                )
            )
            if cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.ENABLED:
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
            import time as _t

            from detectron2.data.build import get_detection_dataset_dicts
            from detectron2.data.samplers import TrainingSampler

            print("[DEBUG] building Hdf5CocoInstanceDatasetMapper", flush=True)
            mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
            print("[DEBUG] mapper built", flush=True)

            _t0 = _t.time()
            dataset = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            )
            print(
                f"[DEBUG] get_detection_dataset_dicts returned in {_t.time() - _t0:.2f}s, type={type(dataset)}",
                flush=True,
            )

            _t0 = _t.time()
            n = len(dataset)
            print(f"[DEBUG] len(dataset)={n} took {_t.time() - _t0:.2f}s", flush=True)

            _t0 = _t.time()
            sampler = TrainingSampler(n, seed=cfg.SEED)
            print(
                f"[DEBUG] TrainingSampler built in {_t.time() - _t0:.2f}s", flush=True
            )

            _t0 = _t.time()
            loader = build_detection_train_loader(
                cfg, dataset=dataset, mapper=mapper, sampler=sampler
            )
            print(
                f"[DEBUG] build_detection_train_loader returned in {_t.time() - _t0:.2f}s",
                flush=True,
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
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

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
    # NUM_CLASSES == -1 means "size the class head to the dataset": fill it in from
    # the registered dataset's effective class count. No-op for stock datasets.
    _ds_for_classes = (
        cfg.DATASETS.TEST[0] if args.eval_only else cfg.DATASETS.TRAIN[0]
    )
    set_num_classes_from_metadata(cfg, _ds_for_classes)
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
