"""Shared setup for the mapper-driven preview/dump tools in this directory.

tools/augmentation/dump_training_samples.py needs the same four steps - build a cfg, resolve the training dataset dicts, construct
Hdf5CocoInstanceDatasetMapper in training mode, and wire the copy-paste sources -
and it must do so *without* silently overriding what the config file says. That
matters for A/B arm configs, whose augmentation knobs are the thing under test:
a tool that forces its own BLEND_MODES/CLUSTER_* on top of the file is previewing
a condition that will never be trained.

Import this only after the repo root is on sys.path (every tool in here does a
sys.path.insert at the top), since it imports `maskdino`.
"""
import logging

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

from maskdino.config import add_maskdino_config
from maskdino.data.dataset_mappers.hdf5_coco_instance_dataset_mapper import (
    Hdf5CocoInstanceDatasetMapper,
)

__all__ = ["build_cfg", "build_preview_mapper"]

logger = logging.getLogger(__name__)


def build_cfg(config_file, opts=(), force_copy_paste=True):
    """Merge `config_file` + trailing `opts` (detectron2 KEY VALUE pairs) into a
    frozen cfg. `force_copy_paste` only flips INPUT.COPY_PASTE.ENABLED on - every
    other copy-paste knob is left exactly as the config file set it."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(list(opts))
    if force_copy_paste and not cfg.INPUT.COPY_PASTE.ENABLED:
        logger.info("[preview] forcing INPUT.COPY_PASTE.ENABLED on for this preview")
        cfg.INPUT.COPY_PASTE.ENABLED = True
    cfg.freeze()
    return cfg


def build_preview_mapper(config_file, opts=(), force_copy_paste=True):
    """Returns (cfg, mapper, dataset_dicts, class_names) for cfg.DATASETS.TRAIN[0],
    with the mapper in training mode and its copy-paste sources wired in."""
    cfg = build_cfg(config_file, opts, force_copy_paste=force_copy_paste)

    name = cfg.DATASETS.TRAIN[0]
    dataset_dicts = DatasetCatalog.get(name)
    class_names = MetadataCatalog.get(name).thing_classes

    mapper = Hdf5CocoInstanceDatasetMapper(cfg, True)
    mapper.set_copy_paste_sources(dataset_dicts)

    return cfg, mapper, dataset_dicts, class_names
