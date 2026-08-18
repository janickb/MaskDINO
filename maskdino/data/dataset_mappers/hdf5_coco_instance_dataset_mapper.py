# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import copy
import logging

import h5py
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from sgdata import schema

from .coco_instance_new_baseline_dataset_mapper import build_transform_gen

__all__ = ["Hdf5CocoInstanceDatasetMapper"]


class Hdf5CocoInstanceDatasetMapper:
    """
    A callable which takes a dataset dict pointing at a BlenderProc-style .hdf5 frame
    (see sgdata.schema for the `coco_annotations` schema) and maps it
    into the format used by MaskDINO for instance segmentation.

    Unlike COCOInstanceNewBaselineDatasetMapper, the image ("colors") is read directly
    out of the .hdf5 file named by dataset_dict["file_name"] — there is no separate
    image file. Annotations are expected to already be present in
    dataset_dict["annotations"] (populated from the file's `coco_annotations` key by
    register_hdf5_instance.list_hdf5_dicts at dataset-listing time), same as a regular
    COCO-format dataset dict — there is no separate COCO JSON either.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        min_visibility=0.0,
    ):
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[Hdf5CocoInstanceDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.min_visibility = min_visibility

    @classmethod
    def from_config(cls, cfg, is_train=True):
        if is_train:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            # build_transform_gen only supports train-time LSJ augmentation; at eval
            # time just resize like Detectron2's own default DatasetMapper does.
            tfm_gens = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")]

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "min_visibility": cfg.INPUT.MIN_VISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        with h5py.File(dataset_dict["file_name"], "r") as f:
            image = f[schema.COLORS][()]

        assert image.ndim == 3 and image.shape[2] == 3, "expected an (H, W, 3) RGB 'colors' array"
        if self.img_format == "BGR":
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)

        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0 and obj.get("visibility_fraction", 1.0) >= self.min_visibility
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        if not instances.has("gt_masks"):  # image has no (visible) instances
            instances.gt_masks = BitMasks(torch.zeros((0, *image_shape), dtype=torch.uint8))
        instances = utils.filter_empty_instances(instances)
        # MaskDINO's prepare_targets expects gt_masks to be a plain (N, H, W) tensor.
        instances.gt_masks = instances.gt_masks.tensor
        dataset_dict["instances"] = instances

        return dataset_dict
