"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import os

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import (
    BaseDatasetBuilder, MultiModalDatasetBuilder)
from lavis.datasets.datasets.coco_caption_datasets import NoCapsEvalDataset

from .cl_coco_caption_dataset import CLCaptionEvalDataset, CLCOCOCaptionDataset
from .cl_textcaps_datasets import CLTextCapsCapDataset, CLTextCapsCapEvalDataset


@registry.register_builder("cl_coco_caption")
class CLCOCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CLCOCOCaptionDataset
    eval_dataset_cls = CLCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cl_coco_defaults_cap.yaml",
    }

    @classmethod
    def default_config_path(cls, type="default"):
        return os.path.join(registry.get_path("project_root"), cls.DATASET_CONFIG_DICT[type])


@registry.register_builder("cl_textcaps_caption")
class CLTextCapsCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CLTextCapsCapDataset
    eval_dataset_cls = CLTextCapsCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cl_textcaps_cap.yaml",
    }
    @classmethod
    def default_config_path(cls, type="default"):
        return os.path.join(registry.get_path("project_root"), cls.DATASET_CONFIG_DICT[type])
