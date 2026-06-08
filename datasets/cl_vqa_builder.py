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

from .cl_coco_vqa_dataset import CLCOCOVQADataset, CLCOCOVQAEvalDataset
from .cl_textvqa_datasets import CLTextVQADataset, CLTextVQAEvalDataset


@registry.register_builder("cl_coco_vqa")
class CLCOCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = CLCOCOVQADataset
    eval_dataset_cls = CLCOCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cl_coco_defaults_vqa.yaml",
    }

    @classmethod
    def default_config_path(cls, type="default"):
        return os.path.join(registry.get_path("project_root"), cls.DATASET_CONFIG_DICT[type])


@registry.register_builder("cl_textvqa_vqa")
class CLTextVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = CLTextVQADataset
    eval_dataset_cls = CLTextVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cl_textvqa_vqa.yaml",
    }
    @classmethod
    def default_config_path(cls, type="default"):
        return os.path.join(registry.get_path("project_root"), cls.DATASET_CONFIG_DICT[type])
