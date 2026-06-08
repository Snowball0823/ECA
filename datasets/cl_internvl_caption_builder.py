"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import os

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from .cl_textcaps_internvl_datasets import (
    CLTextCapsInternVLDataset,
    CLTextCapsInternVLEvalDataset,
)


@registry.register_builder("cl_textcaps_caption_internvl")
class CLTextCapsInternVLBuilder(BaseDatasetBuilder):
    train_dataset_cls = CLTextCapsInternVLDataset
    eval_dataset_cls = CLTextCapsInternVLEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cl_textcaps_cap_internvl.yaml",
    }

    @classmethod
    def default_config_path(cls, type="default"):
        return os.path.join(registry.get_path("project_root"), cls.DATASET_CONFIG_DICT[type])
