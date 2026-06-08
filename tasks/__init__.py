"""
 Copyright (c) 2022, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.captioning import CaptionTask
from lavis.tasks.image_text_pretrain import ImageTextPretrainTask
from lavis.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from lavis.tasks.retrieval import RetrievalTask
from lavis.tasks.vqa import VQATask, GQATask, AOKVQATask, DisCRNTask
from lavis.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from lavis.tasks.dialogue import DialogueTask
from lavis.tasks.text_to_image_generation import TextToImageGenerationTask
from .eca_q_captioning import ECAQCaptionTask
from .eca_q_captioning_lwf import ECAQLWFCaptionTask
from .eca_q_vqa import ECAQVQATask
from .eca_q_vqa_lwf import ECAQLWFVQATask
from .eca_llava_captioning import ECALlavaCaptionTask
from .eca_llava_vqa import ECALlavaVQATask
from .eca_llava_captioning_lwf import ECALlavaLWFCaptionTask
from .eca_llava_vqa_lwf import ECALlavaLWFVQATask
from .eca_internvl_captioning import ECAInternVLCaptionTask
from .eca_internvl_vqa import ECAInternVLVQATask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "ECAQCaptionTask",
    "ECAQLWFCaptionTask",
    "ECAQVQATask",
    "ECAQLWFVQATask",
    "ECALlavaCaptionTask",
    "ECALlavaVQATask",
    "ECALlavaLWFCaptionTask",
    "ECALlavaLWFVQATask",
    "ECAInternVLCaptionTask",
    "ECAInternVLVQATask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
    "TextToImageGenerationTask",
    "DisCRNTask"
]
