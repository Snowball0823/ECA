"""
 Copyright (c) 2022, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

from lavis.processors.base_processor import BaseProcessor

from lavis.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
    BlipQuestionProcessor,
)

from .blip_processors import (
    BlipPromptQuestionProcessor,
)

from lavis.processors.blip_diffusion_processors import (
    BlipDiffusionInputImageProcessor,
    BlipDiffusionTargetImageProcessor,
)
from lavis.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from lavis.processors.clip_processors import ClipImageTrainProcessor
from lavis.processors.audio_processors import BeatsAudioProcessor
from lavis.processors.ulip_processors import ULIPPCProcessor
from lavis.processors.instruction_text_processors import BlipInstructionProcessor

from lavis.common.registry import registry
from processors.blip_processors import BlipPromptQuestionProcessor
from processors.llava_processors import LlavaCLIPImageProcessor
from processors.internvl_processors import (
    InternVLImageEvalProcessor,
    InternVLImageTrainProcessor,
)

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "BlipInstructionProcessor",
    "BlipQuestionProcessor",
    "BlipPromptQuestionProcessor",
    # BLIP-Diffusion
    "BlipDiffusionInputImageProcessor",
    "BlipDiffusionTargetImageProcessor",
    # CLIP
    "ClipImageTrainProcessor",
    # LLaVA
    "LlavaCLIPImageProcessor",
    # InternVL
    "InternVLImageTrainProcessor",
    "InternVLImageEvalProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    # AUDIO
    "BeatsAudioProcessor",
    # 3D
    "ULIPPCProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
