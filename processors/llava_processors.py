"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""
Utility processors tailored for LLaVA-style vision preprocessing.
"""

from transformers import CLIPImageProcessor
from omegaconf import OmegaConf

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("llava_clip_image_train")
@registry.register_processor("llava_clip_image_eval")
class LlavaCLIPImageProcessor(BaseProcessor):
    """
    Thin wrapper around HuggingFace CLIPImageProcessor so we can instantiate it
    via the LAVIS registry. By default it loads the projector-aligned processor
    shipped with LLaVA checkpoints.
    """

    def __init__(self, processor: CLIPImageProcessor):
        self.processor = processor

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        visual_encoder_name = cfg.get(
            "visual_encoder_name",
           "openai/clip-vit-large-patch14"
        )
        processor_kwargs = cfg.get("processor_kwargs", {})

        processor = CLIPImageProcessor.from_pretrained(
            visual_encoder_name, **processor_kwargs
        )

        image_size = cfg.get("image_size", None)
        if image_size is not None:
            image_size = int(image_size)
            processor.size = {"shortest_edge": image_size}
            processor.crop_size = {"height": image_size, "width": image_size}

        return cls(processor)

    def __call__(self, image):
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"][0]
        return pixel_values
