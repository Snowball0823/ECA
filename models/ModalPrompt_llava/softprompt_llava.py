"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
from __future__ import annotations

"""LLaVA backbone wrapper providing common utilities for continual alignment models."""

import os
from typing import Optional, Dict, Any

import torch

from lavis.common.registry import registry
from lavis.models.base_model import BaseModel

from .utils import load_eca_pretrained_model




class SoftpromptLlavaBase(BaseModel):
    """Base wrapper for loading and using an official LLaVA model within LAVIS."""

    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        model_name: Optional[str] = None,
        load_pretrained: bool = True,
        use_flash_attn: bool = False,
        pretrained: Optional[str] = None,
        **loader_kwargs,
    ):
        super().__init__()

        # Supported loading scenarios:
        #   * Full checkpoints (e.g. `liuhaotian/llava-v1.5-7b`): set `model_path` to the repo/path only.
        #   * Delta / LoRA checkpoints: provide `model_path` plus `model_base` (base LM repo).
        #   * Projector-only checkpoints: provide `model_path` and `model_base`; only the projector weights are
        #     loaded from `model_path`.
        # Extra loader options can be passed via `loader_kwargs`, e.g.
        #   device / device_map / torch_dtype / load_in_8bit / load_in_4bit /
        #   attn_implementation / low_cpu_mem_usage / mm_use_im_start_end /
        #   mm_use_im_patch_token / mm_hidden_size / mm_vision_select_layer /
        #   mm_vision_tower / tune_mm_mlp_adapter / use_mm_proj, etc.

        if model_path is None and pretrained is not None:
            model_path = pretrained  # legacy parameter support

        if load_pretrained:
            if model_path is None:
                raise ValueError(
                    "Please provide `model_path` (or the deprecated `pretrained`) when load_pretrained=True."
                )

            load_kwargs: Dict[str, Any] = dict(loader_kwargs)

            load_kwargs.setdefault("model_path", model_path)
            load_kwargs.setdefault("model_base", model_base)

            default_name = os.path.basename(model_path.rstrip("/")) or model_path
            load_kwargs.setdefault("model_name", model_name or default_name)
            load_kwargs.setdefault("device", load_kwargs.get("device", "cpu"))
            load_kwargs.setdefault("use_flash_attn", use_flash_attn)

            tokenizer, llava_model, image_processor, context_len = load_eca_pretrained_model(**load_kwargs)
        else:
            raise ValueError("LLaVA backbone must be loaded from a pretrained checkpoint.")

        self.llava_tokenizer = tokenizer
        self.llava_model = llava_model
        self.image_processor = image_processor
        self.context_len = context_len

    @property
    def vision_tower(self):
        return self.get_vision_encoder()

    @property
    def mm_projector(self):
        return self.get_mm_projector()

    @property
    def device(self):
        return next(self.llava_model.parameters()).device

    def maybe_autocast(self, dtype=torch.float16):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return torch.cuda.amp.autocast(enabled=False)

    # ------------------------------------------------------------------
    def get_vision_encoder(self):
        tower = self.llava_model.get_vision_tower()
        if hasattr(tower, "is_loaded") and not tower.is_loaded:
            tower.load_model()
        return tower

    def get_mm_projector(self):
        return self.llava_model.get_model().mm_projector

    def _set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def freeze_vision_modules(self, disable_training: bool = False, freeze_projector: bool = True):
        self._set_requires_grad(self.vision_tower, False)
        if freeze_projector:
            self._set_requires_grad(self.mm_projector, False)
        if disable_training:
            self.vision_tower.eval()
            self.vision_tower.train = disabled_train
            if freeze_projector:
                self.mm_projector.eval()
                self.mm_projector.train = disabled_train

    def freeze_lm_head(self, disable_training: bool = False):
        self._set_requires_grad(self.llava_model.lm_head, False)
        if disable_training:
            self.llava_model.lm_head.eval()
            self.llava_model.lm_head.train = disabled_train

    # ------------------------------------------------------------------
    def encode_image(self, image: torch.Tensor, use_projector: bool = True):
        with self.maybe_autocast():
            feats = self.vision_tower(image)
            if use_projector:
                feats = self.mm_projector(feats)
        return feats

    # ------------------------------------------------------------------
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": 1,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": 1,
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        return list(parameter_group_vars.values())

    # ------------------------------------------------------------------
    @classmethod
    def default_config_path(cls, model_type: str):
        if model_type not in cls.PRETRAINED_MODEL_CONFIG_DICT:
            raise KeyError(f"Unknown model type {model_type}")
        return cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
