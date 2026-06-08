"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""InternVL backbone wrapper providing common utilities for continual alignment models."""

import os

import torch

from lavis.models.base_model import BaseModel

from .utils import load_eca_pretrained_model


class SoftpromptInternVLBase(BaseModel):
    """Base wrapper for loading and using an official InternVL model within LAVIS."""

    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(
        self,
        model_path=None,
        load_pretrained=True,
        use_flash_attn=False,
        pretrained=None,
        **loader_kwargs,
    ):
        super().__init__()

        # Supported loading scenario:
        #   * Full InternVL checkpoints (e.g. `OpenGVLab/InternVL2_5-1B`): set
        #     `model_path` to the local directory or HF-style path.
        # Extra loader options can be passed via `loader_kwargs`, e.g.
        #   device / torch_dtype / low_cpu_mem_usage / trust_remote_code /
        #   add_eos_token / model_max_length /
        #   template / select_layer / force_image_size /
        #   dynamic_image_size / use_thumbnail / ps_version /
        #   min_dynamic_patch / max_dynamic_patch / downsample_ratio /
        #   vision_config / llm_config / vision_config_overrides /
        #   llm_config_overrides / pad2square / normalize_type.

        if model_path is None and pretrained is not None:
            model_path = pretrained  # legacy parameter support

        if load_pretrained:
            if model_path is None:
                raise ValueError(
                    "Please provide `model_path` (or the deprecated `pretrained`) when load_pretrained=True."
                )

            load_kwargs = dict(loader_kwargs)
            load_kwargs.setdefault("model_path", model_path)
            load_kwargs.setdefault("use_flash_attn", use_flash_attn)

            tokenizer, internvl_model, image_processor, context_len = load_eca_pretrained_model(**load_kwargs)
        else:
            raise ValueError("InternVL backbone must be loaded from a pretrained checkpoint.")

        self.internvl_tokenizer = tokenizer
        self.internvl_model = internvl_model
        self.image_processor = image_processor
        self.context_len = context_len

    @property
    def vision_tower(self):
        return self.get_vision_encoder()

    @property
    def mm_projector(self):
        return self.get_mm_projector()

    @property
    def llm_model(self):
        return self.internvl_model.language_model

    @property
    def device(self):
        return next(self.internvl_model.parameters()).device

    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return torch.cuda.amp.autocast(enabled=False)

    def get_vision_encoder(self):
        return self.internvl_model.vision_model

    def get_mm_projector(self):
        return self.internvl_model.mlp1

    def _set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def freeze_vision_modules(self, disable_training=False, freeze_projector=True):
        self._set_requires_grad(self.vision_tower, False)
        if freeze_projector:
            self._set_requires_grad(self.mm_projector, False)
        if disable_training:
            self.vision_tower.eval()
            self.vision_tower.train = disabled_train
            if freeze_projector:
                self.mm_projector.eval()
                self.mm_projector.train = disabled_train

    def freeze_lm_head(self, disable_training=False):
        self._set_requires_grad(self.internvl_model.language_model.lm_head, False)
        if disable_training:
            self.internvl_model.language_model.lm_head.eval()
            self.internvl_model.language_model.lm_head.train = disabled_train

    def freeze_llm(self, disable_training=False):
        self._set_requires_grad(self.llm_model, False)
        if disable_training:
            self.llm_model.eval()
            self.llm_model.train = disabled_train

    def extract_vision_features(self, pixel_values):
        with self.maybe_autocast():
            if self.internvl_model.select_layer == -1:
                vit_embeds = self.vision_tower(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            else:
                vit_embeds = self.vision_tower(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[self.internvl_model.select_layer]

            vit_embeds = vit_embeds[:, 1:, :]
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.internvl_model.pixel_shuffle(
                vit_embeds, scale_factor=self.internvl_model.downsample_ratio
            )
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def project_vision_features(self, features):
        projector = self.mm_projector
        projector_param = next(projector.parameters())
        features = features.to(device=projector_param.device, dtype=projector_param.dtype)
        with self.maybe_autocast():
            return projector(features)

    def encode_image(self, image, use_projector=True):
        feats = self.extract_vision_features(image)
        if use_projector:
            feats = self.project_vision_features(feats)
        return feats

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
                    "lr_scale": lr_scale,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": lr_scale,
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        return list(parameter_group_vars.values())

    @classmethod
    def default_config_path(cls, model_type):
        if model_type not in cls.PRETRAINED_MODEL_CONFIG_DICT:
            raise KeyError(f"Unknown model type {model_type}")
        return cls.PRETRAINED_MODEL_CONFIG_DICT[model_type]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to keep train/eval mode fixed."""
    return self
