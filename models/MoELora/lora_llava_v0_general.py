"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
from __future__ import annotations

"""LLaVA continual alignment model with minimal ECA pipeline."""

import logging
from copy import deepcopy
import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adapters import LoRAConfig, ParBnConfig
from transformers import StoppingCriteriaList
from lavis.common.registry import registry
from collections import defaultdict

from ..custom_adapters import adapter_init
from ..custom_adapters.utils import (
    Fuse,
    Average,
    freeze_adapter,
    init_adapter,
    print_trainable_parameters,
    iter_adapter_named_parameters,
)
from .softprompt_llava import SoftpromptLlavaBase
from .utils import (
    clean_generation,
    compute_image_offsets,
    orthogonal_svd_init,
    tokenize_with_image_support,
)

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)
from llava.mm_utils import KeywordsStoppingCriteria


@registry.register_model("lora_llava_v0_general")
class LORALlavaV0General(SoftpromptLlavaBase):
    """Baseline LLaVA continual model with soft prompts and adapter/LoRA expansion."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "cl_caption_llava_v0": "configs/models/pa_caption_llava_v0.yaml",
        "cl_vqa_llava_v0": "configs/models/pa_vqa_llava_v0.yaml",
    }

    def __init__(
        self,
        freeze_vit: bool = True,
        train_projector: bool = False,
        soft_prompt_len: int = 16,
        mix_query: bool = True,
        kd_weight: float = 0.0,
        ortho: bool = False,
        ortho_weight: float = 0.1,
        alignment_layers: int = 4,
        mh_pa_r: float = 32.0,
        mh_pa_dropout: float = 0.0,
        mh_pa_scale: float = 4.0,
        ffn_pa_r: float = 1.0,
        ffn_pa_dropout: float = 0.0,
        ffn_pa_scale: float = 4.0,
        vision_lora_r: int = 16,
        vision_lora_alpha: int = 32,
        mm_projector_lr: Optional[float] = None,
        max_txt_len: Optional[int] = None,
        max_answer_len: int = 64,
        prompt: str = "",
        conversation_template: str = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions."
            "### Human: {}"
            "### Assistant:"
        ),
        kd_prompt: str = "",
        use_prompt_anchor: bool = False,
        load_pretrained: bool = True,
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        mm_use_im_patch_token: bool = False,
        mm_use_im_start_end: bool = False,
        **loader_kwargs,
    ):
        """
        Args:
            freeze_vit: Whether to freeze vision tower and projector.
            soft_prompt_len: Length of MoQ soft prompt tokens per task.
            mix_query: Enable mixture-of-query routing.
            kd_weight: Weight for dictionary replay loss.
            ortho: Enable orthogonality regularisation over MoQ keys/queries.
            ortho_weight: Factor for orthogonality penalty.
            alignment_layers: Number of lowest LLaMA layers kept trainable with adapters.
            mh_pa_r/ffn_pa_r: Reduction factors for attention / FFN parallel adapters.
            mh_pa_dropout/ffn_pa_dropout: Dropout applied inside adapters.
            mh_pa_scale/ffn_pa_scale: Scaling factors for adapter outputs.
            vision_lora_r / vision_lora_alpha: LoRA rank and alpha for vision tower.
            max_txt_len / max_answer_len: Optional token length caps during training.
            prompt: Task prompt prefix for captioning/vqa datasets.
            conversation_template: Conversation prompt template injected before answers.
            kd_prompt: Prompt used when extracting dictionary features.
            use_prompt_anchor: Use shared prompt anchor for MoQ.
            load_pretrained: Whether to load LLaVA backbone weights.
            model_path/model_base/model_name/pretrained: Passed to loader for checkpoint resolution.
            mm_use_im_patch_token/mm_use_im_start_end: Overrides for multimodal special tokens.
            loader_kwargs: Extra keyword arguments forwarded to
                `load_eca_pretrained_model`, supporting options such as
                `device`, `device_map`, `torch_dtype`, `load_in_8bit`, `load_in_4bit`,
                `quantization_config`, `attn_implementation`, `low_cpu_mem_usage`,
                and LLaVA-specific overrides like
                `mm_use_im_patch_token`, `mm_use_im_start_end`, `mm_hidden_size`,
                `mm_vision_select_layer`, `mm_vision_tower`,
                `tune_mm_mlp_adapter`, `use_mm_proj`.
        """
        super().__init__(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            load_pretrained=load_pretrained,
            pretrained=pretrained,
            mm_use_im_patch_token=mm_use_im_patch_token,
            mm_use_im_start_end=mm_use_im_start_end,
            **loader_kwargs,
        )

        self.train_projector = train_projector
        self.mm_projector_lr = mm_projector_lr

        if freeze_vit:
            self.freeze_vision_modules(disable_training=True, freeze_projector=not train_projector)

        if self.train_projector:
            self.mm_projector = self.mm_projector.to(dtype=torch.float32)

        self.mix_query = mix_query
        self.kd_weight = kd_weight
        self.ortho = ortho
        self.ortho_weight = ortho_weight
        self.max_txt_len = max_txt_len
        self.max_answer_len = max_answer_len
        self.prompt = prompt.strip()
        self.conversation_template = conversation_template
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_use_im_start_end = mm_use_im_start_end
        self.kd_prompt = kd_prompt or ""

        self.moq_prompt_len = soft_prompt_len
        self.moq_hidden_dim = self.llava_model.config.hidden_size
        self.use_prompt_anchor = use_prompt_anchor
        if use_prompt_anchor:
            anchor = torch.zeros(1, soft_prompt_len, self.moq_hidden_dim)
            nn.init.xavier_uniform_(anchor)
            self.prompt_anchor = nn.Parameter(anchor, requires_grad=False)
        else:
            zeros = torch.zeros(1, soft_prompt_len, self.moq_hidden_dim)
            self.register_buffer("prompt_anchor", zeros, persistent=False)
        self.current_keys: Optional[nn.Parameter] = None
        self.current_queries: Optional[nn.Parameter] = None
        self.keys_history: Optional[Tensor] = None
        self.queries_history: Optional[Tensor] = None

        # Adapter bookkeeping ------------------------------------------------
        self.parallel_adapters_dict = defaultdict(list)
        self.adapter_init = False
        self.leave_out = list(range(alignment_layers, self.llava_model.config.num_hidden_layers))
        self.attn_pa_config = ParBnConfig(
            mh_adapter=True,
            output_adapter=False,
            reduction_factor=mh_pa_r,
            dropout=mh_pa_dropout,
            scaling=mh_pa_scale,
            non_linearity="linear",
            leave_out=self.leave_out,
        )
        # self.ffn_pa_config = ParBnConfig(
        #     reduction_factor=ffn_pa_r,
        #     dropout=ffn_pa_dropout,
        #     scaling=ffn_pa_scale,
        #     non_linearity="linear",
        #     leave_out=leave_out,
        # )
        # for MoELora configs
        self.ffn_pa_config = ParBnConfig(
            reduction_factor=ffn_pa_r,
            dropout=ffn_pa_dropout,
            scaling=ffn_pa_scale,
            non_linearity="linear",
            leave_out=self.leave_out,
        )
        self.attn_pa_prefix = "llm_attn_pa_"
        self.ffn_pa_prefix = "llm_ffn_pa_"

        self.visual_lora_init = False
        self.lora_dict: dict[int, List[str]] = {}
        self.visual_lora_config = LoRAConfig(
            r=vision_lora_r,
            alpha=vision_lora_alpha,
            intermediate_lora=True,
            output_lora=True,
        )
        self.lora_prefix = "vision_lora_"

        # Mimic BLIP-2 interface for compatibility with tasks
        self.teacher_adapters_dict: dict[int, List[str]] = {}
        self.use_teacher: bool = False
        self.projector_snapshot = None

        # MoELora
        ## num of FFN Lora => refer from CoIN
        self.lora_num = 16
        self.moe_lora_prefix = 'moe_ffn_lora'


    # ------------------------------------------------------------------
    # Properties
    @property
    def adapters(self):
        names: List[str] = []
        for group in self.parallel_adapters_dict.values():
            names.extend(group)
        return names

    @property
    def moq_device(self):
        return self.mm_projector.weight.device

    @property
    def loras(self):
        names: List[str] = []
        for group in self.lora_dict.values():
            names.extend(group)
        return names

    @property
    def moq_num(self):
        total = 0
        if self.keys_history is not None:
            total += self.keys_history.size(0)
        if self.current_keys is not None:
            total += self.current_keys.size(0)
        return total

    @property
    def moq_old_kv(self):
        payload = {}
        if self.keys_history is not None:
            payload["keys_history"] = self.keys_history.detach().clone().cpu()
        if self.queries_history is not None:
            payload["queries_history"] = self.queries_history.detach().clone().cpu()
        if self.current_keys is not None:
            payload["current_keys"] = self.current_keys.detach().clone().cpu()
        if self.current_queries is not None:
            payload["current_queries"] = self.current_queries.detach().clone().cpu()
        return payload

    @property
    def visual_tower_freezed(self):
        params = list(self.vision_tower.parameters())
        projector = getattr(self, "mm_projector", None)
        if projector is not None:
            params.extend(projector.parameters())
        return not any(p.requires_grad for p in params)

    @property
    def current_adapter_names(self):
        if not self.parallel_adapters_dict:
            return []
        return self.parallel_adapters_dict[max(self.parallel_adapters_dict.keys())]

    @property
    def current_adapter_index(self):
        if not self.parallel_adapters_dict:
            return -1
        return max(self.parallel_adapters_dict.keys())

    @property
    def visual_feature_dim(self):
        if self.train_projector:
            vision_cfg = getattr(self.vision_tower, "config", None)
            vision_dim = getattr(vision_cfg, "hidden_size", None)
            return vision_dim
        return self.llava_model.config.hidden_size

    @property
    def trainable_adapter_parameters(self):
        llama_core = self.llava_model.get_model()
        adapter_names = getattr(self, "adapters", [])
        if not adapter_names:
            return
        for name, param in iter_adapter_named_parameters(llama_core, adapter_names):
            if param.requires_grad:
                yield name, param

    # ------------------------------------------------------------------
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """Group trainable params into projector / adapter / MoQ buckets."""
        grouped_named_params = {
            "projector": [],
            "adapter": [],
            "vision_lora": [],
            "moq": [],
            "other": [],
        }

        llm_adapter_prefixes = (self.attn_pa_prefix, self.ffn_pa_prefix)
        vision_lora_prefix = self.lora_prefix

        def is_no_decay(name: str):
            return name.endswith(".bias") or "norm" in name.lower()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "mm_projector" in name:
                grouped_named_params["projector"].append((name, param))
            elif any(prefix in name for prefix in llm_adapter_prefixes):
                grouped_named_params["adapter"].append((name, param))
            elif vision_lora_prefix and vision_lora_prefix in name:
                grouped_named_params["vision_lora"].append((name, param))
            elif name in {"current_keys", "current_queries"}:
                grouped_named_params["moq"].append((name, param))
            else:
                grouped_named_params["other"].append((name, param))

        param_groups = []
        proj_lr_scale = self.mm_projector_lr if self.mm_projector_lr is not None else lr_scale

        projector_decay = [p for n, p in grouped_named_params["projector"] if not is_no_decay(n)]
        projector_no_decay = [p for n, p in grouped_named_params["projector"] if is_no_decay(n)]
        adapter_params = [p for _, p in grouped_named_params["adapter"]]
        vision_lora_params = [p for _, p in grouped_named_params["vision_lora"]]
        moq_params = [p for _, p in grouped_named_params["moq"]]
        other_decay = [p for n, p in grouped_named_params["other"] if not is_no_decay(n)]
        other_no_decay = [p for n, p in grouped_named_params["other"] if is_no_decay(n)]

        if projector_decay:
            param_groups.append(
                {
                    "weight_decay": weight_decay,
                    "lr_scale": proj_lr_scale,
                    "params": projector_decay,
                }
            )
        if projector_no_decay:
            param_groups.append(
                {
                    "weight_decay": 0.0,
                    "lr_scale": proj_lr_scale,
                    "params": projector_no_decay,
                }
            )
        if adapter_params:
            param_groups.append(
                {
                    "weight_decay": 0.0,
                    "lr_scale": 1,
                    "params": adapter_params,
                }
            )
        if vision_lora_params:
            param_groups.append(
                {
                    "weight_decay": 0.0,
                    "lr_scale": 1,
                    "params": vision_lora_params,
                }
            )
        if moq_params:
            param_groups.append(
                {
                    "weight_decay": 0.0,
                    "lr_scale": 1,
                    "params": moq_params,
                }
            )
        if other_decay:
            warning_msg = (
                "Unexpected trainable params found outside 'mm projector/LlaMa adapters/CLIP LoRA/MoQ'. "
                "Backbone appears to be updating."
            )
            logging.warning(warning_msg)
            print(f"WARNING: {warning_msg}")
            param_groups.append(
                {
                    "weight_decay": weight_decay,
                    "lr_scale": 1,
                    "params": other_decay,
                }
            )
        if other_no_decay:
            param_groups.append(
                {
                    "weight_decay": 0.0,
                    "lr_scale": 1,
                    "params": other_no_decay,
                }
            )

        def summarize(tag, entries):
            if not entries:
                return
            total_params = sum(param.numel() for _, param in entries)
            names = [name for name, _ in entries[:3]]
            if len(entries) > 3:
                names.append("...")
            layer_ids = set()
            for name, _ in entries:
                if ".layers." in name:
                    try:
                        segment = name.split(".layers.", 1)[1]
                        layer_idx = int(segment.split(".", 1)[0])
                        layer_ids.add(layer_idx)
                    except (ValueError, IndexError):
                        continue
            layer_summary = ""
            if layer_ids:
                sorted_ids = sorted(layer_ids)
                layer_summary = f" | layers: {sorted_ids[0]}-{sorted_ids[-1]} (n={len(sorted_ids)})"
            logging.info(
                "[Optimizer] %s: tensors=%d, params=%.2fM%s | sample=%s",
                tag,
                len(entries),
                total_params / 1e6,
                layer_summary,
                ", ".join(names),
            )

        summarize("projector", grouped_named_params["projector"])
        summarize("adapter", grouped_named_params["adapter"])
        summarize("vision_lora", grouped_named_params["vision_lora"])
        summarize("moq", grouped_named_params["moq"])
        summarize("other", grouped_named_params["other"])

        return param_groups

    # ------------------------------------------------------------------
    # MoQ helpers
    def _prepare_task_queries(self, svd_init: bool = True):
        if self.current_keys is not None and self.current_queries is not None:
            self._stash_current_queries()
        new_keys = torch.zeros(1, self.moq_hidden_dim, device=self.moq_device)
        new_queries = torch.zeros(1, self.moq_prompt_len, self.moq_hidden_dim, device=self.moq_device)
        if self.keys_history is not None and svd_init:
            svd_row = orthogonal_svd_init(self.keys_history.float())
            new_keys.data.copy_(svd_row.to(device=self.moq_device))
        self.current_keys = nn.Parameter(new_keys)
        self.current_queries = nn.Parameter(new_queries)

    def _stash_current_queries(self):
        keys_detached = self.current_keys.detach().clone()
        queries_detached = self.current_queries.detach().clone()
        self.keys_history = (
            keys_detached if self.keys_history is None else torch.cat([self.keys_history, keys_detached], dim=0)
        )
        self.queries_history = (
            queries_detached if self.queries_history is None else torch.cat([self.queries_history, queries_detached], dim=0)
        )
        self.current_keys = None
        self.current_queries = None

    def mixture_of_query(self, image_embeds: Tensor, old_only: bool = False):
        if image_embeds.device != self.moq_device:
            raise RuntimeError(
                "Mixture-of-Query expects image embeddings to share dtype/device with MoQ buffers."
            )
        if not self.mix_query:
            if self.use_prompt_anchor:
                base = self.prompt_anchor
                return base.expand(image_embeds.size(0), -1, -1)
            return None
        key_sources = self.keys_history if self.keys_history is not None else self.current_keys[:0]

        query_sources = self.queries_history if self.queries_history is not None else self.current_queries[:0]
        if not old_only:
            key_sources = torch.cat([key_sources, self.current_keys], dim=0)
            query_sources = torch.cat([query_sources, self.current_queries], dim=0)
        if key_sources.numel() == 0 or query_sources.numel() == 0:
            raise RuntimeError(
                "Mixture-of-Query requested without available key/query history "
                f"(old_only={old_only}). Ensure task adapters are initialized before calling."
            )
        img_feat = image_embeds.mean(dim=1).to(self.current_keys.dtype)
        img_norm = F.normalize(img_feat, p=2, dim=-1)
        key_norm = F.normalize(key_sources, p=2, dim=-1)
        scaled_logits = torch.einsum('bd,nd->bn', img_norm, key_norm)
        attn = F.softmax(scaled_logits, dim=1)
        delta_prompt = torch.einsum("bn,nqd->bqd", attn, query_sources)
        base = self.prompt_anchor.expand(image_embeds.size(0), -1, -1)
        return base + delta_prompt

    # ------------------------------------------------------------------
    def _prepare_inputs_with_feature_dictionary(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_template: torch.Tensor,
        dummy_images: torch.Tensor,
        feature_dictionary: torch.Tensor,
        patch_len: int,
    ):
        """Construct multimodal embeddings and stitch precomputed vision tokens into the sequence."""
        (
            _,
            position_ids,
            extended_attention_mask,
            _,
            inputs_embeds,
            labels,
        ) = self.llava_model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            attention_mask,
            None,
            label_template,
            dummy_images,
        )

        first_positions, min_offset = compute_image_offsets(input_ids)
        text_offset = 0 if min_offset is None else min_offset
        feature_dictionary = feature_dictionary.to(dtype=inputs_embeds.dtype)
        seq_len = inputs_embeds.size(1)
        starts = first_positions.long()
        start_ref = starts[0].item()
        if (starts != start_ref).any():
            raise RuntimeError("Dictionary extraction expects identical image offsets across the batch.")
        if start_ref >= seq_len:
            raise RuntimeError("Image token position exceeds sequence length during dictionary extraction.")
        end_ref = start_ref + patch_len
        if end_ref > seq_len:
            raise RuntimeError("Patch length exceeds available sequence space during dictionary extraction.")
        inputs_embeds[:, start_ref:end_ref, :] = feature_dictionary
        return inputs_embeds, position_ids, extended_attention_mask, labels, text_offset

    def extract_query_dictionary(
        self,
        feature_dictionary: Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
        num_patch: Optional[int] = None,
        extract_type: str = "teacher",
        answer_texts: Optional[List[str]] = None,
    ):
        if extract_type not in {"teacher", "student"}:
            raise ValueError(f"Unsupported extract_type {extract_type}. Expected 'teacher' or 'student'.")
        if not self.max_answer_len or self.max_answer_len <= 0:
            raise ValueError("max_answer_len must be set to a positive value for KD extraction.")

        if feature_dictionary.dim() != 3:
            raise ValueError("feature_dictionary must have shape [B, num_patch, hidden_dim].")

        if self.train_projector:
            projector = self.projector_snapshot if extract_type == "teacher" else self.mm_projector
            if projector is None:
                raise RuntimeError(
                    "Projector snapshot is not initialized. Ensure before_training has been called."
                )
            proj_ctx = torch.no_grad() if extract_type == "teacher" else contextlib.nullcontext()
            with proj_ctx:
                feature_dictionary = projector(
                    feature_dictionary.to(device=projector.weight.device, dtype=projector.weight.dtype)
                )
        patch_len = feature_dictionary.size(1)
        if num_patch is not None and patch_len != num_patch:
            raise ValueError(
                f"feature_dictionary patch length {patch_len} does not match expected {num_patch}."
            )
        batch_size = feature_dictionary.size(0)
        device = feature_dictionary.device
        if image_hw is not None:
            img_h, img_w = image_hw
        else:
            vision_cfg = getattr(self.vision_tower, "config", None)
            size = getattr(vision_cfg, "image_size", None)
            if size is None:
                size = getattr(self.vision_tower, "image_size", None)
            if size is None:
                raise RuntimeError(
                    "Vision tower does not expose an image_size. "
                    "Please pass image_hw explicitly when calling extract_query_dictionary."
            )
            if isinstance(size, (tuple, list)):
                img_h, img_w = size
            else:
                img_h = img_w = int(size)
        dummy_images = torch.zeros(batch_size, 3, img_h, img_w, device=device, dtype=self.vision_tower.dtype)

        base_prompt = (self.kd_prompt or "").strip()
        if self.mm_use_im_start_end:
            image_tokens = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
        else:
            image_tokens = DEFAULT_IMAGE_TOKEN
        human_lines = image_tokens + "\n"
        if base_prompt:
            human_lines += base_prompt
        human_message = human_lines.strip()
        kd_inputs = [self.conversation_template.format(human_message)] * batch_size
        prefix_ids, prefix_mask, _ = tokenize_with_image_support(
            self.llava_tokenizer,
            kd_inputs,
            self.max_txt_len,
            allow_image_token=True,
        )
        prefix_ids = prefix_ids.to(device)
        prefix_mask = prefix_mask.to(device)

        if answer_texts is None:
            if extract_type != "teacher":
                raise ValueError("Student KD requires teacher answer_texts for alignment.")
            with torch.no_grad():
                prefix_labels = torch.full_like(prefix_ids, IGNORE_INDEX)
                (
                    prefix_inputs_embeds,
                    prefix_position_ids,
                    prefix_attention_mask,
                    _,
                    prefix_text_offset,
                ) = self._prepare_inputs_with_feature_dictionary(
                    prefix_ids,
                    prefix_mask,
                    prefix_labels,
                    dummy_images,
                    feature_dictionary,
                    patch_len,
                )
                prompt_tokens_gen = self.mixture_of_query(feature_dictionary, old_only=True)
                if prompt_tokens_gen is None:
                    soft_prompt_gen = None
                else:
                    gen_offset = prefix_text_offset + patch_len + (1 if self.mm_use_im_start_end else 0)
                    soft_prompt_gen = (prompt_tokens_gen.detach(), gen_offset)

                # Greedily decode teacher answers using the stitched embeddings.
                llama_core = self.llava_model.get_model()
                prev_model_mode = self.llava_model.training
                prev_core_mode = llama_core.training
                prev_active = getattr(llama_core, "active_adapters", None)

                self.llava_model.eval()
                llama_core.eval()

                history = self.adapters
                if self.current_adapter_names:
                    history = history[:-len(self.current_adapter_names)]
                attn_history = history[0::2]
                ffn_history = history[1::2]

                if self.use_teacher:
                    teacher_names = self.teacher_adapters_dict.get("teacher")
                    if not teacher_names:
                        raise RuntimeError("Teacher adapters not initialized for current task.")
                    attn_names = attn_history + [teacher_names[0]] if attn_history else [teacher_names[0]]
                    ffn_names = ffn_history + [teacher_names[1]] if ffn_history else [teacher_names[1]]
                else:
                    if not attn_history or not ffn_history:
                        raise RuntimeError("No historical adapters available to serve as KD teacher.")
                    attn_names = attn_history
                    ffn_names = ffn_history

                llama_core.set_active_adapters([
                    Average(*attn_names, weights=[1] * len(attn_names)),
                    Average(*ffn_names, weights=[1] * len(ffn_names)),
                ])
                stop_words = ["###"]
                stopping_criteria = StoppingCriteriaList(
                    [KeywordsStoppingCriteria(stop_words, self.llava_tokenizer, prefix_ids)]
                )
                sequences = self.llava_model.generate(
                    inputs_embeds=prefix_inputs_embeds,
                    attention_mask=prefix_attention_mask,
                    position_ids=prefix_position_ids,
                    soft_prompt=soft_prompt_gen,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_answer_len,
                    stopping_criteria=stopping_criteria,
                )
                # prefix_len = inputs_embeds.size(1)
                # if sequences.size(1) > prefix_len:
                #     new_tokens = sequences[:, prefix_len:]
                # else:
                #     new_tokens = sequences[:, 0:0]
                decoded = self.llava_tokenizer.batch_decode(sequences, skip_special_tokens=True)
                answer_texts = [clean_generation(text, "") for text in decoded]

        if len(answer_texts) != batch_size:
            raise ValueError("answer_texts must match batch size during KD extraction.")

        def _append_stop_token(text: str) -> str:
            text = (text or "").rstrip()
            if not text:
                return text
            if text.endswith("###"):
                return text
            return f"{text}\n###"

        answer_texts = [_append_stop_token(text) for text in answer_texts]

        answer_ids, answer_mask, answer_lengths = tokenize_with_image_support(
            self.llava_tokenizer,
            answer_texts,
            self.max_answer_len,
            allow_image_token=False,
        )
        answer_ids = answer_ids.to(device)
        answer_mask = answer_mask.to(device)
        answer_lengths = answer_lengths.to(device)

        kd_input_ids = torch.cat([prefix_ids, answer_ids], dim=1)
        kd_attention_mask = torch.cat([prefix_mask, answer_mask], dim=1)
        label_template = torch.full_like(kd_input_ids, IGNORE_INDEX)

        (
            inputs_embeds,
            position_ids,
            attention_mask,
            labels,
            text_offset,
        ) = self._prepare_inputs_with_feature_dictionary(
            kd_input_ids,
            kd_attention_mask,
            label_template,
            dummy_images,
            feature_dictionary,
            patch_len,
        )

        if extract_type == "teacher":            
            with torch.no_grad():
                # if prompt_tokens is None:
                #     soft_prompt = None
                # else:
                #     offset = text_offset + patch_len + (1 if self.mm_use_im_start_end else 0)
                #     soft_prompt = (prompt_tokens, offset)

                with self.maybe_autocast():
                    # if later we use MES for hiddent_state, then set output_hidden_states=True
                    outputs = self.llava_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        soft_prompt=soft_prompt_gen,
                        labels=labels,
                        return_dict=True,
                        output_hidden_states=False,
                    )
            logits = outputs.logits.detach()
            # hidden_states = [state.detach() for state in outputs.hidden_states] if outputs.hidden_states else None
            answer_lengths = answer_lengths.detach()
            assert prev_active is not None, "LLaVA active adapters should be set."
            llama_core.active_adapters = prev_active
            if prev_core_mode:
                llama_core.train()
            if prev_model_mode:
                self.llava_model.train()
        else:
            prompt_tokens = self.mixture_of_query(feature_dictionary, old_only=False)
            soft_prompt = None
            if prompt_tokens is not None:
                offset = text_offset + patch_len + (1 if self.mm_use_im_start_end else 0)
                soft_prompt = (prompt_tokens, offset)
            with self.maybe_autocast():
                # if later we use MES for hiddent_state, then set output_hidden_states=True
                outputs = self.llava_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    soft_prompt=soft_prompt,
                    labels=labels,
                    return_dict=True,
                    output_hidden_states=False,
                )
            logits = outputs.logits
            # hidden_states = outputs.hidden_states

        distillable = torch.clamp(answer_lengths - 1, min=0)
        kd_mask = torch.zeros(logits.shape[:2], dtype=torch.bool, device=device)
        if answer_mask.size(1) > 0:
            token_idx = torch.arange(answer_mask.size(1), device=device).unsqueeze(0)
            suffix_mask = token_idx < distillable.unsqueeze(1)
            kd_mask[:, -answer_mask.size(1):] = suffix_mask

        return {
            "logits": logits,
            # "hidden_states": hidden_states,
            "kd_mask": kd_mask,
            "answer_text": answer_texts,
            "answer_lengths": answer_lengths,
        }
    
    @torch.no_grad()
    def extract_visual_feature(self, samples):
        embeds = self.encode_image(samples["image"], use_projector=not self.train_projector)
        return {"feature": embeds}

    # ------------------------------------------------------------------
    # Adapter / LoRA expansion
    def before_training(self, expand_adapters: bool = True, lora_visual: bool = False, **kwargs):
        super().before_training(**kwargs)

        # if expand_adapters:
        #     self.expand_alignment_adapter()
        #     self.use_teacher = False
        # else:
        #     self.use_teacher = self.current_adapter_index >= 0
        #     if self.use_teacher:
        #         self.create_teacher_snapshot()

        if len(self.adapters) == 0:
            self.init_moe_lora()

        if self.train_projector:
            projector = deepcopy(self.mm_projector)
            projector.to(device=self.mm_projector.weight.device, dtype=self.mm_projector.weight.dtype)
            projector.eval()
            for param in projector.parameters():
                param.requires_grad = False
            self.projector_snapshot = projector
        else:
            self.projector_snapshot = None

        if lora_visual:
            if self.visual_tower_freezed:
                raise AssertionError(
                    "Vision tower is frozen; set freeze_vit=False before expanding LoRA."
                )
            self.expand_visual_lora()
        elif self.lora_dict and not self.visual_tower_freezed:
            self.freeze_vision_modules(disable_training=True, freeze_projector=not self.train_projector)

        if self.mix_query:
            self._prepare_task_queries(svd_init=self.keys_history is not None)

        if not hasattr(self, "unchange_keys"):
            self.unchange_keys = []
        if not self.unchange_keys:
            # Snapshot parameters we never plan to tune to ease future filters.
            self.unchange_keys = [
                name
                for name, q in self.named_parameters() if not q.requires_grad
            ]


    def init_moe_lora(self):
        llama_core = self.llava_model.get_model()
        if not self.adapter_init:
            stored_vision = getattr(llama_core, "vision_tower", None)
            stored_projector = getattr(llama_core, "mm_projector", None)
            try:
                if stored_vision is not None:
                    llama_core.vision_tower = None
                if stored_projector is not None:
                    llama_core.mm_projector = None
                adapter_init(llama_core, use_customize=True)
                llama_core.freeze_model()
                self.freeze_lm_head(disable_training=True)
            finally:
                if stored_vision is not None:
                    llama_core.vision_tower = stored_vision
                    llama_core.mm_projector = stored_projector
            self.adapter_init = True
        for i in range(self.lora_num):
            ffn_pa_name = self.ffn_pa_prefix+str(i)
            # add adapter name in dict
            self.parallel_adapters_dict[self.moe_lora_prefix] += [ffn_pa_name]
            llama_core.add_adapter(ffn_pa_name, config=self.ffn_pa_config)
        moe_lora_names = self.parallel_adapters_dict[self.moe_lora_prefix]
        llama_core.add_adapter_fusion(moe_lora_names, 'linear')
        # self.parallel_adapters_dict[self.moe_lora_prefix] += [','.join(moe_lora_names)]
        llama_core.active_adapters = [Fuse(*moe_lora_names)]
        freeze_adapter(llama_core,
                       self.parallel_adapters_dict[self.moe_lora_prefix], freeze=False)
        print_trainable_parameters(llama_core)
        adapter_summary_fn = getattr(llama_core, "adapter_summary", None)
        if callable(adapter_summary_fn):
            logging.info("\n%s", adapter_summary_fn())
        logging.info("Enabled LLaMA MoELora (ffn): %s", moe_lora_names)
        self._freeze_unused_fusion(self.leave_out)

    def expand_alignment_adapter(self):
        llama_core = self.llava_model.get_model()
        if not self.adapter_init:
            stored_vision = getattr(llama_core, "vision_tower", None)
            stored_projector = getattr(llama_core, "mm_projector", None)
            try:
                if stored_vision is not None:
                    llama_core.vision_tower = None
                if stored_projector is not None:
                    llama_core.mm_projector = None
                adapter_init(llama_core, use_customize=True)
                llama_core.freeze_model()
                self.freeze_lm_head(disable_training=True)
            finally:
                if stored_vision is not None:
                    llama_core.vision_tower = stored_vision
                    llama_core.mm_projector = stored_projector
            self.adapter_init = True
        idx = len(self.parallel_adapters_dict)
        if idx > 0:
            freeze_adapter(llama_core, self.adapters)
        attn_name = f"{self.attn_pa_prefix}{idx}"
        ffn_name = f"{self.ffn_pa_prefix}{idx}"
        self.parallel_adapters_dict[idx] = [attn_name, ffn_name]
        llama_core.add_adapter(attn_name, config=self.attn_pa_config)
        llama_core.add_adapter(ffn_name, config=self.ffn_pa_config)
        attn_names = [names[0] for names in self.parallel_adapters_dict.values()]
        ffn_names = [names[1] for names in self.parallel_adapters_dict.values()]
        llama_core.active_adapters = [
            Average(*attn_names, weights=[1] * len(attn_names)),
            Average(*ffn_names, weights=[1] * len(ffn_names)),
        ]
        if idx > 0:
            prev_names = [self.parallel_adapters_dict[i] for i in range(idx)]
            init_adapter(
                llama_core,
                [attn_name, ffn_name],
                [list(group) for group in zip(*prev_names)],
            )
        freeze_adapter(llama_core, [attn_name, ffn_name], freeze=False)
        print_trainable_parameters(llama_core)
        adapter_summary_fn = getattr(llama_core, "adapter_summary", None)
        if callable(adapter_summary_fn):
            logging.info("\n%s", adapter_summary_fn())
        logging.info("Enabled LLaMA adapters (attn): %s", attn_names)
        logging.info("Enabled LLaMA adapters (ffn): %s", ffn_names)

    def _freeze_unused_fusion(self, leave_out: List[int]):
        """
        Freeze adapter fusion layers in blocks listed in leave_out.
        fusion modules exist even when LoRA modules are skipped, so we
        manually disable their gradients to avoid counting them as trainable.
        """
        if not leave_out:
            return
        llama_core = self.llava_model.get_model()
        for layer_idx in leave_out:
            if layer_idx >= len(llama_core.layers):
                continue
            layer = llama_core.layers[layer_idx]
            fusion = getattr(layer.output_adapters, "adapter_fusion_layer", None)
            if fusion is None:
                continue
            for param in fusion.parameters():
                param.requires_grad = False
            setattr(layer.output_adapters, "adapter_fusion_layer", nn.ModuleDict())
        logging.info("Fused adapters frozen on layers: %s", leave_out)

    def expand_visual_lora(self):
        wrapper = self.vision_tower
        vision_core = getattr(wrapper, "vision_tower", wrapper)

        if not self.visual_lora_init:
            adapter_init(vision_core, use_customize=False)
            vision_core.freeze_model()
            self.visual_lora_init = True
        idx = len(self.lora_dict)
        if idx > 0:
            freeze_adapter(vision_core, self.loras)
        name = f"{self.lora_prefix}{idx}"
        self.lora_dict[idx] = [name]
        vision_core.add_adapter(name, config=self.visual_lora_config)
        vision_core.set_active_adapters(self.loras)
        freeze_adapter(vision_core, self.lora_dict[idx], freeze=False)
        print_trainable_parameters(wrapper)
        logging.info("Activated vision adapters: %s", self.loras)

    # ------------------------------------------------------------------
    # Forward & loss
    def _build_sequences(self, samples, batch_size):
        outputs = [
            (output or "").rstrip("\n") + "\n"
            for output in samples.get("text_output", [""] * batch_size)
        ]
        text_inputs = samples.get("text_input", [""] * batch_size)
        ocr_inputs = samples.get("ocr_input", [""] * batch_size)
        prefixes = []
        answers = []
        full_sequences = []
        prompt = self.prompt
        prompt_len = len(prompt)
        for question, ocr, answer in zip(text_inputs, ocr_inputs, outputs):
            question = (question or "").strip()
            ocr = (ocr or "").strip()
            answer_body = answer.rstrip("\n")
            if self.mm_use_im_start_end:
                image_tokens = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
            else:
                image_tokens = DEFAULT_IMAGE_TOKEN
            human_lines = [image_tokens+'\n']
            if ocr:
                human_lines.append(ocr)
            if question:
                human_lines.append(question)
            elif prompt_len > 0:
                human_lines.append(prompt.strip())
                if answer_body.startswith(prompt):
                    answer_body = answer_body[prompt_len:].lstrip()
            else:
                raise ValueError("Either question or prompt must be provided.")
            if answer_body:
                stop_token = "###"
                if not answer_body.rstrip().endswith(stop_token):
                    answer_body = f"{answer_body.rstrip()}\n{stop_token}".rstrip()
            human_message = " ".join(line for line in human_lines if line)
            prefix = self.conversation_template.format(human_message)
            prefixes.append(prefix)
            answers.append(answer_body)
            full_sequences.append(f"{prefix}{answer_body}")
        return prefixes, answers, full_sequences

    # ------------------------------------------------------------------
    def forward(self, samples):
        image: Tensor = samples["image"]
        image_embeds = self.encode_image(image)
        prompt_tokens = self.mixture_of_query(image_embeds)

        batch_size = image.size(0)
        prefixes, answers, _ = self._build_sequences(samples, batch_size)

        device = image_embeds.device
        prefix_ids, prefix_mask, prefix_lengths = tokenize_with_image_support(
            self.llava_tokenizer,
            prefixes,
            self.max_txt_len,
            allow_image_token=True,
        )
        answer_ids, answer_mask, _ = tokenize_with_image_support(
            self.llava_tokenizer,
            answers,
            self.max_answer_len,
            allow_image_token=False,
        )

        _, min_offset = compute_image_offsets(prefix_ids)
        text_offset = 0 if min_offset is None else min_offset
        if prompt_tokens is None:
            soft_prompt = None
        else:
            offset = text_offset + image_embeds.size(1)
            if self.mm_use_im_start_end:
                offset += 1
            soft_prompt = (prompt_tokens, offset)

        if answer_ids.size(1) > 0:
            full_input_ids = torch.cat([prefix_ids, answer_ids], dim=1)
            full_attention_mask = torch.cat([prefix_mask, answer_mask], dim=1)
        else:
            full_input_ids = prefix_ids
            full_attention_mask = prefix_mask

        full_input_ids = full_input_ids.to(device)
        full_attention_mask = full_attention_mask.to(device)
        prefix_lengths = prefix_lengths.to(device)

        labels = full_input_ids.clone()
        labels[full_attention_mask == 0] = IGNORE_INDEX
        for idx, pre_len in enumerate(prefix_lengths.tolist()):
            if pre_len > 0:
                labels[idx, :pre_len] = IGNORE_INDEX

        with self.maybe_autocast():
            outputs = self.llava_model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                labels=labels,
                images=image,
                soft_prompt=soft_prompt,
                return_dict=True,
            )

        loss = outputs.loss
        kd_loss = torch.tensor(0.0, device=loss.device)
        ortho_loss = torch.tensor(0.0, device=loss.device)
        key_task_loss = torch.tensor(0.0, device=loss.device)

        feature_dict = samples.get("feature")
        if feature_dict is not None:
            image_hw = tuple(image.shape[-2:]) if image is not None else None
            num_patch = image_embeds.size(1)
            with torch.no_grad():
                teacher_out = self.extract_query_dictionary(
                    feature_dict,
                    image_hw=image_hw,
                    num_patch=num_patch,
                    extract_type="teacher",
                )
            student_out = self.extract_query_dictionary(
                feature_dict,
                image_hw=image_hw,
                num_patch=num_patch,
                extract_type="student",
                answer_texts=teacher_out["answer_text"],
            )

            mask = teacher_out["kd_mask"]

            if mask.any():
                teacher_logits = teacher_out["logits"]
                student_logits = student_out["logits"]
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits[mask], dim=-1),
                    F.softmax(teacher_logits[mask], dim=-1),
                    reduction="batchmean",
                )
                loss = loss + self.kd_weight * kd_loss

        if self.mix_query and self.ortho:
            img_feat = image_embeds.mean(dim=1).detach().to(self.current_keys.dtype)
            img_norm = F.normalize(img_feat, p=2, dim=-1)
            current_keys_norm = F.normalize(self.current_keys, p=2, dim=-1)
            task_attention_scores = torch.einsum('bd,nd->bn', img_norm, current_keys_norm)
            key_task_loss = torch.mean(1 - task_attention_scores)

            _, k_dim = self.current_keys.shape
            _, _, q_dim = self.current_queries.shape
            history_keys = self.keys_history if self.keys_history is not None else torch.tensor([]).to(image_embeds.device)
            history_queries = self.queries_history if self.queries_history is not None else torch.tensor([]).to(image_embeds.device)
            history_keys = history_keys.view(-1, k_dim)
            history_queries = history_queries.view(1, -1, q_dim)
            # For testing without normalization
            # key_gram_matrix = torch.einsum('bd,nd->bn', F.normalize(self.current_keys, p=2, dim=-1), F.normalize(history_keys, p=2, dim=-1))
            # query_gram_matrix = torch.einsum('bqd,bad->qa', F.normalize(self.current_queries, p=2, dim=-1), F.normalize(history_queries, p=2, dim=-1))
            key_gram_matrix = torch.einsum('bd,nd->bn', self.current_keys, history_keys)
            query_gram_matrix = torch.einsum('bqd,bad->qa', self.current_queries, history_queries)
            key_ortho_loss = torch.norm(key_gram_matrix, p='fro')**2
            query_ortho_loss = torch.norm(query_gram_matrix, p='fro')**2

            ortho_loss = query_ortho_loss + key_ortho_loss
            loss = loss + self.ortho_weight * (ortho_loss + key_task_loss)

        return {
            "loss": loss,
            "output loss": outputs.loss,
            "DR loss": kd_loss,
            "MoQ loss": ortho_loss + key_task_loss,
            # "Key task loss": key_task_loss,
            # "Ortho loss": ortho_loss,
        }

    # ------------------------------------------------------------------
    # Generation
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling: bool = False,
        num_beams: int = 1,
        max_length: int = 64,
        min_length: int = 1,
        top_p: Optional[float] = None,
        num_captions: int = 1,
        temperature: Optional[float] = 0.2,
        **kwargs,
    ):
        with self.maybe_autocast():
            projector_tokens, prompt_tokens, prefixes, _, _ = self._forward_internal_for_generation(samples)
            device = projector_tokens.device

            input_ids, attention_mask, _ = tokenize_with_image_support(
                self.llava_tokenizer,
                prefixes,
                self.max_txt_len,
                add_special_tokens=True,
                allow_image_token=True,
            )
            _, min_offset = compute_image_offsets(input_ids)
            text_offset = 0 if min_offset is None else min_offset
            if prompt_tokens is None:
                soft_prompt = None
            else:
                offset = text_offset + projector_tokens.size(1)
                if self.mm_use_im_start_end:
                    offset += 1  # skip <im_end> so soft prompt sits after the image block
                soft_prompt = (prompt_tokens, offset)
                    
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            temp = 0.0 if temperature is None else temperature
            use_sampling = True if temp > 0 and use_nucleus_sampling else False
            stop_words = ["###"]
            stopping_criteria = StoppingCriteriaList(
                [KeywordsStoppingCriteria(stop_words, self.llava_tokenizer, input_ids)]
            )

            outputs = self.llava_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=samples["image"],
                soft_prompt=soft_prompt,
                do_sample=use_sampling,
                top_p=top_p,
                temperature=temp,
                num_beams=num_beams,
                max_new_tokens=max_length,
                stopping_criteria=stopping_criteria,
            )
            decoded = self.llava_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [clean_generation(text, prefix) for text, prefix in zip(decoded, prefixes)]

    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams: int = 1,
        max_len: int = 32,
        min_len: int = 1,
        use_nucleus_sampling: bool = False,
        top_p: Optional[float] = None,
        temperature: Optional[float] = 0.2,
        prompt: str = "",
        **kwargs,
    ):
        with self.maybe_autocast():
            projector_tokens, prompt_tokens, prefixes, _, _ = self._forward_internal_for_generation(samples, prompt_override=prompt)
            device = projector_tokens.device
            input_ids, attention_mask, _ = tokenize_with_image_support(
                self.llava_tokenizer,
                prefixes,
                self.max_txt_len,
                add_special_tokens=True,
                allow_image_token=True,
            )
            _, min_offset = compute_image_offsets(input_ids)
            text_offset = 0 if min_offset is None else min_offset
            if prompt_tokens is None:
                soft_prompt = None
            else:
                offset = text_offset + projector_tokens.size(1)
                if self.mm_use_im_start_end:
                    offset += 1
                soft_prompt = (prompt_tokens, offset)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            temp = 0.0 if temperature is None else temperature
            use_sampling = True if temp > 0 and use_nucleus_sampling else False
            stop_words = ["###"]
            stopping_criteria = StoppingCriteriaList(
                [KeywordsStoppingCriteria(stop_words, self.llava_tokenizer, input_ids)]
            )

            outputs = self.llava_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=samples["image"],
                soft_prompt=soft_prompt,
                do_sample=use_sampling,
                top_p=top_p,
                temperature=temp,
                num_beams=num_beams,
                max_new_tokens=max_len,
                stopping_criteria=stopping_criteria,
            )
            decoded = self.llava_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [clean_generation(text, prefix) for text, prefix in zip(decoded, prefixes)]

    def _forward_internal_for_generation(self, samples, prompt_override: Optional[str] = None):
        image: Tensor = samples["image"]
        image_embeds = self.encode_image(image)
        projector_tokens = image_embeds
        prompt_tokens = self.mixture_of_query(projector_tokens)

        batch_size = image.size(0)
        if prompt_override:
            questions = samples.get("text_input")
            if not isinstance(questions, list):
                questions = [questions or ""] * batch_size

            if "{}" in prompt_override:
                formatted_questions = [prompt_override.format(q or "") for q in questions]
            else:
                formatted_questions = [
                    f"{prompt_override} {q}".strip() if q else prompt_override for q in questions
                ]
                
            samples["text_input"] = formatted_questions

        prefixes, answers, full_sequences = self._build_sequences(samples, batch_size)

        return projector_tokens, prompt_tokens, prefixes, answers, full_sequences

    # ------------------------------------------------------------------
    # Persistence utilities
    def create_teacher_snapshot(self):
        llama_core = self.llava_model.get_model()
        student_attn, student_ffn = self.parallel_adapters_dict[self.current_adapter_index]
        teacher_names = self.teacher_adapters_dict.get("teacher")
        if teacher_names is None:
            teacher_attn = "llm_attn_pa_teacher"
            teacher_ffn = "llm_ffn_pa_teacher"
            llama_core.add_adapter(teacher_attn, config=self.attn_pa_config)
            llama_core.add_adapter(teacher_ffn, config=self.ffn_pa_config)
            teacher_names = [teacher_attn, teacher_ffn]
            self.teacher_adapters_dict["teacher"] = teacher_names
        init_adapter(llama_core, teacher_names[0], student_attn)
        init_adapter(llama_core, teacher_names[1], student_ffn)
        freeze_adapter(llama_core, teacher_names, freeze=True)
        return {"teacher_adapters": list(teacher_names)}

    @property
    def adapter_structure(self):
        return {
            "adapter": deepcopy(self.parallel_adapters_dict),
            "lora": deepcopy(self.lora_dict),
            "moq_num": self.moq_num,
            "visual_frozen": self.visual_tower_freezed,
        }

    def rebuild_from_config(self, adapter_structure: dict, moq_old_kv: dict, **kwargs):
        if not adapter_structure:
            return self

        lora_info = adapter_structure.get("lora", {})
        if lora_info:
            logging.info('Rebuilding visual LoRA from config...')
            self.lora_dict.clear()
            self.visual_lora_init = False
            for _ in lora_info:
                self.expand_visual_lora()

        if adapter_structure.get("visual_frozen", True):
            logging.info('Freezing vision tower as per config...')
            self.freeze_vision_modules(disable_training=True, freeze_projector=not self.train_projector)
        else:
            self._set_requires_grad(self.vision_tower, True)
            projector = getattr(self, "mm_projector", None)
            if projector is not None:
                self._set_requires_grad(projector, True)
        if len(self.adapters) == 0:
            self.init_moe_lora()

        moq_num = adapter_structure.get("moq_num", 0)
        if moq_num > 0:
            self.keys_history = None
            self.queries_history = None
            self.current_keys = None
            self.current_queries = None
            logging.info('Rebuilding MoQ from config...')
            for _ in range(moq_num):
                self._prepare_task_queries(svd_init=False)
            for key, value in moq_old_kv.items():
                if 'history' in key:
                    logging.info('Copying MoQ current kv:'+str(key))
                    setattr(self, key, value.to(device=self.moq_device))
                elif 'current' in key:
                    logging.info('Skipping MoQ current kv:'+str(key)+", it will be loded by model state dict.")


    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        return cls(
            freeze_vit=cfg.get("freeze_vit", True),
            train_projector=cfg.get("train_projector", False),
            soft_prompt_len=cfg.get("soft_prompt_len", cfg.get("prompt_len", 16)),
            mix_query=cfg.get("mix_query", True),
            kd_weight=cfg.get("kd_weight", 0.0),
            ortho=cfg.get("ortho", False),
            ortho_weight=cfg.get("ortho_weight", 0.1),
            alignment_layers=cfg.get("alignment_layers", 4),
            mh_pa_r=cfg.get("mh_pa_r", 32.0),
            mh_pa_dropout=cfg.get("mh_pa_dropout", 0.0),
            mh_pa_scale=cfg.get("mh_pa_scale", 4.0),
            ffn_pa_r=cfg.get("ffn_pa_r", 1.0),
            ffn_pa_dropout=cfg.get("ffn_pa_dropout", 0.0),
            ffn_pa_scale=cfg.get("ffn_pa_scale", 4.0),
            vision_lora_r=cfg.get("vision_lora_r", 16),
            vision_lora_alpha=cfg.get("vision_lora_alpha", 32),
            mm_projector_lr=cfg.get("mm_projector_lr"),
            max_txt_len=cfg.get("max_txt_len"),
            max_answer_len=cfg.get("max_answer_len", 64),
            prompt=cfg.get("prompt", ""),
            conversation_template=cfg.get(
                "conversation_template",
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
                "### Human: {}\n"
                "### Assistant: ",
            ),
            kd_prompt=cfg.get("kd_prompt", ""),
            use_prompt_anchor=cfg.get("use_prompt_anchor", False),
            load_pretrained=cfg.get("load_pretrained", True),
            model_path=cfg.get("model_path"),
            model_base=cfg.get("model_base"),
            model_name=cfg.get("model_name"),
            pretrained=cfg.get("pretrained"),
            mm_use_im_patch_token=cfg.get("mm_use_im_patch_token", False),
            mm_use_im_start_end=cfg.get("mm_use_im_start_end", False),
            **cfg.get("pretrained_kwargs", {}),
        )
