"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""InternVL continual alignment model with minimal ECA pipeline."""

import contextlib
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from adapters import ParBnConfig
from lavis.common.registry import registry
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from internvl.conversation import get_conv_template
from internvl.train.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN
from internvl.train.dataset import IGNORE_TOKEN_ID, preprocess_internvl2_5

from ..custom_adapters import adapter_init
from ..custom_adapters.utils import (
    Average,
    freeze_adapter,
    init_adapter,
    iter_adapter_named_parameters,
    print_trainable_parameters,
)
from .softprompt_internvl import SoftpromptInternVLBase
from .utils import orthogonal_svd_init


@registry.register_model("pa_internvl_general")
class PAInternVLGeneral(SoftpromptInternVLBase):
    """Baseline InternVL continual model with soft prompts and adapter expansion."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "cl_caption_internvl2_5": "configs/models/pa_caption_internvl2_5.yaml",
        "cl_vqa_internvl2_5": "configs/models/pa_vqa_internvl2_5.yaml",
    }

    def __init__(
        self,
        freeze_vit=True,
        train_projector=False,
        soft_prompt_len=16,
        mix_query=True,
        kd_weight=0.0,
        ortho=False,
        ortho_weight=0.1,
        alignment_layers=4,
        mh_pa_r=32.0,
        mh_pa_dropout=0.0,
        mh_pa_scale=4.0,
        ffn_pa_r=1.0,
        ffn_pa_dropout=0.0,
        ffn_pa_scale=4.0,
        vision_lora_r=16,
        vision_lora_alpha=32,
        vision_lora_dropout=0.05,
        dict_patch_blocks=1,
        mm_projector_lr=None,
        max_txt_len=None,
        max_answer_len=64,
        prompt="",
        kd_prompt="",
        use_ocr_input=True,
        use_prompt_anchor=False,
        load_pretrained=True,
        model_path=None,
        pretrained=None,
        use_flash_attn=False,
        **loader_kwargs,
    ):
        """
        Args:
            freeze_vit: Whether to freeze InternVL vision tower and projector.
            train_projector: Keep projector trainable and use pre-projector features for dictionary replay.
            soft_prompt_len: Length of MoQ soft prompt tokens per task.
            mix_query: Enable mixture-of-query routing.
            kd_weight: Weight for dictionary replay loss.
            ortho: Enable orthogonality regularisation over MoQ keys/queries.
            ortho_weight: Factor for orthogonality penalty.
            alignment_layers: Number of lowest Qwen2 layers kept trainable with adapters.
            mh_pa_r/ffn_pa_r: Reduction factors for attention / FFN parallel adapters.
            mh_pa_dropout/ffn_pa_dropout: Dropout applied inside adapters.
            mh_pa_scale/ffn_pa_scale: Scaling factors for adapter outputs.
            vision_lora_r / vision_lora_alpha / vision_lora_dropout: Reserved PEFT LoRA settings for the
                visual encoder. The hook is intentionally single-step only in this first InternVL variant.
            dict_patch_blocks: Fixed number of image blocks used when replay features are reshaped back into
                InternVL visual token sequences for dictionary replay.
            mm_projector_lr: Optional learning-rate scale for the projector bucket.
            max_txt_len / max_answer_len: Optional caps used during generation / KD extraction.
            prompt: Task prompt prefix for captioning/VQA datasets.
            kd_prompt: Prompt used when extracting dictionary features.
            use_prompt_anchor: Use shared prompt anchor for MoQ.
            load_pretrained / model_path / pretrained / use_flash_attn / loader_kwargs:
                forwarded to `SoftpromptInternVLBase`, which in turn loads the official InternVL backbone.
        """
        super().__init__(
            model_path=model_path,
            load_pretrained=load_pretrained,
            use_flash_attn=use_flash_attn,
            pretrained=pretrained,
            **loader_kwargs,
        )

        self.train_projector = train_projector
        self.mm_projector_lr = mm_projector_lr
        self.max_txt_len = max_txt_len
        self.max_answer_len = max_answer_len
        self.dict_patch_blocks = int(dict_patch_blocks)
        self.vision_patch_len = self.internvl_model.num_image_token * self.dict_patch_blocks
        self.prompt = (prompt or "").strip()
        self.kd_prompt = (kd_prompt or "").strip()
        self.use_ocr_input = bool(use_ocr_input)
        self.mix_query = mix_query
        self.kd_weight = kd_weight
        self.ortho = ortho
        self.ortho_weight = ortho_weight
        self.template = self.internvl_model.template
        setattr(self.vision_tower, "patch_len", self.vision_patch_len)

        if freeze_vit:
            self.freeze_vision_modules(disable_training=True, freeze_projector=not train_projector)

        if self.train_projector:
            projector_dtype = next(self.mm_projector.parameters()).dtype
            self.mm_projector = self.mm_projector.to(dtype=projector_dtype)

        self.moq_prompt_len = soft_prompt_len
        self.moq_hidden_dim = self.llm_model.config.hidden_size
        self.use_prompt_anchor = use_prompt_anchor
        if use_prompt_anchor:
            anchor = torch.zeros(1, soft_prompt_len, self.moq_hidden_dim)
            nn.init.xavier_uniform_(anchor)
            self.prompt_anchor = nn.Parameter(anchor, requires_grad=False)
        else:
            zeros = torch.zeros(1, soft_prompt_len, self.moq_hidden_dim)
            self.register_buffer("prompt_anchor", zeros, persistent=False)
        self.current_keys = None
        self.current_queries = None
        self.keys_history = None
        self.queries_history = None

        self.parallel_adapters_dict = {}
        self.adapter_init = False
        leave_out = list(range(alignment_layers, self.llm_model.config.num_hidden_layers))
        self.attn_pa_config = ParBnConfig(
            mh_adapter=True,
            output_adapter=False,
            reduction_factor=mh_pa_r,
            dropout=mh_pa_dropout,
            scaling=mh_pa_scale,
            non_linearity="linear",
            leave_out=leave_out,
        )
        self.ffn_pa_config = ParBnConfig(
            reduction_factor=ffn_pa_r,
            dropout=ffn_pa_dropout,
            scaling=ffn_pa_scale,
            non_linearity="linear",
            leave_out=leave_out,
        )
        logging.info(
            "[Adapter] InternVL alignment_layers=%d, leave_out=%s",
            alignment_layers,
            leave_out,
        )
        self.attn_pa_prefix = "llm_attn_pa_"
        self.ffn_pa_prefix = "llm_ffn_pa_"

        self.visual_lora_init = False
        self.lora_dict = {}
        self.vision_lora_r = vision_lora_r
        self.vision_lora_alpha = vision_lora_alpha
        self.vision_lora_dropout = vision_lora_dropout
        self.lora_prefix = "vision_lora_"

        self.teacher_adapters_dict = {}
        self.use_teacher = False
        self.projector_snapshot = None
        self.unchange_keys = []

    @property
    def adapters(self):
        names = []
        for group in self.parallel_adapters_dict.values():
            names.extend(group)
        return names

    @property
    def moq_device(self):
        return next(self.mm_projector.parameters()).device

    @property
    def loras(self):
        names = []
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
        return not any(param.requires_grad for param in params)

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
            return int(self.mm_projector[0].normalized_shape[0])
        return self.llm_model.config.hidden_size

    @property
    def trainable_adapter_parameters(self):
        llm_core = self.llm_model.model
        adapter_names = getattr(self, "adapters", [])
        if not adapter_names:
            return
        for name, param in iter_adapter_named_parameters(llm_core, adapter_names):
            if param.requires_grad:
                yield name, param

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """Group trainable params into projector / adapter / visual LoRA / MoQ buckets."""
        grouped_named_params = {
            "projector": [],
            "adapter": [],
            "vision_lora": [],
            "moq": [],
            "other": [],
        }

        llm_adapter_prefixes = (self.attn_pa_prefix, self.ffn_pa_prefix)

        def is_no_decay(name):
            return name.endswith(".bias") or "norm" in name.lower()

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "internvl_model.mlp1" in name:
                grouped_named_params["projector"].append((name, param))
            elif any(prefix in name for prefix in llm_adapter_prefixes):
                grouped_named_params["adapter"].append((name, param))
            elif "internvl_model.vision_model" in name and "lora_" in name:
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
            param_groups.append({"weight_decay": weight_decay, "lr_scale": proj_lr_scale, "params": projector_decay})
        if projector_no_decay:
            param_groups.append({"weight_decay": 0.0, "lr_scale": proj_lr_scale, "params": projector_no_decay})
        if adapter_params:
            param_groups.append({"weight_decay": 0.0, "lr_scale": 1, "params": adapter_params})
        if vision_lora_params:
            param_groups.append({"weight_decay": 0.0, "lr_scale": 1, "params": vision_lora_params})
        if moq_params:
            param_groups.append({"weight_decay": 0.0, "lr_scale": 1, "params": moq_params})
        if other_decay:
            warning_msg = (
                "Unexpected trainable params found outside 'projector/LLM adapters/vision LoRA/MoQ'. "
                "Backbone appears to be updating."
            )
            logging.warning(warning_msg)
            print(f"WARNING: {warning_msg}")
            param_groups.append({"weight_decay": weight_decay, "lr_scale": 1, "params": other_decay})
        if other_no_decay:
            param_groups.append({"weight_decay": 0.0, "lr_scale": 1, "params": other_no_decay})

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

    def _normalize_batch_text(self, values, batch_size):
        if values is None:
            return [""] * batch_size
        if isinstance(values, str):
            return [values] * batch_size
        normalized = []
        for value in values:
            normalized.append("" if value is None else str(value))
        if len(normalized) != batch_size:
            raise ValueError("Batch text fields do not match the image batch size.")
        return normalized

    def _prepare_task_queries(self, svd_init=True):
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

    def mixture_of_query(self, image_embeds, old_only=False):
        if image_embeds.device != self.moq_device:
            raise RuntimeError("Mixture-of-Query expects image embeddings to share device with MoQ buffers.")
        if not self.mix_query:
            if self.use_prompt_anchor:
                return self.prompt_anchor.expand(image_embeds.size(0), -1, -1)
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

        if image_embeds.dim() == 3:
            img_feat = image_embeds.mean(dim=1).to(self.current_keys.dtype)
        elif image_embeds.dim() == 2:
            img_feat = image_embeds.to(self.current_keys.dtype)
        else:
            raise ValueError("Mixture-of-Query expects image embeddings with shape [B, N, D] or [B, D].")
        img_norm = F.normalize(img_feat, p=2, dim=-1)
        key_norm = F.normalize(key_sources, p=2, dim=-1)
        scaled_logits = torch.einsum("bd,nd->bn", img_norm, key_norm)
        attn = F.softmax(scaled_logits, dim=1)
        delta_prompt = torch.einsum("bn,nqd->bqd", attn, query_sources)
        base = self.prompt_anchor.expand(image_embeds.size(0), -1, -1)
        return base + delta_prompt

    def _get_num_patches_list(self, samples, image):
        num_patches_list = samples.get("num_patches_list")
        if num_patches_list is None:
            if image.dim() != 4:
                raise ValueError("Missing num_patches_list for dynamic InternVL image batch.")
            return [1] * int(image.size(0))

        if torch.is_tensor(num_patches_list):
            num_patches_list = num_patches_list.tolist()

        num_patches_list = [int(value) for value in num_patches_list]
        if sum(num_patches_list) != int(image.size(0)):
            raise ValueError(
                f"Dynamic image batch mismatch: sum(num_patches_list)={sum(num_patches_list)} "
                f"but image batch has {int(image.size(0))} tiles."
            )
        return num_patches_list

    def _get_image_token_counts(self, num_patches_list):
        return [self.internvl_model.num_image_token * int(count) for count in num_patches_list]

    def _pool_visual_features_for_moq(self, image_embeds, num_patches_list):
        if image_embeds.dim() != 3:
            raise ValueError("InternVL visual features for MoQ must have shape [N_tiles, N_tokens, D].")

        if len(num_patches_list) == int(image_embeds.size(0)):
            return image_embeds.mean(dim=1)

        pooled_tiles = image_embeds.mean(dim=1)
        pooled_samples = []
        start = 0
        for count in num_patches_list:
            end = start + int(count)
            pooled_samples.append(pooled_tiles[start:end].mean(dim=0))
            start = end
        return torch.stack(pooled_samples, dim=0)

    def _build_messages(self, samples, batch_size, answer_texts=None, prompt_override=None):
        outputs = self._normalize_batch_text(answer_texts, batch_size) if answer_texts is not None else [
            output or "" for output in self._normalize_batch_text(samples.get("text_output"), batch_size)
        ]
        text_inputs = self._normalize_batch_text(samples.get("text_input"), batch_size)
        if self.use_ocr_input:
            ocr_inputs = self._normalize_batch_text(samples.get("ocr_input"), batch_size)
        else:
            ocr_inputs = [""] * batch_size

        human_messages = []
        answers = []
        base_prompt = self.prompt.strip()
        prompt_len = len(base_prompt)

        for question, ocr, answer in zip(text_inputs, ocr_inputs, outputs):
            question = (question or "").strip()
            ocr = (ocr or "").strip()
            answer_body = (answer or "").strip()

            if prompt_override:
                if "{}" in prompt_override:
                    question = prompt_override.format(question)
                else:
                    question = f"{prompt_override} {question}".strip() if question else prompt_override.strip()

            human_lines = ["<image>\n"]
            if ocr:
                human_lines.append(ocr + " ")
            if question:
                human_lines.append(question)
            elif base_prompt:
                human_lines.append(base_prompt)
                if answer_body.startswith(base_prompt):
                    answer_body = answer_body[prompt_len:].lstrip()
            else:
                raise ValueError("Either question or prompt must be provided.")

            human_messages.append("".join(line for line in human_lines if line).strip())
            answers.append(answer_body)

        return human_messages, answers

    def _decode_generations(self, outputs, sep_token):
        decoded = self.internvl_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.split(sep_token)[0].strip() for text in decoded]

    def _build_sources(self, human_messages, answer_texts=None):
        if answer_texts is not None and len(human_messages) != len(answer_texts):
            raise ValueError("human_messages and answer_texts must have the same batch size.")

        sources = []
        for idx, human_message in enumerate(human_messages):
            conv = [{"from": "human", "value": human_message}]
            if answer_texts is not None:
                conv.append({"from": "gpt", "value": answer_texts[idx]})
            sources.append(conv)
        return sources

    def _preprocess_sources(self, sources, num_image_tokens, text_only=False):
        if isinstance(num_image_tokens, int):
            token_counts = [num_image_tokens] * len(sources)
        else:
            token_counts = list(num_image_tokens)
        if len(token_counts) != len(sources):
            raise ValueError("num_image_tokens must either be an int or match the batch size.")

        sample_batches = []
        for source, token_count in zip(sources, token_counts):
            sample = preprocess_internvl2_5(
                self.template,
                [source],
                self.internvl_tokenizer,
                [int(token_count)],
                text_only=text_only,
                group_by_length=True,
                num_image=0 if text_only else 1,
            )
            sample_batches.append(
                {
                    "input_ids": sample["input_ids"].squeeze(0),
                    "labels": sample["labels"].squeeze(0),
                    "attention_mask": sample["attention_mask"].squeeze(0),
                }
            )

        pad_token_id = self.internvl_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("InternVL tokenizer must define a pad_token_id for batch padding.")

        max_len = max(sample["input_ids"].size(0) for sample in sample_batches)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}

        for sample in sample_batches:
            pad_len = max_len - sample["input_ids"].size(0)
            batch["input_ids"].append(F.pad(sample["input_ids"], (0, pad_len), value=pad_token_id))
            batch["labels"].append(F.pad(sample["labels"], (0, pad_len), value=IGNORE_TOKEN_ID))
            batch["attention_mask"].append(F.pad(sample["attention_mask"], (0, pad_len), value=False))

        return {
            "input_ids": torch.stack(batch["input_ids"], dim=0),
            "labels": torch.stack(batch["labels"], dim=0),
            "attention_mask": torch.stack(batch["attention_mask"], dim=0),
        }

    def _compute_soft_prompt_offset(self, input_ids):
        img_end_token_id = self.internvl_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        positions = []
        for row in input_ids:
            matched = (row == img_end_token_id).nonzero(as_tuple=False)
            if matched.numel() == 0:
                raise RuntimeError("Failed to locate the InternVL image block when computing soft prompt offsets.")
            positions.append(int(matched[0].item()) + 1)
        return torch.tensor(positions, dtype=torch.long)

    def _expand_answer_token_mask(self, answer_token_mask, logits_shape, prompt_tokens=None, prompt_offsets=None):
        if prompt_tokens is None:
            return answer_token_mask

        prompt_len = int(prompt_tokens.size(1))
        expanded_answer_mask = torch.zeros(logits_shape[:2], dtype=torch.bool, device=answer_token_mask.device)
        for row_idx, insert_after in enumerate(prompt_offsets.tolist()):
            left = answer_token_mask[row_idx, :insert_after]
            right = answer_token_mask[row_idx, insert_after:]
            expanded_answer_mask[row_idx] = torch.cat(
                [
                    left,
                    torch.zeros(prompt_len, dtype=torch.bool, device=answer_token_mask.device),
                    right,
                ],
                dim=0,
            )
        return expanded_answer_mask

    def _build_kd_mask(self, answer_token_mask, logits_shape):
        answer_lengths = answer_token_mask.sum(dim=1)
        kd_mask = torch.zeros(logits_shape[:2], dtype=torch.bool, device=answer_token_mask.device)
        for row_idx in range(answer_token_mask.size(0)):
            answer_positions = answer_token_mask[row_idx].nonzero(as_tuple=False).squeeze(-1)
            if answer_positions.numel() > 1:
                kd_mask[row_idx, answer_positions[:-1]] = True
        return kd_mask, answer_lengths

    def _prepare_generation_inputs(self, human_messages, num_image_tokens, prompt_tokens=None):
        tokenizer = self.internvl_tokenizer
        previous_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        try:
            queries = []
            eos_token_id = None
            sep_token = None
            for human_message, token_count in zip(human_messages, num_image_tokens):
                template = get_conv_template(self.template)
                template.system_message = self.internvl_model.system_message
                template.append_message(template.roles[0], human_message)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(token_count) + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                queries.append(query)
                sep_token = template.sep.strip()
                eos_token_id = tokenizer.convert_tokens_to_ids(sep_token)

            tokenize_kwargs = {"return_tensors": "pt", "padding": True}
            if self.max_txt_len is not None:
                tokenize_kwargs["max_length"] = self.max_txt_len
                tokenize_kwargs["truncation"] = True
            model_inputs = tokenizer(queries, **tokenize_kwargs)
        finally:
            tokenizer.padding_side = previous_padding_side

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        soft_prompt = None
        if prompt_tokens is not None:
            offset = self._compute_soft_prompt_offset(input_ids)
            soft_prompt = (prompt_tokens, offset)
        return input_ids, attention_mask, soft_prompt, eos_token_id, sep_token

    def _run_with_sources(self, sources, visual_features, prompt_tokens=None, use_grad=True, num_image_tokens=None):
        if num_image_tokens is None:
            num_image_tokens = visual_features.size(1)
        batch = self._preprocess_sources(sources, num_image_tokens)
        device = visual_features.device
        visual_features = visual_features.to(device=device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        soft_prompt = None
        if prompt_tokens is not None:
            offset = self._compute_soft_prompt_offset(batch["input_ids"]).to(device)
            soft_prompt = (prompt_tokens, offset)

        grad_ctx = contextlib.nullcontext() if use_grad else torch.no_grad()
        with grad_ctx:
            with self.maybe_autocast():
                outputs = self.internvl_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    visual_features=visual_features,
                    soft_prompt=soft_prompt,
                    return_dict=True,
                )

        answer_token_mask = self._expand_answer_token_mask(
            labels != IGNORE_TOKEN_ID,
            outputs.logits.shape,
            prompt_tokens=prompt_tokens,
            prompt_offsets=soft_prompt[1] if soft_prompt is not None else None,
        )
        kd_mask, answer_lengths = self._build_kd_mask(answer_token_mask, outputs.logits.shape)
        return outputs, kd_mask, answer_lengths

    @torch.no_grad()
    def extract_visual_feature(self, samples):
        embeds = self.encode_image(samples["image"], use_projector=not self.train_projector)
        return {"feature": embeds}

    def before_training(self, expand_adapters=True, lora_visual=False, **kwargs):
        super().before_training(**kwargs)

        if expand_adapters:
            self.expand_alignment_adapter()
            self.use_teacher = False
        else:
            self.use_teacher = self.current_adapter_index >= 0
            if self.use_teacher:
                self.create_teacher_snapshot()

        if self.train_projector:
            projector = deepcopy(self.mm_projector)
            projector.to(device=next(self.mm_projector.parameters()).device, dtype=next(self.mm_projector.parameters()).dtype)
            projector.eval()
            for param in projector.parameters():
                param.requires_grad = False
            self.projector_snapshot = projector
        else:
            self.projector_snapshot = None

        if lora_visual:
            if self.visual_tower_freezed:
                raise AssertionError("Vision tower is frozen; set freeze_vit=False before expanding LoRA.")
            self.expand_visual_lora()
        elif self.lora_dict and not self.visual_tower_freezed:
            self.freeze_vision_modules(disable_training=True, freeze_projector=not self.train_projector)

        if self.mix_query:
            self._prepare_task_queries(svd_init=self.keys_history is not None)

        if not self.unchange_keys:
            self.unchange_keys = [name for name, param in self.named_parameters() if not param.requires_grad]

    def expand_alignment_adapter(self):
        llm_core = self.llm_model.model
        if not self.adapter_init:
            adapter_init(llm_core, use_customize=True)
            llm_core.freeze_model()
            self.freeze_lm_head(disable_training=True)
            self.adapter_init = True

        idx = len(self.parallel_adapters_dict)
        if idx > 0:
            freeze_adapter(llm_core, self.adapters)

        attn_name = f"{self.attn_pa_prefix}{idx}"
        ffn_name = f"{self.ffn_pa_prefix}{idx}"
        self.parallel_adapters_dict[idx] = [attn_name, ffn_name]

        llm_core.add_adapter(attn_name, config=self.attn_pa_config)
        llm_core.add_adapter(ffn_name, config=self.ffn_pa_config)

        attn_names = [names[0] for names in self.parallel_adapters_dict.values()]
        ffn_names = [names[1] for names in self.parallel_adapters_dict.values()]
        llm_core.active_adapters = [
            Average(*attn_names, weights=[1] * len(attn_names)),
            Average(*ffn_names, weights=[1] * len(ffn_names)),
        ]

        if idx > 0:
            prev_names = [self.parallel_adapters_dict[i] for i in range(idx)]
            init_adapter(llm_core, [attn_name, ffn_name], [list(group) for group in zip(*prev_names)])

        freeze_adapter(llm_core, [attn_name, ffn_name], freeze=False)
        print_trainable_parameters(llm_core)
        adapter_summary_fn = getattr(llm_core, "adapter_summary", None)
        if callable(adapter_summary_fn):
            logging.info("\n%s", adapter_summary_fn())
        logging.info("Enabled Qwen2 adapters (attn): %s", attn_names)
        logging.info("Enabled Qwen2 adapters (ffn): %s", ffn_names)

    def expand_visual_lora(self):
        if self.visual_lora_init:
            logging.info("Visual LoRA is already initialized; skipping repeated wrap.")
            return

        lora_config = PeftLoraConfig(
            r=self.vision_lora_r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=self.vision_lora_alpha,
            lora_dropout=self.vision_lora_dropout,
        )
        self.internvl_model.vision_model = get_peft_model(self.internvl_model.vision_model, lora_config)
        self.visual_lora_init = True
        self.lora_dict = {0: [f"{self.lora_prefix}0"]}
        setattr(self.vision_tower, "patch_len", self.vision_patch_len)
        if hasattr(self.internvl_model.vision_model, "print_trainable_parameters"):
            self.internvl_model.vision_model.print_trainable_parameters()
        logging.info("Activated reserved visual LoRA hook: %s", self.lora_dict[0])

    def forward(self, samples):
        image = samples["image"]
        num_patches_list = self._get_num_patches_list(samples, image)
        token_counts = self._get_image_token_counts(num_patches_list)
        image_embeds = self.encode_image(image, use_projector=True)
        prompt_inputs = self._pool_visual_features_for_moq(image_embeds, num_patches_list)
        prompt_tokens = self.mixture_of_query(prompt_inputs)

        batch_size = len(num_patches_list)
        human_messages, answers = self._build_messages(samples, batch_size)
        sources = self._build_sources(human_messages, answers)
        outputs, _, _ = self._run_with_sources(
            sources,
            image_embeds,
            prompt_tokens=prompt_tokens,
            use_grad=True,
            num_image_tokens=token_counts,
        )

        loss = outputs.loss
        kd_loss = torch.tensor(0.0, device=loss.device)
        ortho_loss = torch.tensor(0.0, device=loss.device)
        key_task_loss = torch.tensor(0.0, device=loss.device)

        feature_dict = samples.get("feature")
        if feature_dict is not None:
            if feature_dict.dim() != 3:
                raise ValueError("Dictionary replay features must have shape [B, num_patch, hidden_dim].")
            num_patch = self.vision_patch_len
            with torch.no_grad():
                teacher_out = self.extract_query_dictionary(
                    feature_dict,
                    num_patch=num_patch,
                    extract_type="teacher",
                )
            student_out = self.extract_query_dictionary(
                feature_dict,
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
            img_feat = prompt_inputs.detach().to(self.current_keys.dtype)
            img_norm = F.normalize(img_feat, p=2, dim=-1)
            current_keys_norm = F.normalize(self.current_keys, p=2, dim=-1)
            task_attention_scores = torch.einsum("bd,nd->bn", img_norm, current_keys_norm)
            key_task_loss = torch.mean(1 - task_attention_scores)

            _, k_dim = self.current_keys.shape
            _, _, q_dim = self.current_queries.shape
            history_keys = self.keys_history if self.keys_history is not None else torch.tensor([], device=image_embeds.device)
            history_queries = self.queries_history if self.queries_history is not None else torch.tensor([], device=image_embeds.device)
            history_keys = history_keys.view(-1, k_dim)
            history_queries = history_queries.view(1, -1, q_dim)
            key_gram_matrix = torch.einsum("bd,nd->bn", self.current_keys, history_keys)
            query_gram_matrix = torch.einsum("bqd,bad->qa", self.current_queries, history_queries)
            key_ortho_loss = torch.norm(key_gram_matrix, p="fro") ** 2
            query_ortho_loss = torch.norm(query_gram_matrix, p="fro") ** 2
            ortho_loss = query_ortho_loss + key_ortho_loss
            loss = loss + self.ortho_weight * (ortho_loss + key_task_loss)

        output = {
            "loss": loss,
            "output loss": outputs.loss,
            "DR loss": kd_loss,
            "MoQ loss": ortho_loss + key_task_loss,
        }
        if samples.get("return_logits"):
            output["logits"] = outputs.logits
        return output

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=8,
        top_p=None,
        num_captions=1,
        temperature=0.2,
        **kwargs,
    ):
        with self.maybe_autocast():
            image = samples["image"]
            num_patches_list = self._get_num_patches_list(samples, image)
            token_counts = self._get_image_token_counts(num_patches_list)
            image_embeds = self.encode_image(image, use_projector=True)
            prompt_inputs = self._pool_visual_features_for_moq(image_embeds, num_patches_list)
            prompt_tokens = self.mixture_of_query(prompt_inputs)

            batch_size = len(num_patches_list)
            human_messages, _ = self._build_messages(samples, batch_size, answer_texts=[""] * batch_size)
            input_ids, attention_mask, soft_prompt, eos_token_id, sep_token = self._prepare_generation_inputs(
                human_messages,
                token_counts,
                prompt_tokens=prompt_tokens,
            )
            input_ids = input_ids.to(image_embeds.device)
            attention_mask = attention_mask.to(image_embeds.device)

            temp = 0.0 if temperature is None else temperature
            use_sampling = True if temp > 0 and use_nucleus_sampling else False
            outputs = self.internvl_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=image_embeds,
                soft_prompt=soft_prompt,
                do_sample=use_sampling,
                top_p=top_p,
                temperature=temp,
                num_beams=num_beams,
                num_return_sequences=num_captions,
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                eos_token_id=eos_token_id,
                **kwargs,
            )
            return self._decode_generations(outputs, sep_token)

    @torch.no_grad()
    def predict_answers(
        self,
        samples,
        num_beams=1,
        max_len=10,
        min_len=1,
        use_nucleus_sampling=False,
        top_p=None,
        temperature=0.2,
        prompt="",
        **kwargs,
    ):
        with self.maybe_autocast():
            image = samples["image"]
            num_patches_list = self._get_num_patches_list(samples, image)
            token_counts = self._get_image_token_counts(num_patches_list)
            image_embeds = self.encode_image(image, use_projector=True)
            prompt_inputs = self._pool_visual_features_for_moq(image_embeds, num_patches_list)
            prompt_tokens = self.mixture_of_query(prompt_inputs)

            batch_size = len(num_patches_list)
            human_messages, _ = self._build_messages(
                samples,
                batch_size,
                answer_texts=[""] * batch_size,
                prompt_override=prompt,
            )
            input_ids, attention_mask, soft_prompt, eos_token_id, sep_token = self._prepare_generation_inputs(
                human_messages,
                token_counts,
                prompt_tokens=prompt_tokens,
            )
            input_ids = input_ids.to(image_embeds.device)
            attention_mask = attention_mask.to(image_embeds.device)

            temp = 0.0 if temperature is None else temperature
            use_sampling = True if temp > 0 and use_nucleus_sampling else False
            outputs = self.internvl_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=image_embeds,
                soft_prompt=soft_prompt,
                do_sample=use_sampling,
                top_p=top_p,
                temperature=temp,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_new_tokens=min_len,
                eos_token_id=eos_token_id,
                **kwargs,
            )
            return self._decode_generations(outputs, sep_token)

    def teacher_forward(
        self,
        samples,
        use_old_moq=True,
        use_teacher_adapters=True,
        use_grad=False,
    ):
        llm_core = self.llm_model.model
        prev_model_mode = self.internvl_model.training
        prev_core_mode = llm_core.training
        prev_active = getattr(llm_core, "active_adapters", None)

        no_grad_ctx = torch.no_grad() if not use_grad else contextlib.nullcontext()
        with no_grad_ctx:
            if not use_grad:
                self.internvl_model.eval()
                llm_core.eval()

            if use_teacher_adapters:
                history = self.adapters
                if self.current_adapter_names:
                    history = history[:-len(self.current_adapter_names)]
                attn_names = history[0::2]
                ffn_names = history[1::2]
                teacher_names = self.teacher_adapters_dict.get("teacher")
                if teacher_names:
                    attn_names = attn_names + [teacher_names[0]] if attn_names else [teacher_names[0]]
                    ffn_names = ffn_names + [teacher_names[1]] if ffn_names else [teacher_names[1]]
                if attn_names and ffn_names:
                    llm_core.set_active_adapters(
                        [
                            Average(*attn_names, weights=[1] * len(attn_names)),
                            Average(*ffn_names, weights=[1] * len(ffn_names)),
                        ]
                    )

            image = samples["image"]
            num_patches_list = self._get_num_patches_list(samples, image)
            token_counts = self._get_image_token_counts(num_patches_list)
            image_embeds = self.encode_image(image, use_projector=True)
            prompt_inputs = self._pool_visual_features_for_moq(image_embeds, num_patches_list)
            use_old_only = use_old_moq and self.keys_history is not None and self.queries_history is not None
            prompt_tokens = self.mixture_of_query(prompt_inputs, old_only=use_old_only) if self.mix_query else None

            batch_size = len(num_patches_list)
            human_messages, answers = self._build_messages(samples, batch_size)
            sources = self._build_sources(human_messages, answers)
            batch = self._preprocess_sources(sources, token_counts)
            device = image_embeds.device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            soft_prompt = None
            if prompt_tokens is not None:
                offset = self._compute_soft_prompt_offset(batch["input_ids"]).to(device)
                soft_prompt = (prompt_tokens, offset)

            with self.maybe_autocast():
                outputs = self.internvl_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    visual_features=image_embeds,
                    soft_prompt=soft_prompt,
                    return_dict=True,
                )

        answer_token_mask = self._expand_answer_token_mask(
            labels != IGNORE_TOKEN_ID,
            outputs.logits.shape,
            prompt_tokens=prompt_tokens,
            prompt_offsets=soft_prompt[1] if soft_prompt is not None else None,
        )
        kd_mask, answer_lengths = self._build_kd_mask(answer_token_mask, outputs.logits.shape)

        if prev_active is not None:
            llm_core.active_adapters = prev_active
        if prev_core_mode:
            llm_core.train()
        if prev_model_mode:
            self.internvl_model.train()

        return {
            "logits": outputs.logits,
            "kd_mask": kd_mask,
            "answer_lengths": answer_lengths,
        }

    def extract_query_dictionary(
        self,
        feature_dictionary,
        num_patch=None,
        extract_type="teacher",
        answer_texts=None,
    ):
        if extract_type not in {"teacher", "student"}:
            raise ValueError(f"Unsupported extract_type {extract_type}. Expected 'teacher' or 'student'.")
        if self.max_answer_len <= 0:
            raise ValueError("max_answer_len must be set to a positive value for KD extraction.")
        if feature_dictionary.dim() != 3:
            raise ValueError("feature_dictionary must have shape [B, num_patch, hidden_dim].")

        if self.train_projector:
            projector = self.projector_snapshot if extract_type == "teacher" else self.mm_projector
            if projector is None:
                raise RuntimeError("Projector snapshot is not initialized. Ensure before_training has been called.")
            proj_ctx = torch.no_grad() if extract_type == "teacher" else contextlib.nullcontext()
            with proj_ctx:
                proj_param = next(projector.parameters())
                feature_dictionary = projector(
                    feature_dictionary.to(device=proj_param.device, dtype=proj_param.dtype)
                )

        patch_len = feature_dictionary.size(1)
        if num_patch is not None and patch_len != num_patch:
            raise ValueError(f"feature_dictionary patch length {patch_len} does not match expected {num_patch}.")

        batch_size = feature_dictionary.size(0)
        feature_dictionary = feature_dictionary.to(device=self.device)

        base_prompt = self.kd_prompt or ""
        kd_samples = {"text_input": [base_prompt] * batch_size, "ocr_input": [""] * batch_size, "text_output": [""] * batch_size}
        human_messages, _ = self._build_messages(kd_samples, batch_size, answer_texts=[""] * batch_size)

        llm_core = self.llm_model.model
        prev_model_mode = self.internvl_model.training
        prev_core_mode = llm_core.training
        prev_active = getattr(llm_core, "active_adapters", None)

        try:
            if extract_type == "teacher":
                with torch.no_grad():
                    self.internvl_model.eval()
                    llm_core.eval()

                    history = self.adapters
                    if self.current_adapter_names:
                        history = history[:-len(self.current_adapter_names)]
                    attn_names = history[0::2]
                    ffn_names = history[1::2]
                    if self.use_teacher:
                        teacher_names = self.teacher_adapters_dict.get("teacher")
                        if not teacher_names:
                            raise RuntimeError("Teacher adapters not initialized for current task.")
                        attn_names = attn_names + [teacher_names[0]] if attn_names else [teacher_names[0]]
                        ffn_names = ffn_names + [teacher_names[1]] if ffn_names else [teacher_names[1]]
                    if attn_names and ffn_names:
                        llm_core.set_active_adapters(
                            [
                                Average(*attn_names, weights=[1] * len(attn_names)),
                                Average(*ffn_names, weights=[1] * len(ffn_names)),
                            ]
                        )

                    prompt_tokens = self.mixture_of_query(feature_dictionary, old_only=True)

                    if answer_texts is None:
                        input_ids, attention_mask, soft_prompt, eos_token_id, sep_token = self._prepare_generation_inputs(
                            human_messages,
                            [patch_len] * batch_size,
                            prompt_tokens=prompt_tokens,
                        )
                        input_ids = input_ids.to(feature_dictionary.device)
                        attention_mask = attention_mask.to(feature_dictionary.device)
                        with self.maybe_autocast():
                            outputs = self.internvl_model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                visual_features=feature_dictionary,
                                soft_prompt=soft_prompt,
                                do_sample=False,
                                num_beams=1,
                                max_new_tokens=self.max_answer_len,
                                eos_token_id=eos_token_id,
                            )
                        answer_texts = self._decode_generations(outputs, sep_token)

                sources = self._build_sources(human_messages, answer_texts)
                outputs, kd_mask, answer_lengths = self._run_with_sources(
                    sources,
                    feature_dictionary,
                    prompt_tokens=prompt_tokens,
                    use_grad=False,
                )
                logits = outputs.logits.detach()
                answer_lengths = answer_lengths.detach()
            else:
                if answer_texts is None:
                    raise ValueError("Student KD requires teacher answer_texts for alignment.")
                prompt_tokens = self.mixture_of_query(feature_dictionary, old_only=False)
                sources = self._build_sources(human_messages, answer_texts)
                outputs, kd_mask, answer_lengths = self._run_with_sources(
                    sources,
                    feature_dictionary,
                    prompt_tokens=prompt_tokens,
                    use_grad=True,
                )
                logits = outputs.logits
        finally:
            if prev_active is not None:
                llm_core.active_adapters = prev_active
            if prev_core_mode:
                llm_core.train()
            if prev_model_mode:
                self.internvl_model.train()

        return {
            "logits": logits,
            "kd_mask": kd_mask,
            "answer_text": answer_texts,
            "answer_lengths": answer_lengths,
        }

    def create_teacher_snapshot(self):
        llm_core = self.llm_model.model
        student_attn, student_ffn = self.parallel_adapters_dict[self.current_adapter_index]
        teacher_names = self.teacher_adapters_dict.get("teacher")
        if teacher_names is None:
            teacher_attn = "llm_attn_pa_teacher"
            teacher_ffn = "llm_ffn_pa_teacher"
            llm_core.add_adapter(teacher_attn, config=self.attn_pa_config)
            llm_core.add_adapter(teacher_ffn, config=self.ffn_pa_config)
            teacher_names = [teacher_attn, teacher_ffn]
            self.teacher_adapters_dict["teacher"] = teacher_names
        init_adapter(llm_core, teacher_names[0], student_attn)
        init_adapter(llm_core, teacher_names[1], student_ffn)
        freeze_adapter(llm_core, teacher_names, freeze=True)
        return {"teacher_adapters": list(teacher_names)}

    @property
    def adapter_structure(self):
        return {
            "adapter": deepcopy(self.parallel_adapters_dict),
            "lora": deepcopy(self.lora_dict),
            "moq_num": self.moq_num,
            "visual_frozen": self.visual_tower_freezed,
        }

    def rebuild_from_config(self, adapter_structure, moq_old_kv, **kwargs):
        del kwargs
        if not adapter_structure:
            return self

        lora_info = adapter_structure.get("lora", {})
        if lora_info:
            logging.info("Rebuilding visual LoRA from config...")
            self.lora_dict.clear()
            self.visual_lora_init = False
            self.expand_visual_lora()

        if adapter_structure.get("visual_frozen", True):
            logging.info("Freezing vision tower as per config...")
            self.freeze_vision_modules(disable_training=True, freeze_projector=not self.train_projector)
        else:
            self._set_requires_grad(self.vision_tower, True)
            projector = getattr(self, "mm_projector", None)
            if projector is not None:
                self._set_requires_grad(projector, True)

        adapter_info = adapter_structure.get("adapter", {})
        if adapter_info:
            logging.info("Rebuilding adapters from config...")
            self.parallel_adapters_dict.clear()
            self.adapter_init = False
            for _ in adapter_info:
                self.expand_alignment_adapter()

        moq_num = adapter_structure.get("moq_num", 0)
        if moq_num > 0:
            self.keys_history = None
            self.queries_history = None
            self.current_keys = None
            self.current_queries = None
            logging.info("Rebuilding MoQ from config...")
            for _ in range(moq_num):
                self._prepare_task_queries(svd_init=False)
            for key, value in moq_old_kv.items():
                if "history" in key:
                    logging.info("Copying MoQ historical state: %s", key)
                    setattr(self, key, value.to(device=self.moq_device))
                elif "current" in key:
                    logging.info("Skipping MoQ current state %s; it will be restored by model state_dict.", key)
        return self

    @classmethod
    def from_config(cls, cfg):
        pretrained_kwargs = dict(cfg.get("pretrained_kwargs", {}))
        for key in [
            "template",
            "select_layer",
            "force_image_size",
            "dynamic_image_size",
            "use_thumbnail",
            "ps_version",
            "min_dynamic_patch",
            "max_dynamic_patch",
            "model_max_length",
            "drop_path_rate",
            "downsample_ratio",
            "pad2square",
            "vision_config",
            "llm_config",
            "vision_config_overrides",
            "llm_config_overrides",
        ]:
            if key in cfg and key not in pretrained_kwargs:
                pretrained_kwargs[key] = cfg.get(key)

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
            vision_lora_dropout=cfg.get("vision_lora_dropout", 0.05),
            dict_patch_blocks=cfg.get("dict_patch_blocks", 1),
            mm_projector_lr=cfg.get("mm_projector_lr"),
            max_txt_len=cfg.get("max_txt_len"),
            max_answer_len=cfg.get("max_answer_len", 64),
            prompt=cfg.get("prompt", ""),
            kd_prompt=cfg.get("kd_prompt", ""),
            use_ocr_input=cfg.get("use_ocr_input", True),
            use_prompt_anchor=cfg.get("use_prompt_anchor", False),
            load_pretrained=cfg.get("load_pretrained", True),
            model_path=cfg.get("model_path"),
            pretrained=cfg.get("pretrained"),
            use_flash_attn=cfg.get("use_flash_attn", False),
            **pretrained_kwargs,
        )
