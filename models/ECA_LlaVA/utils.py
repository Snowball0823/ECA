"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import logging
import os
import shutil
import tempfile
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from lavis.common.registry import registry
from torch.autograd import Variable
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import tokenizer_image_token
from llava.train.train import smart_tokenizer_and_embedding_resize
from .softprompt_llava_model import SoftPromptLlavaForCausalLM


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p  


def freeze_parameters(model: nn.Module, names: list):
    for n, p in model.named_parameters():
        for name in names:
            if name in n:
                p.requires_grad = False


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("project_root"), rel_path)



def orthogonal_svd_init(matrix):
    n, m = matrix.shape
    # U, S, V = torch.svd(matrix)
    U, S, V = torch.linalg.svd(matrix)
    num_singular_values = S.numel()
    if num_singular_values < m:
        new_row = torch.randn(m).to(matrix.device)
        for i in range(n):
            row = matrix[i]
            dot_product = torch.dot(new_row, row)
            new_row = new_row - (dot_product / torch.norm(row)**2) * row
    else:
        new_row = torch.randn(m).to(matrix.device) @ V[:, num_singular_values:]

    row_norms = torch.norm(matrix, dim=1)
    input_scale = row_norms.mean()

    new_row = new_row / torch.norm(new_row) * input_scale

    return new_row.unsqueeze(0)


def clean_generation(text: str, prefix: str):
    """Minimal cleaning that keeps descriptive content intact."""
    text = (text or "").strip()
    prefix = (prefix or "").strip()
    stop_token = "###"
    if stop_token in text:
        idx = text.rfind(stop_token)
        if idx >= 0:
            tail = text[idx + len(stop_token):]
            if not tail.strip():
                text = text[:idx].rstrip()

    def strip_prefix(value: str):
        if not prefix:
            return value
        while value.startswith(prefix):
            value = value[len(prefix):].lstrip()
        return value

    text = strip_prefix(text)

    role_tokens = ("assistant", "human", "user", "system", "ocr")
    changed = True
    while changed and text:
        changed = False
        lower = text.lower()
        for token in role_tokens:
            if lower.startswith(token):
                offset = len(token)
                if lower[offset:offset + 1] == ":":
                    offset += 1
                text = text[offset:].lstrip(" :.-")
                text = strip_prefix(text)
                changed = True
                break

    # Normalize whitespace and trim simple punctuation at the beginning.
    text = " ".join(text.replace("\n", " ").split())
    while text.startswith(("#", ",", ".", ";", ":", "-")):
        text = text[1:].lstrip()
    text = strip_prefix(text)

    return text.strip()


def tokenize_with_image_support(
    tokenizer,
    texts: List[str],
    max_length: Optional[int],
    add_special_tokens: bool = False,
    allow_image_token: bool = True,
):
    """Tokenize text sequences, optionally preserving <image> placeholders."""
    seqs: List[List[int]] = []
    lengths: List[int] = []

    for text in texts:
        if allow_image_token and DEFAULT_IMAGE_TOKEN in text:
            tokens = tokenizer_image_token(text, tokenizer)
            if max_length and max_length > 0:
                num_images = text.count(DEFAULT_IMAGE_TOKEN)
                limit = max_length + num_images
                tokens = tokens[:limit]
        else:
            if max_length and max_length > 0:
                enc = tokenizer(
                    text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_length,
                )
            else:
                enc = tokenizer(
                    text,
                    add_special_tokens=add_special_tokens,
                    truncation=False,
                )
            tokens = enc.input_ids

        seqs.append(tokens)
        lengths.append(len(tokens))

    max_seq_len = max(lengths) if lengths else 0
    batch_size = len(seqs)
    device = torch.device("cpu")

    input_ids = torch.full((batch_size, max_seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    for idx, seq in enumerate(seqs):
        if not seq:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[idx, : seq_tensor.size(0)] = seq_tensor
        attention_mask[idx, : seq_tensor.size(0)] = 1

    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    return input_ids, attention_mask, lengths_tensor


def compute_image_offsets(token_ids: torch.Tensor, image_token_index: int = IMAGE_TOKEN_INDEX):
    """Return shared <image> offset for the batch, assuming prompts are aligned."""
    image_mask = token_ids == image_token_index
    if not image_mask.any():
        return None, None
    first_positions = image_mask.float().argmax(dim=1)
    start_ref = first_positions[0].item()
    if (first_positions != start_ref).any():
        raise RuntimeError("Image token positions mismatch within the batch.")
    return first_positions, int(start_ref)


def load_eca_pretrained_model(
    model_path: str,
    model_base: Optional[str],
    model_name: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "auto",
    device: str = "cuda",
    use_flash_attn: bool = False,
    torch_dtype=torch.float16,
    **kwargs,
):
    """Load a LLaVA model with ECA-specific fixes.

    Common keyword overrides (pass via **kwargs):
        device, device_map, torch_dtype, load_in_8bit, load_in_4bit, quantization_config,
        attn_implementation, low_cpu_mem_usage, use_fast_tokenizer, trust_remote_code,
        mm_use_im_patch_token, mm_use_im_start_end, mm_hidden_size, mm_vision_select_layer,
        mm_vision_tower, tune_mm_mlp_adapter, use_mm_proj.

    Supports two scenarios:
        1. Full fine-tuned checkpoints (model_base is None) – mirrors official loader.
        2. Projector-only checkpoints (model_base provided) – correctly restores
           projector and newly introduced image special token embeddings.
    """
    if model_path is None:
        raise ValueError("model_path must be specified for load_eca_pretrained_model.")
    if not model_name or "llava" not in model_name.lower():
        raise AssertionError(
            f"load_eca_pretrained_model currently supports only LLaVA checkpoints (got {model_name})."
        )

    # collect loader kwargs
    load_kwargs = dict(kwargs)
    config_override_keys = (
        "mm_use_im_patch_token",
        "mm_use_im_start_end",
        "mm_hidden_size",
        "mm_vision_select_layer",
        "mm_vision_select_feature",
        "mm_vision_tower",
        "tune_mm_mlp_adapter",
        "use_mm_proj",
        "mm_projector_type",
        "mm_patch_merge_type",
        "image_aspect_ratio",
        "image_grid_pinpoints",
    )
    config_overrides = {}
    for key in config_override_keys:
        if key in load_kwargs:
            config_overrides[key] = load_kwargs.pop(key)
    if device != "cuda":
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs.setdefault("device_map", device_map)

    if load_8bit:
        load_kwargs["load_in_8bit"] = True
    elif load_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs.setdefault("torch_dtype", torch_dtype)

    if use_flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    # disable low_cpu_mem_usage for stability
    load_kwargs.setdefault("low_cpu_mem_usage", True)

    tokenizer = None
    model = None
    embed_weight = None

    if model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        config = AutoConfig.from_pretrained(model_path)
        model = SoftPromptLlavaForCausalLM.from_pretrained(
            model_base,
            config=config,
            **load_kwargs,
        )
        projector_path = os.path.join(model_path, "mm_projector.bin")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"mm_projector checkpoint not found: {projector_path}")
        projector_state = torch.load(projector_path, map_location="cpu")
        for candidate in (
            "model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ):
            if candidate in projector_state:
                embed_weight = projector_state.pop(candidate)
                break
        cast_state = {
            key: (value.to(torch.float16) if torch.is_tensor(value) else value)
            for key, value in projector_state.items()
        }
        model.load_state_dict(cast_state, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        config = AutoConfig.from_pretrained(model_path)
        model = SoftPromptLlavaForCausalLM.from_pretrained(
            model_path,
            config=config,
            **load_kwargs,
        )

    # ensure pad token before adding other tokens
    if tokenizer.pad_token is None:
        logging.info("PAD tokens added for this model configuration.")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    for key, value in config_overrides.items():
        setattr(model.config, key, value)

    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    special_tokens = []
    if mm_use_im_patch_token:
        special_tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
    if mm_use_im_start_end:
        special_tokens.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    num_special_tokens = 0
    if special_tokens:
        num_special_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)

    if num_special_tokens > 0:
        logging.info("Additional special tokens added for this model configuration.")
        prev_vocab_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data
        output_layer = model.get_output_embeddings()
        output_embeddings = output_layer.weight.data if output_layer is not None else None

        if mm_use_im_start_end:
            start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
            end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
            valid_ids = [idx for idx in (start_id, end_id) if idx is not None and idx >= 0]

            base_input = input_embeddings[:prev_vocab_size].mean(dim=0, keepdim=True)
            base_output = (
                output_embeddings[:prev_vocab_size].mean(dim=0, keepdim=True)
                if output_embeddings is not None
                else None
            )
            input_embeddings[valid_ids] = base_input.expand(len(valid_ids), -1)
            if base_output is not None:
                output_embeddings[valid_ids] = base_output.expand(len(valid_ids), -1)

            if embed_weight is not None and valid_ids:
                weight = embed_weight.to(dtype=input_embeddings.dtype, device=input_embeddings.device)
                if weight.shape == input_embeddings.shape:
                    input_embeddings[valid_ids] = weight[valid_ids]
                elif weight.shape[0] == len(valid_ids) == 2:
                    input_embeddings[valid_ids] = weight
                else:
                    logging.warning(
                        "Unexpected embed_tokens weight shape %s for mm projector; expected 2 rows or full vocab.",
                        tuple(weight.shape),
                    )
    else:
        logging.info("No additional special tokens needed for this model configuration.")


    vision_tower = model.get_vision_tower()
    device_map_final = load_kwargs.get("device_map", "auto")
    if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map_final)
    if device_map_final != "auto":
        target_device = device if device != "cuda" else device_map_final
        vision_tower.to(device=target_device, dtype=torch.float16)
    image_processor = getattr(vision_tower, "image_processor", None)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
