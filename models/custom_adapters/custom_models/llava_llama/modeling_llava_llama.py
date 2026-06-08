"""Adapter-enabled modules for the LLaVA LLaMA backbone.

This file mirrors ``adapters.models.llama.modeling_llama`` so that the
``llava_llama`` model type can reuse the same adapter-capable
implementations while keeping compatibility with LLaVA's extensions
(vision tower, projector, etc.).
"""

import inspect
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from adapters.composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
try:
    from transformers.cache_utils import Cache  # transformers >= 4.30
except ImportError:  # Older transformers versions used by LAVIS do not expose Cache
    Cache = Any  # type: ignore
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from .mixin_llava_llama import (
    LlavaLlamaAttentionMixin,
    LlavaLlamaDecoderLayerMixin,
)


logger = logging.get_logger(__name__)

_APPLY_USES_POSITION_IDS = len(inspect.signature(apply_rotary_pos_emb).parameters) >= 5


class LlamaAttentionWithAdapters(LlavaLlamaAttentionMixin, LlamaAttention):
    """Multi-headed attention with adapter support."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not hasattr(past_key_value, "update"):
            kv_seq_len += past_key_value[0].shape[-2]

        rotary_sig = inspect.signature(self.rotary_emb.forward)
        if "position_ids" in rotary_sig.parameters:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if _APPLY_USES_POSITION_IDS:
            rope_pos_ids = position_ids
            if rope_pos_ids is None:
                rope_pos_ids = torch.arange(kv_seq_len, device=value_states.device, dtype=torch.long)
                rope_pos_ids = rope_pos_ids.unsqueeze(0).expand(bsz, -1)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                rope_pos_ids,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        present_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                present_key_value = past_key_value
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_dropout = getattr(self, "attention_dropout", getattr(self, "dropout", 0.0))
        attn_weights = nn.functional.dropout(attn_weights, p=attn_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp))
        else:
            attn_output = self.o_proj(attn_output)

        if use_cache and present_key_value is None:
            present_key_value = (key_states, value_states)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


class LlamaFlashAttention2WithAdapters(LlavaLlamaAttentionMixin, LlamaAttention):
    """FlashAttention-optimised attention with adapter support."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not hasattr(past_key_value, "update"):
            kv_seq_len += past_key_value[0].shape[-2]

        rotary_sig = inspect.signature(self.rotary_emb.forward)
        if "position_ids" in rotary_sig.parameters:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if _APPLY_USES_POSITION_IDS:
            rope_pos_ids = position_ids
            if rope_pos_ids is None:
                rope_pos_ids = torch.arange(kv_seq_len, device=value_states.device, dtype=torch.long)
                rope_pos_ids = rope_pos_ids.unsqueeze(0).expand(bsz, -1)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                rope_pos_ids,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        present_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                present_key_value = past_key_value
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        base_dropout = getattr(self, "attention_dropout", getattr(self, "dropout", 0.0))
        dropout_rate = base_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to the fact"
                " you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if use_cache and present_key_value is None:
            present_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        return attn_output, attn_weights, present_key_value


class LlamaSdpaAttentionWithAdapters(LlavaLlamaAttentionMixin, LlamaAttention):
    """Scaled dot-product attention with adapter support."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                "LLaVA LLaMA is using SDPA attention, but `torch.nn.functional.scaled_dot_product_attention` does"
                " not support `output_attentions=True`. Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states, value_states = match_attn_matrices_for_parallel(
            query_states, key_states, value_states
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_states, attention_mask)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not hasattr(past_key_value, "update"):
            kv_seq_len += past_key_value[0].shape[-2]

        rotary_sig = inspect.signature(self.rotary_emb.forward)
        if "position_ids" in rotary_sig.parameters:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if _APPLY_USES_POSITION_IDS:
            rope_pos_ids = position_ids
            if rope_pos_ids is None:
                rope_pos_ids = torch.arange(kv_seq_len, device=value_states.device, dtype=torch.long)
                rope_pos_ids = rope_pos_ids.unsqueeze(0).expand(bsz, -1)
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                rope_pos_ids,
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        present_key_value = None
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                present_key_value = past_key_value
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=(
                getattr(self, "attention_dropout", getattr(self, "dropout", 0.0))
                if self.training
                else 0.0
            ),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if use_cache and present_key_value is None:
            present_key_value = (key_states, value_states)

        return attn_output, None, present_key_value


class LlamaDecoderLayerWithAdapters(LlavaLlamaDecoderLayerMixin, LlamaDecoderLayer):
    """Decoder layer that mirrors the upstream implementation while adding adapters."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in a future Transformers release."
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.attention_adapters(hidden_states, residual, None)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.output_adapters(hidden_states, residual, None)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
