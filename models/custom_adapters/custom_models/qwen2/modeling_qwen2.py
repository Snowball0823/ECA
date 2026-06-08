# coding=utf-8
"""Adapter-enabled Qwen2 attention and decoder modules.

The eager path is implemented first because it is the most direct path to
debugging adapter expansion and Fisher computation. FlashAttention2 and SDPA
need custom adapter-aware forwards as well, but those are intentionally left
as explicit TODO stubs for a later pass.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from adapters.composition import adjust_tensors_for_parallel, match_attn_matrices_for_parallel
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

from .mixin_qwen2 import (
    Qwen2AttentionMixin,
    Qwen2DecoderLayerMixin,
)


logger = logging.get_logger(__name__)


class Qwen2AttentionWithAdapters(Qwen2AttentionMixin, Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
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
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention "
                    "class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states, value_states, attention_mask = self.prefix_tuning(
            key_states, value_states, hidden_states, attention_mask
        )
        (query_states,) = adjust_tensors_for_parallel(key_states, query_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        bsz = key_states.shape[0]
        kv_seq_len = key_states.shape[-2]

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is "
                f"{attn_weights.size()}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :kv_seq_len]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is "
                f"{attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention2WithAdapters(Qwen2AttentionMixin, Qwen2FlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # TODO: Implement adapter-aware FlashAttention2 by porting the original
        # Qwen2FlashAttention2 forward and inserting:
        # 1) match/adjust parallel tensor utilities
        # 2) prefix tuning on key/value states
        # 3) any dtype handling kept by the upstream implementation
        raise NotImplementedError(
            "Qwen2FlashAttention2WithAdapters is not implemented yet. "
            "Load Qwen2/InternVL with attn_implementation='eager' or use_flash_attn=False for now."
        )


class Qwen2SdpaAttentionWithAdapters(Qwen2AttentionMixin, Qwen2SdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # TODO: Implement adapter-aware SDPA by porting the original
        # Qwen2SdpaAttention forward and inserting:
        # 1) match/adjust parallel tensor utilities
        # 2) prefix tuning on key/value states
        # 3) SDPA-specific mask and contiguity handling
        raise NotImplementedError(
            "Qwen2SdpaAttentionWithAdapters is not implemented yet. "
            "Load Qwen2 with attn_implementation='eager' for now."
        )


class Qwen2DecoderLayerWithAdapters(Qwen2DecoderLayerMixin, Qwen2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
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
