"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
from __future__ import annotations

"""LLaVA language model that accepts optional soft prompt embeddings."""

from typing import List, Optional

import torch
from torch import Tensor

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)


class SoftPromptLlavaForCausalLM(LlavaLlamaForCausalLM):
    """Subclass of LlavaLlamaForCausalLM that prepends soft prompts to the input sequence."""

    def _prepend_soft_prompt(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        labels: Optional[Tensor],
        soft_prompt,
        soft_prompt_mask: Optional[Tensor] = None,
    ):
        if isinstance(soft_prompt, tuple):
            soft_prompt, insert_after = soft_prompt
        else:
            insert_after = 0

        B, S_prompt, _ = soft_prompt.shape
        device = inputs_embeds.device

        front = inputs_embeds[:, :insert_after, :]
        back = inputs_embeds[:, insert_after:, :]
        inputs_embeds = torch.cat([front, soft_prompt, back], dim=1)

        if attention_mask is not None:
            if soft_prompt_mask is None:
                soft_prompt_mask = torch.ones((B, S_prompt), dtype=attention_mask.dtype, device=device)
            front_mask = attention_mask[:, :insert_after]
            back_mask = attention_mask[:, insert_after:]
            attention_mask = torch.cat([front_mask, soft_prompt_mask, back_mask], dim=1)

        if labels is not None:
            prompt_labels = torch.full((B, S_prompt), IGNORE_INDEX, dtype=labels.dtype, device=device)
            front_labels = labels[:, :insert_after]
            back_labels = labels[:, insert_after:]
            labels = torch.cat([front_labels, prompt_labels, back_labels], dim=1)

        if position_ids is not None:
            left = position_ids[:, :insert_after]
            right = position_ids[:, insert_after:]
            offset = left[:, -1:] + 1 if insert_after > 0 else torch.zeros_like(left[:, :1])
            prompt_pos = torch.arange(S_prompt, device=device, dtype=position_ids.dtype).unsqueeze(0) + offset
            position_ids = torch.cat([left, prompt_pos, right + S_prompt], dim=1)

        return inputs_embeds, attention_mask, position_ids, labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        soft_prompt: Optional[Tensor] = None,
        soft_prompt_mask: Optional[Tensor] = None,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        if soft_prompt is not None:
            inputs_embeds, attention_mask, position_ids, labels = self._prepend_soft_prompt(
                inputs_embeds,
                attention_mask,
                position_ids,
                labels,
                soft_prompt,
                soft_prompt_mask,
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        *args,
        input_ids: Optional[Tensor] = None,
        inputs: Optional[Tensor] = None,
        soft_prompt: Optional[Tensor] = None,
        soft_prompt_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        if input_ids is None:
            input_ids = inputs
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)

        if inputs_embeds is None:
            if images is not None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(input_ids)

        if soft_prompt is not None:
            inputs_embeds, attention_mask, position_ids, _ = self._prepend_soft_prompt(
                inputs_embeds,
                attention_mask,
                position_ids,
                None,
                soft_prompt,
                soft_prompt_mask,
            )
                
        return super(LlavaLlamaForCausalLM, self).generate(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
