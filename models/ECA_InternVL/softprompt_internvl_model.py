"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""InternVL chat model extended with optional soft prompt injection."""

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from internvl.model.internvl_chat import InternVLChatModel


class SoftPromptInternVLChatModel(InternVLChatModel):
    """Subclass of InternVLChatModel that prepends optional soft prompts."""

    def _prepend_soft_prompt(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        labels,
        soft_prompt,
        soft_prompt_mask=None,
    ):
        if soft_prompt is None:
            return inputs_embeds, attention_mask, position_ids, labels

        if isinstance(soft_prompt, tuple):
            soft_prompt, insert_after = soft_prompt
        else:
            insert_after = 0

        batch_size, prompt_len, _ = soft_prompt.shape
        device = inputs_embeds.device
        soft_prompt = soft_prompt.to(device=device, dtype=inputs_embeds.dtype)
        if torch.is_tensor(insert_after):
            insert_after = insert_after.to(device=device).view(-1).tolist()
        elif isinstance(insert_after, (list, tuple)):
            insert_after = [int(value) for value in insert_after]
        else:
            insert_after = [int(insert_after)] * batch_size

        if len(insert_after) != batch_size:
            raise ValueError("soft prompt insertion offsets must match batch size.")

        embed_rows = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None
        pos_rows = [] if position_ids is not None else None

        for idx, offset in enumerate(insert_after):
            offset = int(offset)
            front = inputs_embeds[idx, :offset, :]
            back = inputs_embeds[idx, offset:, :]
            embed_rows.append(torch.cat([front, soft_prompt[idx], back], dim=0))

            if attention_mask is not None:
                if soft_prompt_mask is None:
                    row_prompt_mask = torch.ones(
                        (prompt_len,),
                        dtype=attention_mask.dtype,
                        device=device,
                    )
                else:
                    row_prompt_mask = soft_prompt_mask[idx].to(device=device, dtype=attention_mask.dtype)
                front_mask = attention_mask[idx, :offset]
                back_mask = attention_mask[idx, offset:]
                mask_rows.append(torch.cat([front_mask, row_prompt_mask, back_mask], dim=0))

            if labels is not None:
                prompt_labels = torch.full(
                    (prompt_len,),
                    -100,
                    dtype=labels.dtype,
                    device=device,
                )
                front_labels = labels[idx, :offset]
                back_labels = labels[idx, offset:]
                label_rows.append(torch.cat([front_labels, prompt_labels, back_labels], dim=0))

            if position_ids is not None:
                left = position_ids[idx, :offset]
                right = position_ids[idx, offset:]
                if offset > 0:
                    start = left[-1:] + 1
                else:
                    start = torch.zeros((1,), dtype=position_ids.dtype, device=device)
                prompt_pos = torch.arange(
                    prompt_len,
                    device=device,
                    dtype=position_ids.dtype,
                ) + start
                pos_rows.append(torch.cat([left, prompt_pos, right + prompt_len], dim=0))

        inputs_embeds = torch.stack(embed_rows, dim=0)

        if attention_mask is not None:
            attention_mask = torch.stack(mask_rows, dim=0)

        if labels is not None:
            labels = torch.stack(label_rows, dim=0)

        if position_ids is not None:
            position_ids = torch.stack(pos_rows, dim=0)

        return inputs_embeds, attention_mask, position_ids, labels

    def _prepare_multimodal_inputs(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        pixel_values=None,
        visual_features=None,
    ):
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        if pixel_values is None and visual_features is None:
            return input_embeds, False, 0

        if visual_features is not None:
            vit_embeds = visual_features
            vit_batch_size = visual_features.shape[0]
        else:
            vit_embeds = self.extract_feature(pixel_values)
            vit_batch_size = pixel_values.shape[0]

        if image_flags is None:
            image_flags = torch.ones(
                vit_batch_size,
                dtype=torch.long,
                device=vit_embeds.device,
            )
        image_flags = image_flags.squeeze(-1)
        vit_embeds = vit_embeds[image_flags == 1]

        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == self.img_context_token_id
        vit_embeds = vit_embeds.to(device=flat_embeds.device, dtype=flat_embeds.dtype)

        ignore_flag = False
        try:
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, hidden_size)
        except Exception as exc:
            vit_embeds = vit_embeds.reshape(-1, hidden_size)
            print(
                f"warning: {exc}, flat_embeds[selected].shape={flat_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            num_selected = selected.sum()
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds[:num_selected]
            ignore_flag = True

        input_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_size)
        return input_embeds, ignore_flag, vit_batch_size

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        statistics=None,
        loss_weight=None,
        loss_reduction_all_gather=False,
        visual_features=None,
        soft_prompt=None,
        soft_prompt_mask=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds, ignore_flag, vit_batch_size = self._prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            pixel_values=pixel_values,
            visual_features=visual_features,
        )

        if (
            pixel_values is not None
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        ):
            batch_size, seq_len, _ = input_embeds.shape
            print(
                f"dynamic ViT batch size: {vit_batch_size}, "
                f"images per sample: {vit_batch_size / batch_size}, dynamic token length: {seq_len}"
            )
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(
                    f"total_samples={self.num_samples}, {num_samples=}, "
                    f"{num_padding_tokens=}, {num_padding_images=}"
                )

        input_embeds, attention_mask, position_ids, labels = self._prepend_soft_prompt(
            input_embeds,
            attention_mask,
            position_ids,
            labels,
            soft_prompt,
            soft_prompt_mask,
        )

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        soft_prompt=None,
        soft_prompt_mask=None,
        position_ids=None,
        **generate_kwargs,
    ):
        assert self.img_context_token_id is not None

        input_embeds, _, _ = self._prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=None,
            pixel_values=pixel_values,
            visual_features=visual_features,
        )

        input_embeds, attention_mask, position_ids, _ = self._prepend_soft_prompt(
            input_embeds,
            attention_mask,
            position_ids,
            None,
            soft_prompt,
            soft_prompt_mask,
        )

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
