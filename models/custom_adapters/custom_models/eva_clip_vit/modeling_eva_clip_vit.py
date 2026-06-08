# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CLIP model."""


from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from adapters.composition import (adjust_tensors_for_parallel,
                                  match_attn_matrices_for_parallel)
from torch import nn

from models.ECA_Q.eva_vit import Attention, Block

from .mixin_eva_clip_vit import AttentionAdaptersMixin, BlockAdaptersMixin

# from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer


class AttentionWithAdapters(AttentionAdaptersMixin, Attention):
    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = self.qkv(x)
        qkv = qkv + qkv_bias
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q, k, v = match_attn_matrices_for_parallel(q, k, v)
        k, v, _ = self.prefix_tuning(k, v, x)
        (q,) = adjust_tensors_for_parallel(k, q)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
        

class BlockWithAdapters(BlockAdaptersMixin, Block):
    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            residual = x
            _x = self.norm1(x)
            _x = self.attn(_x, rel_pos_bias=rel_pos_bias)
            _x = self.drop_path(_x)
            _x = self.attention_adapters(_x, residual, layer_norm=None)
            
            residual = _x
            _x = self.norm2(_x)
            _x = self.mlp(_x)
            _x = self.drop_path(_x)
            _x = self.output_adapters(_x, residual, layer_norm=None)
        else:
            residual = x
            _x = self.norm1(x)
            _x = self.attn(_x, rel_pos_bias=rel_pos_bias)
            _x = self.drop_path(self.gamma_1 * _x)
            _x = self.attention_adapters(_x, residual, layer_norm=None)
                
            residual = _x
            _x = self.norm2(_x)
            _x = self.mlp(_x)
            _x = self.drop_path(self.gamma_2 * _x)
            _x = self.output_adapters(_x, residual, layer_norm=None)
        return _x