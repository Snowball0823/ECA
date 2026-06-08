# Based on EVA, BEIT, timm and DeiT code bases
# https://github.com/baaivision/EVA
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from lavis.common.dist_utils import download_cached_file
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import transformers
from transformers.modeling_utils import PreTrainedModel
from .configs import EvaViTConfig


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, config: EvaViTConfig):
        super().__init__()
        self.config = config
        in_features = config.in_features
        out_features = config.out_features
        hidden_features = config.hidden_features
        act_layer = config.act_layer
        drop = config.drop
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: EvaViTConfig):
            # dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            # proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        dim = config.dim
        num_heads = config.num_heads
        attn_head_dim = config.attn_head_dim
        qk_scale = config.qk_scale
        qkv_bias = config.qkv_bias
        window_size = config.window_size if config.use_rel_pos_bias else None
        attn_drop = config.attn_drop
        proj_drop = config.proj_drop

        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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


class Block(nn.Module):

    def __init__(self, config:EvaViTConfig, drop_path=0.):
        super().__init__()
        self.config = config
        dim = config.dim
        norm_layer = config.norm_layer
        init_values = config.init_values
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(config=config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(config=config)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            residual = x
            _x = self.norm1(x)
            _x = self.attn(_x, rel_pos_bias=rel_pos_bias)
            _x = self.drop_path(_x)
            _x = _x + residual
            
            residual = _x
            _x = self.norm2(_x)
            _x = self.mlp(_x)
            _x = self.drop_path(_x)
            _x = _x + residual
        else:
            residual = x
            _x = self.norm1(x)
            _x = self.attn(_x, rel_pos_bias=rel_pos_bias)
            _x = self.drop_path(self.gamma_1 * _x)
            _x = _x + residual
                
            residual = _x
            _x = self.norm2(_x)
            _x = self.mlp(_x)
            _x = self.drop_path(self.gamma_2 * _x)
            _x = _x + residual
        return _x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, config:EvaViTConfig):
        super().__init__()
        self.config = config
        img_size = to_2tuple(config.img_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(config.in_chans, config.embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, config:EvaViTConfig):
        super().__init__()
        self.config = config
        window_size = config.window_size
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww



class EvaVisionTransformerPreTrainedModel(PreTrainedModel):
    config_class = EvaViTConfig
    base_model_prefix = "eva_clip_vit"
    supports_gradient_checkpointing = True




class EvaVisionTransformer(EvaVisionTransformerPreTrainedModel):
    """ 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, config: EvaViTConfig,):
        super().__init__(config)
        self.config = config
        self.image_size = config.img_size
        self.num_classes = config.num_classes
        self.num_features = self.embed_dim = config.embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(config=config)
        if hasattr(config, "window_size"):
            config.window_size = self.patch_embed.patch_shape
        else:
            setattr(config, "window_size", self.patch_embed.patch_shape)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if config.use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(config=config)
        else:
            self.rel_pos_bias = None
        self.use_checkpoint = config.use_checkpoint
        
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = config.use_rel_pos_bias
        self.blocks = nn.ModuleList([Block(config, drop_path=dpr[i]) for i in range(config.depth)])

#         self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
#         self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
#         if isinstance(self.head, nn.Linear):
#             trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()
#         if isinstance(self.head, nn.Linear):
#             self.head.weight.data.mul_(init_scale)
#             self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def active_embedding(self):
        '''
        Usage: only be used after freeze the whole model to activate part of ViT
        '''
        for n, p in self.patch_embed.named_parameters():
            p.requires_grad = True
        self.cls_token.requires_grad = True
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = True
        if self.rel_pos_bias is not None:
            for n, p in self.rel_pos_bias.named_parameters():
                p.requires_grad = True


    def active_block(self, layer_num:int =-1):
        '''
        Usage: only be used after freeze the whole model to activate part of ViT
        '''
        if layer_num == -1:
            for n, p in self.blocks.named_parameters():
                p.requires_grad = True
        else:
            for layer_id, layer in enumerate(self.blocks):
                if layer_id == layer_num:
                    for n, p in layer.named_parameters():
                        p.requires_grad = True


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)
        return x
#         x = self.norm(x)

#         if self.fc_norm is not None:
#             t = x[:, 1:, :]
#             return self.fc_norm(t.mean(1))
#         else:
#             return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
#         x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        return features
    
    def get_num_layer(self, var_name=""):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.blocks)
        
            
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
            
def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

#         if isinstance(l, (nn.MultiheadAttention, Attention)):
#             for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
#                 tensor = getattr(l, attr)
#                 if tensor is not None:
#                     tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)
    
    
def create_eva_vit_g(img_size=224,drop_path_rate=0.4,use_checkpoint=False,precision="fp16"):
    eva_vit_config = EvaViTConfig(img_size=img_size, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint)
    model = EvaVisionTransformer(eva_vit_config)  
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")  
    interpolate_pos_embed(model,state_dict)
    
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    # print(incompatible_keys)
    
    if precision == "fp16":
#         model.to("cuda") 
        convert_weights_to_fp16(model)
    return model