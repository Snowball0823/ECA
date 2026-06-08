from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class EvaViTConfig(PretrainedConfig):
    model_type = "eva_clip_vit"
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=0.4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=False,
        in_chans=3,
        num_classes=1000,
        qk_scale=None,
        drop_rate=0.,
        attn_drop=0.,
        drop=None,
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        init_scale=0.001,
        act_layer=nn.GELU,
        window_size=None,
        attn_head_dim=None,
        in_features = None,
        hidden_features=None,
        out_features=None,
        proj_drop=None, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        # common config
        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.window_size = window_size
        # ==> use with depth, to setup the different dropout in each block
        self.drop_path_rate = drop_path_rate
        self.embed_dim = embed_dim
        self.dim = embed_dim
        self.hidden_size = embed_dim
        self.drop_rate = drop_rate
        self.norm_layer = norm_layer
        self.init_values = init_values
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        # pacth embedding config
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        # attention config
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.attn_head_dim = attn_head_dim
        self.proj_drop = proj_drop if proj_drop is not None else self.drop_rate
        # MLP config
        self.drop = drop if drop is not None else self.drop_rate
        self.act_layer = act_layer
        self.mlp_ratio = mlp_ratio
        self.in_features = in_features if in_features is not None else self.embed_dim
        if mlp_ratio is not None:
            self.hidden_features = hidden_features if hidden_features is not None else int(self.embed_dim * self.mlp_ratio)
        else:
            self.hidden_features = hidden_features if hidden_features is not None else self.in_features
        self.out_features = out_features if out_features is not None else self.in_features
        # others
        self.use_mean_pooling = use_mean_pooling
        self.num_classes = num_classes
        self.init_scale = init_scale