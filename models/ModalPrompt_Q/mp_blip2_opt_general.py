"""
 Copyright (c) 2023, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from adapters import BnConfig, LoRAConfig, MAMConfig, ParBnConfig
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from packaging import version
from torch.cuda.amp import autocast as autocast
from transformers import (AutoTokenizer, CLIPImageProcessor, CLIPTextModel,
                          CLIPTokenizer, CLIPVisionModelWithProjection,
                          OPTConfig, OPTForCausalLM)

from ..custom_adapters import adapter_init
from ..custom_adapters.utils import (Average, freeze_adapter, init_adapter,
                                     print_trainable_parameters)
from .blip2 import Blip2Base, disabled_train
from .utils import freeze_parameters, orthogonal_svd_init, tensor_prompt


@registry.register_model("mp_blip2_opt")
class MPBlip2OPT(Blip2Base):
    """
    ModalPrompt baseline adapted for the BLIP-2 OPT backbone.

    This implementation is adapted from the official ModalPrompt repository:
    https://github.com/AuroraZengfh/ModalPrompt

    It is integrated with the training, checkpointing, and backbone interfaces
    used in this repository.

    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "cl_caption_coco_opt2.7b": "configs/models/pa_blip2_caption_opt2.7b.yaml",
        "cl_caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
        "cl_vqa_coco_opt2.7b": "configs/models/pa_blip2_vqa_opt2.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=32,
        apply_lemmatizer=False,
        qformer_text_input=True,
        mh_pa_r=25.0,
        mh_pa_drop_out=0.0,
        mh_pa_s=4.0,
        ffn_pa_r=1.5,
        ffn_pa_drop_out=0.0,
        ffn_pa_s=4.0,
        r=8,
        alpha=16,
        mix_query=True,
        kd_weight=1.0,
        ortho=True,
        ortho_weight=0.1,
        use_modal_prompt=False,
        mp_prefix_len=10,
        mp_transfer_num=3,
        mp_lam=0.5,
        mp_loss_weight=1.0,
        mp_clip_model_name="openai/clip-vit-large-patch14",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.freeze_vit = freeze_vit
        if freeze_vit:
            self.freeze_visual_encoder_ln(disable_training=True)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        if not qformer_text_input:
            # decrease the model size
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        # max_txt_len for question length.
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        # that prompt is only for caption while eval, while training is only for mask.
        # 1. the prompt for training, both vqa and caption, are set in the dataset.
        # 2. the prompt for eval, vqa is set in run/task, caption set here.
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None  

        self.qformer_text_input = qformer_text_input
        # for adapter
        self.parallel_adapters_dict = defaultdict(list)
        self.adapter_init = False
        # => rank = dim_in/r, adapter: dim_in, rank, relu, rank, dim_out
        self.mh_pa_r = mh_pa_r
        self.mh_pa_drop_out = mh_pa_drop_out
        self.mh_pa_s = mh_pa_s
        self.ffn_pa_r = ffn_pa_r
        self.ffn_pa_drop_out = ffn_pa_drop_out
        self.ffn_pa_s = ffn_pa_s
        # for adapter configs
        self.attn_pa_config = ParBnConfig(mh_adapter=True, output_adapter=False,
                                          reduction_factor=self.mh_pa_r, dropout=self.mh_pa_drop_out, scaling=self.mh_pa_s, non_linearity='linear')
        self.ffn_pa_config = ParBnConfig(
            reduction_factor=self.ffn_pa_r, dropout=self.ffn_pa_drop_out, scaling=self.ffn_pa_s, non_linearity='linear')
        self.mh_pa_prefix = "pa_adapter_attn_"
        self.ffn_pa_prefix = "pa_adapter_ffn_"

        # for lora
        self.lora_dict = defaultdict(list)
        self.visual_lora_init = False
        # => rank = r; s = alpha/r, lora: dim_in, rank, rank, dim_out
        # original is r=8, aplha=8/16/32
        self.r = r
        self.alpha = alpha
        # for lora configs
        self.visual_lora_config = LoRAConfig(r=self.r, alpha=self.alpha, intermediate_lora=True, output_lora=True)
        self.lora_prefix = "lora_"
        # for MoQ
        self.mix_query = mix_query
        # for kd weight
        self.kd_weight = kd_weight
        # for ortho
        self.ortho = ortho
        self.ortho_weight = ortho_weight
        # for saving
        self.unchange_keys = list()
        # ModalPrompt
        self.use_modal_prompt = use_modal_prompt
        self.mp_prefix_len = mp_prefix_len
        self.mp_transfer_num = mp_transfer_num
        self.mp_lam = mp_lam
        self.mp_loss_weight = mp_loss_weight
        self.mp_clip_model_name = mp_clip_model_name
        self.mp_prompt_tokens: List[List[str]] = []
        self.mp_prompt_transform = nn.ModuleList()
        self.mp_prompt_embeddings = nn.ParameterList()
        self.mp_current_task: Optional[int] = None
        self._mp_clip_ready = False
        self.mp_clip_processor = None
        self.mp_clip_vision = None
        self.mp_clip_text = None
        self.mp_clip_tokenizer = None


    @property
    def nlp_proj(self):
        return self.opt_proj

    @property
    def adapters(self):
        _adapters = []
        for name_list in self.parallel_adapters_dict.values():
            _adapters += name_list
        return _adapters
    
    @property
    def loras(self):
        _loras = []
        for name_list in self.lora_dict.values():
            _loras += name_list
        return _loras

    # ------------------------------------------------------------------
    # ModalPrompt helpers
    def _init_modal_prompt_modules(self):
        if self._mp_clip_ready:
            return
        self.mp_clip_processor = CLIPImageProcessor.from_pretrained(self.mp_clip_model_name)
        self.mp_clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            self.mp_clip_model_name
        )
        self.mp_clip_text = CLIPTextModel.from_pretrained(self.mp_clip_model_name)
        self.mp_clip_tokenizer = CLIPTokenizer.from_pretrained(self.mp_clip_model_name)
        for module in (self.mp_clip_vision, self.mp_clip_text):
            module.requires_grad_(False)
            module.eval()
        self._mp_clip_ready = True

    def _mp_ensure_clip_device(self, device: torch.device):
        if not self._mp_clip_ready:
            self._init_modal_prompt_modules()
        if self.mp_clip_vision.device != device:
            self.mp_clip_vision.to(device)
        if self.mp_clip_text.device != device:
            self.mp_clip_text.to(device)

    def _mp_add_new_task_prompt(self, task_id: int):
        if self.mp_prefix_len <= 0:
            return
        prompt_name = f"PRE{task_id + 1}_"
        tokens_list = [f"[{prompt_name}{i}]" for i in range(1, self.mp_prefix_len + 1)]
        self.mp_prompt_tokens.append(tokens_list)

        hidden_dim = self.query_tokens.size(-1)
        with torch.no_grad():
            source = self.query_tokens.detach().float().reshape(-1, hidden_dim)
            if source.numel() == 0:
                init_embeds = torch.randn(self.mp_prefix_len, hidden_dim, device=self.device) * 0.02
            else:
                rand_idx = torch.randint(
                    0,
                    source.size(0),
                    (self.mp_prefix_len,),
                    device=source.device,
                )
                init_embeds = source[rand_idx].clone()
        prompt_embeds = nn.Parameter(init_embeds)
        self.mp_prompt_embeddings.append(prompt_embeds)
        proj_dim = 768
        if self._mp_clip_ready and self.mp_clip_text is not None:
            proj_dim = int(self.mp_clip_text.config.projection_dim)
        transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.mp_prompt_transform.append(transform)
        self.mp_prompt_transform.to(device=self.device, dtype=torch.float32)
        self.mp_prompt_embeddings.to(device=self.device, dtype=torch.float32)

    def _mp_rebuild_prompts(self, prompt_tokens: List[List[str]]):
        self.mp_prompt_tokens = []
        self.mp_prompt_transform = nn.ModuleList()
        self.mp_prompt_embeddings = nn.ParameterList()
        if not prompt_tokens:
            return
        if self.mp_prefix_len <= 0:
            self.mp_prefix_len = len(prompt_tokens[0])
        for tokens in prompt_tokens:
            self.mp_prompt_tokens.append(tokens)
            hidden_dim = self.query_tokens.size(-1)
            with torch.no_grad():
                source = self.query_tokens.detach().float().reshape(-1, hidden_dim)
                if source.numel() == 0:
                    init_embeds = torch.randn(self.mp_prefix_len, hidden_dim, device=self.device) * 0.02
                else:
                    rand_idx = torch.randint(
                        0,
                        source.size(0),
                        (self.mp_prefix_len,),
                        device=source.device,
                    )
                    init_embeds = source[rand_idx].clone()
            prompt_embeds = nn.Parameter(init_embeds)
            self.mp_prompt_embeddings.append(prompt_embeds)
            proj_dim = 768
            if self._mp_clip_ready and self.mp_clip_text is not None:
                proj_dim = int(self.mp_clip_text.config.projection_dim)
            transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, proj_dim),
            )
            self.mp_prompt_transform.append(transform)
        self.mp_prompt_transform.to(device=self.device, dtype=torch.float32)
        self.mp_prompt_embeddings.to(device=self.device, dtype=torch.float32)

    def _mp_set_current_task(self, cur_task: int):
        if not self.mp_prompt_transform:
            return
        self.mp_current_task = cur_task
        for idx, module in enumerate(self.mp_prompt_transform):
            for p in module.parameters():
                p.requires_grad = idx == cur_task
        for idx, param in enumerate(self.mp_prompt_embeddings):
            param.requires_grad = idx == cur_task
        self.mp_prompt_transform.to(device=self.device, dtype=torch.float32)
        self.mp_prompt_embeddings.to(device=self.device, dtype=torch.float32)

    def _mp_log_parameter_summary(self):
        total_token_params = sum(p.numel() for p in self.mp_prompt_embeddings)
        trainable_token_params = sum(
            p.numel() for p in self.mp_prompt_embeddings if p.requires_grad
        )
        total_transform_params = sum(
            p.numel() for p in self.mp_prompt_transform.parameters()
        )
        trainable_transform_params = sum(
            p.numel() for p in self.mp_prompt_transform.parameters() if p.requires_grad
        )
        logging.info(
            "ModalPrompt params: tokens=%d (trainable=%d), transform=%d (trainable=%d)",
            total_token_params,
            trainable_token_params,
            total_transform_params,
            trainable_transform_params,
        )

    def _mp_encode_image(self, images: torch.Tensor):
        self._mp_ensure_clip_device(images.device)
        pixel_values = images
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()
        expected = int(self.mp_clip_vision.config.image_size)
        if pixel_values.shape[-1] != expected or pixel_values.shape[-2] != expected:
            pixel_values = F.interpolate(
                pixel_values, size=(expected, expected), mode="bilinear", align_corners=False
            )
        pixel_values = pixel_values.to(dtype=self.mp_clip_vision.dtype, device=self.mp_clip_vision.device)
        with torch.no_grad():
            outputs = self.mp_clip_vision(pixel_values=pixel_values)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def _mp_encode_text(self, texts: List[str]):
        self._mp_ensure_clip_device(self.device)
        tokenized = self.mp_clip_tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.mp_clip_text.device)
        with torch.no_grad():
            outputs = self.mp_clip_text(**tokenized)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def _mp_compute_prototypes(self, device: torch.device):
        if not self.mp_prompt_embeddings:
            return torch.empty(0, 768, device=device)
        prompt_embeds = torch.stack([emb for emb in self.mp_prompt_embeddings], dim=0).to(device)
        pooled = prompt_embeds.mean(dim=1)
        prototypes = []
        for idx, transform in enumerate(self.mp_prompt_transform):
            proto = transform(pooled[idx])
            prototypes.append(proto)
        proto_embeddings = torch.stack(prototypes, dim=0)
        return proto_embeddings

    def _mp_select_prompt_ids(
        self,
        proto_embeddings: torch.Tensor,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        training: bool,
    ):
        if proto_embeddings.numel() == 0:
            return None
        num_tasks = proto_embeddings.size(0)
        batch_size = image_feats.size(0)
        lam = float(self.mp_lam)
        img_norm = F.normalize(image_feats, dim=-1)
        txt_norm = F.normalize(text_feats, dim=-1)
        proto_norm = F.normalize(proto_embeddings, dim=-1)
        guide_coef = lam * torch.matmul(img_norm, proto_norm.t()) + (1.0 - lam) * torch.matmul(
            txt_norm, proto_norm.t()
        )
        k = max(int(self.mp_transfer_num), 1)
        if training:
            if num_tasks <= k:
                selected_idx = torch.arange(num_tasks, device=guide_coef.device).unsqueeze(0).expand(batch_size, -1)
            else:
                if k <= 1:
                    selected_idx = torch.full((batch_size, 1), num_tasks - 1, device=guide_coef.device, dtype=torch.long)
                else:
                    hist_scores = guide_coef[:, :-1]
                    topk_idx = torch.topk(hist_scores, k - 1, dim=-1)[1]
                    topk_idx = torch.flip(topk_idx, dims=[1])
                    current_idx = torch.full((batch_size, 1), num_tasks - 1, device=guide_coef.device, dtype=torch.long)
                    selected_idx = torch.cat([topk_idx, current_idx], dim=1)
        else:
            if num_tasks <= k:
                selected_idx = torch.arange(num_tasks, device=guide_coef.device).unsqueeze(0).expand(batch_size, -1)
            else:
                topk_idx = torch.topk(guide_coef, k, dim=-1)[1]
                selected_idx = torch.flip(topk_idx, dims=[1])
        return selected_idx

    def _mp_prepare_prompts(self, images: torch.Tensor, text_inputs: List[str], training: bool):
        if not self.use_modal_prompt or not self.mp_prompt_embeddings:
            return None, torch.tensor(0.0, device=images.device)
        image_feats = self._mp_encode_image(images)
        text_feats = self._mp_encode_text(text_inputs)
        proto_embeddings = self._mp_compute_prototypes(image_feats.device)
        selected_prompt_idx = self._mp_select_prompt_ids(
            proto_embeddings, image_feats, text_feats, training=training
        )
        mp_loss = torch.tensor(0.0, device=image_feats.device)
        if training and self.mp_loss_weight > 0:
            current_idx = (
                self.mp_current_task
                if self.mp_current_task is not None
                else proto_embeddings.size(0) - 1
            )
            if current_idx >= 0 and current_idx < proto_embeddings.size(0):
                proto_current = proto_embeddings[current_idx].unsqueeze(0)
                img_loss = 1.0 - F.cosine_similarity(image_feats, proto_current, dim=-1)
                txt_loss = 1.0 - F.cosine_similarity(text_feats, proto_current, dim=-1)
                mp_loss = (img_loss + txt_loss).mean()
        if selected_prompt_idx is None:
            return None, mp_loss
        prompt_bank = torch.stack([emb for emb in self.mp_prompt_embeddings], dim=0).to(image_feats.device)
        selected = prompt_bank[selected_prompt_idx]
        prompt_embeds = selected.reshape(image_feats.size(0), -1, prompt_bank.size(-1))
        return prompt_embeds, mp_loss

    def _mp_get_text_inputs(self, samples: dict, batch_size: int):
        if "text_input" in samples:
            text_input = samples["text_input"]
            if isinstance(text_input, str):
                text_input = [text_input]
            if "ocr_input" in samples:
                text_input = [
                    " ".join([samples["ocr_input"][i], text_input[i]]).strip()
                    for i in range(len(text_input))
                ]
            return text_input
        if "prompt" in samples:
            prompt = samples["prompt"]
            if isinstance(prompt, list):
                return prompt
            return [prompt for _ in range(batch_size)]
        if self.prompt:
            return [self.prompt for _ in range(batch_size)]
        return ["" for _ in range(batch_size)]
    

    @property
    def current_adapter_names(self):
        adapters_num = len(self.parallel_adapters_dict)
        return self.parallel_adapters_dict[adapters_num-1] if adapters_num>0 else []
    

    @property
    def current_adapter_index(self):
        return len(self.parallel_adapters_dict)-1 if len(self.parallel_adapters_dict)>0 else -1
    

    @property
    def visual_encoder_freezed(self):
        _status = True
        for n, p in self.visual_encoder.named_parameters():
            if p.requires_grad:
                _status = False
                break
        return _status


    @property
    def ln_freezed(self):
        _status = True
        for _, p in self.ln_vision.named_parameters():
            if p.requires_grad:
                _status = False
                break
        return _status
    

    @property
    def qformer_embedding_freezed(self):
        _status = True
        for _, p in self.Qformer.bert.embeddings.named_parameters():
            if p.requires_grad:
                _status = False
                break
        return _status
    
    
    @property
    def moq_num(self):
        new_num = len(getattr(self, "current_keys", torch.tensor([], dtype=torch.float)))
        old_num = len(getattr(self, "keys_set", torch.tensor([], dtype=torch.float)))
        return new_num+old_num

    @property
    def adapter_structure(self):
        structure = {
            'lora': self.lora_dict,
            'adapter': self.parallel_adapters_dict,
            'v_freeze': self.visual_encoder_freezed,
            'ln_freeze': self.ln_freezed,
            'q_embedding_freeze': self.qformer_embedding_freezed,
            'moq_num': self.moq_num,
        }
        if self.use_modal_prompt or self.mp_prompt_tokens:
            structure["modal_prompt"] = {
                "enabled": self.use_modal_prompt,
                "prefix_len": self.mp_prefix_len,
                "transfer_num": self.mp_transfer_num,
                "lam": self.mp_lam,
                "loss_weight": self.mp_loss_weight,
                "task_count": len(self.mp_prompt_tokens),
                "prompt_tokens": deepcopy(self.mp_prompt_tokens),
            }
        return structure

    @property
    def moq_old_kv(self):
        device = self.query_tokens.device
        keys = getattr(self, 'keys_set', torch.tensor([], device=device, dtype=torch.float))
        queries = getattr(self, 'queries_set', torch.tensor([], device=device, dtype=torch.float))
        return {'keys_set': deepcopy(keys.clone().detach().cpu()), 'queries_set': deepcopy(queries.clone().detach().cpu())}


    def freeze_visual_encoder_ln(self, disable_training=False):
        # Add self._status change
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        if disable_training:
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
        logging.info("freeze vision encoder")
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        if disable_training:
            self.ln_vision.eval()
            self.ln_vision.train = disabled_train
        logging.info("freeze layer norm after vision encoder")


    def freeze_qformer_proj(self):
        # may not be used
        for name, param in self.opt_proj.named_parameters():
            param.requires_grad = False


    def freeze_qformer_embedding(self, freeze=True):
        for name, param in self.Qformer.bert.embeddings.named_parameters():
            param.requires_grad = not freeze


    def get_optimizer_params(self, weight_decay, lr_scale=1):
        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = [lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2)]

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_tag = "no_decay"
                this_weight_decay = 0.0
            else:
                group_tag = "decay"
                this_weight_decay = weight_decay

            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.', ''))
                group_name = f"vit_layer_{layer_id}_{group_tag}"
            else:
                layer_id = None
                group_name = group_tag

            if group_name not in parameter_group_names:
                scale = lr_scales[layer_id] if layer_id is not None else 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        optim_params = list(parameter_group_vars.values())

        grouped_named_params = {
            "projector": [],
            "adapter": [],
            "vision_lora": [],
            "moq": [],
            "modal_prompt": [],
            "other": [],
        }

        adapter_prefixes = (self.mh_pa_prefix, self.ffn_pa_prefix)
        vision_lora_prefix = self.lora_prefix

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "opt_proj" in name:
                grouped_named_params["projector"].append((name, param))
            elif any(prefix in name for prefix in adapter_prefixes):
                grouped_named_params["adapter"].append((name, param))
            elif vision_lora_prefix and vision_lora_prefix in name:
                grouped_named_params["vision_lora"].append((name, param))
            elif name in {"current_keys", "current_queries"}:
                grouped_named_params["moq"].append((name, param))
            elif self.use_modal_prompt and (
                "mp_prompt_embeddings" in name or "mp_prompt_transform" in name
            ):
                grouped_named_params["modal_prompt"].append((name, param))
            else:
                grouped_named_params["other"].append((name, param))

        def summarize(tag, entries):
            if not entries:
                return
            total_params = sum(param.numel() for _, param in entries)
            names = [name for name, _ in entries[:3]]
            if len(entries) > 3:
                names.append("...")
            layer_ids = set()
            for name, _ in entries:
                for token in (".layer.", ".layers.", ".block.", ".blocks."):
                    if token in name:
                        try:
                            idx = int(name.split(token, 1)[1].split(".", 1)[0])
                            layer_ids.add(idx)
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
        summarize("modal_prompt", grouped_named_params["modal_prompt"])

        other_entries = grouped_named_params["other"]
        if other_entries:
            warning_msg = (
                "Unexpected trainable params found outside projector/adapters/vision_lora/MoQ/ModalPrompt. "
                "Backbone appears to be updating."
            )
            logging.warning(warning_msg)
            print(f"WARNING: {warning_msg}")
            summarize("other", other_entries)

        return optim_params


    def before_training(self, expand_q_former=True, lora_visual=False, **kwargs):
        super().before_training(**kwargs)
        if len(self.unchange_keys)==0:
            self.unchange_keys += [k for k in self.state_dict() if 'opt' in k and 'opt_proj' not in k]

        if expand_q_former:
            self.expand_q_adapter()

        if self.qformer_text_input:
            if lora_visual:
                self.freeze_qformer_embedding(freeze=False)
            else:
                self.freeze_qformer_embedding()

        if lora_visual:
            assert not self.freeze_vit, "Freeze the ViT alreday, set `freeze_vit=False` at mode yaml first."
            self.expand_visual_lora()
        elif (not lora_visual) and len(self.lora_dict)>0 and (not self.ln_freezed):
            self.freeze_visual_encoder_ln(disable_training=True)

        if self.use_modal_prompt and self.mix_query:
            logging.warning(
                "ModalPrompt and MoQ are both enabled. This is a hybrid setting, "
                "not the pure ModalPrompt baseline."
            )

        if self.mix_query:
            self.query_mixture()

        if self.use_modal_prompt:
            if not self._mp_clip_ready:
                self._init_modal_prompt_modules()
            cur_task = 0 if self.mp_current_task is None else self.mp_current_task + 1
            while len(self.mp_prompt_embeddings) <= cur_task:
                self._mp_add_new_task_prompt(len(self.mp_prompt_embeddings))
            self._mp_set_current_task(cur_task)
            self._mp_log_parameter_summary()


    def rebuild_from_config(self, adapter_structure: dict, moq_old_kv: dict, **kwargs):
        if len(adapter_structure['lora'])>0:
            assert not self.freeze_vit, "Freeze the ViT alreday, set `freeze_vit=False` at mode yaml first."
            for _ in adapter_structure['lora']:
                self.expand_visual_lora()
        if adapter_structure['ln_freeze']:
            self.freeze_visual_encoder_ln(disable_training=True)

        if adapter_structure['q_embedding_freeze']:
            self.freeze_qformer_embedding()
        else:
            self.freeze_qformer_embedding(freeze=False)
        
        if len(adapter_structure['adapter'])>0:
            for _ in adapter_structure['adapter']:
                self.expand_q_adapter()

        if adapter_structure['moq_num']>0:
            for _ in range(adapter_structure['moq_num']):
                self.query_mixture(svd_init=False)
            if hasattr(self, 'keys_set'):
                keys_set = getattr(self, 'keys_set')
                for key in moq_old_kv:
                    setattr(self, key, moq_old_kv[key].to(keys_set.device))

        modal_info = adapter_structure.get("modal_prompt", {})
        if modal_info:
            self.use_modal_prompt = modal_info.get("enabled", self.use_modal_prompt)
            self.mp_prefix_len = modal_info.get("prefix_len", self.mp_prefix_len)
            self.mp_transfer_num = modal_info.get("transfer_num", self.mp_transfer_num)
            self.mp_lam = modal_info.get("lam", self.mp_lam)
            self.mp_loss_weight = modal_info.get("loss_weight", self.mp_loss_weight)
            if self.use_modal_prompt and not self._mp_clip_ready:
                self._init_modal_prompt_modules()
            prompt_tokens = modal_info.get("prompt_tokens", [])
            if prompt_tokens:
                self._mp_rebuild_prompts(prompt_tokens)
                self.mp_current_task = len(self.mp_prompt_embeddings) - 1


    def expand_q_adapter(self):
        if not self.adapter_init:
            adapter_init(self.Qformer.bert, use_customize=True)
            self.Qformer.bert.freeze_model()
            self.adapter_init = True
        adapter_num = len(self.parallel_adapters_dict)
        if adapter_num > 0:
            freeze_adapter(self.Qformer.bert, self.adapters)
        attn_pa_name = self.mh_pa_prefix+str(adapter_num)
        ffn_pa_name = self.ffn_pa_prefix+str(adapter_num)
        # add adapter name in dict
        self.parallel_adapters_dict[adapter_num] += [attn_pa_name, ffn_pa_name]
        self.Qformer.bert.add_adapter(attn_pa_name, config=self.attn_pa_config)
        self.Qformer.bert.add_adapter(ffn_pa_name, config=self.ffn_pa_config)
        # self.Qformer.bert.active_adapters = self.adapters
        attn_pa_names = [self.parallel_adapters_dict[num][0] for num in self.parallel_adapters_dict]
        ffn_pa_names = [self.parallel_adapters_dict[num][1] for num in self.parallel_adapters_dict]
        self.Qformer.bert.active_adapters = [Average(*attn_pa_names, weights=[1]*len(
            attn_pa_names)), Average(*ffn_pa_names, weights=[1]*len(ffn_pa_names))]
        if adapter_num > 0:
            pa_names = [self.parallel_adapters_dict[i] for i in range(adapter_num)]
            init_adapter(self.Qformer.bert, self.parallel_adapters_dict[adapter_num], [list(item) for item in zip(*pa_names)])
        freeze_adapter(self.Qformer.bert,
                       self.parallel_adapters_dict[adapter_num], freeze=False)
        adapter_sum = self.Qformer.bert.adapter_summary()
        print_trainable_parameters(self.Qformer)
        logging.info('\n'+adapter_sum)


    def expand_visual_lora(self):
        if not self.visual_lora_init:
            adapter_init(self.visual_encoder, use_customize=True)
            self.visual_encoder.freeze_model()
            self.visual_lora_init = True
        lora_num = len(self.lora_dict)
        if lora_num > 0:
            freeze_adapter(self.visual_encoder, self.loras)
        lora_name = self.lora_prefix+str(lora_num)
        self.lora_dict[lora_num] += [lora_name]
        self.visual_encoder.add_adapter(lora_name, config=self.visual_lora_config)
        self.visual_encoder.active_adapters = self.loras
        freeze_adapter(self.visual_encoder, self.lora_dict[lora_num], freeze=False)
        self.visual_encoder.active_embedding()
        lora_sum = self.visual_encoder.adapter_summary()
        print_trainable_parameters(self.visual_encoder)
        logging.info('\n'+lora_sum)


    @torch.no_grad()
    def merge_lora(self):
        for lora in self.loras:
            self.visual_encoder.merge_adapter(lora)
        self.lora_dict.clear()


    def query_mixture(self, svd_init=True):
        device = self.query_tokens.device
        freeze_parameters(self, ['query_tokens'])
        # b=1
        b, q_num, dim = self.query_tokens.shape
        key_dim = self.visual_encoder.config.hidden_size
        # new_keys = torch.FloatTensor(b, key_dim)
        # new_keys = torch.empty(b, key_dim, device=device, dtype=torch.float)
        new_keys = torch.zeros(b, key_dim, device=device, dtype=torch.float)
        # nn.init.uniform_(new_keys)
        # nn.init.orthogonal_(new_keys)
        # new_queries = torch.FloatTensor(b, q_num, dim)
        # new_queries = torch.empty(b, q_num, dim, device=device, dtype=torch.float)
        new_queries = torch.zeros(b, q_num, dim, device=device, dtype=torch.float)
        # nn.init.orthogonal_(new_queries)
        if hasattr(self, 'current_keys') and hasattr(self, 'current_queries'):
            base_keys = getattr(self, 'keys_set', torch.tensor([], device=device, dtype=torch.float))
            base_queries = getattr(self, 'queries_set', torch.tensor([], device=device, dtype=torch.float))
            base_keys = torch.cat([base_keys, self.current_keys.detach().clone()], dim=0)
            base_queries = torch.cat([base_queries, self.current_queries.detach().clone()], dim=0)
            setattr(self, 'keys_set', base_keys)
            setattr(self, 'queries_set', base_queries)
            if svd_init:
                # using svd to init as orthogonal
                new_key_init = orthogonal_svd_init(base_keys)
                new_keys.data.copy_(new_key_init)

        self.current_keys = nn.Parameter(new_keys)
        self.current_queries = nn.Parameter(new_queries)


    def forward(self, samples: dict):
        feature_diction, query_diction = None, None
        if 'feature' in samples and 'query' in samples:
            feature_diction = samples['feature']
            query_diction = samples['query']
            pre_query_tokens = samples['query_token']

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        mp_loss = torch.tensor(0.0, device=image.device)
        modal_prompt = None
        if self.use_modal_prompt and self.mp_prompt_embeddings:
            text_inputs = self._mp_get_text_inputs(samples, image.size(0))
            modal_prompt, mp_loss = self._mp_prepare_prompts(
                images=image,
                text_inputs=text_inputs,
                training=self.training,
            )

        # Q-Former part
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.mixture_of_query(image_embeds, self.query_tokens)
        if modal_prompt is not None:
            modal_prompt = modal_prompt.to(dtype=query_tokens.dtype, device=query_tokens.device)
            query_tokens = torch.cat([modal_prompt, query_tokens], dim=1)
        if not self.qformer_text_input:
            # caption case
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            # VQA and TextCap case
            # ==> input: question, max_txt_len=128, answer, max_output_txt_len=32; input: caption, max_output_txt_len=32
            assert "text_input" in samples, 'The data must has `text_input`.'

            if 'ocr_input' in samples:
                text_input = [' '.join([samples['ocr_input'][i], samples["text_input"][i]]).strip() for i in range(len(samples["text_input"]))]
            else:
                text_input = samples["text_input"]

            text_Qformer = self.tokenizer(
                text_input,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(self.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )


        # LLM part
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        # ==> if caption case: text_output==prompt+caption; vqa case: text_input==question, text_output==answer
        ## '\n' is the special token of opt

        if "text_input" not in samples:
            # caption case
            text = [t + "\n" for t in samples["text_output"]]

            opt_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)

            targets = opt_tokens.input_ids.masked_fill(
                opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if len(self.prompt)>0:
                targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

            empty_targets = (
                torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        else:
            # vqa case or textcaps
            text = [t + "\n" for t in samples["text_output"]]

            if 'ocr_input' in samples:
                text_input = [' '.join([samples['ocr_input'][i], samples["text_input"][i]]).strip() for i in range(len(samples["text_input"]))]
            else:
                text_input = samples["text_input"]

            opt_input_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            opt_output_tokens = self.opt_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(image.device)
            targets = opt_output_tokens.input_ids.masked_fill(
                opt_output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if len(self.prompt)>0:
                targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

            query_empty_targets = (
                torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
            )
            question_empty_targets = opt_input_tokens.input_ids.masked_fill(
                opt_input_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            targets = torch.cat([query_empty_targets, question_empty_targets, targets], dim=1)

            inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_input_tokens.input_ids)
            outputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_output_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds, outputs_embeds], dim=1)

            attention_mask = torch.cat([atts_opt, opt_input_tokens.attention_mask, opt_output_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss
        # Knowledge Distillation
        mse_loss = 0
        # MoQ Ortho
        ortho_loss = 0
        key_task_loss = 0
        if feature_diction is not None and query_diction is not None:
            query_connect, feature_query_tokens = self.extract_query_dictionary(feature_diction)
            mse_loss_fn = nn.MSELoss()
            mse_loss = mse_loss_fn(query_connect, query_diction)

            # mse_loss += mse_loss_fn(feature_query_tokens, pre_query_tokens)
            loss = loss + self.kd_weight*mse_loss
        if self.mix_query and self.ortho:
            previous_keys = getattr(self, 'keys_set', torch.tensor([]).to(self.device))
            previous_queries = getattr(self, 'queries_set', torch.tensor([]).to(self.device))
            # for MoQ L2 norm weight
            l2_weight = 5e-4
            # ortho loss for keys
            key_ortho_loss = 0
            _, k_dim = self.current_keys.shape
            previous_keys = previous_keys.view(-1, k_dim)
            key_gram_matrix = torch.einsum('bd,nd->bn', self.current_keys, previous_keys)
            # key_ortho_loss = torch.norm(key_gram_matrix, p='fro')
            # key_gram_matrix = torch.einsum('bd,nd->bn', F.normalize(self.current_keys, p=2, dim=-1), F.normalize(previous_keys, p=2, dim=-1))
            key_ortho_loss = torch.norm(key_gram_matrix, p='fro')**2
            # key_ortho_loss = key_ortho_loss + l2_weight*torch.norm(self.current_keys, 2)
            # key task loss
            task_keys = torch.cat([previous_keys, self.current_keys], dim=0)
            img_patches = torch.mean(image_embeds[:,1:,:], dim=1).detach()
            # task_attention_scores = F.softmax(scaled_logits, dim=-1)
            # for cos sim
            # img_norm = F.normalize(img_patches, p=2, dim=-1)
            # keys_norm = F.normalize(task_keys, p=2, dim=-1)
            # task_attention_scores = torch.einsum('bd,nd->bn', img_norm, keys_norm)
            # bs, task_num = task_attention_scores.shape
            # task_attention_scores += 1
            # if task_num == 1:
            #     task_attention_scores = torch.cat([task_attention_scores, 2-task_attention_scores], dim=-1)
            # direct cos opt
            img_patches = torch.mean(image_embeds[:,1:,:], dim=1).detach()
            img_norm = F.normalize(img_patches, p=2, dim=-1)
            keys_norm = F.normalize(self.current_keys, p=2, dim=-1)
            task_attention_scores = torch.einsum('bd,nd->bn', img_norm, keys_norm)
            key_task_loss = torch.mean(1-task_attention_scores)

            # ortho loss for queries
            _, _, q_dim = self.current_queries.shape
            query_ortho_loss = 0
            previous_queries = previous_queries.view(1, -1, q_dim)
            query_gram_matrix = torch.einsum('bqd,bad->qa', self.current_queries, previous_queries)
            # query_gram_matrix = torch.einsum('bqd,bad->qa', F.normalize(self.current_queries, p=2, dim=-1), F.normalize(previous_queries, p=2, dim=-1))
            query_ortho_loss = torch.norm(query_gram_matrix, p='fro')**2
            
            ortho_loss = key_ortho_loss+query_ortho_loss

            loss = loss + self.ortho_weight*(ortho_loss+key_task_loss)

        if self.use_modal_prompt and self.mp_loss_weight > 0:
            loss = loss + self.mp_loss_weight * mp_loss

        return {
            "loss": loss,
            "output loss": outputs.loss,
            "DR loss": mse_loss,
            "MoQ loss": (ortho_loss + key_task_loss),
            "MP loss": mp_loss,
        }
    

    def mixture_of_query(self, image_embeds: torch.Tensor, query_tokens: nn.Parameter, old_only=False):
        if hasattr(self, 'current_keys') and hasattr(self, 'current_queries'):
            task_keys = getattr(self, 'keys_set', torch.tensor([]).to(self.device))
            task_queries = getattr(self, 'queries_set', torch.tensor([]).to(self.device))
            if not old_only:
                task_keys = torch.cat([task_keys, self.current_keys], dim=0)
                task_queries = torch.cat([task_queries, self.current_queries], dim=0)
            
            # Step 1: Compute attention scores using mean of image patches and task keys
            _, p, _ = image_embeds.shape
            if p>1 and (int((p-1)**0.5)**2 == (p-1)):
                 img_patches = image_embeds[:,1:,:]
            img_patches = torch.mean(image_embeds, dim=1)
            


            # get cos sim
            img_norm = F.normalize(img_patches, p=2, dim=-1)
            keys_norm = F.normalize(task_keys, p=2, dim=-1)
            scaled_logits = torch.einsum('bd,nd->bn', img_norm, keys_norm)

            # get task_attention_scores
            task_attention_scores = F.softmax(scaled_logits, dim=1)

            
            # Step 3: Apply attention task attention scores to task queries to get delta query
            # [bs, n] @ [n, q_num, dim_b] -> [bs, q_num, dim_b]
            delta_query = torch.einsum('bn,nqd->bqd', task_attention_scores, task_queries)
            
            _query_tokens = query_tokens + delta_query  # [bs, q_num, dim_b]
        else:
            _query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1) # [bs, q_num, dim]
        return _query_tokens
    

    def extract_query_dictionary(self, feature_dictionary: torch.Tensor):
        if len(feature_dictionary.shape)==2:
            feature_dictionary = feature_dictionary.unsqueeze(1)
        feature_atts = torch.ones(feature_dictionary.size()[:-1], dtype=torch.long).to(
            feature_dictionary.device
        )

        query_tokens = self.mixture_of_query(feature_dictionary, self.query_tokens, old_only=False)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=feature_dictionary,
            encoder_attention_mask=feature_atts,
            return_dict=True,
        )
        query_connect = self.opt_proj(query_output.last_hidden_state)
        return query_connect, query_tokens
    

    @torch.no_grad()
    def extract_visual_feature(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        return {'feature':image_embeds}


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        **kwargs
    ):
        """
        NOTE:
            This function is for genertaing the caption specially.
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.mixture_of_query(image_embeds, self.query_tokens)
            modal_prompt = None
            if self.use_modal_prompt and self.mp_prompt_embeddings:
                text_inputs = self._mp_get_text_inputs(samples, image.size(0))
                modal_prompt, _ = self._mp_prepare_prompts(
                    images=image,
                    text_inputs=text_inputs,
                    training=False,
                )
            if modal_prompt is not None:
                modal_prompt = modal_prompt.to(dtype=query_tokens.dtype, device=query_tokens.device)
                query_tokens = torch.cat([modal_prompt, query_tokens], dim=1)
            if not self.qformer_text_input:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                # TextCap case
                # ==> input: ocr, max_txt_len=128, prompt, max_output_txt_len=32
                assert "text_input" in samples, 'The data must has `text_input`.'

                if 'ocr_input' in samples:
                    text_input = [' '.join([samples['ocr_input'][i], samples["text_input"][i]]).strip() for i in range(len(samples["text_input"]))]
                else:
                    text_input = samples["text_input"]

                text_Qformer = self.tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]

            if "text_input" in samples:
                if 'ocr_input' in samples:
                    text_input = [' '.join([samples['ocr_input'][i], samples["text_input"][i]]).strip() for i in range(len(samples["text_input"]))]
                else:
                    text_input = samples["text_input"]
            else:
                text_input = [''] * image.size(0)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [' '.join([text_input[i], prompt]).strip() for i in range(len(text_input))]

            self.opt_tokenizer.padding_side = "left"

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        '''
        NOTE:
            This function is especially for the VQA. Depend on the qformer_input_text, and also
            the input text with prompt is prompt+question => "Question: {}. Answer:" and then use 
            that to generate the answer. Need to set the prompt at val and test step.
        '''
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.mixture_of_query(image_embeds, self.query_tokens)

            if not self.qformer_text_input:
                # no pre-knowledge in qformer
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                # VQA/TextVQA case
                # ==> input: question, max_txt_len=128, answer, max_output_txt_len=32; input: caption, max_output_txt_len=32
                if isinstance(samples["text_input"], str):
                    samples["text_input"] = [samples["text_input"]]
                
                if prompt:
                    text_input = [prompt.format(question) for question in samples["text_input"]]
                else:
                    text_input = samples["text_input"]

                # for TextVQA
                if 'ocr_input' in samples:
                    text_input = [' '.join([samples['ocr_input'][i], text_input[i]]).strip() for i in range(len(text_input))]
                
                # text_input = samples["text_input"]

                text_Qformer = self.tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(self.device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
                
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            # for TextVQA
                if 'ocr_input' in samples:
                    text_input = [' '.join([samples['ocr_input'][i], text_input[i]]).strip() for i in range(len(text_input))]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        qformer_text_input = cfg.get("qformer_text_input", False)
        # for adapter
        mh_pa_r = cfg.get("mh_pa_r", 25.0)
        mh_pa_drop_out = cfg.get("mh_pa_drop_out", 0.0)
        mh_pa_s = cfg.get("mh_pa_s", 4.0)
        ffn_pa_r = cfg.get("ffn_pa_r", 1.5)
        ffn_pa_drop_out = cfg.get("ffn_pa_drop_out", 0.0)
        ffn_pa_s = cfg.get("ffn_pa_s", 4.0)
        # for visual lora
        r = cfg.get("r", 8)
        alpha = cfg.get("alpha", 16)
        # for MoQ
        mix_query = cfg.get("mix_query", True)
        # for kd weight
        kd_weight = cfg.get("kd_weight", 1.0)
        # for ortho
        ortho = cfg.get("ortho", True)
        ortho_weight = cfg.get("ortho_weight", 0.1)
        use_modal_prompt = cfg.get("use_modal_prompt", False)
        mp_prefix_len = cfg.get("mp_prefix_len", 10)
        mp_transfer_num = cfg.get("mp_transfer_num", 3)
        mp_lam = cfg.get("mp_lam", 0.5)
        mp_loss_weight = cfg.get("mp_loss_weight", 1.0)
        mp_clip_model_name = cfg.get("mp_clip_model_name", "openai/clip-vit-large-patch14")

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            mh_pa_r=mh_pa_r,
            mh_pa_drop_out=mh_pa_drop_out,
            mh_pa_s=mh_pa_s,
            ffn_pa_r=ffn_pa_r,
            ffn_pa_drop_out=ffn_pa_drop_out,
            ffn_pa_s=ffn_pa_s,
            r=r,
            alpha=alpha,
            mix_query=mix_query,
            kd_weight=kd_weight,
            ortho=ortho,
            ortho_weight=ortho_weight,
            use_modal_prompt=use_modal_prompt,
            mp_prefix_len=mp_prefix_len,
            mp_transfer_num=mp_transfer_num,
            mp_lam=mp_lam,
            mp_loss_weight=mp_loss_weight,
            mp_clip_model_name=mp_clip_model_name,
        )
        model.load_checkpoint_from_config(cfg)

        return model
