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

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from adapters import BnConfig, LoRAConfig, MAMConfig, ParBnConfig
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from packaging import version
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

from ..custom_adapters import adapter_init
from ..custom_adapters.utils import (Average, freeze_adapter, freeze_dropout,
                                     init_adapter, print_trainable_parameters)
from .blip2 import Blip2Base, disabled_train
from .utils import freeze_parameters, orthogonal_svd_init, tensor_prompt


@registry.register_model("pa_blip2_opt_cap")
class PABlip2OPTCap(Blip2Base):
    """
    BLIP2 OPT model.
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
        max_txt_len=32,
        apply_lemmatizer=False,
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
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.freeze_vit = freeze_vit
        if freeze_vit:
            self.freeze_visual_encoder_ln(disable_training=True)

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

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

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None  

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
        # self.sim_collection = list()
        # # ============== #


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
        for _, p in self.visual_encoder.named_parameters():
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
    def moq_num(self):
        new_num = len(getattr(self, "current_keys", torch.tensor([], dtype=torch.float)))
        old_num = len(getattr(self, "keys_set", torch.tensor([], dtype=torch.float)))
        return new_num+old_num

    @property
    def adapter_structure(self):
        structure = {'lora': self.lora_dict, 'adapter': self.parallel_adapters_dict, 
                                  'v_freeze': self.visual_encoder_freezed, 'ln_freeze': self.ln_freezed,
                                  'moq_num': self.moq_num}
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

        other_entries = grouped_named_params["other"]
        if other_entries:
            warning_msg = (
                "Unexpected trainable params found outside projector/adapters/vision_lora/MoQ. "
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

        if lora_visual:
            assert not self.freeze_vit, "Freeze the ViT alreday, set `freeze_vit=False` at mode yaml first."
            self.expand_visual_lora()
        elif (not lora_visual) and len(self.lora_dict)>0 and (not self.ln_freezed):
            self.freeze_visual_encoder_ln(disable_training=True)

        if self.mix_query:
            self.query_mixture()

    
    def rebuild_from_config(self, adapter_structure: dict, moq_old_kv: dict, **kwargs):
        if len(adapter_structure['lora'])>0:
            assert not self.freeze_vit, "Freeze the ViT alreday, set `freeze_vit=False` at mode yaml first."
            for _ in adapter_structure['lora']:
                self.expand_visual_lora()
        if adapter_structure['ln_freeze']:
            self.freeze_visual_encoder_ln(disable_training=True)
        
        if len(adapter_structure['adapter'])>0:
            for _ in adapter_structure['adapter']:
                self.expand_q_adapter()

        if adapter_structure['moq_num']>0:
            # TODO: like LlaVA version, save the current_keys/queries and keys_set/queries_set for restore.
            # Current is only suitable for the saved the same task's 1 epoch ckpt.
            for _ in range(adapter_structure['moq_num']):
                self.query_mixture(svd_init=False)
            if hasattr(self, 'keys_set'):
                keys_set = getattr(self, 'keys_set')
                for key in moq_old_kv:
                    setattr(self, key, moq_old_kv[key].to(keys_set.device))


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


    def forward(self, samples):
        feature_diction, query_diction = None, None
        if 'feature' in samples and 'query' in samples:
            feature_diction = samples['feature']
            query_diction = samples['query']
            pre_query_tokens = samples['query_token']

        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.mixture_of_query(image_embeds, self.query_tokens)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_output"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
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
            # key_task_loss = 0
            # ============== #

            # ortho loss for queries
            _, _, q_dim = self.current_queries.shape
            query_ortho_loss = 0
            previous_queries = previous_queries.view(1, -1, q_dim)
            query_gram_matrix = torch.einsum('bqd,bad->qa', self.current_queries, previous_queries)
            # query_gram_matrix = torch.einsum('bqd,bad->qa', F.normalize(self.current_queries, p=2, dim=-1), F.normalize(previous_queries, p=2, dim=-1))
            query_ortho_loss = torch.norm(query_gram_matrix, p='fro')**2
            
            ortho_loss = key_ortho_loss+query_ortho_loss
            # ortho_loss = 0
            # # ============== #

            loss = loss + self.ortho_weight*(ortho_loss+key_task_loss)

        return {"loss": loss, 'output loss': outputs.loss, 'DR loss': mse_loss, 'MoQ loss': (ortho_loss+key_task_loss)}
    

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
                image_embeds = image_embeds[:,1:,:]
            img_patches = torch.mean(image_embeds, dim=1)
            


            # get cos sim
            img_norm = F.normalize(img_patches, p=2, dim=-1)
            keys_norm = F.normalize(task_keys, p=2, dim=-1)
            scaled_logits = torch.einsum('bd,nd->bn', img_norm, keys_norm)
            # self.sim_collection.append(scaled_logits.detach().cpu().clone())
            # # ============== #


            # get task_attention_scores
            task_attention_scores = F.softmax(scaled_logits, dim=1)

            
            # Step 3: Apply attention task attention scores to task queries to get delta query
            # [bs, n] @ [n, q_num, dim_b] -> [bs, q_num, dim_b]
            delta_query = torch.einsum('bn,nqd->bqd', task_attention_scores, task_queries)
            # delta_query = task_queries[torch.argmax(task_attention_scores, dim=1)]
            # ============== #
            
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
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

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
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.mixture_of_query(image_embeds, self.query_tokens)
            query_output = self.Qformer.bert(
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
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
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
            apply_lemmatizer=apply_lemmatizer,
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
            ortho_weight=ortho_weight
        )
        model.load_checkpoint_from_config(cfg)

        return model
