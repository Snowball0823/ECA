import types

import adapters
import torch
import torch.nn as nn
import torch.optim as optim
from adapters import BnConfig, MAMConfig, ParBnConfig, init
from lavis.common.registry import registry
from transformers import BertConfig, BertModel, BertTokenizer

from models.custom_adapters.adapter_warp_model import adapter_init
from models.custom_adapters.utils import (freeze_adapter,
                                          print_trainable_parameters)
# from lavis.models.blip2_models.blip2 import (Blip2Base, compute_sim_matrix,
#                                              disabled_train)
from models.ECA_Q.blip2 import Blip2Base, compute_sim_matrix, disabled_train

# config = BertConfig()
# bert = BertModel(config=config)
# bert.cuda()
# init(bert)
# bert.freeze_model()
# attn_pa_config = ParBnConfig(mh_adapter=True, output_adapter=False, reduction_factor=25)
# bert.add_adapter("pa_adapter_attn_1", config=attn_pa_config)
# ffn_pa_config = ParBnConfig(reduction_factor=1.5)
# bert.add_adapter("pa_adapter_ffn_1", config=ffn_pa_config)
# bert.active_adapters = ["pa_adapter_attn_1", "pa_adapter_ffn_1"]
# bert.train()
# image_embeds = torch.randn(2,196,768).cuda()
# image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
# query_tokens = torch.randn(2,32,768).cuda()
# query_output = bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
# print(query_output)
# input()




test_blip2 = Blip2Base()
Qformer, query_tokens = test_blip2.init_Qformer(
            32, 768
        )
Qformer.cls = None
# Qformer.bert.embeddings.word_embeddings = None
# Qformer.bert.embeddings.position_embeddings = None
# for layer in Qformer.bert.encoder.layer:
#     layer.output = None
#     layer.intermediate = None
for name, param in Qformer.named_parameters():
    param.requires_grad = False
Qformer = Qformer.eval()
# Qformer.train = disabled_train
Qformer.train = types.MethodType(disabled_train, Qformer)
# test
'''
patch=32, with_project=True, local_pretrian_file=False, 
target_modules=["q_proj", "v_proj", "out_proj"], rank=8, drop_out=0,
'''
# print(Qformer.bert)
# input()
adapter_init(Qformer.bert, use_customize=True)
Qformer.bert.freeze_model()
attn_pa_config = ParBnConfig(mh_adapter=True, output_adapter=False, reduction_factor=25, non_linearity='linear')
Qformer.bert.add_adapter("pa_adapter_attn_1", config=attn_pa_config)
ffn_pa_config = ParBnConfig(reduction_factor=1.5, non_linearity='linear')
Qformer.bert.add_adapter("pa_adapter_ffn_1", config=ffn_pa_config)
Qformer.bert.active_adapters = ["pa_adapter_attn_1", "pa_adapter_ffn_1"]
Qformer.train()
print(Qformer.bert)
input()
freeze_adapter(Qformer.bert, ["pa_adapter_attn_1", "pa_adapter_ffn_1"], freeze=False)
print_trainable_parameters(Qformer.bert)
adapter_sum = Qformer.bert.adapter_summary()
print(adapter_sum)
Qformer = Qformer.cuda()
query_tokens = query_tokens.cuda()
image_embeds = torch.randn(2,196,768).cuda()
image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).cuda()
# query_output = Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
# print(query_output)
# input()
# print('Update')
# optimizer = optim.SGD([p for n,p in Qformer.bert.named_parameters() if p.requires_grad], lr=0.1)
# criterion = nn.MSELoss()
# for i in range(10):
#     query_output = Qformer.bert(
#                 query_embeds=query_tokens,
#                 encoder_hidden_states=image_embeds,
#                 encoder_attention_mask=image_atts,
#                 return_dict=True,
#             )
#     optimizer.zero_grad()
#     gt = torch.randn_like(query_output.last_hidden_state).cuda()
#     loss = criterion(query_output.last_hidden_state, gt)
#     print(loss)
#     loss.backward()
#     optimizer.step()

# query_output = Qformer.bert(
#                 query_embeds=query_tokens,
#                 encoder_hidden_states=image_embeds,
#                 encoder_attention_mask=image_atts,
#                 return_dict=True,
#             )
# print(query_output)
# input()
freeze_adapter(Qformer.bert, ["pa_adapter_attn_1", "pa_adapter_ffn_1"])
input()
adapter_sum = Qformer.bert.adapter_summary()
print(adapter_sum)
input()


Qformer.bert.add_adapter("pa_adapter_attn_2", config=attn_pa_config)
Qformer.bert.add_adapter("pa_adapter_ffn_2", config=ffn_pa_config)
# Qformer.bert.active_adapters = ["pa_adapter_attn_2", "pa_adapter_ffn_2"]
print_trainable_parameters(Qformer)
input()
adapter_sum = Qformer.bert.adapter_summary()
print(adapter_sum)
input()
# print(Qformer)
# input()
# Qformer = Qformer.cuda()
# query_output = Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
# print(query_output)
# input()
print('Open all')
# Qformer.bert.active_adapters = ["pa_adapter_attn_1", "pa_adapter_ffn_1", "pa_adapter_attn_2", "pa_adapter_ffn_2"]
freeze_adapter(Qformer.bert, ["pa_adapter_attn_1", "pa_adapter_ffn_1", "pa_adapter_attn_2", "pa_adapter_ffn_2"], freeze=False)
import adapters.composition as ac
# Qformer.bert.active_adapters = [ac.Average(*["pa_adapter_ffn_1","pa_adapter_ffn_2"], weights=[1,1]), ac.Average(*["pa_adapter_attn_1","pa_adapter_attn_2"], weights=[1,1])]
# ac.Average("m", "n", "o", weights=[0.1, 0.6, 0.3])
Qformer.bert.add_adapter_fusion(["pa_adapter_ffn_1","pa_adapter_ffn_2"], 'dynamic')
Qformer.bert.add_adapter_fusion(["pa_adapter_attn_1","pa_adapter_attn_2"], 'dynamic')
Qformer.bert.active_adapters = [ac.Fuse(*["pa_adapter_ffn_1","pa_adapter_ffn_2"])]
print(Qformer.bert)
adapter_sum = Qformer.bert.adapter_summary()
print(adapter_sum)
input()
# Qformer = Qformer.cuda()
# query_output = Qformer.bert(
#             query_embeds=query_tokens,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=image_atts,
#             return_dict=True,
#         )
# print(query_output)
# input()

import torch
import torch.nn as nn
import torch.optim as optim

Qformer = Qformer.cuda()
opt_names = [n for n,p in Qformer.bert.named_parameters() if p.requires_grad and 'fusion' in n]
print(opt_names)
input()
optimizer = optim.SGD([p for n,p in Qformer.bert.named_parameters() if p.requires_grad], lr=0.01)
criterion = nn.MSELoss()
for i in range(10):
    kwargs = dict(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    query_output = Qformer.bert(
                **kwargs
            )

    optimizer.zero_grad()
    gt = torch.randn_like(query_output.last_hidden_state).cuda()
    loss = criterion(query_output.last_hidden_state, gt)
    loss.backward()
    optimizer.step()

    for n,p in Qformer.bert.named_parameters():
        if 'fusion' in n:
            print(n)
            print(p)
            print(p.requires_grad)
            print(p.grad)
            input()
        # if p.grad is not None:
        #     print(n)
        #     print(p)
        #     print(torch.sum(p.grad))
        #     input()