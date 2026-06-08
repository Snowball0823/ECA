import adapters
import torch
from lavis.common.registry import registry
# from lavis.models.blip2_models.blip2 import Blip2Base
from models.ECA_Q.blip2 import Blip2Base as Blip2Base
from lavis.models.blip2_models.blip2_opt import Blip2OPT
from adapters import BnConfig, ParBnConfig, MAMConfig, LoRAConfig
from transformers import BertModel, BertTokenizer, CLIPVisionModel, CLIPVisionConfig, CLIPModel, CLIPConfig
from models.custom_adapters.adapter_warp_model import adapter_init
from models.custom_adapters.utils import print_trainable_parameters, freeze_adapter
from lavis.models.blip2_models.blip2 import (
    compute_sim_matrix,
    disabled_train,
)

# clip_config = CLIPConfig()
# clip = CLIPModel(config=clip_config)
# lora_config = LoRAConfig(r=16, alpha=32)
# adapter_init(clip, use_customize=True)
# print(clip)
# input()


# clip_vis_config = CLIPVisionConfig()
# o_vit = CLIPVisionModel(config=clip_vis_config)
# lora_config = LoRAConfig(r=16, alpha=32)
# adapter_init(o_vit, use_customize=True)
# o_vit.add_adapter("lora", config=lora_config)
# print('Original VIT')
# print(o_vit)
# input()



test_blip2 = Blip2Base()
visual_encoder, ln_vision = test_blip2.init_vision_encoder(
            "eva_clip_g", 224, 0, False, "fp16"
        )
# visual_encoder.config

for name, param in visual_encoder.named_parameters():
    param.requires_grad = False
ViT = visual_encoder.eval()

# Qformer.train = disabled_train
# test
'''
patch=32, with_project=True, local_pretrian_file=False, 
target_modules=["q_proj", "v_proj", "out_proj"], rank=8, drop_out=0,
'''
# TODO: test for merge
adapter_init(ViT, use_customize=True)
ViT.freeze_model()
# original is r=8, aplha=8/16
lora_config = LoRAConfig(r=16, alpha=32, intermediate_lora=True, output_lora=True)
ViT.add_adapter("lora", config=lora_config)
ViT.active_adapters = ["lora"]
freeze_adapter(ViT, ["lora"], freeze=False)
ViT.active_embedding()
print(ViT)
input()
print_trainable_parameters(ViT)
adapter_sum = ViT.adapter_summary()
print(adapter_sum)
freeze_adapter(ViT, ["lora"])
input()
adapter_sum = ViT.adapter_summary()
print(adapter_sum)
input()
