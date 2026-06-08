from adapters.models import MODEL_MIXIN_MAPPING
from .llava_llama.mixin_llava_llama import LlavaLlamaModelAdapterMixin
from .q_former.mixin_q_former import BertLayerAdaptersMixin, BertModelAdaptersMixin
from .qwen2.mixin_qwen2 import Qwen2ModelAdapterMixin
from .eva_clip_vit.mixin_eva_clip_vit import EvaVisionTransformerAdaptersMixin
custom_mixin_mapping = {
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "EvaVisionTransformer": EvaVisionTransformerAdaptersMixin,
    "LlavaLlamaModel": LlavaLlamaModelAdapterMixin,
    "Qwen2Model": Qwen2ModelAdapterMixin,
    }

MODEL_MIXIN_MAPPING.update(custom_mixin_mapping)
