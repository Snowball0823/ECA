from typing import Iterable, Tuple

import torch.nn as nn
from adapters.composition import adjust_tensors_for_parallel_
from adapters.methods.bottleneck import BottleneckLayer
from adapters.methods.lora import LoRALinear, LoRAMergedLinear
from adapters.methods.prefix_tuning import PrefixTuningLayer
from adapters.model_mixin import ModelBaseAdaptersMixin
from adapters.utils import patch_forward


class AttentionAdaptersMixin:
    """Adds adapters to the Attention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.qkv = LoRAMergedLinear.wrap(self.qkv, "selfattn", model_config, adapters_config)
        self.prefix_tuning = PrefixTuningLayer(
            "self_prefix", model_config, adapters_config, add_model_type_to_key=True
        )
        patch_forward(self)


class BlockAdaptersMixin:
    """Adds adapters to the Block module of EvaVisionTransformer."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.mlp.fc1 = LoRALinear.wrap(self.mlp.fc1, "intermediate", model_config, adapters_config)
        self.mlp.fc2 = LoRALinear.wrap(self.mlp.fc2, "output", model_config, adapters_config)

        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")

        patch_forward(self)


class EvaVisionTransformerAdaptersMixin(ModelBaseAdaptersMixin):
    """Adds adapters to the EvaVisionTransformer class."""

    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config, add_prefix_tuning_pool=False):
        super().init_adapters(model_config, adapters_config, add_prefix_tuning_pool=add_prefix_tuning_pool)


    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.blocks):
            yield i, layer

