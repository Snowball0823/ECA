import adapters
import logging
from typing import Iterable, Tuple

import torch.nn as nn

from adapters.composition import adjust_tensors_for_parallel_
from adapters.methods.bottleneck import BottleneckLayer
from adapters.methods.lora import LoRALinear
from adapters.methods.prefix_tuning import PrefixTuningLayer
from adapters.model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from adapters.utils import patch_forward


logger = logging.getLogger(__name__)


class BertSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")
        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
        patch_forward(self)


# For backwards compatibility, BertSelfOutput inherits directly from BottleneckLayer
class BertSelfOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


# For backwards compatibility, BertOutput inherits directly from BottleneckLayer
class BertOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        if hasattr(self.intermediate,'dense') and self.intermediate.dense is not None:
            self.intermediate.dense = LoRALinear.wrap(
                self.intermediate.dense, "intermediate", model_config, adapters_config
            )
        if hasattr(self.output,'dense') and self.output.dense is not None:
            self.output.dense = LoRALinear.wrap(self.output.dense, "output", model_config, adapters_config)
            
        if hasattr(self, "intermediate_query"):
            if hasattr(self.intermediate_query,'dense') and self.intermediate_query.dense is not None:
                self.intermediate_query.dense = LoRALinear.wrap(
                    self.intermediate_query.dense, "intermediate", model_config, adapters_config
                )
        if hasattr(self, "output_query"):
            if hasattr(self.output_query,'dense') and self.output_query.dense is not None:
                self.output_query.dense = LoRALinear.wrap(self.output_query.dense, "output", model_config, adapters_config)
        

        # Set location keys for prefix tuning
        self.attention.self.location_key = "self"
        if hasattr(self, "has_cross_attention") and self.has_cross_attention:
            self.crossattention.self.location_key = "cross"


class BertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def init_adapters(self, model_config, adapters_config, add_prefix_tuning_pool=False):
        super().init_adapters(model_config, adapters_config, add_prefix_tuning_pool=add_prefix_tuning_pool)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        # Register hook for post embedding forward
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
