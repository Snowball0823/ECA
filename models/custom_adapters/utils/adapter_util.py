from adapters.model_mixin import ModelAdaptersMixin, AdapterLayerBase
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch.nn as nn

def _disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_adapter(adapter_model:ModelAdaptersMixin, names:Union[str, list], freeze=True):
    if isinstance(names, str):
        names = [names]
    for name in names:
        # use a custom index to ensure numbering is from 0 to N layers
        for _, layer in adapter_model.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    adapter_module = module.get_adapter(name)
                    if adapter_module is not None:
                        adapter_module = adapter_module.eval() if freeze else adapter_module.train()
                        for p in adapter_module.parameters():
                            p.requires_grad = not freeze


def freeze_dropout(adapter_model:ModelAdaptersMixin, freeze=True):
    # use a custom index to ensure numbering is from 0 to N layers
    for _, layer in adapter_model.iter_layers():
        for module in layer.modules():
            if isinstance(module, nn.Dropout):
                if freeze:
                    module.eval()
                    module.train = _disabled_train
                else:
                    module.train = nn.Module.train
                


def set_training(adapter_model:ModelAdaptersMixin, names:Union[str, list], training=True):
    if isinstance(names, str):
        names = [names]
    for name in names:
        # use a custom index to ensure numbering is from 0 to N layers
        for _, layer in adapter_model.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    adapter_module = module.get_adapter(name)
                    if adapter_module is not None:
                        adapter_module = adapter_module.eval() if not training else adapter_module.train()




def init_adapter(adapter_model:ModelAdaptersMixin, target_names:Union[str, list], source_names:Union[str, list]):
    if isinstance(target_names, str):
        target_names = [target_names]
    if isinstance(source_names, str):
        source_names = [source_names]

    for target_name, source_name in zip(target_names, source_names):
        # use a custom index to ensure numbering is from 0 to N layers
        for _, layer in adapter_model.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    if isinstance(source_name, str):
                        source_name = [source_name]
                    parameters, avg_parameters = [], []
                    source_adapter_num = len(source_name)
            
                    for name in source_name:
                        _adapter_module = module.get_adapter(name)
                        if _adapter_module is not None:
                            parameters.append(_adapter_module.parameters())

                    for params in zip(*parameters):
                        avg_param = sum(p.detach() for p in params) / source_adapter_num
                        avg_parameters.append(avg_param)

                    adapter_module = module.get_adapter(target_name)
                    if adapter_module is not None:
                        for target_param, avg_param in zip(adapter_module.parameters(), avg_parameters):
                            target_param.data.copy_(avg_param)


def iter_adapter_named_parameters(
    adapter_model: ModelAdaptersMixin,
    adapter_names: Union[str, Iterable[str]],
    prefix: str = "",
):
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]
    adapter_names = list(adapter_names)
    if not adapter_names:
        return
    for name in adapter_names:
        for layer_idx, layer in adapter_model.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    adapter_module = module.get_adapter(name)
                    if adapter_module is None:
                        continue
                    for param_name, param in adapter_module.named_parameters():
                        full_name = f"{prefix}.layers.{layer_idx}.{name}.{param_name}" if prefix else f"layers.{layer_idx}.{name}.{param_name}"
                        yield full_name, param
