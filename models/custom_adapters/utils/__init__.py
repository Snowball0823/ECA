import logging
from .adapter_util import freeze_adapter, set_training, init_adapter, freeze_dropout, iter_adapter_named_parameters
from adapters.composition import Average, Fuse, Stack

class CombinedModule:
    @classmethod
    def combine_modules(cls, modules_list):
        for moudle in modules_list:
            for attr in dir(moudle):
                if not attr.startswith('__'):
                    setattr(cls, attr, getattr(moudle, attr))
        return cls
    

import inspect

def list_module_attributes(module):
    attributes = dir(module)
    for attr in attributes:
        attr_value = getattr(module, attr)
        print(f"Attribute: {attr}, Type: {type(attr_value)}")
        if inspect.isfunction(attr_value):
            print(f" - Function: {attr}")
        elif inspect.isclass(attr_value):
            print(f" - Class: {attr}")
        elif inspect.ismodule(attr_value):
            print(f" - Submodule: {attr}")
        else:
            print(f" - Value: {attr_value}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # print(
    #     f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    # )
    logging.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

