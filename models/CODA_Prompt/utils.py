import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from lavis.common.registry import registry
from torch.autograd import Variable


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p  


def freeze_parameters(model: nn.Module, names: list):
    for n, p in model.named_parameters():
        for name in names:
            if name in n:
                p.requires_grad = False


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("project_root"), rel_path)



def orthogonal_svd_init(matrix):
    n, m = matrix.shape
    # U, S, V = torch.svd(matrix)
    U, S, V = torch.linalg.svd(matrix)
    num_singular_values = S.numel()
    if num_singular_values < m:
        new_row = torch.randn(m).to(matrix.device)
        for i in range(n):
            row = matrix[i]
            dot_product = torch.dot(new_row, row)
            new_row = new_row - (dot_product / torch.norm(row)**2) * row
    else:
        new_row = torch.randn(m).to(matrix.device) @ V[:, num_singular_values:]

    row_norms = torch.norm(matrix, dim=1)
    input_scale = row_norms.mean()

    new_row = new_row / torch.norm(new_row) * input_scale

    return new_row.unsqueeze(0)



def prefix_attention_mask(attention_mask, prompt_length, dim: int = 3, prefix_value: int = 0):
    if attention_mask is not None:
        ones_shape = list(attention_mask.shape)
        ones_shape[dim] = prompt_length
        prefix_mask = torch.full(ones_shape, prefix_value, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat((prefix_mask, attention_mask), dim=dim)
    return attention_mask
