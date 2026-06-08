"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import logging
import os

import dcor
import numpy
import torch
import hashlib
import torch.nn.functional as F
import webdataset as wds
from collections import defaultdict
from lavis.common.dist_utils import (get_rank, get_world_size,
                                     is_dist_avail_and_initialized,
                                     is_main_process, main_process)
from lavis.common.registry import registry
from lavis.datasets.datasets.dataloader_utils import (IterLoader,
                                                      MultiIterLoader,
                                                      PrefetchLoader)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataset import ChainDataset

from .dictionary_learner import MiniBatchScaleCodeDictionaryLearner

def _flatten_fim(fim, norm=False):
    flat_fim = []
    epsilon=1e-15
    for key, value in fim.items():
        _value = value[1].view(-1)
        flat_fim.append(_value.view(-1))
    _flat_fim = torch.cat(flat_fim)
    if norm:
        _flat_fim = F.normalize(_flat_fim+epsilon, p=1,dim=0)
    return _flat_fim

def fim_distiance_correlation(fim_A, fim_B):
    '''
    args:
        fim_A: a fisher information matrix dict {name: param_grade} on cpu
        fim_B: the same as the fim_A
    return:
        the distance_correlation between flatten fim_A and flatten fim_B.
    '''
    fim_a_flat = _flatten_fim(fim_A).numpy()
    fim_b_flat = _flatten_fim(fim_B).numpy()
    distance_corr = dcor.distance_correlation(fim_a_flat, fim_b_flat)
    distance_corr_tensor = torch.tensor(distance_corr)
    return distance_corr_tensor

def fim_jsd(fim_A, fim_B):

    p = _flatten_fim(fim_A, norm=True)
    q = _flatten_fim(fim_B, norm=True)

    m = 0.5 * (p + q)
    
    kl_pm = F.kl_div(m.log(), p, reduction='sum')
    kl_qm = F.kl_div(m.log(), q, reduction='sum')
    jsd = 0.5 * (kl_pm + kl_qm)

    return 1-jsd


def fim_overlap(fim_A, fim_B):
    p = _flatten_fim(fim_A, norm=True)
    q = _flatten_fim(fim_B, norm=True)
    mean_overlap = torch.mean(torch.min(p, q))
    mean_upperbound = torch.mean(torch.max(p, q))
    overlap_rate = mean_overlap/mean_upperbound
    return overlap_rate


def fim_project_jsd(fim_A, fim_B):
    '''
    NOTE:
        project fim_B on fim_A, and compare the JSD of rest direction of .
    '''
    for key, value in fim_A.items():
        grad1 = value[0].view(-1)
        grad2 = fim_B[key][0].view(-1)
        cos_similarity = F.cosine_similarity(grad1, grad2, dim=0)
        # fim_B[key][1] *= (1-cos_similarity)
        fim_B[key][1] *= -cos_similarity
        fim_B[key][1] = torch.clamp_min(fim_B[key][1], 0.0)

    p = _flatten_fim(fim_A, norm=True)
    q = _flatten_fim(fim_B, norm=True)

    m = 0.5 * (p + q)
    
    kl_pm = F.kl_div(m.log(), p, reduction='sum')
    kl_qm = F.kl_div(m.log(), q, reduction='sum')
    jsd = 0.5 * (kl_pm + kl_qm)

    return 1-jsd

def fim_impact(fim_A, fim_B):
    '''
    NOTE:
        Get the impact of performance on dataset A after updating the model on dataset B.
    '''
    theta_impact = []
    for key, value in fim_A.items():
        grad1 = value[0].view(-1)
        grad2 = fim_B[key][0].view(-1)
        first_order = -(grad1*grad2)
        fim = value[1].view(-1)
        sencond_order = 0.5*fim*grad2.pow(2)
        theta_impact.append(first_order+sencond_order)

    flat_theta_impact = torch.cat(theta_impact)
    pos_part = torch.sum(torch.relu(flat_theta_impact))
    neg_part = torch.sum(torch.relu(-flat_theta_impact))

    return pos_part/(pos_part+neg_part)

FIM_METRIC={'overlap': fim_overlap, 'jsd': fim_jsd, 'dis_cor': fim_distiance_correlation, 'p-jsd': fim_project_jsd, 'i_fim': fim_impact}



def unwrap_dist_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model
    

def reload_best_model(model):
    """
    Load the best checkpoint for evaluation.
    """
    output_dir = registry.get_path('output_dir')
    checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")

    logging.info("Loading checkpoint from {}.".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as e:
        logging.warning(
            """
            Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
            Trying to load the model with strict=False.
            """
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def md5check(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def adapter_fim(task_fim: dict, cuda_enabled=False, multitask=True):
    '''
    args:
        task_fim: a dict with every task fim on this adapter -> {task_num:{fim_dict,},...}
        cuda_enabled: if True, then return the fim on GPU.
    return:
        adapter_fim: {name:fim, ...}
    '''
    adapter_fim = {}

    if not multitask:
        tasks = list(task_fim.keys())
        task_fim = {tasks[-1]:task_fim[tasks[-1]]}

    for _task_fim in task_fim.values():
        for n, f in _task_fim.items():
            if n not in adapter_fim:
                adapter_fim[n] = f[1].clone()
            else:
                adapter_fim[n] += f[1]
    if cuda_enabled:
        adapter_fim = {n: f.cuda() for n, f in adapter_fim.items()}
    return adapter_fim

def freeze_model(model: torch.nn.Module):
    for n, p in model.named_parameters():
        if p.grad is not None:
            p.requires_grad = False
    model.eval()


def cat_samples(batch_samples):
    merged_dict = defaultdict(list)
    final_dict = dict()

    for sample in batch_samples:
        for key, value in sample.items():
            merged_dict[key].append(value)

    for key, value in merged_dict.items():
        if isinstance(value[0], list):
            cat_value = sum(value, [])
        elif isinstance(value[0], ()):
            cat_value = sum(value, ())
        elif isinstance(value[0], torch.Tensor):
            cat_value = torch.cat(value, dim=0)
        final_dict[key] = cat_value
    return final_dict


    

def create_train_loader(
        datasets,
        num_workers,
        batch_sizes,
        collate_fns,
        dataset_ratios=None,
        use_distributed=True,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=True,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=collate_fn,
                    drop_last=True,
                )
                loader = PrefetchLoader(loader)
                loader = IterLoader(loader, use_distributed=use_distributed)

            return loader

        if isinstance(datasets, list) or isinstance(datasets, tuple):
            loader = MultiIterLoader(
                loaders=[
                    _create_loader(d, num_workers, batch_sizes, collate_fns[i])
                    for i, d in enumerate(datasets)
                ],
                ratios=dataset_ratios,
            )
        else:
            loader = _create_loader(datasets, num_workers, batch_sizes, collate_fns)


        return loader


class DictionaryDataset(Dataset):
    def __init__(self, feature_dictonary, repeat_factor=1):
        self.feature_dictonary = feature_dictonary
        self.base_len = len(feature_dictonary)
        self.repeat_factor = max(int(repeat_factor), 1)
        self.index_map = self._build_index_map()

    def _build_index_map(self):
        if self.base_len == 0:
            return torch.empty(0, dtype=torch.long)
        if self.repeat_factor == 1:
            return torch.arange(self.base_len, dtype=torch.long)
        blocks = [torch.randperm(self.base_len) for _ in range(self.repeat_factor)]
        return torch.cat(blocks, dim=0)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        real_idx = int(self.index_map[idx])
        return {"feature": self.feature_dictonary[real_idx]}
