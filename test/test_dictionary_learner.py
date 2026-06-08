import argparse
import logging
import multiprocessing as mp
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from lavis.common.config import Config
from lavis.common.dist_utils import (get_rank, get_world_size,
                                     init_distributed_mode, main_process)
from lavis.common.logger import setup_logger
from lavis.common.optims import (LinearWarmupCosineLRScheduler,
                                 LinearWarmupStepLRScheduler)
from lavis.common.registry import registry
from lavis.common.utils import now
from lavis.models.clip_models.model import trace_model
from lavis.processors import *
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

# import lavis.tasks as tasks
import tasks as tasks
# imports modules for registration
# from lavis.datasets.builders import *
from datasets import *
# from lavis.models import *
from models import *
from runners import *
# from lavis.runners import *
# from lavis.tasks import *
from tasks import *
from tasks.utils.dictionary_learner import \
    MiniBatchScaleCodeDictionaryLearner


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--env-cfg-path", default='',
                        help="path to customize environment configuration file, default path at `path/to/lavis/configs/default.yaml`.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls

def replace_env_cfg(args):
    env_cfg_path = args.env_cfg_path
    if env_cfg_path != '':
        if os.path.isfile(env_cfg_path):
            env_cfg_path = os.path.abspath(env_cfg_path)
            env_cfg = OmegaConf.load(env_cfg_path)
            project_root = os.path.dirname(os.path.abspath(__file__))
            registry.register_path("project_root", project_root)
            cache_root = os.path.join(project_root, env_cfg.env.cache_root)
            o_cache_root = registry.get_path('cache_root')
            registry.replace_path("cache_root", cache_root)
            print("Change `cache_root` from: "+str(o_cache_root)+" to: "+str(cache_root))



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(1408, 1408)

    def forward(self, x):
        return self.fc(x)


def test():
    # ===========================
    accum_batch = 2
    dim = 1408
    n_samples = 40000
    device = torch.device("cuda")
    # for dictionary leaner (include out_dim)
    atoms_num = 5000
    alpha = 1.0
    regress = "lars"
    n_jobs = -1
    fit_iter = 10
    # ===========================
    model = Encoder().to(device)
    out_dim = model.fc.weight.shape[0]

    model = DDP(model, device_ids=[device])

    X = torch.randn(n_samples, dim).to(device)

    dataset = torch.utils.data.TensorDataset(X)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=get_world_size(),
                        rank=get_rank(),)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=25)

    dictionary_tensor = torch.empty(atoms_num, out_dim).to(device)



    if get_rank() == 0:
        dict_learner = MiniBatchScaleCodeDictionaryLearner(n_components=atoms_num, alpha=alpha, fit_algorithm=regress, n_jobs=n_jobs)
        

    model.eval()
    for epoch in range(fit_iter):
        for batch in dataloader:
            inputs = batch[0].to(device)
            with torch.no_grad():
                outputs = model(inputs)

            gathered_outputs = [torch.zeros_like(outputs) for _ in range(get_world_size())]
            dist.all_gather(gathered_outputs, outputs)

            if get_rank() == 0:
                gathered_outputs = torch.cat(gathered_outputs, dim=0).cpu().numpy()
                print(gathered_outputs.shape)
                print("start dictionary")
                dict_learner.partial_fit(gathered_outputs)
                print("Finish dictionary learn")

            dist.barrier()

    if get_rank() == 0:
        dictionary = dict_learner.components_
        dictionary_tensor = torch.tensor(dictionary).to(device).contiguous()
    
    dist.broadcast(dictionary_tensor, src=0)

    if get_rank()==0:
        print(dictionary_tensor)



def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()

    replace_env_cfg(args)

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    test()


if __name__ == "__main__":
    main()