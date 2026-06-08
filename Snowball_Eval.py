"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import argparse
import os
import random
import logging
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import is_main_process, init_distributed_mode
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# Import local modules for registry side effects.
from datasets import *
from models import *
from processors import *
from tasks import *
from runners import *

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")

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

    return args

def setup_logger():
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

def setup_seeds(config):
    seed = config.run_cfg.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

def main():
    # Set before distributed initialization so all ranks share the same run id.
    job_id = now()

    args = parse_args()

    replace_env_cfg(args)

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # Configure logging after distributed initialization.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)

if __name__ == "__main__":
    main()