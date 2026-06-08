"""
 Copyright (c) 2022, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

import datetime
import gc
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (download_cached_file, get_rank,
                                     get_world_size, is_main_process,
                                     main_process)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (IterLoader,
                                                      MultiIterLoader,
                                                      PrefetchLoader)
from lavis.runners.runner_base import RunnerBase
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset

@registry.register_runner("runner_cl")
class RunnerCL(RunnerBase):
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, job_id, datasets):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets_backup = datasets
        self.datasets = self.datasets_backup

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.start_task = 0

        self.setup_output_dir()
        self.setup_tasks()

        # Redirect logs to the per-run output directory.
        self.set_direction = False

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu]
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def cpu_model(self):
        """
        A property to get the original model on the cpu.
        """
        self._model = self._model.to(torch.device("cpu"))
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            lr_scale = self.config.run_cfg.get("lr_layer_decay", 1)
            weight_decay = self.config.run_cfg.get("weight_decay", 0.05)
            optim_params = self._model.get_optimizer_params(weight_decay, lr_scale)

            num_parameters = 0
            for p_group in optim_params:
                for p in p_group["params"]:
                    num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: {}".format(num_parameters))

            beta2 = self.config.run_cfg.get("beta2", 0.999)

            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = self.max_epoch
            min_lr = self.min_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self):
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            # reorganize the datasets to {'train':[],'val':[],'test':[]}
            datasets = reorg_datasets_by_split(self.datasets)
            # concat the muliple datasets in 'train'
            self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def save_freq(self):
        save_freq = self.config.run_cfg.get("save_freq", 5)
        return int(save_freq)

    @property
    def val_freq(self):
        val_freq = self.config.run_cfg.get("val_freq", 1)
        return int(val_freq)

    @property
    def save_last(self):
        save_last = self.config.run_cfg.get("save_last", True)
        return int(save_last)

    @property
    def save_final(self):
        save_final = self.config.run_cfg.get("save_final", True)
        return int(save_final)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def lora_visual_first(self):
        return self.config.run_cfg.get("lora_visual_first", True)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    @property
    def init_topic_task(self):
        return int(self.config.run_cfg.init_topic)

    @property
    def topic_per_task(self):
        return int(self.config.run_cfg.topic_per_task)

    @property
    def total_task_num(self):
        return len(self.split_tasks)

    @property
    def cache_root(self):
        return registry.get_path('cache_root')

    @property
    def joint_training(self):
        return self.config.run_cfg.get("joint_training", False)


    def _dived_tasks(self, task_list):
        divide_tasks = []
        index = 0
        n = len(task_list)
        if n >= self.init_topic_task:
            divide_tasks.append(task_list[index:index + self.init_topic_task])
            index += self.init_topic_task
        else:
            raise ValueError("Not enough elements for the first task")

        while index < n:
            if n - index <= self.topic_per_task:
                divide_tasks.append(task_list[index:])
                break
            else:
                divide_tasks.append(task_list[index:index + self.topic_per_task])
                index += self.topic_per_task
        return divide_tasks


    def reset_seeds(self, with_rank=False):
        if with_rank:
            seed = self.config.run_cfg.seed + get_rank()
        else:
            seed = self.config.run_cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def setup_output_dir(self):
        proj_root = Path(registry.get_path("project_root"))

        output_dir = proj_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

        # set up the log file
        if is_main_process():
            log_file = os.path.join(self.output_dir, 'training.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_formatter)
            logger = logging.getLogger()
            logger.addHandler(file_handler)

    def setup_tasks(self):
        assert len(self.datasets) == 1, "Can not support more than one datasets."
        dataset_name = list(self.datasets.keys())[0]
        splits = list(self.datasets[dataset_name].keys())
        dataset = self.datasets[dataset_name][splits[0]]
        self.tasks_order = list(dataset.cats)
        self.tasks_order.sort()
        random.shuffle(self.tasks_order)
        self.split_tasks = self._dived_tasks(self.tasks_order)

    def rebuild_dataset(self, cur_task, only_test=False, test_type='now'):
        self.reset_seeds()
        if hasattr(self._dataloaders, "clear"):
            self._dataloaders.clear()
        del self.datasets
        del self._dataloaders
        self._dataloaders = None
        train_tasks_list = self.split_tasks[cur_task]
        if self.joint_training:
            train_tasks_list = [task for tasks in self.split_tasks[:cur_task+1] for task in tasks]
        if test_type == 'now':
            test_tasks_list = [task for tasks in self.split_tasks[:cur_task+1] for task in tasks]
        elif test_type == 'up2now':
            test_tasks_list = self.split_tasks[cur_task]
        else:
            raise ValueError("`test_type` can be only 'now' or 'up2now'.")
        for name in self.datasets_backup:
            for split in self.datasets_backup[name]:
                if not only_test:
                    if split in self.train_splits or split in self.valid_splits:
                        self.datasets_backup[name][split].rebuild(train_tasks_list)
                        logging.info('Rebuilding {}-{} Dataset => Topics: {}'.format(name, split, train_tasks_list))
                    elif split in self.test_splits:
                        self.datasets_backup[name][split].rebuild(test_tasks_list)
                        logging.info('Rebuilding {}-{} Dataset => Topics: {}'.format(name, split, test_tasks_list))
                else:
                    if split in self.test_splits:
                        self.datasets_backup[name][split].rebuild(test_tasks_list)
                        logging.info('Rebuilding {}-{} Dataset => Topics: {}'.format(name, split, test_tasks_list))
        self.datasets = self.datasets_backup

        if hasattr(self.task, 'rebuild_dict_dataset'):
            self.task.rebuild_dict_dataset(cur_task)

        self.reset_seeds(with_rank=True)

    def task_initial(self, cur_task=None):
        '''
        Release the wrapped model, optimizer, and scheduler before task-specific structural changes.
        '''
        assert cur_task is not None, "Initial method needs a task number, `cur_task` is required."
        del self._wrapped_model
        del self._optimizer
        del self._lr_sched
        self._wrapped_model = None
        self._optimizer = None
        self._lr_sched = None
        self.rebuild_dataset(cur_task=cur_task)
        self.reset_seeds(with_rank=True)
        # Move the base model to CPU before rebuilding wrappers.
        self._model = self.cpu_model
        gc.collect()
        torch.cuda.empty_cache()

    def train(self):
        if not self.set_direction:
            self.log_config()
            self.set_direction = True
        # resume from checkpoint if specified: only support resume from 'final ckpt' now.
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            best_epoch, best_agg_metric = self._load_checkpoint(self.resume_ckpt_path)
        for cur_task in range(self.start_task, self.total_task_num):
            if self.start_epoch == 0 and not self.evaluate_only:
                # rebuild the datasets
                self.rebuild_dataset(cur_task=cur_task)

                # before training
                self.task.before_training(
                    cur_task=cur_task,
                    model=self.model,
                    dataset=None,
                    dataloader=self.train_loader,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    cuda_enabled=self.cuda_enabled,
                )
                best_agg_metric = 0
                best_epoch = 0

            # reload the model, optimizer, lr_schedule
            self.task_initial(cur_task)

            start_time = time.time()
            # start training the current task
            for cur_epoch in range(self.start_epoch, self.max_epoch):
                if not self.evaluate_only:
                    logging.info("Start training")
                    # Keep the LAVIS training order here; see https://github.com/salesforce/LAVIS/issues/449
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)

                if len(self.valid_splits) > 0 and (self.evaluate_only or cur_epoch % self.val_freq == 0):
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))
                        val_log = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )
                        if val_log is not None:
                            if is_main_process():
                                assert (
                                    "agg_metrics" in val_log
                                ), "No agg_metrics found in validation log."

                                agg_metrics = val_log["agg_metrics"]
                                if agg_metrics >= best_agg_metric and split_name == "val":
                                    best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                    if not self.evaluate_only:
                                        self._save_checkpoint(cur_task, cur_epoch, best_epoch, best_agg_metric, is_best=True)

                                val_log.update({"best_epoch": best_epoch})
                                self.log_stats(val_log, split_name)

                else:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_task, cur_epoch, best_epoch, best_agg_metric)

                if self.evaluate_only:
                    break

                if self.save_freq > 0 and cur_epoch % self.save_freq == 0:
                    self._save_checkpoint(cur_task, cur_epoch, best_epoch, best_agg_metric)

                dist.barrier()

            if self.save_last and not self.evaluate_only:
                self._save_checkpoint(cur_task, cur_epoch, best_epoch, best_agg_metric)

            if self.start_epoch != 0:
                self.start_epoch = 0

            test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
            self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only, current_task=cur_task)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            logging.info("Task: {}, Training time {}".format(cur_task+1, total_time_str))
            dist.barrier()
            if hasattr(self.task, "final") and not self.evaluate_only:
                self.task.final(
                    cur_task=cur_task,
                    model=self.model,
                    dataloader=self.train_loader,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    cuda_enabled=self.cuda_enabled,
                )
            if self.save_final and not self.evaluate_only:
                # Save the final checkpoint with the task-final state.
                self._save_checkpoint(cur_task, cur_epoch, best_epoch, best_agg_metric, is_final=True)



    def evaluate(self, cur_epoch="best", skip_reload=False, current_task=None):
        if not self.set_direction:
            self.log_config()
            self.set_direction = True
        # TODO: rebuild the ECA network before eval, if eval only
        test_logs = defaultdict(dict)
        if current_task is None:
            for task in range(self.total_task_num):
                self.rebuild_dataset(task, only_test=True)
                self.evaluate(cur_epoch, skip_reload, task)
            return None
        logging.info("Test:  model on task {}".format(current_task+1))
        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                up2now_test_logs, record_up2now_test_logs = self.up2now_evaluate(split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload, current_task=None)

                test_logs.update(up2now_test_logs)

                up2now_avg_scores = {}
                _logs = record_up2now_test_logs[:current_task+1]
                if is_main_process():
                    for key in _logs[0].keys():
                        up2now_avg_scores[key] = sum(log[key] for log in _logs) / len(_logs)
                test_logs['CurrentAvgTest-Task '+str(current_task+1)] = up2now_avg_scores

                for k, v in test_logs.items():
                    self.log_stats(v, split_name+'_'+k)
            return test_logs


    def up2now_evaluate(self, cur_epoch="best", skip_reload=False, current_task=None, split_name=None):
        up2now_logs = defaultdict(dict)
        record_logs = []
        current_task = current_task if current_task is not None else self.total_task_num-1
        logging.info('-'*10+"Up2Now Test"+'-'*10)
        for cur_task in range(current_task+1):
            # rebuild the datasets
            logging.info("Up2Now Test: model on task {}".format(cur_task+1))
            self.rebuild_dataset(cur_task=cur_task, only_test=True, test_type='up2now')
            up2now_logs["Up2nowTest-Task "+str(cur_task+1)] = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                    )
            record_logs.append(up2now_logs["Up2nowTest-Task "+str(cur_task+1)])
        return up2now_logs, record_logs

    def train_epoch(self, epoch):
        self.model.train()
        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )


    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, use provided weights and skip reloading the best checkpoint.
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model


    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
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
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None
                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_task, cur_epoch, best_epoch, best_agg_metric, is_best=False, is_final=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        unchanged_key = model_no_ddp.unchange_keys

        # Save only state that the model marks as changed or trainable.
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in unchanged_key:
                del state_dict[k]


        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
            "task": cur_task,
            "best_epoch": best_epoch,
            "best_agg_metric": best_agg_metric,
        }
        if hasattr(self.task, "save_info"):
            save_obj.update({"save_info": self.task.save_info})
        if hasattr(model_no_ddp, "adapter_structure"):
            save_obj.update({"adapter_structure": model_no_ddp.adapter_structure})
        if hasattr(model_no_ddp, "moq_old_kv"):
            save_obj.update({"moq_old_kv": model_no_ddp.moq_old_kv})

        post_fix = cur_epoch
        if is_best or is_final:
            post_fix = 'best' if is_best else 'final'
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format(post_fix),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

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

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        unwrap_model = self.unwrap_dist_model(self.model)
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        self.task.reload_from_ckpt(checkpoint, unwrap_model)

        try:
            unwrap_model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            unwrap_model.load_state_dict(checkpoint["model"], strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.start_task = checkpoint["task"]
        if self.start_epoch == self.max_epoch:
            self.start_task += 1
            self.start_epoch = 0
        logging.info("Resume checkpoint from {}".format(url_or_filename))
        return checkpoint["best_epoch"], checkpoint["best_agg_metric"]

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        from .utils import ReDirectSTD
        def time_str(fmt=None):
            if fmt is None:
                fmt = '%Y-%m-%d_%H:%M:%S'
            return datetime.datetime.today().strftime(fmt)
        stdout_file = os.path.join(self.output_dir, 'stdout_{}.txt'.format(time_str()))
        stderr_file = os.path.join(self.output_dir, 'stderr_{}.txt'.format(time_str()))
        ReDirectSTD(stdout_file, 'stdout', True)
        ReDirectSTD(stderr_file, 'stderr', True)
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
