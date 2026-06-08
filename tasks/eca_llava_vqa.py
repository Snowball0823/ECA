"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""ECA continual VQA task for LLaVA alignment models."""

import datetime
import logging
import time
from collections import defaultdict

import torch
import torch.distributed as dist
from lavis.common.dist_utils import (
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
)
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from tqdm import tqdm

from .eca_q_vqa import ECAQVQATask
from .utils import (
    DictionaryDataset,
    adapter_fim,
    create_train_loader,
    unwrap_dist_model,
)


@registry.register_task("eca_llava_vqa")
class ECALlavaVQATask(ECAQVQATask):
    """VQA task variant tailored for the LLaVA-based ECA model."""

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)
    
    @classmethod
    def setup_task(cls, cfg):
        base = super().setup_task(cfg)
        # ensure dictionary learner points to correct class
        return base

    def init_dictionary(self, num_features: int = None):
        super().init_dictionary(num_features=num_features)

    def reload_from_ckpt(self, ckpt, unwrap_model, cuda_enabled=True):
        assert hasattr(
            unwrap_model, "rebuild_from_config"
        ), "The model is not supported to rebuild by this task."
        model_adapter_structure = ckpt.get("adapter_structure")
        model_moq_old_kv = ckpt.get("moq_old_kv")
        unwrap_model.rebuild_from_config(
            model_adapter_structure, model_moq_old_kv, task_num=ckpt["task"]
        )

        index = unwrap_model.current_adapter_index
        save_info = ckpt["save_info"]
        self.DICT.clear()
        self.DICT.update(save_info[self.dict_prefix])

        self.FIM.clear()
        self.FIM.update(save_info[self.fim_prefix])

        if self.local_ewc:
            self.ewc_paramters.clear()
            params = {
                name: param.clone().detach().to(self.device)
                for name, param in unwrap_model.trainable_adapter_parameters
            }
            self.ewc_paramters.update(params)
            adapter_fisher = adapter_fim(
                self.FIM[index], cuda_enabled=cuda_enabled, multitask=False
            )
            setattr(self, "adapter_fisher", adapter_fisher)
            self.save_info.update({"adapter_fisher": getattr(self, "adapter_fisher")})

        if self.use_dictionary:
            num_features = unwrap_model.visual_feature_dim
            self.init_dictionary(num_features=num_features)
            if "dict_learner_inner_state" in save_info:
                state_dict = save_info["dict_learner_inner_state"]
                self.diction_learner.reload_inner_state(state_dict)

    # ------------------------------------------------------------------
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None
        unwrap_model = unwrap_dist_model(model)

        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        dictionary_loader = getattr(self, "dictionary_loader", None)
        if dictionary_loader is not None and not hasattr(dictionary_loader, "__next__"):
            dictionary_loader = iter(dictionary_loader)
            setattr(self, "dictionary_loader", dictionary_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            if dictionary_loader is not None:
                dictionaries = next(dictionary_loader)
                dictionaries = prepare_sample(dictionaries, cuda_enabled=cuda_enabled)
                replay_feature = dictionaries.get("feature")
                patch_len = getattr(self, "_dict_patch_len", None)
                if replay_feature is not None and replay_feature.dim() == 2 and patch_len:
                    replay_feature = replay_feature.view(
                        -1, patch_len, replay_feature.size(-1)
                    )
                    dictionaries["feature"] = replay_feature
                samples.update(dictionaries)

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                if self.local_ewc:
                    ewc_loss = 0.0
                    adapter_fisher = getattr(self, "adapter_fisher")
                    for name, param in unwrap_model.trainable_adapter_parameters:
                        if name in adapter_fisher and name in self.ewc_paramters:
                            ref_param = self.ewc_paramters[name]
                            ewc_loss += (
                                torch.sum(adapter_fisher[name] * (param - ref_param).pow(2))
                                / 2
                            )
                    loss += ewc_loss
                    loss_dict["loss"] = loss
                    loss_dict.update({"EWC loss": ewc_loss})

                loss /= accum_grad_iters
                loss_dict["loss"] = loss

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    # ------------------------------------------------------------------
    def _calculate_adapters_fisher_step(self, iter_idx, unwrap_model, samples, fisher, scaler=None):
        _, loss_dict = self.train_step(model=unwrap_model, samples=samples)
        loss = loss_dict["output loss"]

        trainable_params = unwrap_model.trainable_adapter_parameters
        if not trainable_params:
            return fisher

        parameters = [(name, value) for name, value in trainable_params]
        names, params = zip(*parameters)
        grads = torch.autograd.grad(loss, params, allow_unused=True)

        world = dist.get_world_size() if dist.is_initialized() else 1
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            if dist.is_initialized():
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad /= world
            grad_det = grad.detach().clone()
            fim = grad_det.pow(2)
            if len(fisher[name]) > 0:
                prev_grad, prev_fim = fisher[name]
                grad_det = (prev_grad * iter_idx + grad_det) / (iter_idx + 1)
                fim = (prev_fim * iter_idx + fim) / (iter_idx + 1)
            fisher[name].clear()
            fisher[name].append(grad_det)
            fisher[name].append(fim)
        return fisher

    # ------------------------------------------------------------------
    def before_training(
        self,
        cur_task,
        model,
        dataloader,
        optimizer,
        scaler=None,
        cuda_enabled=True,
        **kwargs,
    ):
        fim_dc = -1
        expand = False
        lora_visual = False
        unwrap_model = unwrap_dist_model(model)
        patch_len = getattr(self, "_dict_patch_len", None)
        if self.use_dictionary and patch_len is None:
            vision = unwrap_model.vision_tower
            if vision is None:
                raise RuntimeError(
                    "Vision tower is not initialized; cannot determine patch number."
                )
            patch_len = getattr(vision, "num_patches", None)
            if patch_len is None:
                raise RuntimeError(
                    "Vision tower does not expose num_patches; please ensure LLaVA v0 vision tower is used."
                )
            self._dict_patch_len = patch_len

        index = unwrap_model.current_adapter_index
        if cur_task > 0 and self.use_dictionary:
            if hasattr(self, "dictionary_loader"):
                delattr(self, "dictionary_loader")
            assert hasattr(
                self, "dict_dataset_index"
            ), "rebuild dict dataset first."
            batch_sizes = self.train_batch_size // 2 if self.train_batch_size > 1 else 1
            if patch_len:
                batch_sizes *= patch_len
            dictionary = self.DICT[self.feature_dict_prefix].cpu()
            dict_index = getattr(self, "dict_dataset_index")
            repeat_factor = max(int(getattr(self, "dict_cfg", {}).get("repeat_factor", 1)), 1)
            dictionary_dataset = DictionaryDataset(dictionary[dict_index], repeat_factor=repeat_factor)
            dictionary_dataloader = create_train_loader(
                dictionary_dataset,
                num_workers=self.num_workers,
                batch_sizes=batch_sizes,
                use_distributed=is_dist_avail_and_initialized(),
                collate_fns=None,
            )
            setattr(self, "dictionary_loader", dictionary_dataloader)

        if self.lora_visual_first and cur_task == 0:
            lora_visual = True
        if self.expand_pa and cur_task == 0:
            expand = True
        if cur_task > 0 and self.expand_pa:
            logging.info("Start calculating Initial FIM for Task {}".format(cur_task + 1))
            fisher_start = time.time()
            unwrap_model.train()
            self.init_fisher = self.update_adapters_fisher(
                cur_task, unwrap_model, dataloader, optimizer, scaler=scaler, cuda_enabled=cuda_enabled
            )
            fisher_time = time.time() - fisher_start
            fisher_time_str = str(datetime.timedelta(seconds=int(fisher_time)))
            logging.info(
                "Task: {}, Fetching Initial FIM time {}".format(cur_task + 1, fisher_time_str)
            )
            for task in self.FIM[index]:
                task_num = int(task.replace("task_", ""))
                if task_num != cur_task:
                    previous_fisher = self.FIM[index][task]
                    fim_dc = self.fim_metric(previous_fisher, self.init_fisher)
                    logging.info(
                        "Impact factor with Task {} is: {:.4f}.".format(
                            task_num + 1, fim_dc
                        )
                    )
                    if fim_dc > self.fim_thr:
                        expand = True
                        logging.info(
                            "Impact factor with Task {} exceed. Expanding Parallel Adapter".format(
                                task_num + 1
                            )
                        )
                        task_num += 1
                        break
            if not expand:
                logging.info("Keep Parallel Adapter")

            del self.FIM[index]["task_" + str(cur_task)]

        if fim_dc != -1:
            logging.info(
                "The Fisher Distance Correlation between Task %d and Task %d is: %.4f",
                task_num,
                cur_task + 1,
                fim_dc,
            )

        self.local_ewc = self.global_ewc & (not expand) & (cur_task > 0)
        unwrap_model.before_training(expand_adapters=expand, lora_visual=lora_visual)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if self.local_ewc:
            self.ewc_paramters.clear()
            params = {
                name: param.clone().detach().to(self.device)
                for name, param in unwrap_model.trainable_adapter_parameters
            }
            self.ewc_paramters.update(params)
            adapter_fisher = adapter_fim(
                self.FIM[index], cuda_enabled=cuda_enabled, multitask=False
            )
            setattr(self, "adapter_fisher", adapter_fisher)
            self.save_info.update({"adapter_fisher": getattr(self, "adapter_fisher")})

        if self.use_dictionary and not hasattr(self, "dictionary"):
            num_features = unwrap_model.visual_feature_dim
            self.init_dictionary(num_features=num_features)

    # ------------------------------------------------------------------
    def update_adapters_fisher(
        self,
        task_num,
        unwrap_model,
        data_loader,
        optimizer,
        scaler=None,
        cuda_enabled=True,
    ):
        iter_num = len(data_loader)
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)
        fisher = defaultdict(list)
        iterator = (
            tqdm(range(iter_num), desc="Fetch LLaMA Fisher", mininterval=0.2)
            if is_main_process()
            else range(iter_num)
        )
        for idx in iterator:
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            fisher = self._calculate_adapters_fisher_step(
                idx, unwrap_model, samples, fisher, scaler
            )

        world = get_world_size() if is_dist_avail_and_initialized() else 1
        for name, value in fisher.items():
            grad, fim = value
            if is_dist_avail_and_initialized():
                dist.all_reduce(fim, op=dist.ReduceOp.SUM)
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                fim /= world
                grad /= world
            fisher[name] = [grad.cpu(), fim.cpu()]

        index = unwrap_model.current_adapter_index
        self.fishers[index].update({f"task_{task_num}": fisher})
        return fisher

    # ------------------------------------------------------------------
    def final(self, cur_task, model, dataloader, optimizer, scaler=None, cuda_enabled=True):
        if hasattr(self, "dictionary_loader"):
            delattr(self, "dictionary_loader")

        unwrap_model = model.module if hasattr(model, "module") else model
        if self.expand_pa or self.global_ewc:
            logging.info("Start calculating Final FIM for Task %d", cur_task + 1)
            fisher_start = time.time()
            unwrap_model.train()
            self.update_adapters_fisher(
                cur_task,
                unwrap_model,
                dataloader,
                optimizer,
                scaler=scaler,
                cuda_enabled=cuda_enabled,
            )
            elapsed = time.time() - fisher_start
            logging.info(
                "Task: %d, Fetching Final FIM time %s",
                cur_task + 1,
                str(datetime.timedelta(seconds=int(elapsed))),
            )

        if self.use_dictionary:
            unwrap_model.eval()
            logging.info("Updating Feature Dictionary for Task %d", cur_task + 1)
            self.update_dictionary(unwrap_model, dataloader, cuda_enabled=cuda_enabled)
            self.save_info.update(
                {"dict_learner_inner_state": self.diction_learner.get_inner_state()}
            )

    # ------------------------------------------------------------------
    def _calculate_dictionary_step(self, unwrap_model, samples):
        assert hasattr(
            unwrap_model, "extract_visual_feature"
        ), "Model must implement extract_visual_feature."
        output_dict = unwrap_model.extract_visual_feature(samples)
        output = output_dict["feature"]
        output = output.reshape(-1, output.shape[-1]).to(self.dictionary.dtype)
        loss = self.diction_learner.minibatch_fit(output)
        return loss
