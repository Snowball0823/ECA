"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""LLaVA LwF captioning task."""

import logging
import torch
import torch.nn.functional as F
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

from .eca_llava_captioning import ECALlavaCaptionTask


@registry.register_task("eca_llava_lwf_captioning")
class ECALlavaLWFCaptionTask(ECALlavaCaptionTask):
    """Captioning task with lightweight LwF distillation for LLaVA."""

    @classmethod
    def setup_task(cls, cfg):
        base = super().setup_task(cfg)
        base._current_task = 0
        return base

    def before_training(self, cur_task, model, dataloader, optimizer, scaler=None, cuda_enabled=True, **kwargs):
        self._current_task = cur_task
        return super().before_training(
            cur_task=cur_task,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            cuda_enabled=cuda_enabled,
            **kwargs,
        )

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
        unwrap_model = model.module if hasattr(model, "module") else model

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
                samples["return_logits"] = True
                outputs = model(samples)
                loss = outputs["loss"]
                loss_dict = {k: v for k, v in outputs.items() if "loss" in k}
                student_logits = outputs.get("logits")

                lwf_weight = float(getattr(unwrap_model, "kd_weight", 0.0))
                if lwf_weight > 0 and self._current_task > 0 and student_logits is not None:
                    with torch.no_grad():
                        teacher_out = unwrap_model.teacher_forward(
                            samples,
                            use_old_moq=True,
                            use_teacher_adapters=True,
                            use_grad=False,
                        )

                    kd_mask = teacher_out.get("kd_mask")
                    if kd_mask is not None and kd_mask.any().item():
                        teacher_logits = teacher_out["logits"].detach()
                        lwf_loss = F.kl_div(
                            F.log_softmax(student_logits[kd_mask], dim=-1),
                            F.softmax(teacher_logits[kd_mask], dim=-1),
                            reduction="batchmean",
                        )
                        loss = loss + lwf_weight * lwf_loss
                        loss_dict["LwF loss"] = lwf_loss
                        loss_dict["loss"] = loss

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
