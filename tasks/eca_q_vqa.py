"""
 Copyright (c) 2022, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

import datetime
import json
import logging
import os
import os.path
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lavis.common.dist_utils import (get_rank, get_world_size,
                                     is_dist_avail_and_initialized,
                                     is_main_process, main_process)
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.common.utils import cache_url, is_convertible_to_int, is_url
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask
from lavis.common.vqa_tools.vqa_eval import VQAEval
from omegaconf import OmegaConf
from tqdm import tqdm

from .coco_vqa import VQA, convert_to_coco_gt
from .utils import (FIM_METRIC, DictionaryDataset,
                    MiniBatchScaleCodeDictionaryLearner, adapter_fim,
                    cat_samples, create_train_loader, freeze_model,
                    unwrap_dist_model)


@registry.register_task("eca_q_vqa")
class ECAQVQATask(BaseTask):
    def __init__(
            self,
            device,
            train_batch_size,
            num_workers,
            num_beams,
            max_len,
            min_len,
            evaluate,
            report_metric=True,
            sample_id_key="instance_id",
            ques_files=dict(),
            anno_files=dict(),
            prompt="",
            valid_splits=["val"],
            test_splits=None,
            expand_pa=True,
            fim_metric='overlap',
            fim_thr=0.5,
            global_ewc=True,
            lora_visual_first=True,
            dict_cfg: OmegaConf =None,
    ):
        super().__init__()
        self.device = device

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric
        # generalize to non coco data
        self.sample_id_key = sample_id_key

        self.ques_files = ques_files
        self.anno_files = anno_files

        # set for vqa validation's prompot
        self.prompt = prompt

        assert len(valid_splits) == 1, "Only support one val split for evaluation CL now."
        self.valid_splits = valid_splits[0]

        if test_splits is not None:
            assert len(test_splits) == 1, "Only support one test split for evaluation CL now."
            test_splits = test_splits[0]
        self.test_splits = test_splits

        self.expand_pa = expand_pa
        assert fim_metric in FIM_METRIC, "The FIM Metric methods: '{}' is not supported.".format(fim_metric)
        self.fim_metric = FIM_METRIC[fim_metric]
        self.fim_thr = fim_thr
        self.global_ewc = global_ewc
        self.local_ewc = global_ewc
        self.fim_prefix = 'FIMs'
        self.dict_prefix = 'Dictionary'
        self.feature_dict_prefix = 'feature_dict'
        self.ewc_paramters = defaultdict(list)
        self.fishers = defaultdict(dict)
        self.save_info = defaultdict(dict)
        self.save_info[self.fim_prefix] = self.fishers
        self.save_info[self.dict_prefix] = dict()
        self.init_fisher = None
        # dictionary setting
        if dict_cfg is not None:
            self.use_dictionary = dict_cfg.get('use_dictionary', False)
            self.dict_cfg = OmegaConf.to_container(dict_cfg)
            if 'use_dictionary' in self.dict_cfg:
                del(self.dict_cfg['use_dictionary'])
        else:
            self.use_dictionary = False
        # visual encoder expand
        self.lora_visual_first = lora_visual_first

    
    def _dump_dictionary(self):
        '''
        Dump the dictionary after updating the dictionary.
        '''
        self.DICT.update({self.feature_dict_prefix : self.dictionary})


    def _calculate_fisher_step(self, iter, unwrap_model, samples, fisher:dict, scaler=None):
        '''
        args:
            unwrap_model: just model (if using ddp, model.module)
            optimizer: optimizer
            samples: batch_samples
            fisher: inplace fisher dict
        return:
            a fisher dict on cpu
        '''
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss_dict = self.train_step(model=unwrap_model, samples=samples)
            loss = loss_dict['output loss']
            if use_amp:
                loss = scaler.scale(loss)

        bert_parameters = [(n, p) for n, p in unwrap_model.Qformer.bert.named_parameters() if p.requires_grad and 'embeddings' not in n]
        bert_p_name, bert_p_val = zip(*bert_parameters)
        bert_grads = torch.autograd.grad(loss, list(bert_p_val))
        if use_amp:
            bert_grads = [g / scaler.get_scale() for g in bert_grads]

        for n, g in zip(list(bert_p_name), bert_grads):
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= dist.get_world_size()
            _grad = g.detach().clone()
            _fim_diag = _grad.pow(2)
            if len(fisher[n]) > 0:
                _previous_grad = fisher[n][0]
                _previous_fim_diag = fisher[n][1]
                _grad = (_previous_grad*iter+_grad)/(iter+1)
                _fim_diag = (_previous_fim_diag*iter+_fim_diag)/(iter+1)
            fisher[n].clear()
            fisher[n].append(_grad)
            fisher[n].append(_fim_diag)
        return fisher
    
    def _calculate_dictionary_step(self, unwrap_model, samples):
        assert hasattr(unwrap_model, "extract_visual_feature"), "The model must have `extract_visual_feature` function."
        output_dict = unwrap_model.extract_visual_feature(samples)
        output = output_dict['feature'][:,1:,:]
        output = output.reshape(-1, output.shape[-1])
        loss = self.diction_learner.minibatch_fit(output)
        return loss


    @property
    def FIM(self):
        return self.save_info[self.fim_prefix]
    
    @property
    def DICT(self):
        return self.save_info[self.dict_prefix]
    

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        device = torch.device(run_cfg.get("device", "cuda"))
        # for generating
        train_batch_size = run_cfg.batch_size_train
        num_workers = run_cfg.num_workers
        num_beams = run_cfg.get("num_beams", 5)
        max_len = run_cfg.get("max_len", 30)
        min_len = run_cfg.get("min_len", 1)
        evaluate = run_cfg.evaluate
        # for building datasets
        report_metric = run_cfg.get("report_metric", True)

        prompt = run_cfg.get("prompt", "")
        # generalize to non coco data
        sample_id_key = run_cfg.get("sample_id_key", "instance_id")
        ques_files = run_cfg.get("ques_files", dict())
        anno_files = run_cfg.get("anno_files", dict())
        valid_splits = run_cfg.get("valid_splits", ["val"])
        test_splits = run_cfg.get("test_splits", None)
        # for FeDEx
        expand_pa = run_cfg.get("expand_pa", True)
        fim_metric = run_cfg.get("fim_metric", 'p-jsd')
        fim_thr = run_cfg.get("fim_threshold", 0.5)
        # for EWC penalty
        global_ewc = run_cfg.get("use_ewc", True)
        # for dictionary
        dict_cfg = run_cfg.get('dictionary_learning', None)
        # for visual encoder expand
        lora_visual_first = run_cfg.get("lora_visual_first", True)

        return cls(
            device=device,
            train_batch_size=train_batch_size,
            num_workers=num_workers,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
            sample_id_key=sample_id_key,
            valid_splits=valid_splits,
            test_splits=test_splits,
            prompt=prompt,
            ques_files=ques_files,
            anno_files=anno_files,
            expand_pa=expand_pa,
            fim_metric=fim_metric,
            fim_thr=fim_thr,
            global_ewc=global_ewc,
            dict_cfg=dict_cfg,
            lora_visual_first=lora_visual_first,
        )
    

    def reload_from_ckpt(self, ckpt, unwrap_model, cuda_enabled=True):
        # update unwarp_model first
        assert hasattr(unwrap_model, 'rebuild_from_config'), 'The model is not supported to rebuild by this task.'
        model_adapter_structure = ckpt['adapter_structure'] if 'adapter_structure' in ckpt else None
        model_moq_old_kv = ckpt['moq_old_kv'] if 'moq_old_kv' in ckpt else None
        unwrap_model.rebuild_from_config(model_adapter_structure, model_moq_old_kv, task_num=ckpt['task'])
        
        index = unwrap_model.current_adapter_index
        save_info = ckpt['save_info']
        self.DICT.clear()
        self.DICT.update(save_info[self.dict_prefix])

        self.FIM.clear()
        self.FIM.update(save_info[self.fim_prefix])

        if self.local_ewc:
            self.ewc_paramters.clear()
            parameters = {
                n: p.clone().detach().to(self.device)
                for n, p in unwrap_model.Qformer.bert.named_parameters()
                if p.requires_grad
            }
            self.ewc_paramters.update(parameters)
            # original EWC using multitask=False
            adapter_fisher = adapter_fim(self.FIM[index], cuda_enabled=cuda_enabled, multitask=False)
            setattr(self, 'adapter_fisher', adapter_fisher)
            self.save_info.update({'adapter_fisher': getattr(self, 'adapter_fisher')})

        # init dictionary
        if self.use_dictionary:
            num_features = unwrap_model.visual_encoder.config.hidden_size
            self.init_dictionary(num_features=num_features)
            if 'dict_learner_inner_state' in save_info:
                state_dict = save_info['dict_learner_inner_state']
                self.diction_learner.reload_inner_state(state_dict)
    

    def rebuild_dict_dataset(self, cur_task):
        if cur_task > 0 and self.use_dictionary:
            atom_num = self.dict_cfg.get('n_components', None)
            dict_index = torch.tensor(range(atom_num))

            setattr(self, 'dict_dataset_index', dict_index)
    


    def build_datasets(self, cfg):
        '''
        Build the CL datasets
        NOTE:
            Assume all dataset do not have test dataset, which means all metric on validation. And set the test dataset as the same as val.
        '''
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        assert len(datasets_config) < 2, "Only support one dataset for CL."
        name = list(datasets_config.keys())[0]
        assert 'cl' in name, "Please use the `cl` version datasets."
        dataset_config = datasets_config[name]
        builder = registry.get_builder_class(name)(dataset_config)
        dataset = builder.build_datasets()
        datasets[name] = dataset

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            splits = [self.valid_splits] if self.test_splits is None else [self.valid_splits, self.test_splits]
            for split in splits:
                if split not in dataset:
                    print(f"Split {split} not found in {ds_name}.")
                if (hasattr(dataset[split], "coco_fmt_qust_file") and dataset[split].coco_fmt_qust_file is not None):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file
                else:
                    if split not in self.ques_files: # precomputed and passed in task builder
                        self.ques_files[split] = os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt', f'{ds_name}_{split}_questions.json')
                        self.anno_files[split] = os.path.join(registry.get_path("cache_root"), f'{ds_name}_gt', f'{ds_name}_{split}_annotations.json')
                        if is_main_process() and (not os.path.exists(self.ques_files[split]) or not os.path.exists(self.anno_files[split])):
                            os.makedirs(os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt'), exist_ok=True)
                            try:
                                convert_to_coco_gt(dataset, self.ques_files[split], self.anno_files[split], split, self.sample_id_key)
                            except:
                                logging.info("{ds_name} do not have Ground Truth Answer.")
                                # tasks like vizwiz with no gt answer
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    logging.info("{ds_name} {split} do not have answer list.")
                    # if answer_list is not provided, then set it to None
    

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Number of question files and annotation files are not equal."

        return datasets

    

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        assert "blip2".lower() in model_config.arch, "The `eca_q_vqa` only support BLIP2-based model."
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

        
    def init_dictionary(self, num_features: int=None):
        old_dictionary = None
        if len(self.DICT)>0:
            old_dictionary = self.DICT[self.feature_dict_prefix]
        atom_num = self.dict_cfg.get('n_components', None)
        assert atom_num is not None, 'The atom number has to be setted! Please go to config file, and set the atom number!'
        self.dictionary = old_dictionary.to(self.device) if old_dictionary is not None else torch.nn.init.orthogonal_(torch.empty(atom_num, num_features).to(self.device))
        old_dictionary = old_dictionary.T if old_dictionary is not None else old_dictionary
        self.diction_learner = self.init_dictionary_learner(old_dict=old_dictionary)


    def valid_step(self, model, samples):
        # top_p, temperature always be default top_p=0.9, temperature=1.0 same in VQA and Cap
        # length_penalty for OPT, T5 with Cap, is 1.0, OPT for VQA=0, T5 for VQA=-1
        # repetition_penalty for OPT with Cap, is 1.15, not sure for T5 cap. For VQA, no repetition_penalty.
        # For open-end generation task, inference_method and num_ans_candidates are not supported.
        # For muli-classification, then inference_method='rank', num_ans_candidates=128
        # prompt is setted at task for VQA, but is setted at model for Cap.
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            if ques_id != int and is_convertible_to_int(ques_id):
                ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs


    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )
        if self.report_metric:
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = {"agg_metrics": 0.0}
        return metrics
    

    def before_training(self, cur_task, model, dataloader, optimizer, scaler=None, cuda_enabled=True, **kwargs):
        fim_dc = -1
        expand = False
        lora_visual = False
        unwrap_model = unwrap_dist_model(model)
        index = unwrap_model.current_adapter_index
        if cur_task > 0:
            if self.use_dictionary:
                if hasattr(self, 'dictionary_loader'):
                    delattr(self, 'dictionary_loader')
                # creat the dictionary dataloader
                assert hasattr(self, 'dict_dataset_index'), 'rebuild dict dataset first.'
                batch_sizes = self.train_batch_size
                dictionary = self.DICT[self.feature_dict_prefix].cpu()

                dict_index = getattr(self, 'dict_dataset_index')

                repeat_factor = max(int(getattr(self, "dict_cfg", {}).get("repeat_factor", 1)), 1)
                dictionary_dataset = DictionaryDataset(dictionary[dict_index], repeat_factor=repeat_factor)
                dictionary_dataloader = create_train_loader(
                                            dictionary_dataset,
                                            num_workers=self.num_workers,
                                            batch_sizes=batch_sizes,
                                            use_distributed=is_dist_avail_and_initialized(),
                                            collate_fns=None,
                                        )
                setattr(self, 'dictionary_loader', dictionary_dataloader)

            # TODO: change to Class later -> old_model: .qformer_bert, .query_tokens, .keys_set, .queries_set, .nlp_proj
            if hasattr(self, 'old_model_dict'):
                delattr(self, 'old_model_dict')
            old_model_dict = dict()
            old_model_dict['qformer_bert'] = deepcopy(unwrap_model.Qformer.bert)
            freeze_model(old_model_dict['qformer_bert'])
            old_model_dict['query_tokens'] = deepcopy(unwrap_model.query_tokens)
            if hasattr(unwrap_model, 'keys_set') and hasattr(unwrap_model, 'queries_set'):
                old_model_dict['keys_set'] = deepcopy(unwrap_model.keys_set.detach().clone())
                old_model_dict['queries_set'] = deepcopy(unwrap_model.queries_set.detach().clone())
            if hasattr(unwrap_model, 'current_keys') and hasattr(unwrap_model, 'current_queries'):
                old_model_dict['current_keys'] = deepcopy(unwrap_model.current_keys.detach().clone())
                old_model_dict['current_queries'] = deepcopy(unwrap_model.current_queries.detach().clone())
            old_model_dict['nlp_proj'] = deepcopy(unwrap_model.nlp_proj)
            freeze_model(old_model_dict['nlp_proj'])
            setattr(self, 'old_model_dict', old_model_dict)

        if self.lora_visual_first and cur_task == 0:
            lora_visual = True
        if self.expand_pa and cur_task == 0:
            expand = True
        if cur_task > 0 and self.expand_pa:
            logging.info("Start calculating Initial FIM for Task {}".format(cur_task+1))
            fisher_start = time.time()
            unwrap_model.train()
            self.init_fisher = self.update_Q_fisher(cur_task, unwrap_model, dataloader, optimizer, scaler=scaler, cuda_enabled=cuda_enabled)
            fisher_time = time.time()-fisher_start
            fisher_time_str = str(datetime.timedelta(seconds=int(fisher_time)))
            logging.info("Task: {}, Fetching Initial FIM time {}".format(cur_task+1, fisher_time_str))
            for task in self.FIM[index]:
                task_num = int(task.replace('task_',''))
                if task_num != cur_task:
                    previous_fisher = self.FIM[index][task]
                    fim_dc = self.fim_metric(previous_fisher, self.init_fisher)
                    logging.info("Impact factor with Task {} is: {:.4f}.".format(task_num+1, fim_dc))
                    if fim_dc > self.fim_thr:
                        expand = True
                        logging.info("Impact factor with Task {} exceed. Expanding Parallel Adapter".format(task_num+1))
                        task_num+=1
                        break
            if not expand:
                logging.info("Keep Parallel Adapter")

            del(self.FIM[index]['task_'+str(cur_task)])

        if fim_dc != -1:
            logging.info("The Fisher Dictiance Correlation between Task {} and Task {} is: {:.4f}".format(task_num, cur_task+1, fim_dc))

        self.local_ewc = self.global_ewc & (not expand) & (cur_task > 0)

        unwrap_model.before_training(expand_q_former=expand, lora_visual=lora_visual)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if self.local_ewc:
            self.ewc_paramters.clear()
            parameters = {
                n: p.clone().detach().to(self.device)
                for n, p in unwrap_model.Qformer.bert.named_parameters()
                if p.requires_grad
            }
            self.ewc_paramters.update(parameters)
            # original EWC using multitask=False
            adapter_fisher = adapter_fim(self.FIM[index], cuda_enabled=cuda_enabled, multitask=False)
            setattr(self, 'adapter_fisher', adapter_fisher)
            self.save_info.update({'adapter_fisher': getattr(self, 'adapter_fisher')})

        # init dictionary
        if self.use_dictionary and not hasattr(self, "dictionary"):
            num_features = unwrap_model.visual_encoder.config.hidden_size
            self.init_dictionary(num_features=num_features)


    def final(self, cur_task, model, dataloader, optimizer, scaler=None, cuda_enabled=True):
        # delete dictionary loader
        if hasattr(self, 'dictionary_loader'):
            delattr(self, 'dictionary_loader')

        unwrap_model = unwrap_dist_model(model)
        # final is based on "best" model or cur_epoch model
        if self.expand_pa or self.global_ewc:
            logging.info("Start calculating Final FIM for Task {}".format(cur_task+1))
            fisher_start = time.time()
            unwrap_model.train()
            self.update_Q_fisher(cur_task, unwrap_model, dataloader, optimizer, scaler=scaler, cuda_enabled=cuda_enabled)
            fisher_time = time.time()-fisher_start
            fisher_time_str = str(datetime.timedelta(seconds=int(fisher_time)))
            logging.info("Task: {}, Fetching Final FIM time {}".format(cur_task+1, fisher_time_str))

        if self.use_dictionary:
            unwrap_model.eval()
            logging.info("Updating Feature Dictionary for Task {}".format(cur_task+1))
            self.update_dictionary(unwrap_model, dataloader, cuda_enabled=cuda_enabled)
            self.save_info.update({'dict_learner_inner_state':self.diction_learner.get_inner_state()})

    def update_Q_fisher(self, task_num:int, unwrap_model, data_loader, optimizer, scaler=None, cuda_enabled=True):
        """
        Fetch the FIM without amp.
        """
        iter_num = len(data_loader)
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)
        _fisher = defaultdict(list)
        # set progress bar
        if is_main_process():
            progress = tqdm(range(iter_num), desc="Fetch QFormer Fisher", mininterval=0.2)
        else:
            progress = range(iter_num)

        for _iter in progress:
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            _fisher = self._calculate_fisher_step(_iter, unwrap_model, samples, _fisher)


        for n, p in _fisher.items():
            _grad = p[0]
            _fim = p[1]
            dist.all_reduce(_fim, op=dist.ReduceOp.SUM)
            dist.all_reduce(_grad, op=dist.ReduceOp.SUM)
            _fim /= get_world_size()
            _grad /= get_world_size()
            _fisher[n] = [_grad.cpu(), _fim.cpu()]

        index = unwrap_model.current_adapter_index
        self.fishers[index].update({'task_'+str(task_num):_fisher})
        return _fisher
    

    def update_dictionary(self, unwrap_model, data_loader, cuda_enabled=True):
        """
        Update dictionary.
        """
        iter_num = len(data_loader)
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)
        # if iter-based runner, schedule lr based on inner epoch.
        batchs_samples = []
        accum_batch = self.dict_cfg.get('accum_batch', 1)
        total_iter = self.dict_cfg.get('iter', 1)


        for _ in range(total_iter):
            for _iter in range(iter_num):
                samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                batchs_samples.append(samples)
                if (_iter+1) % accum_batch == 0 or iter_num-1 < _iter+accum_batch:
                    _samples = cat_samples(batchs_samples)
                    loss = self._calculate_dictionary_step(unwrap_model, _samples)
                    dictionary = self.diction_learner.dictionary.T
                    dictionary = dictionary.to_dense().contiguous()
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(dictionary, op=dist.ReduceOp.SUM)
                    world_size = dist.get_world_size()
                    loss_mean = loss / world_size
                    dictionary_mean = dictionary / world_size
                    if is_main_process():
                        if (_iter+1) % 50 == 0:
                            logging.info('It {}/{} Dictionary Learning Loss: {:.3f}'.format(_iter+1, iter_num, loss_mean.data.cpu().item()))
                    
                    self.dictionary = dictionary_mean.to_dense().contiguous()
                    self.diction_learner.dictionary = self.dictionary.T
                    batchs_samples.clear()


        self._dump_dictionary()


    @torch.no_grad()
    def extract_query_diction(self, unwarp_model, feature_dictionary: torch.Tensor):
        assert hasattr(self, 'old_model_dict'), 'Must set the arttribution `old_model_dict` first.'
        old_model_dict = getattr(self, 'old_model_dict')
        if len(feature_dictionary.shape)==2:
            feature_dictionary = feature_dictionary.unsqueeze(1)
        feature_atts = torch.ones(feature_dictionary.size()[:-1], dtype=torch.long).to(
            feature_dictionary.device
        )       
        query_tokens = old_model_dict['query_tokens']
        query_tokens = unwarp_model.mixture_of_query(feature_dictionary, query_tokens, old_only=True)
        bert = old_model_dict['qformer_bert']
        nlp_proj = old_model_dict['nlp_proj']
        query_output = bert(
            query_embeds=query_tokens,
            encoder_hidden_states=feature_dictionary,
            encoder_attention_mask=feature_atts,
            return_dict=True,
        )
        query_connect = nlp_proj(query_output.last_hidden_state)
        return query_connect, query_tokens
        


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
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        unwrap_model = unwrap_dist_model(model)

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        dictionary_loader = None
        if hasattr(self, 'dictionary_loader'):
            dictionary_loader = getattr(self, 'dictionary_loader')
            if not hasattr(dictionary_loader, "__next__"):
                # convert to iterator if not already
                dictionary_loader = iter(dictionary_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            ## notify model that sample is empty (error occured)
            if not isinstance(samples, dict):
                samples = {"is_empty":True}

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
                query_dictionaries, query_token = self.extract_query_diction(unwrap_model, dictionaries['feature'])
                dictionaries.update({'query': query_dictionaries, 'query_token': query_token})
                samples.update(dictionaries)

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                # EWC panelty
                if self.local_ewc:
                    ewc_loss = 0
                    adapter_fisher = getattr(self, 'adapter_fisher')
                    for n, p in unwrap_model.Qformer.bert.named_parameters():
                        if n in adapter_fisher.keys():
                            ewc_loss += (torch.sum((adapter_fisher[n]) * (
                                p[: len(self.ewc_paramters[n])] - self.ewc_paramters[n]).pow(2)) / 2)
                    loss += ewc_loss
                    loss_dict['loss'] = loss
                    loss_dict.update({'EWC loss': ewc_loss})

                loss /= accum_grad_iters
                loss_dict['loss'] = loss

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()

                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

        

    # @main_process
    def init_dictionary_learner(self, old_dict=None):
        return MiniBatchScaleCodeDictionaryLearner(
            dict_init=old_dict,
            device=self.device,
            **self.dict_cfg
        )
    

    @main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )
            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate(quesIds=vqa_result.getQuesIds())

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics
