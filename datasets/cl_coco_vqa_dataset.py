"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

import json
import logging
import os
import random
from collections import OrderedDict
from copy import deepcopy

import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class CLCOCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}


    def rebuild(self, coco_cats: list):
        # regenerate the self.annotation
        if not hasattr(self, 'original_annotation'):
            setattr(self, 'original_annotation', deepcopy(self.annotation))
        self.annotation.clear()
        anns = [i for i in self.original_annotation if 'label' in i]
        self.annotation.extend([i for i in anns if i['label'] in coco_cats])
        self._add_instance_ids()


    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None # image does not exist

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = random.choice(answers)

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "answers": answers,
            "weights": weights,
        }
    
    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if 'label' not in ann:
                no_label += 1
        assert no_label/total < 0.8, 'The rate of data without label is higher than 80%! Please check CL data again!'
        if no_label > 0:
            logging.warn("The rate of data without label is: "+str(no_label/total))


    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        image_list, question_list, answer_list, weight_list = [], [], [], []

        num_answers, text_output = [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            text_output.extend(random.choices(answers, weights=sample["weights"], k=1))

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "text_output": text_output,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
        }
            

class CLCOCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        self.vis_root = vis_root
        self.annotation = json.load(open(ann_paths[0]))

        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

    def rebuild(self, coco_cats: list):
        # regenerate the self.annotation
        if not hasattr(self, 'original_annotation'):
            setattr(self, 'original_annotation', deepcopy(self.annotation))
        self.annotation.clear()
        # regenerate the self.annotation
        anns = [i for i in self.original_annotation if 'label' in i]
        self.annotation += [i for i in anns if i['label'] in coco_cats]

    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if 'label' not in ann:
                no_label += 1
        assert no_label/total < 0.5, 'The rate of data without label is higher than 50%! Please check CL data again!'
        if no_label > 0:
            logging.warn("The rate of data without label is: "+str(no_label/total))