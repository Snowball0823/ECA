"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""
InternVL-specific continual TextVQA datasets.
"""

import logging
import os
import random
from collections import OrderedDict
from copy import deepcopy

import torch
from lavis.common.registry import registry
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


def _unpack_processed_image(processed):
    if isinstance(processed, dict):
        pixel_values = processed["pixel_values"]
        num_patches_list = processed.get("num_patches_list", [pixel_values.size(0)])
        if isinstance(num_patches_list, (list, tuple)):
            if len(num_patches_list) != 1:
                raise ValueError("Per-sample InternVL preprocessing must return a single patch count.")
            num_patches = int(num_patches_list[0])
        else:
            num_patches = int(num_patches_list)
        return pixel_values, num_patches

    if torch.is_tensor(processed):
        if processed.dim() == 3:
            return processed.unsqueeze(0), 1
        if processed.dim() == 4:
            return processed, int(processed.size(0))
    raise TypeError(f"Unsupported InternVL image payload: {type(processed)}")


class CLTextVQAInternVLDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.ocr_processer = registry.get_processor_class("blip_caption").from_config()
        self.ocr_prompt = "Based on OCR: {}."
        self.annotation = self.annotation[3]["data"]
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1
            assert "image_path" in ann, "Update the dataset file first, add 'image_path' into it."
            ann["image"] = ann["image_path"]

        self.__check_cat()
        self.cats = {ann["label"] for ann in self.annotation if "label" in ann}

    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if "label" not in ann:
                no_label += 1
        assert no_label / total < 0.8, "The rate of data without label is higher than 80%! Please check CL data again!"
        if no_label > 0:
            logging.warning("The rate of data without label is: %s", no_label / total)

    def rebuild(self, coco_cats: list):
        if not hasattr(self, "original_annotation"):
            setattr(self, "original_annotation", deepcopy(self.annotation))
        self.annotation.clear()
        anns = [i for i in self.original_annotation if "label" in i]
        self.annotation.extend([i for i in anns if i["label"] in coco_cats])
        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        processed = self.vis_processor(image)
        image, num_patches = _unpack_processed_image(processed)

        ocr_tokens = " ".join(ann["ocr_tokens"][:30])
        question = self.text_processor(ann["question"])
        ocr = self.ocr_prompt.format(self.ocr_processer(ocr_tokens))

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight:
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())
        answer = random.choice(answers)

        return {
            "image": image,
            "num_patches_list": num_patches,
            "text_input": question,
            "text_output": answer,
            "ocr_input": ocr,
            "answers": answers,
            "weights": weights,
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        image_list, num_patches_list = [], []
        question_list, answer_list, weight_list = [], [], []
        ocr_list, num_answers, text_output = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            num_patches_list.append(int(sample["num_patches_list"]))
            question_list.append(sample["text_input"])
            ocr_list.append(sample["ocr_input"])
            weight_list.extend(sample["weights"])
            answers = sample["answers"]
            text_output.extend(random.choices(answers, weights=sample["weights"], k=1))
            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.cat(image_list, dim=0),
            "num_patches_list": num_patches_list,
            "text_input": question_list,
            "answer": answer_list,
            "text_output": text_output,
            "ocr_input": ocr_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
        }


class CLTextVQAInternVLEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.ocr_processer = registry.get_processor_class("blip_caption").from_config()
        self.ocr_prompt = "Based on OCR: {}."

        self.annotation = self.annotation[3]["data"]
        self.annotation = [ann for ann in self.annotation if "answers" in ann]

        self.__check_cat()
        self.cats = {ann["label"] for ann in self.annotation if "label" in ann}
        self.original_annotation = deepcopy(self.annotation)

        self.answer_list = None
        self.coco_fmt_qust_file = None
        self.coco_fmt_anno_file = None

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1
            assert "image_path" in ann, "Update the dataset file first, add 'image_path' into it."
            ann["image"] = ann["image_path"]
        self._add_instance_ids()
        self.original_annotation = deepcopy(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        processed = self.vis_processor(image)
        image, num_patches = _unpack_processed_image(processed)

        ocr_tokens = " ".join(ann["ocr_tokens"][:30])
        ocr = self.ocr_prompt.format(self.ocr_processer(ocr_tokens))
        question = self.text_processor(ann["question"])
        answers = list(ann["answers"])

        return {
            "image": image,
            "num_patches_list": num_patches,
            "text_input": question,
            "ocr_input": ocr,
            "answers": answers,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}

        return {
            "image": torch.cat([sample["image"] for sample in samples], dim=0),
            "num_patches_list": [int(sample["num_patches_list"]) for sample in samples],
            "text_input": [sample["text_input"] for sample in samples],
            "ocr_input": [sample["ocr_input"] for sample in samples],
            "answers": [sample["answers"] for sample in samples],
            "question_id": [sample["question_id"] for sample in samples],
            "instance_id": [sample["instance_id"] for sample in samples],
        }

    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if "label" not in ann:
                no_label += 1
        assert no_label / total < 0.5, "The rate of data without label is higher than 50%! Please check CL data again!"
        if no_label > 0:
            logging.warning("The rate of data without label is: %s", no_label / total)

    def rebuild(self, coco_cats: list):
        self.annotation.clear()
        anns = [i for i in self.original_annotation if "label" in i]
        self.annotation += [i for i in anns if i["label"] in coco_cats]
