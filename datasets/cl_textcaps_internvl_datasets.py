"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""InternVL-specific continual TextCaps datasets."""

import logging
import os
from copy import deepcopy

import torch
from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from PIL import Image


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


class CLTextCapsInternVLDataset(CaptionDataset):
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
            ann["image"] = ann["image_path"]
            ann["caption"] = ann["caption_str"]
            del ann["caption_str"]

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
        caption = self.text_processor(ann["caption"])
        ocr = self.ocr_prompt.format(self.ocr_processer(ocr_tokens))

        return {
            "image": image,
            "num_patches_list": num_patches,
            "text_input": "",
            "text_output": caption,
            "ocr_input": ocr,
            "image_id": ann["image_id"],
            "label": ann["label"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        return {
            "image": torch.cat([sample["image"] for sample in samples], dim=0),
            "num_patches_list": [int(sample["num_patches_list"]) for sample in samples],
            "text_input": [sample["text_input"] for sample in samples],
            "text_output": [sample["text_output"] for sample in samples],
            "ocr_input": [sample["ocr_input"] for sample in samples],
            "image_id": [sample["image_id"] for sample in samples],
            "label": [sample["label"] for sample in samples],
        }


class CLTextCapsInternVLEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.ocr_processer = registry.get_processor_class("blip_caption").from_config()
        self.ocr_prompt = "Based on OCR: {}."

        self.annotation = self.annotation[3]["data"]
        self.annotation = [ann for ann in self.annotation if "caption_str" in ann]

        self.__check_cat()
        self.cats = {ann["label"] for ann in self.annotation if "label" in ann}
        self.original_annotation = deepcopy(self.annotation)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids:
                self.img_ids[img_id] = n
                n += 1
            ann["image"] = ann["image_path"]
            ann["caption"] = ann["caption_str"]
            del ann["caption_str"]
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

        return {
            "image": image,
            "num_patches_list": num_patches,
            "text_input": "",
            "ocr_input": ocr,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
            "label": ann["label"],
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
            "image_id": [sample["image_id"] for sample in samples],
            "instance_id": [sample["instance_id"] for sample in samples],
            "label": [sample["label"] for sample in samples],
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
