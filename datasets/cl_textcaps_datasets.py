"""
 Copyright (c) 2022, salesforce.com, inc.
 Modifications Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Original LAVIS code remains under the BSD-3-Clause license.
 ECA modifications are released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import logging
import os
import random
from copy import deepcopy

from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import (CaptionDataset,
                                                      CaptionEvalDataset)
from PIL import Image


class CLTextCapsCapDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.ocr_processer = registry.get_processor_class('blip_caption').from_config()
        self.ocr_prompt = "Based on OCR: {}."
        self.annotation = self.annotation[3]['data']
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
            ann["image"] = ann["image_path"]
            ann["caption"] = ann["caption_str"]
            del ann["caption_str"]

        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}

    
    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if 'label' not in ann:
                no_label += 1
        assert no_label/total < 0.8, 'The rate of data without label is higher than 80%! Please check CL data again!'
        if no_label > 0:
            logging.warn("The rate of data without label is: "+str(no_label/total))


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
        
        ocr_tokens = " ".join(ann["ocr_tokens"][:30])
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        ocr = self.ocr_prompt.format(self.ocr_processer(ocr_tokens))


        return {
            "image": image,
            "text_input": "",
            "text_output": caption,
            "ocr_input": ocr,
            "image_id": ann["image_id"],
            'label': ann['label']
        }




class CLTextCapsCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        BaseDataset.__init__(self, vis_processor, text_processor, vis_root, ann_paths)
        self.ocr_processer = registry.get_processor_class('blip_caption').from_config()
        self.ocr_prompt = "Based on OCR: {}."

        self.annotation = self.annotation[3]['data']
        self.annotation = [ann for ann in self.annotation if "caption_str" in ann] # only keep annotations with captions

        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}
        self.original_annotation = deepcopy(self.annotation)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
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

        image = self.vis_processor(image)

        ocr_tokens = " ".join(ann["ocr_tokens"][:30])
        ocr = self.ocr_prompt.format(self.ocr_processer(ocr_tokens))

        return {
            "image": image,
            "text_input": "",
            "ocr_input": ocr,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
            'label': ann['label'],
        }

    def __check_cat(self):
        total = len(self.annotation)
        no_label = 0
        for ann in self.annotation:
            if 'label' not in ann:
                no_label += 1
        assert no_label/total < 0.5, 'The rate of data without label is higher than 50%! Please check CL data again!'
        if no_label > 0:
            logging.warn("The rate of data without label is: "+str(no_label/total))

    
    def rebuild(self, coco_cats: list):
        self.annotation.clear()
        # regenerate the self.annotation
        anns = [i for i in self.original_annotation if 'label' in i]
        self.annotation += [i for i in anns if i['label'] in coco_cats]