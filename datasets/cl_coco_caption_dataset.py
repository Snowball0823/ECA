"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""

import os
from collections import OrderedDict

from PIL import Image
from lavis.datasets.datasets.base_dataset import BaseDataset
import logging
from copy import deepcopy


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CLCOCOCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}

        self._add_instance_ids()
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1


    def rebuild(self, coco_cats: list):
        # regenerate the self.annotation
        if not hasattr(self, 'original_annotation'):
            setattr(self, 'original_annotation', deepcopy(self.annotation))
        self.annotation.clear()
        anns = [i for i in self.original_annotation if 'label' in i]
        self.annotation.extend([i for i in anns if i['label'] in coco_cats])
        self._add_instance_ids()
        
        if self.img_ids:
            self.img_ids.clear()

        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1


    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None # image does not exist

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_output": caption,
            "image_id": ann["image_id"],
            'label': ann['label']
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
            

class CLCaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.__check_cat()
        self.cats = {ann['label'] for ann in self.annotation if "label" in ann}
        self.original_annotation = deepcopy(self.annotation)

    def rebuild(self, coco_cats: list):
        self.annotation.clear()
        # regenerate the self.annotation
        anns = [i for i in self.original_annotation if 'label' in i]
        self.annotation += [i for i in anns if i['label'] in coco_cats]

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
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