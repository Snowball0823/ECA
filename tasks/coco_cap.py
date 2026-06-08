import json
import os

import pandas as pd
from lavis.common.registry import registry
from lavis.common.utils import cache_url, is_convertible_to_int, is_url
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url
from tqdm import tqdm


def load_gt_file(file_path):
    if is_url(file_path):
        file_path = cache_url(file_path, registry.get_path("cache_root"))
    data = []
    if any(ext in file_path for ext in ['csv', 'tsv']):
        df = pd.read_csv(file_path)
        data.extend(df.to_dict(orient="records"))
        
    elif 'jsonl' in file_path:
        with open(file_path, "r") as f:
            data.extend([json.loads(line) for line in f])
    else:
        with open(file_path, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data.extend(loaded)
            elif isinstance(loaded, dict):
                # assume that loaded data in file  is the corresponding caption to the key
                data.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "caption": v} for k, v in loaded.items()])
    return data


def convert_to_coco_gt(data, outpath, caption_key, sample_id_key, split, load_gt_from_file=False, img_ids=[]):
    gt_data = {"annotations":[], "images":[]}
    if load_gt_from_file:
        print(f"Generating ground truth file for evaluation from {load_gt_from_file}....")
        data = load_gt_file(load_gt_from_file)
        for ann in data:
            captions = ann[caption_key]
            img_id = int(ann[sample_id_key]) if is_convertible_to_int(ann[sample_id_key]) else ann[sample_id_key]
            if img_ids and img_id not in img_ids: # only include specified img_ids if specified
                continue
            gt_data["images"].append({"id":img_id})
            if isinstance(captions, str):
                gt_data["annotations"].append({"image_id":img_id, "caption":captions, "id":img_id})
            else:   
                gt_data["annotations"].extend([{"image_id":img_id, "caption":c, "id":img_id} for c in captions])
    else:
        print(f"Generating ground truth file for evaluation....")
        for i,ann in tqdm(enumerate(data[split])):
            captions = data[split].annotation[i][caption_key]
            img_id = int(ann[sample_id_key]) if is_convertible_to_int(ann[sample_id_key]) else ann[sample_id_key]
            if img_ids and img_id not in img_ids: # only include specified img_ids if specified
                continue
            gt_data["images"].append({"id":img_id})
            if isinstance(captions, str):
                gt_data["annotations"].append({"image_id":img_id, "caption":captions, "id":img_id})
            else:   
                gt_data["annotations"].extend([{"image_id":img_id, "caption":c, "id":img_id} for c in captions])
    json.dump(gt_data, open(outpath, 'w'))
    print(f"Saved annotations at {outpath}")


def coco_caption_eval(coco_gt_root, results_file, split, annotation_file=None, img_ids=[]):

    if annotation_file == None:
        # TODO: Modify the dataset
        urls = {
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
        }
        filenames = {
            "val": "coco_karpathy_val_gt.json",
            "test": "coco_karpathy_test_gt.json",
        }

        download_url(urls[split], coco_gt_root)
        annotation_file = os.path.join(coco_gt_root, filenames[split])
    if is_url(annotation_file):
        annotation_file = cache_url(annotation_file, registry.get_path("cache_root"))
        
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    if img_ids:
        coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
