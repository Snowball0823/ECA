"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
"""
InternVL image processors aligned with the official image transform and
dynamic tiling logic.
"""

import io
import os

import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode

from internvl.train.constants import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, SIGLIP_MEAN, SIGLIP_STD
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


QUALITIES = list(range(75, 101))


def expand_to_square(pil_image, background_color):
    width, height = pil_image.size
    if width == height:
        return pil_image
    if width > height:
        result = Image.new(pil_image.mode, (width, width), background_color)
        result.paste(pil_image, (0, (width - height) // 2))
        return result
    result = Image.new(pil_image.mode, (height, height), background_color)
    result.paste(pil_image, ((height - width) // 2, 0))
    return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(image):
        with io.BytesIO() as output:
            image.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)
            return Image.open(output).copy()

    return jpeg_degrade


JPEG_DEGRADE_FUNCTIONS = {quality: simulate_jpeg_degradation(quality) for quality in QUALITIES}


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def build_image_transform(is_train, input_size, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    if is_train:
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomChoice([T.Lambda(JPEG_DEGRADE_FUNCTIONS[quality]) for quality in QUALITIES]),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    else:
        if not pad2square:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                    T.Lambda(lambda img: expand_to_square(img, tuple(int(x * 255) for x in mean))),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ]
            )
    return transform


class InternVLImageProcessor:
    def __init__(
        self,
        input_size=448,
        dynamic_image_size=True,
        use_thumbnail=True,
        max_num=12,
        min_num=1,
        pad2square=False,
        normalize_type="imagenet",
        is_train=False,
    ):
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.min_num = min_num
        self.pad2square = pad2square
        self.normalize_type = normalize_type
        self.is_train = is_train
        self.transform = build_image_transform(
            is_train=is_train,
            input_size=input_size,
            pad2square=pad2square,
            normalize_type=normalize_type,
        )

    def _load_pil_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path not found: {image}")
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)}")

    def preprocess(
        self,
        images,
        return_tensors="pt",
        max_num=None,
        min_num=None,
        use_thumbnail=None,
        dynamic_image_size=None,
    ):
        if return_tensors != "pt":
            raise ValueError("InternVLImageProcessor currently supports only return_tensors='pt'.")

        if not isinstance(images, (list, tuple)):
            images = [images]

        max_num = self.max_num if max_num is None else max_num
        min_num = self.min_num if min_num is None else min_num
        use_thumbnail = self.use_thumbnail if use_thumbnail is None else use_thumbnail
        dynamic_image_size = self.dynamic_image_size if dynamic_image_size is None else dynamic_image_size

        pixel_values = []
        num_patches_list = []
        for image in images:
            pil_image = self._load_pil_image(image)
            if dynamic_image_size:
                image_tiles = dynamic_preprocess(
                    pil_image,
                    min_num=min_num,
                    max_num=max_num,
                    image_size=self.input_size,
                    use_thumbnail=use_thumbnail,
                )
            else:
                image_tiles = [pil_image]

            num_patches_list.append(len(image_tiles))
            pixel_values.extend(self.transform(tile) for tile in image_tiles)

        pixel_values = torch.stack(pixel_values)
        return {"pixel_values": pixel_values, "num_patches_list": num_patches_list}

    __call__ = preprocess


class _InternVLImageProcessorWrapper(BaseProcessor):
    def __init__(self, processor):
        self.processor = processor
        self.input_size = processor.input_size
        self.dynamic_image_size = processor.dynamic_image_size
        self.use_thumbnail = processor.use_thumbnail
        self.min_num = processor.min_num
        self.max_num = processor.max_num
        self.pad2square = processor.pad2square
        self.normalize_type = processor.normalize_type
        self.is_train = processor.is_train

    @classmethod
    def from_config(cls, cfg=None, is_train=False):
        if cfg is None:
            cfg = OmegaConf.create()

        input_size = int(cfg.get("image_size", cfg.get("input_size", 448)))
        dynamic_image_size = cfg.get("dynamic_image_size", True)
        use_thumbnail = cfg.get("use_thumbnail", True)
        min_num = int(cfg.get("min_dynamic_patch", cfg.get("min_num", 1)))
        max_num = int(cfg.get("max_dynamic_patch", cfg.get("max_num", 12)))
        pad2square = cfg.get("pad2square", False)
        normalize_type = cfg.get("normalize_type", "imagenet")

        processor = InternVLImageProcessor(
            input_size=input_size,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_num=min_num,
            max_num=max_num,
            pad2square=pad2square,
            normalize_type=normalize_type,
            is_train=is_train,
        )
        return cls(processor)

    def __call__(self, image):
        processed = self.processor(image, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"],
            "num_patches_list": processed["num_patches_list"],
        }


@registry.register_processor("internvl_image_train")
class InternVLImageTrainProcessor(_InternVLImageProcessorWrapper):
    @classmethod
    def from_config(cls, cfg=None):
        return super().from_config(cfg=cfg, is_train=True)


@registry.register_processor("internvl_image_eval")
class InternVLImageEvalProcessor(_InternVLImageProcessorWrapper):
    @classmethod
    def from_config(cls, cfg=None):
        return super().from_config(cfg=cfg, is_train=False)
