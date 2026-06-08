"""
 Copyright (c) 2026, Jiangtao Kong.
 Contact: Jiangtao Kong <tinysnowball0823@gmail.com>
 Released for non-commercial research use only.
 For license details, see the LICENSE and NOTICE files in the repo root.
"""
import logging
import os
import io
from collections.abc import Mapping

import torch
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as T
from transformers import AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

from internvl.model.internvl_chat import InternVLChatConfig
from internvl.train.constants import (
    CLIP_MEAN,
    CLIP_STD,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    SIGLIP_MEAN,
    SIGLIP_STD,
)
from .softprompt_internvl_model import SoftPromptInternVLChatModel


LOGGER = logging.getLogger(__name__)
DEFAULT_INTERNVL_CHECKPOINT = os.path.join("checkpoints", "InternVL", "InternVL2_5-1B")
QUALITIES = list(range(75, 101))
CONFIG_OVERRIDE_KEYS = (
    "template",
    "select_layer",
    "dynamic_image_size",
    "use_thumbnail",
    "ps_version",
    "min_dynamic_patch",
    "max_dynamic_patch",
    "force_image_size",
    "downsample_ratio",
    "pad2square",
)
VISION_CONFIG_OVERRIDE_KEYS = (
    "drop_path_rate",
    "image_size",
    "patch_size",
    "use_flash_attn",
)
LLM_CONFIG_OVERRIDE_KEYS = (
    "attn_implementation",
    "_attn_implementation",
)
TOKENIZER_OVERRIDE_KEYS = (
    "add_eos_token",
    "model_max_length",
)


def orthogonal_svd_init(matrix):
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix for orthogonal SVD initialization.")

    n, m = matrix.shape
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    num_singular_values = vh.size(0)

    if num_singular_values < m:
        new_row = torch.randn(m, device=matrix.device)
        for i in range(n):
            row = matrix[i]
            dot_product = torch.dot(new_row, row)
            new_row = new_row - (dot_product / torch.norm(row) ** 2) * row
    else:
        new_row = torch.randn(m, device=matrix.device) @ vh[num_singular_values:, :]

    row_norms = torch.norm(matrix, dim=1)
    input_scale = row_norms.mean()
    new_row = new_row / torch.norm(new_row) * input_scale
    return new_row.unsqueeze(0)


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
        return {
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
        }

    __call__ = preprocess


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
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def resolve_checkpoint_path(model_path=None):
    model_path = model_path or DEFAULT_INTERNVL_CHECKPOINT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"InternVL checkpoint not found: {model_path}")
    return model_path


def _pop_override_dict(load_kwargs, key):
    value = load_kwargs.pop(key, None)
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a dict, got {type(value)}")
    return dict(value)


def _collect_overrides(load_kwargs, allowed_keys):
    overrides = {}
    for key in allowed_keys:
        if key in load_kwargs:
            overrides[key] = load_kwargs.pop(key)
    return overrides


def build_internvl_config(
    model_path,
    use_flash_attn=False,
    config_overrides=None,
    vision_config_overrides=None,
    llm_config_overrides=None,
):
    config = InternVLChatConfig.from_pretrained(model_path)
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"

    config_overrides = dict(config_overrides or {})
    vision_config_overrides = dict(vision_config_overrides or {})
    llm_config_overrides = dict(llm_config_overrides or {})

    for key, value in config_overrides.items():
        setattr(config, key, value)

    config.vision_config.use_flash_attn = bool(use_flash_attn)
    for key, value in vision_config_overrides.items():
        setattr(config.vision_config, key, value)

    if getattr(config.llm_config, "model_type", None) == "internlm2":
        config.llm_config.attn_implementation = attn_implementation
    else:
        setattr(config.llm_config, "attn_implementation", attn_implementation)
        setattr(config.llm_config, "_attn_implementation", attn_implementation)

    for key, value in llm_config_overrides.items():
        setattr(config.llm_config, key, value)

    return config


def ensure_special_tokens(tokenizer, model):
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    return num_new_tokens, img_context_token_id


def build_image_transform(input_size=448, is_train=False, pad2square=False, normalize_type="imagenet"):
    if normalize_type == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == "clip":
        mean, std = CLIP_MEAN, CLIP_STD
    elif normalize_type == "siglip":
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError(f"Unsupported normalize_type: {normalize_type}")

    if is_train:
        return T.Compose(
            [
                T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                T.RandomChoice([T.Lambda(JPEG_DEGRADE_FUNCTIONS[quality]) for quality in QUALITIES]),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    transforms = [T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image)]
    if pad2square:
        transforms.append(T.Lambda(lambda image: expand_to_square(image, tuple(int(x * 255) for x in mean))))
    transforms.extend(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return T.Compose(transforms)


def load_image_to_pixel_values(
    image,
    input_size=448,
    max_num=12,
    min_num=1,
    use_thumbnail=True,
    dynamic_image_size=True,
    pad2square=False,
    normalize_type="imagenet",
):
    image_processor = InternVLImageProcessor(
        input_size=input_size,
        dynamic_image_size=dynamic_image_size,
        use_thumbnail=use_thumbnail,
        max_num=max_num,
        min_num=min_num,
        pad2square=pad2square,
        normalize_type=normalize_type,
        is_train=False,
    )
    return image_processor(image, return_tensors="pt")


def load_eca_pretrained_model(
    model_path=None,
    torch_dtype=torch.float32,
    device="cuda",
    use_flash_attn=False,
    low_cpu_mem_usage=True,
    use_fast_tokenizer=False,
    trust_remote_code=True,
    add_eos_token=False,
    model_max_length=None,
    dynamic_image_size=None,
    use_thumbnail=None,
    max_num=None,
    min_num=None,
    pad2square=False,
    normalize_type="imagenet",
    **kwargs,
):
    """Load an InternVL checkpoint with ECA-compatible overrides.

    This loader only handles model-config values that must be decided before or
    during checkpoint initialization, such as tokenizer limits, visual/input
    settings, and config/vision_config/llm_config overrides.

    Post-load model-config changes used by training recipes (for example
    freeze_llm, freeze_mlp, freeze_backbone, use_llm_lora, use_backbone_lora)
    are intentionally not handled here. Those belong to the later model-wrapper
    / training setup stage.

    Common keyword overrides (pass via **kwargs):
        tokenizer:
            add_eos_token, model_max_length

        top-level config:
            template, select_layer, dynamic_image_size, use_thumbnail,
            ps_version, min_dynamic_patch, max_dynamic_patch,
            force_image_size, downsample_ratio, pad2square

        vision_config:
            drop_path_rate, image_size, patch_size, use_flash_attn
            - pass either directly in **kwargs, via vision_config={...},
              or via vision_config_overrides={...}

        llm_config:
            attn_implementation, _attn_implementation
            - pass either directly in **kwargs, via llm_config={...},
              or via llm_config_overrides={...}

    Remaining **kwargs are forwarded to SoftPromptInternVLChatModel.from_pretrained(...).
    """
    # NOTE(ECA_InternVL): official InternVL training scripts do override several
    # checkpoint defaults such as model_max_length, max_dynamic_patch, and
    # drop_path_rate. Keep those explicit in our model config rather than relying
    # on checkpoint defaults.
    model_path = resolve_checkpoint_path(model_path)
    load_kwargs = dict(kwargs)

    config_overrides = _pop_override_dict(load_kwargs, "config_overrides")
    vision_config_overrides = _pop_override_dict(load_kwargs, "vision_config_overrides")
    llm_config_overrides = _pop_override_dict(load_kwargs, "llm_config_overrides")
    tokenizer_overrides = _collect_overrides(load_kwargs, TOKENIZER_OVERRIDE_KEYS)

    # Accept both explicit override dicts and official config subtree aliases.
    vision_config_overrides.update(_pop_override_dict(load_kwargs, "vision_config"))
    llm_config_overrides.update(_pop_override_dict(load_kwargs, "llm_config"))

    config_overrides.update(_collect_overrides(load_kwargs, CONFIG_OVERRIDE_KEYS))
    vision_config_overrides.update(_collect_overrides(load_kwargs, VISION_CONFIG_OVERRIDE_KEYS))
    llm_config_overrides.update(_collect_overrides(load_kwargs, LLM_CONFIG_OVERRIDE_KEYS))

    if dynamic_image_size is not None:
        config_overrides["dynamic_image_size"] = dynamic_image_size
    if use_thumbnail is not None:
        config_overrides["use_thumbnail"] = use_thumbnail
    if min_num is not None:
        config_overrides["min_dynamic_patch"] = min_num
    if max_num is not None:
        config_overrides["max_dynamic_patch"] = max_num
    if "image_size" in vision_config_overrides and "force_image_size" not in config_overrides:
        config_overrides["force_image_size"] = vision_config_overrides["image_size"]

    add_eos_token = tokenizer_overrides.get("add_eos_token", add_eos_token)
    model_max_length = tokenizer_overrides.get("model_max_length", model_max_length)

    config = build_internvl_config(
        model_path,
        use_flash_attn=use_flash_attn,
        config_overrides=config_overrides,
        vision_config_overrides=vision_config_overrides,
        llm_config_overrides=llm_config_overrides,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        add_eos_token=add_eos_token,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = model_path
    resolved_model_max_length = model_max_length
    if resolved_model_max_length is None:
        resolved_model_max_length = getattr(tokenizer, "model_max_length", None)
    if resolved_model_max_length is None or resolved_model_max_length > 100000:
        resolved_model_max_length = getattr(config.llm_config, "max_position_embeddings", 2048)
    tokenizer.model_max_length = resolved_model_max_length

    model = SoftPromptInternVLChatModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_flash_attn=use_flash_attn,
        **load_kwargs,
    )

    llm_config = getattr(model, "language_model", None)
    if llm_config is not None and hasattr(model.language_model, "config"):
        attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
        setattr(model.language_model.config, "_attn_implementation", attn_implementation)
        setattr(model.language_model.config, "attn_implementation", attn_implementation)
        if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "config"):
            setattr(model.language_model.model.config, "_attn_implementation", attn_implementation)
            setattr(model.language_model.model.config, "attn_implementation", attn_implementation)

    target_image_size = getattr(model.config, "force_image_size", None) or model.config.vision_config.image_size
    patch_size = model.config.vision_config.patch_size
    if model.config.vision_config.image_size != target_image_size:
        LOGGER.info(
            "Resizing InternVL vision position embeddings from %s to %s",
            model.config.vision_config.image_size,
            target_image_size,
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=target_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = target_image_size
    model.config.force_image_size = target_image_size
    model.num_image_token = int((target_image_size // patch_size) ** 2 * (model.config.downsample_ratio ** 2))

    ensure_special_tokens(tokenizer, model)

    processor_dynamic_image_size = getattr(model.config, "dynamic_image_size", False)
    processor_use_thumbnail = getattr(model.config, "use_thumbnail", False)
    processor_min_num = getattr(model.config, "min_dynamic_patch", 1)
    processor_max_num = getattr(model.config, "max_dynamic_patch", 12)
    processor_pad2square = getattr(model.config, "pad2square", pad2square)
    image_processor = InternVLImageProcessor(
        input_size=target_image_size,
        dynamic_image_size=processor_dynamic_image_size,
        use_thumbnail=processor_use_thumbnail,
        max_num=processor_max_num,
        min_num=processor_min_num,
        pad2square=processor_pad2square,
        normalize_type=normalize_type,
        is_train=False,
    )

    if device is not None:
        model = model.to(device)
    model.eval()

    context_len = tokenizer.model_max_length

    LOGGER.info(
        "Loaded InternVL checkpoint from %s with attention=%s",
        model_path,
        "flash_attention_2" if use_flash_attn else "eager",
    )
    return tokenizer, model, image_processor, context_len
