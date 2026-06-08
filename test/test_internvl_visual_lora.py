"""Smoke test official InternVL visual LoRA on a local InternVL2.5-1B checkpoint.

Usage:
    python test/test_internvl_visual_lora.py

Environment variables:
    INTERNVL_MODEL_PATH
"""

import contextlib
import os
import sys

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

from models.ECA_InternVL.utils import load_eca_pretrained_model


def build_dummy_image(image_size):
    rng = np.random.default_rng(seed=7)
    array = rng.uniform(0, 255, size=(image_size, image_size, 3)).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def has_nonzero_grad(named_params):
    for _, param in named_params:
        if param.grad is not None and torch.any(param.grad != 0):
            return True
    return False


def wrap_backbone_lora(model, r=128, lora_alpha=256, lora_dropout=0.05):
    lora_config = LoraConfig(
        r=r,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.vision_model = get_peft_model(model.vision_model, lora_config)
    model.vision_model.print_trainable_parameters()


def main():
    model_path = os.environ.get("INTERNVL_MODEL_PATH", "checkpoints/InternVL/InternVL2_5-1B")
    if not os.path.exists(model_path):
        print(f"InternVL checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if has_cuda else torch.float32)
    autocast_ctx = (lambda: torch.cuda.amp.autocast(dtype=dtype)) if has_cuda else (lambda: contextlib.nullcontext())

    tokenizer, model, image_processor, context_len = load_eca_pretrained_model(
        model_path=model_path,
        torch_dtype=dtype,
        device=device,
        use_flash_attn=False,
    )
    model.train()

    for param in model.parameters():
        param.requires_grad = False

    wrap_backbone_lora(model, r=16, lora_alpha=32)
    model.train()

    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("wrap_backbone_lora() did not expose any trainable parameters.")
    if not all("lora_" in name for name, _ in trainable_params):
        bad = [name for name, _ in trainable_params if "lora_" not in name]
        raise RuntimeError(f"Non-LoRA parameters remained trainable after wrap_backbone_lora(): {bad[:8]}")
    if not any(name.startswith("vision_model") for name, _ in trainable_params):
        raise RuntimeError("No trainable visual LoRA parameters were found under vision_model.")

    image_size = image_processor.input_size
    dummy_image = build_dummy_image(image_size)
    processed = image_processor(dummy_image, return_tensors="pt", dynamic_image_size=False)
    pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)
    if processed["num_patches_list"] != [1]:
        raise RuntimeError(f"Expected one patch for dummy image, got {processed['num_patches_list']}.")

    with autocast_ctx():
        features = model.extract_feature(pixel_values)
        if features.ndim != 3 or features.size(0) != pixel_values.size(0):
            raise RuntimeError(f"Unexpected feature shape from extract_feature: {tuple(features.shape)}")
        if torch.isnan(features).any():
            raise RuntimeError("Visual LoRA extract_feature produced NaN values.")
        loss = features.sum()
    loss.backward()

    lora_grads = [(name, param) for name, param in trainable_params if "lora_" in name]
    if not has_nonzero_grad(lora_grads):
        raise RuntimeError("Visual LoRA parameters did not receive gradients.")

    print(f"Loaded tokenizer/model from {model_path}")
    print(f"Context length: {context_len}")
    print(f"Visual image processor size: {image_size}")
    print(f"Trainable visual LoRA params: {len(trainable_params)}")
    print(f"InternVL visual LoRA smoke test completed successfully. Loss={float(loss.detach().float().cpu()):.6f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available. InternVL visual LoRA smoke test will fall back to CPU.",
            file=sys.stderr,
        )
    main()
