"""Smoke test InternVL2.5-1B loading, preprocessing, and Qwen2 adapter injection.

Usage:
    python test/test_internvl_forward.py

Environment variables:
    INTERNVL_MODEL_PATH
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from adapters import ParBnConfig

from models.custom_adapters.adapter_warp_model import adapter_init
from models.custom_adapters.utils import freeze_adapter, print_trainable_parameters
from models.ECA_InternVL.utils import load_eca_pretrained_model


def sync_module_device_dtype(module, device, target_dtype):
    module.to(device=device, dtype=target_dtype)


def build_dummy_image(image_size):
    rng = np.random.default_rng(seed=42)
    array = rng.uniform(0, 255, size=(image_size, image_size, 3)).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def main():
    model_path = os.environ.get("INTERNVL_MODEL_PATH", "checkpoints/InternVL/InternVL2_5-1B")
    if not os.path.exists(model_path):
        print(f"InternVL checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if has_cuda else torch.float32)

    tokenizer, model, image_processor, context_len = load_eca_pretrained_model(
        model_path=model_path,
        torch_dtype=dtype,
        device=device,
        use_flash_attn=False,
    )

    if not hasattr(model, "language_model"):
        raise RuntimeError("Loaded InternVL model does not expose `language_model`.")
    if type(model.language_model).__name__ != "Qwen2ForCausalLM":
        raise RuntimeError(f"Expected a Qwen2 LLM, got {type(model.language_model).__name__}.")

    adapter_model = model.language_model.model
    if getattr(adapter_model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError(
            f"InternVL language_model was not forced to eager attention. "
            f"Got {getattr(adapter_model.config, '_attn_implementation', None)}."
        )

    image_size = image_processor.input_size
    dummy_image = build_dummy_image(image_size)
    processed = image_processor(dummy_image, return_tensors="pt", dynamic_image_size=False)
    pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)
    if tuple(pixel_values.shape[-2:]) != (image_size, image_size):
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")
    if processed["num_patches_list"] != [1]:
        raise RuntimeError(f"Expected a single patch for the dummy image, got {processed['num_patches_list']}.")

    with torch.no_grad():
        features = model.extract_feature(pixel_values)
    if features.ndim != 3 or features.size(0) != pixel_values.size(0):
        raise RuntimeError(f"Unexpected visual feature shape: {tuple(features.shape)}")
    if torch.isnan(features).any():
        raise RuntimeError("InternVL visual features contain NaN.")

    adapter_init(adapter_model, use_customize=True)
    adapter_model.freeze_model()
    adapter_model.train()
    sync_module_device_dtype(adapter_model, device, dtype)

    pa_cfg = ParBnConfig(
        mh_adapter=True,
        output_adapter=False,
        reduction_factor=16.0,
        dropout=0.0,
        scaling=4.0,
        non_linearity="linear",
    )
    adapter_model.add_adapter("internvl_qwen_pa", config=pa_cfg)
    adapter_model.set_active_adapters("internvl_qwen_pa")
    freeze_adapter(adapter_model, "internvl_qwen_pa", freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[InternVL/Qwen2] Trainables after adapter injection:")
    print_trainable_parameters(adapter_model)
    print(adapter_model.adapter_summary())

    vocab_size = model.language_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 8), device=device)
    labels = input_ids.clone()
    outputs = model.language_model(input_ids=input_ids, labels=labels, use_cache=False, output_attentions=False)
    loss = outputs.loss
    if loss is None or torch.isnan(loss):
        raise RuntimeError("InternVL Qwen2 adapter smoke test produced an invalid loss.")
    loss.backward()

    if not any(param.grad is not None for param in adapter_model.parameters() if param.requires_grad):
        raise RuntimeError("InternVL Qwen2 adapter parameters did not receive gradients.")

    print(f"Loaded tokenizer/model from {model_path}")
    print(f"Context length: {context_len}")
    print(f"InternVL image processor size: {image_size}")
    print(f"InternVL Qwen2 adapter smoke test completed successfully. Loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
