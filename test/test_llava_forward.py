"""Quick smoke test to ensure the LLaVA LLM path (text prompt + projector output) works.

Usage:
    python test/test_llava_forward.py \
        --base checkpoints/LLaVA/vicuna-7b-v0 \
        --projector checkpoints/LLaVA/llava-7b-v0

Environment variables (optional overrides):
    LLAVA_BASE_MODEL
    LLAVA_PROJECTOR_PATH
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

from llava.mm_utils import tokenizer_image_token
from models.ECA_LlaVA.utils import load_eca_pretrained_model


def build_prompt(question: str) -> str:
    """Wrap question with the default LLaVA conversation template."""
    template = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
        "### Human: {question}\n"
        "### Assistant:"
    )
    return template.format(question=question)


def load_or_dummy_image(path: Optional[str], image_size: int) -> Image.Image:
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path not found: {path}")
        return Image.open(path).convert("RGB")

    rng = np.random.default_rng(seed=42)
    array = rng.uniform(0, 255, size=(image_size, image_size, 3)).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Smoke test LLaVA LLM generation.")
    parser.add_argument("--base", type=str, default=os.environ.get("LLAVA_BASE_MODEL"))
    parser.add_argument("--projector", type=str, default=os.environ.get("LLAVA_PROJECTOR_PATH"))
    parser.add_argument("--image", type=str, default=None, help="Optional image path. If omitted, uses a random image.")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mm-use-im-start-end", action="store_true", default=True)
    parser.add_argument("--mm-use-im-patch-token", action="store_true", default=False)
    args = parser.parse_args()

    if not args.base or not os.path.exists(args.base):
        print("Base Vicuna/LLaMA checkpoint not found. Provide --base or set LLAVA_BASE_MODEL.", file=sys.stderr)
        sys.exit(1)
    projector_dir = args.projector or "checkpoints/LLaVA/llava-7b-v0"
    projector_file = os.path.join(projector_dir, "mm_projector.bin")
    if not os.path.exists(projector_file):
        print(f"Projector weights not found at {projector_file}.", file=sys.stderr)
        sys.exit(1)

    torch_dtype = getattr(torch, args.dtype)
    tokenizer, llava_model, image_processor, _ = load_eca_pretrained_model(
        model_path=projector_dir,
        model_base=args.base,
        model_name="llava-smoke-test",
        torch_dtype=torch_dtype,
        device=args.device,
        device_map="auto" if args.device == "cpu" else "auto",
        mm_use_im_start_end=args.mm_use_im_start_end,
        mm_use_im_patch_token=args.mm_use_im_patch_token,
    )

    llava_model = llava_model.to(device=args.device, dtype=torch_dtype)
    llava_model.eval()

    vision = llava_model.get_vision_tower()
    if hasattr(vision, "is_loaded") and not vision.is_loaded:
        vision.load_model()
    if hasattr(vision, "vision_tower"):
        vision = vision.vision_tower
    image_size = getattr(getattr(vision, "config", None), "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        image_h = image_w = int(image_size[0])
    else:
        image_h = image_w = int(image_size)

    pil_image = load_or_dummy_image(args.image, image_h)
    image_tensor = image_processor(pil_image, return_tensors="pt")["pixel_values"].to(args.device, dtype=torch_dtype)

    if args.mm_use_im_start_end:
        image_placeholder = "<im_start><image><im_end>"
    else:
        image_placeholder = "<image>"
    conversation = build_prompt(f"{image_placeholder} {args.prompt}")

    input_ids = tokenizer_image_token(conversation, tokenizer, return_tensors="pt").unsqueeze(0).to(args.device)

    with torch.no_grad():
        generation = llava_model.generate(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
    decoded = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    print("----- Prompt -----")
    print(conversation)
    print("----- Generated -----")
    print(decoded)

    if len(decoded.strip()) == 0:
        raise RuntimeError("Generation returned an empty string; LLM path might be broken.")

    print("\nLLaVA LLM smoke test completed successfully.")


if __name__ == "__main__":
    main()
