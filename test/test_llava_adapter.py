"""Smoke test adapters on a projector-only LLaVA v0 setup.

Usage:
    python test/test_llava_adapter.py

Environment variables:
    LLAVA_BASE_MODEL      Path to base Vicuna/LLaMA checkpoint (default: checkpoints/LLaVA/vicuna-7b-v0)
    LLAVA_PROJECTOR_PATH  Path to directory containing mm_projector.bin (default: checkpoints/LLaVA/llava-pretrained-projectors)
"""

import os
import sys
import contextlib

import torch
import torch.nn.functional as F
from adapters import LoRAConfig, ParBnConfig
from adapters.composition import Average

from models.custom_adapters.adapter_warp_model import adapter_init
from models.custom_adapters.utils import (
    freeze_adapter,
    print_trainable_parameters,
    init_adapter,
    iter_adapter_named_parameters,
)
from models.ECA_LlaVA.utils import load_eca_pretrained_model


def resolve_vision_module(llava_model):
    vision = llava_model.get_vision_tower()
    if vision is None:
        raise RuntimeError("LLaVA model does not expose a vision tower.")
    # some implementations lazily load vision tower
    if hasattr(vision, "is_loaded") and not vision.is_loaded:
        vision.load_model()
    if hasattr(vision, "vision_tower"):
        vision = vision.vision_tower
    return vision


def main():
    base_model_path = os.environ.get("LLAVA_BASE_MODEL", "checkpoints/LLaVA/vicuna-7b-v0")
    projector_path = os.environ.get("LLAVA_PROJECTOR_PATH", "checkpoints/LLaVA/llava-7b-v0")

    if not os.path.exists(base_model_path):
        print(f"Base model path not found: {base_model_path}", file=sys.stderr)
        sys.exit(1)
    projector_file = os.path.join(projector_path, "mm_projector.bin")
    if not os.path.exists(projector_file):
        print(f"Projector weights not found at {projector_file}", file=sys.stderr)
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    dtype = torch.float32 if has_cuda else torch.float32
    autocast_ctx = (lambda: torch.cuda.amp.autocast(dtype=dtype)) if has_cuda else (lambda: contextlib.nullcontext())

    def sync_dtype(module: torch.nn.Module, target_dtype: torch.dtype):
        module.to(target_dtype)
        # for child in module.modules():
        #     if isinstance(child, torch.nn.LayerNorm):
        #         child.to(target_dtype)

        # Handle adapter fusion layers created during Average/Stack operations
        adapter_ctrl = getattr(module, "adapters", None)
        if adapter_ctrl is not None and hasattr(adapter_ctrl, "get_fusion"):
            active = getattr(adapter_ctrl, "active_setup", None)
            if active is not None and hasattr(active, "fusions"):
                for fusion_cfg in active.fusions:
                    fusion_mod = adapter_ctrl.get_fusion(fusion_cfg)
                    if fusion_mod is not None:
                        fusion_mod.to(target_dtype)
    device = torch.device("cuda" if has_cuda else "cpu")

    print(f"Loading base model from {base_model_path}")
    print(f"Loading projector from {projector_path}")
    tokenizer, llava, image_processor, context_len = load_eca_pretrained_model(
        model_path=projector_path,
        model_base=base_model_path,
        model_name="llava-v0",
        torch_dtype=dtype,
        device="cpu",
        device_map="auto",
        mm_use_im_start_end=True,
        mm_use_im_patch_token=False,
    )
    llava = llava.to(device=device, dtype=dtype)
    llava.train()

    print(llava.get_model().mm_projector)
    input()
    # Vision tower
    vision = resolve_vision_module(llava)
    vision.to(device=device, dtype=dtype)
    vision.train()

    # Adapter configs
    pa_cfg = ParBnConfig(
        mh_adapter=True,
        output_adapter=False,
        reduction_factor=16.0,
        dropout=0.0,
        scaling=4.0,
        non_linearity="linear",
    )
    lora_cfg = LoRAConfig(r=8, alpha=16)

    # Attach adapters to vision tower
    adapter_init(vision, use_customize=False)
    vision.freeze_model()
    vision.add_adapter("vision_pa", config=pa_cfg)
    vision.set_active_adapters("vision_pa")
    freeze_adapter(vision, "vision_pa", freeze=False)
    vision.to(device=device, dtype=dtype)
    print("[Vision] Parallel adapter trainables:")
    print_trainable_parameters(vision)
    adapter_sum = vision.adapter_summary()
    print('\n'+adapter_sum)

    pixel_values = torch.randn(
        2,
        3,
        getattr(getattr(vision, "config", None), "image_size", 224),
        getattr(getattr(vision, "config", None), "image_size", 224),
    )
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    with autocast_ctx():
        out = vision(pixel_values=pixel_values)
        assert out.last_hidden_state.shape[0] == pixel_values.size(0), "Vision forward batch mismatch"
        assert not torch.isnan(out.last_hidden_state).any(), "Vision features contain NaN"
        loss = out.last_hidden_state.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in vision.named_parameters() if p.requires_grad), (
        "Parallel adapter on vision tower did not receive gradients."
    )
    vision.zero_grad()

    vision.add_adapter("vision_pa_extra", config=pa_cfg)
    vision.set_active_adapters(["vision_pa", "vision_pa_extra"])
    freeze_adapter(vision, ["vision_pa", "vision_pa_extra"], freeze=False)
    vision.to(device=device, dtype=dtype)
    print("[Vision] Second parallel adapter trainables:")
    print_trainable_parameters(vision)
    adapter_sum = vision.adapter_summary()
    print('\n'+adapter_sum)

    with autocast_ctx():
        out = vision(pixel_values=pixel_values)
        assert out.last_hidden_state.shape[0] == pixel_values.size(0), "Vision forward batch mismatch (extra PA)"
        assert not torch.isnan(out.last_hidden_state).any(), "Vision features contain NaN with extra PA"
        loss = out.last_hidden_state.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in vision.named_parameters() if p.requires_grad), (
        "Second parallel adapter on vision tower did not receive gradients."
    )
    vision.zero_grad()

    # # Average composition test on vision PAs
    # vision.set_active_adapters(Average("vision_pa", "vision_pa_extra", weights=[1, 1]))
    # vision.to(device=device, dtype=dtype)
    # sync_dtype(vision, dtype)
    # out = vision(pixel_values=pixel_values)
    # assert out.last_hidden_state.shape[0] == pixel_values.size(0), "Vision forward batch mismatch (Average PA)"
    # assert not torch.isnan(out.last_hidden_state).any(), "Vision features contain NaN with Average PA"
    # loss = out.last_hidden_state.sum()
    # loss.backward()
    # assert any(p.grad is not None for _, p in vision.named_parameters() if p.requires_grad), (
    #     "Average fusion on vision PAs did not receive gradients."
    # )
    # vision.zero_grad()

    # LoRA test (separate)
    vision.add_adapter("vision_lora", config=lora_cfg)
    vision.set_active_adapters(["vision_lora"])
    freeze_adapter(vision, "vision_lora", freeze=False)
    vision.to(device=device, dtype=dtype)
    print("[Vision] LoRA trainables:")
    print_trainable_parameters(vision)
    adapter_sum = vision.adapter_summary()
    print('\n'+adapter_sum)

    with autocast_ctx():
        out = vision(pixel_values=pixel_values)
        assert out.last_hidden_state.shape[0] == pixel_values.size(0), "Vision forward batch mismatch (LoRA)"
        assert not torch.isnan(out.last_hidden_state).any(), "Vision features contain NaN after LoRA"
        loss = out.last_hidden_state.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in vision.named_parameters() if p.requires_grad), (
        "LoRA adapter on vision tower did not receive gradients."
    )
    vision.zero_grad()
    print("Vision tower adapters OK.\n")

    # ------------------------------------------------------------------
    # Simple multimodal forward without additional adapters
    llava.eval()
    vision.eval()
    image_size = getattr(getattr(vision, "config", None), "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        img_h, img_w = image_size
    else:
        img_h = img_w = int(image_size)
    dummy_image = torch.randn(1, 3, img_h, img_w, device=device, dtype=dtype)
    dummy_prompt = "USER: Describe the image in one word.\nASSISTANT:"
    tokenized_prompt = tokenizer(
        [dummy_prompt],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        with autocast_ctx():
            forward_out = llava(
                input_ids=tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                images=dummy_image,
                return_dict=True,
            )
    print(f"Multimodal forward successful, logits shape: {tuple(forward_out.logits.shape)}\n")
    llava.train()
    vision.train()

    # Language model (Vicuna/LLaMA)
    llava_core = llava.get_model()

    # 暂时移除 vision_tower，避免 adapter_init 浏览视觉子模块时报错
    stored_vision = getattr(llava_core, "vision_tower", None)
    try:
        if stored_vision is not None:
            llava_core.vision_tower = None
        adapter_init(llava_core, use_customize=True)
    finally:
        if stored_vision is not None:
            llava_core.vision_tower = stored_vision

    num_llama_layers = len(llava_core.layers)
    alignment_layers = min(4, num_llama_layers)
    leave_out = list(range(alignment_layers, num_llama_layers))
    pa_cfg_llm = ParBnConfig(
        mh_adapter=True,
        output_adapter=False,
        reduction_factor=16.0,
        dropout=0.0,
        scaling=4.0,
        non_linearity="linear",
        leave_out=leave_out,
    )

    llava_core.freeze_model()
    llava_core.add_adapter("llm_pa", config=pa_cfg_llm)
    llava_core.set_active_adapters("llm_pa")
    freeze_adapter(llava_core, "llm_pa", freeze=False)
    llava_core.to(device=device, dtype=dtype)
    # sync_dtype(llava_core, dtype)
    print("[LLM] Parallel adapter trainables:")
    print_trainable_parameters(llava_core)
    adapter_sum = llava_core.adapter_summary()
    print('\n'+adapter_sum)
    print(f"Activated LLM adapters on first {alignment_layers} / {num_llama_layers} layers (leave_out={leave_out}).")

    input_ids = torch.randint(0, llava.config.vocab_size, (2, 16), device=device)
    with autocast_ctx():
        outputs = llava(input_ids=input_ids)
        assert outputs.logits.shape[:2] == input_ids.shape, "LLM logits shape mismatch"
        assert not torch.isnan(outputs.logits).any(), "LLM logits contain NaN"
        loss = outputs.logits.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in llava_core.named_parameters() if p.requires_grad), (
        "Parallel adapter on LLM did not receive gradients."
    )
    llava_core.zero_grad()

    llava_core.add_adapter("llm_pa_extra", config=pa_cfg_llm)
    llava_core.set_active_adapters(["llm_pa", "llm_pa_extra"])
    freeze_adapter(llava_core, ["llm_pa", "llm_pa_extra"], freeze=False)
    llava_core.to(device=device, dtype=dtype)
    # sync_dtype(llava_core, dtype)
    print("[LLM] Second parallel adapter trainables:")
    print_trainable_parameters(llava_core)
    adapter_sum = llava_core.adapter_summary()
    print('\n'+adapter_sum)

    # Inspect per-layer adapter presence
    print("\n[LLM] Adapter distribution per decoder layer (first 10 shown):")
    for idx, layer in enumerate(llava_core.layers):
        layer_adapters = set()
        for module in layer.modules():
            adapter_ctrl = getattr(module, "adapter_ctrl", None)
            # Handle LoRA / bottleneck controllers
            if adapter_ctrl is None and hasattr(module, "adapter_modules"):
                adapter_ctrl = module
            if adapter_ctrl is not None:
                names = []
                if hasattr(adapter_ctrl, "adapter_modules"):
                    names.extend(adapter_ctrl.adapter_modules.keys())
                if hasattr(adapter_ctrl, "adapters") and hasattr(adapter_ctrl.adapters, "keys"):
                    names.extend(adapter_ctrl.adapters.keys())
                layer_adapters.update(names)
        if idx < 10 or layer_adapters:
            print(f"Layer {idx:02d}: {sorted(layer_adapters)}")

    def collect_logits(active_setup):
        if active_setup is None:
            llava_core.adapters_config.active_setup = None
            llava_core.active_adapters = None
        else:
            llava_core.set_active_adapters(active_setup)
        llava_core.to(device=device, dtype=dtype)
        prev_mode = llava.training
        llava.eval()
        llava_core.eval()
        llava_core.zero_grad(set_to_none=True)
        with torch.no_grad():
            with autocast_ctx():
                result = llava(input_ids=input_ids)
        logits = result.logits.detach().cpu()
        if prev_mode:
            llava.train()
            llava_core.train()
        return logits

    logits_base = collect_logits(None)
    logits_a = collect_logits("llm_pa")
    logits_b = collect_logits("llm_pa_extra")

    avg_a = Average("llm_pa", "llm_pa_extra", weights=[1.0, 0.0], normalize_weights=False)
    avg_b = Average("llm_pa", "llm_pa_extra", weights=[0.0, 1.0], normalize_weights=False)

    logits_avg_a = collect_logits(avg_a)
    logits_avg_b = collect_logits(avg_b)

    def assert_weighted_match(candidate, weights, references, label):
        delta = torch.zeros_like(logits_base)
        for w, ref in zip(weights, references):
            delta += w * (ref - logits_base)
        expected = logits_base + delta / branch_count
        max_diff = torch.max(torch.abs(candidate - expected)).item()
        assert torch.allclose(candidate, expected, atol=5e-5, rtol=1e-4), (
            f"{label}: mismatch (max diff {max_diff:.6f})"
        )

    branch_count = 2
    assert_weighted_match(logits_avg_a, [1.0, 0.0], [logits_a, logits_b], "Average weights [1,0]")
    assert_weighted_match(logits_avg_b, [0.0, 1.0], [logits_a, logits_b], "Average weights [0,1]")

    avg_mix = Average("llm_pa", "llm_pa_extra", weights=[0.5, 0.5], normalize_weights=False)
    logits_avg_mix = collect_logits(avg_mix)
    assert_weighted_match(logits_avg_mix, [0.5, 0.5], [logits_a, logits_b], "Average weights [0.5,0.5]")

    # restore gradients check with balanced average
    with autocast_ctx():
        outputs = llava(input_ids=input_ids)
        assert outputs.logits.shape[:2] == input_ids.shape, "LLM logits shape mismatch (extra PA)"
        assert not torch.isnan(outputs.logits).any(), "LLM logits contain NaN with extra PA"
        loss = outputs.logits.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in llava_core.named_parameters() if p.requires_grad), (
        "Second parallel adapter on LLM did not receive gradients."
    )
    llava_core.zero_grad()

    # Average composition test on language PAs
    llava_core.set_active_adapters(Average("llm_pa", "llm_pa_extra", weights=[1, 1]))
    llava_core.to(device=device, dtype=dtype)
    llava_core.train()
    llava_core.zero_grad(set_to_none=True)
    with autocast_ctx():
        outputs = llava(input_ids=input_ids)
        assert outputs.logits.shape[:2] == input_ids.shape, "LLM logits shape mismatch (Average PA)"
        assert not torch.isnan(outputs.logits).any(), "LLM logits contain NaN with Average PA"
        loss = outputs.logits.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in llava_core.named_parameters() if p.requires_grad), (
        "Average fusion on LLM PAs did not receive gradients."
    )
    llava_core.zero_grad()

    # KD-style check with third adapter
    third_adapter = "llm_pa_third"
    llava_core.add_adapter(third_adapter, config=pa_cfg_llm)
    all_adapters = ["llm_pa", "llm_pa_extra", third_adapter]
    llava_core.set_active_adapters(all_adapters)
    freeze_adapter(llava_core, ["llm_pa", "llm_pa_extra"], freeze=True)
    freeze_adapter(llava_core, third_adapter, freeze=False)
    llava_core.to(device=device, dtype=dtype)
    print("[LLM] Added third adapter (student configuration):")
    print_trainable_parameters(llava_core)
    adapter_sum = llava_core.adapter_summary()
    print('\n'+adapter_sum)

    with torch.no_grad():
        for name, param in llava_core.named_parameters():
            if third_adapter in name and param.requires_grad:
                param.data = param.data.normal_(mean=0.0, std=1e-3)

    def register_adapter_activation_hooks(adapter_name: str):
        flag = {"executed": False}
        handles = []

        def make_hook(target):
            def hook(module, input, output, *_args):
                target["executed"] = True
            return hook

        for mod_name, module in llava_core.named_modules():
            if adapter_name in mod_name:
                handles.append(module.register_forward_hook(make_hook(flag)))
        return flag, handles

    # Teacher forward with first two adapters only
    with torch.no_grad():
        prev_model_mode = llava.training
        prev_core_mode = llava_core.training
        llava.eval()
        llava_core.eval()
        llava_core.set_active_adapters(["llm_pa", "llm_pa_extra"])
        print("[LLM] KD teacher active adapters:", llava_core.adapters_config.active_setup)
        print('\n'+llava_core.adapter_summary())
        teacher_flag, teacher_handles = register_adapter_activation_hooks(third_adapter)
        with autocast_ctx():
            teacher_out = llava(input_ids=input_ids)
        teacher_logits = teacher_out.logits.detach()
        for handle in teacher_handles:
            handle.remove()
        assert not teacher_flag["executed"], "Third adapter unexpectedly executed during teacher forward."
        if prev_model_mode:
            llava.train()
        if prev_core_mode:
            llava_core.train()

    # Student forward with new adapter active
    llava_core.set_active_adapters(all_adapters)
    print("[LLM] KD student active adapters:", llava_core.adapters_config.active_setup)
    print('\n'+llava_core.adapter_summary())
    student_flag, student_handles = register_adapter_activation_hooks(third_adapter)
    llava_core.zero_grad(set_to_none=True)
    with autocast_ctx():
        student_out = llava(input_ids=input_ids)
        kd_loss = F.mse_loss(student_out.logits, teacher_logits.to(device=device, dtype=student_out.logits.dtype))
    kd_loss.backward()
    for handle in student_handles:
        handle.remove()
    assert student_flag["executed"], "Third adapter did not execute during student forward."

    # Ensure gradients flow only through the third adapter
    third_grad = False
    frozen_grad = False
    for name, param in llava_core.named_parameters():
        if not param.requires_grad:
            frozen_grad = frozen_grad or (param.grad is not None)
            continue
        if third_adapter in name:
            third_grad = third_grad or (param.grad is not None and torch.any(param.grad != 0))
    assert third_grad, "Third adapter did not receive gradients during KD pass."
    assert not frozen_grad, "Frozen adapters accumulated gradients unexpectedly."
    llava_core.zero_grad()

    # Clone adapter for teacher/student split using the trained third adapter
    clone_teacher = "llm_pa_third_teacher"
    llava_core.add_adapter(clone_teacher, config=pa_cfg_llm)
    init_adapter(llava_core, clone_teacher, third_adapter)
    freeze_adapter(llava_core, clone_teacher, freeze=True)
    freeze_adapter(llava_core, third_adapter, freeze=False)
    llava_core.to(device=device, dtype=dtype)
    print("[LLM] Teacher/Student clone setup:")
    print_trainable_parameters(llava_core)
    print('\n'+llava_core.adapter_summary())

    # Step 1: activate history + student, ensure teacher clone not executed
    llava_core.set_active_adapters(["llm_pa", "llm_pa_extra", third_adapter])
    hook_flag, hook_handles = register_adapter_activation_hooks(clone_teacher)
    with torch.no_grad():
        with autocast_ctx():
            _ = llava(input_ids=input_ids)
    for handle in hook_handles:
        handle.remove()
    assert not hook_flag["executed"], "Teacher clone executed when inactive."

    # Helper to collect logits for a given adapter list
    def collect_clone_logits(active_list):
        llava_core.set_active_adapters(active_list)
        prev_mode = llava.training
        llava.eval()
        with torch.no_grad():
            with autocast_ctx():
                logits = llava(input_ids=input_ids).logits.detach()
        if prev_mode:
            llava.train()
        return logits

    teacher_clone_logits = collect_clone_logits([clone_teacher])
    student_clone_logits = collect_clone_logits([third_adapter])
    assert torch.allclose(
        student_clone_logits.to(dtype=teacher_clone_logits.dtype),
        teacher_clone_logits,
        atol=1e-4,
        rtol=1e-3,
    ), "Student adapter logits diverge from teacher clone logits."

    # Step 3: ensure gradients only flow through student adapter
    llava_core.set_active_adapters([third_adapter])
    llava_core.zero_grad(set_to_none=True)
    with autocast_ctx():
        clone_out = llava(input_ids=input_ids)
        clone_loss = clone_out.logits.sum()
    clone_loss.backward()

    def adapter_has_grad(adapter_name: str):
        for _, param in iter_adapter_named_parameters(llava_core, adapter_name):
            if param.grad is not None and torch.any(param.grad != 0):
                return True
        return False

    assert adapter_has_grad(third_adapter), "Student adapter did not receive gradients."
    assert not adapter_has_grad(clone_teacher), "Teacher clone unexpectedly received gradients."
    llava_core.zero_grad()

    # LoRA test (separate)
    llava_core.add_adapter("llm_lora", config=lora_cfg)
    llava_core.set_active_adapters(["llm_lora"])
    freeze_adapter(llava_core, "llm_lora", freeze=False)
    llava_core.to(device=device, dtype=dtype)
    # sync_dtype(llava_core, dtype)
    print("[LLM] LoRA trainables:")
    print_trainable_parameters(llava_core)
    adapter_sum = llava_core.adapter_summary()
    print('\n'+adapter_sum)

    with autocast_ctx():
        outputs = llava(input_ids=input_ids)
        assert outputs.logits.shape[:2] == input_ids.shape, "LLM logits shape mismatch (LoRA)"
        assert not torch.isnan(outputs.logits).any(), "LLM logits contain NaN after LoRA"
        loss = outputs.logits.sum()
    loss.backward()
    assert any(p.grad is not None for _, p in llava_core.named_parameters() if p.requires_grad), (
        "LoRA adapter on LLM did not receive gradients."
    )

    print("LLM adapters OK.\nAll adapter checks passed.")
    input()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available. Loading full 7B model on CPU may require large memory.",
            file=sys.stderr,
        )
    main()
