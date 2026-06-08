"""Smoke test Qwen2 adapters on the Qwen2 backbone inside a local InternVL2.5-1B checkpoint.

Usage:
    python test/test_internvl_qwen2_adapters.py

Environment variables:
    INTERNVL_MODEL_PATH
"""

import contextlib
import os
import sys

import torch
import torch.nn.functional as F
from adapters import LoRAConfig, ParBnConfig
from adapters.composition import Average

from models.custom_adapters.adapter_warp_model import adapter_init
from models.custom_adapters.utils import (
    freeze_adapter,
    init_adapter,
    iter_adapter_named_parameters,
    print_trainable_parameters,
)
from models.ECA_InternVL.utils import load_eca_pretrained_model


def sync_module_device_dtype(module, device, target_dtype):
    module.to(device=device, dtype=target_dtype)

    # # Average/Stack setups may create fusion modules lazily; keep them aligned.
    # adapter_ctrl = getattr(module, "adapters", None)
    # if adapter_ctrl is not None and hasattr(adapter_ctrl, "get_fusion"):
    #     active = getattr(adapter_ctrl, "active_setup", None)
    #     if active is not None and hasattr(active, "fusions"):
    #         for fusion_cfg in active.fusions:
    #             fusion_mod = adapter_ctrl.get_fusion(fusion_cfg)
    #             if fusion_mod is not None:
    #                 fusion_mod.to(device=device, dtype=target_dtype)


def adapter_has_grad(adapter_model, adapter_names):
    for _, param in iter_adapter_named_parameters(adapter_model, adapter_names):
        if param.grad is not None and torch.any(param.grad != 0):
            return True
    return False


def register_adapter_activation_hooks(adapter_model, adapter_name):
    flag = {"executed": False}
    handles = []

    def make_hook(target):
        def hook(module, _input, _output, *_args):
            target["executed"] = True

        return hook

    for module_name, module in adapter_model.named_modules():
        if adapter_name in module_name:
            handles.append(module.register_forward_hook(make_hook(flag)))
    return flag, handles


def remove_handles(handles):
    for handle in handles:
        handle.remove()


def collect_logits(language_model, adapter_model, input_ids, active_setup, autocast_ctx):
    if active_setup is None:
        adapter_model.adapters_config.active_setup = None
        adapter_model.active_adapters = None
    else:
        adapter_model.set_active_adapters(active_setup)
    sync_module_device_dtype(adapter_model, input_ids.device, next(language_model.parameters()).dtype)

    prev_mode = language_model.training
    language_model.eval()
    adapter_model.eval()
    adapter_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        with autocast_ctx():
            outputs = language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
    logits = outputs.logits.detach().cpu()
    if prev_mode:
        language_model.train()
        adapter_model.train()
    return logits


def assert_weighted_match(base_logits, candidate, weights, references, branch_count, label):
    delta = torch.zeros_like(base_logits)
    for weight, ref in zip(weights, references):
        delta += weight * (ref - base_logits)
    expected = base_logits + delta / branch_count
    max_diff = torch.max(torch.abs(candidate - expected)).item()
    assert torch.allclose(candidate, expected, atol=5e-5, rtol=1e-4), (
        f"{label}: mismatch (max diff {max_diff:.6f})"
    )


def main():
    model_path = os.environ.get("INTERNVL_MODEL_PATH", "checkpoints/InternVL/InternVL2_5-1B")
    if not os.path.exists(model_path):
        print(f"InternVL checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    # dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if has_cuda else torch.float32)
    dtype = torch.float32
    autocast_ctx = (lambda: torch.cuda.amp.autocast(dtype=dtype)) if has_cuda else (lambda: contextlib.nullcontext())

    tokenizer, model, image_processor, context_len = load_eca_pretrained_model(
        model_path=model_path,
        torch_dtype=dtype,
        device=device,
        use_flash_attn=False,
    )
    model.train()

    if not hasattr(model, "language_model"):
        raise RuntimeError("InternVL model does not expose `language_model`.")
    if type(model.language_model).__name__ != "Qwen2ForCausalLM":
        raise RuntimeError(f"Expected Qwen2ForCausalLM, got {type(model.language_model).__name__}.")

    adapter_model = model.language_model.model
    if getattr(adapter_model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError(
            f"Expected eager attention for adapter testing, got "
            f"{getattr(adapter_model.config, '_attn_implementation', None)}."
        )

    adapter_init(adapter_model, use_customize=True)
    adapter_model.freeze_model()
    adapter_model.train()
    sync_module_device_dtype(adapter_model, device, dtype)

    num_layers = len(adapter_model.layers)
    alignment_layers = min(4, num_layers)
    leave_out = list(range(alignment_layers, num_layers))
    pa_cfg = ParBnConfig(
        mh_adapter=True,
        output_adapter=False,
        reduction_factor=16.0,
        dropout=0.0,
        scaling=4.0,
        non_linearity="linear",
        leave_out=leave_out,
    )
    lora_cfg = LoRAConfig(r=8, alpha=16)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model.language_model.config.vocab_size, (batch_size, seq_len), device=device)

    print(f"Loaded tokenizer/model from {model_path}")
    print(f"Context length: {context_len}")
    print(f"Image processor input size: {image_processor.input_size}")
    print(f"Activated adapters on first {alignment_layers} / {num_layers} layers (leave_out={leave_out}).")

    adapter_model.add_adapter("qwen_pa", config=pa_cfg)
    adapter_model.set_active_adapters("qwen_pa")
    freeze_adapter(adapter_model, "qwen_pa", freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[Qwen2] First PA trainables:")
    print_trainable_parameters(adapter_model)
    print("\n" + adapter_model.adapter_summary())

    with autocast_ctx():
        outputs = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        assert outputs.logits.shape[:2] == input_ids.shape, "Qwen2 logits shape mismatch"
        assert not torch.isnan(outputs.logits).any(), "Qwen2 logits contain NaN"
        loss = outputs.logits.sum()
    loss.backward()
    assert adapter_has_grad(adapter_model, "qwen_pa"), "First Qwen2 PA did not receive gradients."
    adapter_model.zero_grad(set_to_none=True)

    adapter_model.add_adapter("qwen_pa_extra", config=pa_cfg)
    adapter_model.set_active_adapters(["qwen_pa", "qwen_pa_extra"])
    freeze_adapter(adapter_model, ["qwen_pa", "qwen_pa_extra"], freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[Qwen2] Second PA trainables:")
    print_trainable_parameters(adapter_model)
    print("\n" + adapter_model.adapter_summary())

    logits_base = collect_logits(model.language_model, adapter_model, input_ids, None, autocast_ctx)
    logits_a = collect_logits(model.language_model, adapter_model, input_ids, "qwen_pa", autocast_ctx)
    logits_b = collect_logits(model.language_model, adapter_model, input_ids, "qwen_pa_extra", autocast_ctx)

    avg_a = Average("qwen_pa", "qwen_pa_extra", weights=[1.0, 0.0], normalize_weights=False)
    avg_b = Average("qwen_pa", "qwen_pa_extra", weights=[0.0, 1.0], normalize_weights=False)
    avg_mix = Average("qwen_pa", "qwen_pa_extra", weights=[0.5, 0.5], normalize_weights=False)

    logits_avg_a = collect_logits(model.language_model, adapter_model, input_ids, avg_a, autocast_ctx)
    logits_avg_b = collect_logits(model.language_model, adapter_model, input_ids, avg_b, autocast_ctx)
    logits_avg_mix = collect_logits(model.language_model, adapter_model, input_ids, avg_mix, autocast_ctx)

    branch_count = 2
    assert_weighted_match(logits_base, logits_avg_a, [1.0, 0.0], [logits_a, logits_b], branch_count, "Average [1,0]")
    assert_weighted_match(logits_base, logits_avg_b, [0.0, 1.0], [logits_a, logits_b], branch_count, "Average [0,1]")
    assert_weighted_match(
        logits_base, logits_avg_mix, [0.5, 0.5], [logits_a, logits_b], branch_count, "Average [0.5,0.5]"
    )

    adapter_model.set_active_adapters(["qwen_pa", "qwen_pa_extra"])
    with autocast_ctx():
        outputs = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        assert outputs.logits.shape[:2] == input_ids.shape, "Qwen2 logits shape mismatch (two PAs)"
        assert not torch.isnan(outputs.logits).any(), "Qwen2 logits contain NaN with two PAs"
        loss = outputs.logits.sum()
    loss.backward()
    assert adapter_has_grad(adapter_model, ["qwen_pa", "qwen_pa_extra"]), "Second Qwen2 PA did not receive gradients."
    adapter_model.zero_grad(set_to_none=True)

    adapter_model.set_active_adapters(Average("qwen_pa", "qwen_pa_extra", weights=[1, 1]))
    sync_module_device_dtype(adapter_model, device, dtype)
    with autocast_ctx():
        outputs = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        assert outputs.logits.shape[:2] == input_ids.shape, "Qwen2 logits shape mismatch (Average)"
        assert not torch.isnan(outputs.logits).any(), "Qwen2 logits contain NaN with Average"
        loss = outputs.logits.sum()
    loss.backward()
    assert adapter_has_grad(
        adapter_model, ["qwen_pa", "qwen_pa_extra"]
    ), "Average fusion on Qwen2 PAs did not receive gradients."
    adapter_model.zero_grad(set_to_none=True)

    third_adapter = "qwen_pa_third"
    adapter_model.add_adapter(third_adapter, config=pa_cfg)
    all_adapters = ["qwen_pa", "qwen_pa_extra", third_adapter]
    adapter_model.set_active_adapters(all_adapters)
    freeze_adapter(adapter_model, ["qwen_pa", "qwen_pa_extra"], freeze=True)
    freeze_adapter(adapter_model, third_adapter, freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[Qwen2] Added third adapter (student configuration):")
    print_trainable_parameters(adapter_model)
    print("\n" + adapter_model.adapter_summary())

    with torch.no_grad():
        for name, param in adapter_model.named_parameters():
            if third_adapter in name and param.requires_grad:
                param.data = param.data.normal_(mean=0.0, std=1e-3)

    with torch.no_grad():
        prev_mode = model.language_model.training
        model.language_model.eval()
        adapter_model.eval()
        adapter_model.set_active_adapters(["qwen_pa", "qwen_pa_extra"])
        sync_module_device_dtype(adapter_model, device, dtype)
        teacher_flag, teacher_handles = register_adapter_activation_hooks(adapter_model, third_adapter)
        with autocast_ctx():
            teacher_out = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        teacher_logits = teacher_out.logits.detach()
        remove_handles(teacher_handles)
        assert not teacher_flag["executed"], "Third adapter unexpectedly executed during teacher forward."
        if prev_mode:
            model.language_model.train()
            adapter_model.train()

    adapter_model.set_active_adapters(all_adapters)
    sync_module_device_dtype(adapter_model, device, dtype)
    student_flag, student_handles = register_adapter_activation_hooks(adapter_model, third_adapter)
    adapter_model.zero_grad(set_to_none=True)
    with autocast_ctx():
        student_out = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        kd_loss = F.mse_loss(student_out.logits, teacher_logits.to(device=device, dtype=student_out.logits.dtype))
    kd_loss.backward()
    remove_handles(student_handles)
    assert student_flag["executed"], "Third adapter did not execute during student forward."

    third_grad = adapter_has_grad(adapter_model, third_adapter)
    frozen_grad = adapter_has_grad(adapter_model, ["qwen_pa", "qwen_pa_extra"])
    assert third_grad, "Third adapter did not receive gradients during KD pass."
    assert not frozen_grad, "Frozen adapters accumulated gradients unexpectedly."
    adapter_model.zero_grad(set_to_none=True)

    clone_teacher = "qwen_pa_third_teacher"
    adapter_model.add_adapter(clone_teacher, config=pa_cfg)
    init_adapter(adapter_model, clone_teacher, third_adapter)
    freeze_adapter(adapter_model, clone_teacher, freeze=True)
    freeze_adapter(adapter_model, third_adapter, freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[Qwen2] Teacher/Student clone setup:")
    print_trainable_parameters(adapter_model)
    print("\n" + adapter_model.adapter_summary())

    adapter_model.set_active_adapters(["qwen_pa", "qwen_pa_extra", third_adapter])
    hook_flag, hook_handles = register_adapter_activation_hooks(adapter_model, clone_teacher)
    with torch.no_grad():
        with autocast_ctx():
            _ = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
    remove_handles(hook_handles)
    assert not hook_flag["executed"], "Teacher clone executed when inactive."

    teacher_clone_logits = collect_logits(model.language_model, adapter_model, input_ids, [clone_teacher], autocast_ctx)
    student_clone_logits = collect_logits(model.language_model, adapter_model, input_ids, [third_adapter], autocast_ctx)
    assert torch.allclose(
        student_clone_logits.to(dtype=teacher_clone_logits.dtype),
        teacher_clone_logits,
        atol=1e-4,
        rtol=1e-3,
    ), "Student adapter logits diverge from teacher clone logits."

    adapter_model.set_active_adapters([third_adapter])
    adapter_model.zero_grad(set_to_none=True)
    with autocast_ctx():
        clone_out = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        clone_loss = clone_out.logits.sum()
    clone_loss.backward()
    assert adapter_has_grad(adapter_model, third_adapter), "Student adapter did not receive gradients."
    assert not adapter_has_grad(adapter_model, clone_teacher), "Teacher clone unexpectedly received gradients."
    adapter_model.zero_grad(set_to_none=True)

    adapter_model.add_adapter("qwen_lora", config=lora_cfg)
    adapter_model.set_active_adapters(["qwen_lora"])
    freeze_adapter(adapter_model, "qwen_lora", freeze=False)
    sync_module_device_dtype(adapter_model, device, dtype)
    print("[Qwen2] LoRA trainables:")
    print_trainable_parameters(adapter_model)
    print("\n" + adapter_model.adapter_summary())

    with autocast_ctx():
        outputs = model.language_model(input_ids=input_ids, use_cache=False, output_attentions=False)
        assert outputs.logits.shape[:2] == input_ids.shape, "Qwen2 logits shape mismatch (LoRA)"
        assert not torch.isnan(outputs.logits).any(), "Qwen2 logits contain NaN after LoRA"
        loss = outputs.logits.sum()
    loss.backward()
    assert adapter_has_grad(adapter_model, "qwen_lora"), "LoRA adapter on Qwen2 did not receive gradients."

    print("Qwen2 adapter checks passed.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available. Loading InternVL2.5-1B on CPU may require large memory.",
            file=sys.stderr,
        )
    main()
