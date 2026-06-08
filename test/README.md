# Smoke Tests

This folder keeps manual smoke tests for the patched ECA components. They are not benchmark scripts and are not meant to replace full training or evaluation.

## How To Run

Run every test from the repository root. The test files live under `test/`, so direct execution from inside this folder cannot find local packages such as `models`, `tasks`, `datasets`, and `runners`.

```bash
cd /path/to/ECA
conda activate eca_test
PYTHONPATH=. python test/test_internvl_qwen2_adapters.py
```

If you run from inside `test/`, use `PYTHONPATH=..` instead.

```bash
cd /path/to/ECA/test
PYTHONPATH=.. python test_internvl_qwen2_adapters.py
```

## BLIP-2 Adapter Checks

These scripts check the customized adapter path used by the BLIP-2 Q-Former and ViT components. They are older development checks and may pause at `input()` statements.

```bash
PYTHONPATH=. python test/test_blip2_qformer_adapter.py
PYTHONPATH=. python test/test_blip2_vit_adapter.py
```

## InternVL Checks

These scripts use `INTERNVL_MODEL_PATH`. If it is not set, they default to `checkpoints/InternVL/InternVL2_5-1B`.

```bash
PYTHONPATH=. python test/test_internvl_forward.py
PYTHONPATH=. python test/test_internvl_qwen2_adapters.py
PYTHONPATH=. python test/test_internvl_visual_lora.py
```

Optional override:

```bash
INTERNVL_MODEL_PATH=checkpoints/InternVL/InternVL2_5-4B \
PYTHONPATH=. python test/test_internvl_qwen2_adapters.py
```

## LLaVA Checks

These scripts use `LLAVA_BASE_MODEL` and `LLAVA_PROJECTOR_PATH`. If they are not set, they default to the paths under `checkpoints/LLaVA/`.

```bash
PYTHONPATH=. python test/test_llava_forward.py
PYTHONPATH=. python test/test_llava_adapter.py
```

Optional override:

```bash
LLAVA_BASE_MODEL=checkpoints/LLaVA/vicuna-7b-v0 \
LLAVA_PROJECTOR_PATH=checkpoints/LLaVA/llava-7b-v0 \
PYTHONPATH=. python test/test_llava_adapter.py
```

## Dictionary Learner Check

This script follows the normal config loading path and requires a training config.

```bash
PYTHONPATH=. python test/test_dictionary_learner.py \
  --cfg-path configs/train/ecaq_cl_train_cap.yaml
```

## Notes

- Run these tests on a GPU machine with the same environment used for ECA experiments.
- Keep `PYTHONPATH=.` in the command unless the tests are converted to a package-style runner later.
- Passing these checks only verifies key import, forward, adapter, and gradient paths. It does not validate final benchmark performance.
