#!/usr/bin/env bash
set -euo pipefail

# ECA does not use BLIP-Diffusion; keep optional diffusion registration off unless explicitly enabled.
export ECA_ENABLE_BLIP_DIFFUSION=${ECA_ENABLE_BLIP_DIFFUSION:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

CFG_PATH=${1:-configs/train/ecaq_cl_train_cap.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29500}

python -W ignore \
  -m torch.distributed.run \
  --master_port "${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  Snowball_Run.py \
  --cfg-path "${CFG_PATH}" \
  --env-cfg-path configs/env.yaml
