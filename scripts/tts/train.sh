#!/usr/bin/env bash
# Full fine-tuning / pretraining launcher (single node).
#
# Usage:
#   scripts/tts/train.sh [config.yaml]              # default: configs/tts/train/pretrain.yaml
#   NUM_GPUS=4 scripts/tts/train.sh configs/tts/train/sft.yaml
#   ACCELERATE_CONFIG=... MASTER_PORT=29501 scripts/tts/train.sh ...
#
# NUM_GPUS defaults to the nvidia-smi GPU count; override via env.
# Auto-resume: re-launching with the same training.output_dir resumes from the
# last checkpoint automatically (get_last_checkpoint in train.py).

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/tts/train/pretrain.yaml}"
ACCEL="${ACCELERATE_CONFIG:-configs/tts/accelerate/single_node_fsdp.yaml}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — training requires NVIDIA GPUs." >&2
    exit 1
fi
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"

# Inductor/dynamo tuning for the max-autotune compile path
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-1}"
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-1}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Generous heartbeat: the first max-autotune compile pass can stall ranks for a long time
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / NUM_GPUS ))}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Uncomment on air-gapped clusters (models/tokenizers must already be cached or local):
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec accelerate launch \
    --config_file "$ACCEL" \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --machine_rank 0 \
    --main_process_port "${MASTER_PORT:-29500}" \
    -m bodhan_genai.tts.training.train "$CONFIG"
