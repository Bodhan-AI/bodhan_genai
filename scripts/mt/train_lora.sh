#!/usr/bin/env bash
# LoRA fine-tuning launcher for bodhan_genai.mt (single node).
#
# Usage:
#   scripts/mt/train_lora.sh [config.yaml]        # default: configs/mt/train/lora.yaml
#   NUM_GPUS=4 scripts/mt/train_lora.sh configs/mt/train/lora.yaml
#   ACCELERATE_CONFIG=... MASTER_PORT=29501 scripts/mt/train_lora.sh ...
#
# NUM_GPUS defaults to the nvidia-smi GPU count; override via env.
# Auto-resume: re-launching with the same training.output_dir resumes from the
# last checkpoint (set `resume: false` in the config to start fresh).
#
# The recipe is PROVISIONAL — see docs/mt/training.md.

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/mt/train/lora.yaml}"
ACCEL="${ACCELERATE_CONFIG:-configs/mt/accelerate/single_node.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — training requires NVIDIA GPUs." >&2
    exit 1
fi
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"

# MT needs transformers >= 5.12 (the release that knows Gemma 4). Fail here with a
# pointer rather than 15 GB into a model load with an opaque architecture error.
python - <<'EOF' || exit 1
import sys
try:
    import transformers, trl  # noqa: F401
except ImportError as exc:
    sys.exit(
        f"missing MT training dependency ({exc.name}). MT installs into its own "
        f"environment:\n    ./install.sh && source .venv/bin/activate"
    )
major, minor = (int(p) for p in transformers.__version__.split(".")[:2])
if (major, minor) < (5, 12):
    sys.exit(
        f"transformers {transformers.__version__} < 5.12, which is the Gemma 4 floor. "
        f"You are probably in the TTS environment; use:\n"
        f"    ./install.sh && source .venv/bin/activate"
    )
EOF

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / NUM_GPUS ))}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
# Offline by default: a training node usually has no egress, and wandb blocking
# on a network call is a confusing way to discover that.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Uncomment on air-gapped clusters (models/tokenizers must already be cached):
# export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "config    : $CONFIG"
echo "accelerate: $ACCEL"
echo "GPUs      : $NUM_GPUS"

exec accelerate launch \
    --config_file "$ACCEL" \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    --machine_rank 0 \
    --main_process_port "${MASTER_PORT:-29500}" \
    -m bodhan_genai.mt.training.train "$CONFIG"
