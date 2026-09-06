#!/usr/bin/env bash
# Train IndicDocLayout (detection + reading order).
#
#   scripts/ocr/train.sh                     # full run, config as written
#   scripts/ocr/train.sh --max-steps 20      # smoke-test the recipe on a few batches
#   GPUS=0,1 scripts/ocr/train.sh            # data-parallel over two GPUs
#
# Multi-GPU goes through accelerate. Note SLURM presets CUDA_VISIBLE_DEVICES on an
# allocated step, and it wins over GPUS unless you export it yourself.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/ocr/train/layout.yaml}"
GPUS="${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
NUM_GPUS="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"

if [ "$NUM_GPUS" -gt 1 ]; then
    exec accelerate launch --num_processes "$NUM_GPUS" \
        -m bodhan_genai.ocr.training.train --config "$CONFIG" "$@"
fi
exec python -m bodhan_genai.ocr.training.train --config "$CONFIG" "$@"
