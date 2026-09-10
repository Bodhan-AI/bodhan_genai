#!/usr/bin/env bash
# Offline batch ASR inference launcher (IndicTranscribe).
#
# Usage:
#   MODEL_DIR=/path/to/indic-transcribe-hf \   # optional; defaults to the published repo
#   scripts/asr/infer.sh --manifest manifests/eval.jsonl --out-dir out/asr_run [more flags]
#
# Multi-GPU: one shard per GPU, each its own process (the model is ~1.2B params
# and each shard writes its own hyp_shard<N>.jsonl, so shards never contend).
#   MODEL_DIR=... NUM_SHARDS=8 scripts/asr/infer.sh --manifest m.jsonl --out-dir out/
#
# Budget 4-8 CPU cores per GPU: audio decode and feature-batch assembly are CPU
# work, and a CPU-starved node starves the GPUs (measured: with 2 CPUs for 8
# GPUs, per-shard encode time went 18 s -> 437 s).
set -euo pipefail

cd "$(dirname "$0")/../.."

# MODEL_DIR from the environment is honoured ONLY inside a deployment image, where it is how
# mounted weights under /models are addressed. Outside one, an inherited MODEL_DIR silently
# transcribing with different weights is a correctness bug that presents as a model regression.
# Pass --model-dir instead: "$@" is forwarded and argparse takes the later occurrence.
if [ "${BODHAN_GENAI_DEPLOYMENT:-}" != "1" ]; then
    unset MODEL_DIR
fi
# Empty is fine: the CLI resolves the published default (asr.checkpoints).
MODEL_DIR="${MODEL_DIR:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"

if [ "${NUM_SHARDS}" -le 1 ]; then
    exec python -m bodhan_genai.asr.inference.transcribe \
        ${MODEL_DIR:+--model-dir "${MODEL_DIR}"} "$@"
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES="${shard}" python -m bodhan_genai.asr.inference.transcribe \
        ${MODEL_DIR:+--model-dir "${MODEL_DIR}"} \
        --shard "${shard}" --num-shards "${NUM_SHARDS}" "$@" &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
exit "${status}"
