#!/usr/bin/env bash
# Offline batch MT inference launcher.
#
# Usage:
#   scripts/mt/infer.sh --tgt-lang Hindi --input-file segments.txt --output-file out.jsonl
#   scripts/mt/infer.sh --tgt-lang Tamil --document --input-file article.txt --max-new-tokens 8192
#
# Defaults come from configs/mt/infer/offline_vllm.yaml; any flag given here
# overrides the config (e.g. --model <your merged checkpoint>).
set -euo pipefail
cd "$(dirname "$0")/../.."

# Keep vLLM's engine in-process; matches the validated single-GPU configuration.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

exec python -m bodhan_genai.mt.inference.offline_vllm \
    --config configs/mt/infer/offline_vllm.yaml \
    "$@"
