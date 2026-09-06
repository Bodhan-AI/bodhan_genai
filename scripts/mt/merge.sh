#!/usr/bin/env bash
# Merge a LoRA adapter into its base model and make the result servable.
#
# Usage:
#   scripts/mt/merge.sh <adapter-dir> <output-dir> [extra flags]
#
# Runs the merge, stages the tokenizer + processor, then adds the KV-shared
# k_norm sidecar that stock vLLM requires. The result loads with `vllm serve`.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ $# -lt 2 ]]; then
    echo "usage: scripts/mt/merge.sh <adapter-dir> <output-dir> [extra flags]" >&2
    exit 2
fi
ADAPTER="$1"; OUTPUT="$2"; shift 2

exec python -m bodhan_genai.mt.training.merge \
    --adapter-path "$ADAPTER" \
    --output-dir "$OUTPUT" \
    --vllm-ready \
    "$@"
