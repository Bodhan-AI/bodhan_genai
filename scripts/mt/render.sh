#!/usr/bin/env bash
# Render bitext into instruction chat rows for finetuning.
#
# Usage:
#   scripts/mt/render.sh [config.yaml] [extra flags]   # default: configs/mt/data/render.yaml
#   scripts/mt/render.sh configs/mt/data/render.yaml --dry-run
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/mt/data/render.yaml}"
[[ $# -gt 0 ]] && shift

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec python -m bodhan_genai.mt.data.render --config "$CONFIG" "$@"
