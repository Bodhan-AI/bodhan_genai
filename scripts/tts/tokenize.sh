#!/usr/bin/env bash
# Stage 1: SNAC-tokenize audio datasets to sharded Parquet.
# Usage: scripts/tts/tokenize.sh [config] [extra args...]
set -euo pipefail

cd "$(dirname "$0")/../.."

export RAY_ADDRESS="${RAY_ADDRESS:-local}"

CONFIG="${1:-configs/tts/data/tokenize.yaml}"
shift || true

exec python -m bodhan_genai.tts.data.tokenize --config "$CONFIG" "$@"
