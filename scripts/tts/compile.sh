#!/usr/bin/env bash
# Stage 2: compile tokenized Parquet into training-ready sequences.
# Usage: scripts/tts/compile.sh [config] [extra args...]
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${1:-configs/tts/data/compile.yaml}"
shift || true

exec python -m bodhan_genai.tts.data.compile --config "$CONFIG" "$@"
