#!/usr/bin/env bash
# Stage 1 only: page images -> <name>.layout.json. Loads torch, never vLLM.
#
#   scripts/ocr/layout.sh pages/ -o out/
#
# Inspect or correct the layout, then run stage 2 against it:
#   python -m bodhan_genai.ocr.inference.cli ocr out/page.layout.json -o out/
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python -m bodhan_genai.ocr.inference.cli layout \
    --config configs/ocr/infer/layout.yaml \
    "$@"
