#!/usr/bin/env bash
# IndicOCR: page images -> markdown + per-block JSON.
#
#   scripts/ocr/parse.sh pages/ -o out/
#   scripts/ocr/parse.sh page.png -o out/ --save-layout --table-format markdown
#
# Prefer passing a folder: the engine takes minutes to start and every block of every page goes
# through it in one batch. Defaults come from configs/ocr/infer/parse.yaml; any flag overrides.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python -m bodhan_genai.ocr.inference.cli parse \
    --config configs/ocr/infer/parse.yaml \
    "$@"
