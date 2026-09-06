#!/usr/bin/env bash
# Score a layout checkpoint: detection mAP, plus reading order against the raster
# baseline. Raster is strong on single-column pages, so read `tau_model_hard` — the
# non-raster slice — not just the aggregate.
#
#   scripts/ocr/eval.sh --ckpt runs/layout/final
#   scripts/ocr/eval.sh --ckpt runs/layout/final/ema --limit 200
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python -m bodhan_genai.ocr.eval.layout \
    --manifest data/manifests/layout_test.json \
    --out runs/layout/metrics.json \
    "$@"
