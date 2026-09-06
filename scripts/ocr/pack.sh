#!/usr/bin/env bash
# Pack the training manifest into the blob cache the trainer reads. Slow once, then
# every epoch avoids two filesystem opens per page — which on a shared parallel
# filesystem costs more than the forward pass.
#
#   scripts/ocr/pack.sh
#   scripts/ocr/pack.sh --sources hw-vs ncert-scert    # a subset, for a trial run
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python -m bodhan_genai.ocr.data.blob \
    --manifest data/manifests/layout_train.json \
    --cache-prefix data/cache/layout_train \
    --max-side 1024 \
    "$@"
