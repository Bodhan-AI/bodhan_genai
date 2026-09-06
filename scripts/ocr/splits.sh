#!/usr/bin/env bash
# Build train/val/test manifests for the layout mix. Reads only the jsons/ directories,
# so it is cheap and safe to re-run: the hash policy is deterministic, so a rebuild
# reproduces the same split rather than reshuffling it.
#
#   scripts/ocr/splits.sh
#   scripts/ocr/splits.sh --data-root /mnt/corpora
set -euo pipefail
cd "$(dirname "$0")/../.."

exec python -m bodhan_genai.ocr.data.splits \
    --config configs/ocr/data/sources.yaml \
    --out-dir data/manifests \
    "$@"
