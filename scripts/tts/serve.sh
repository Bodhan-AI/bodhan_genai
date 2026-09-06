#!/usr/bin/env bash
# Launch the streaming TTS server (Ray Serve): one merged WS-ingress + vLLM
# AsyncLLM + in-process SNAC replica per GPU.
#
# Usage:
#   ./scripts/tts/serve.sh [extra --flags for serving.app]
# Optional env: CHECKPOINT (default bodhan-ai/indic-speak — public Hub repo),
#               TOKENIZER (default $CHECKPOINT), SNAC (default hubertsiuzdak/snac_24khz),
#               NUM_REPLICAS (default = GPU count), PORT (default 8000).
# Example:
#   CHECKPOINT=/path/to/checkpoints/my-tts-ckpt PORT=8000 ./scripts/tts/serve.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — the serving stack requires a GPU node." >&2
    exit 1
fi
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "ERROR: no GPUs visible to nvidia-smi." >&2
    exit 1
fi

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

CHECKPOINT="${CHECKPOINT:-bodhan-ai/indic-speak}"
exec python -m bodhan_genai.tts.serving.app \
    --checkpoint_path "$CHECKPOINT" \
    --tokenizer_path "${TOKENIZER:-$CHECKPOINT}" \
    --snac_model_path "${SNAC:-hubertsiuzdak/snac_24khz}" \
    --num_replicas "${NUM_REPLICAS:-$NUM_GPUS}" \
    --port "${PORT:-8000}" \
    "$@"
