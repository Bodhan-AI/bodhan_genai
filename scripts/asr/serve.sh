#!/usr/bin/env bash
# Launch the ASR server (Ray Serve): one IndicTranscribe engine per GPU replica,
# serving buffered streaming + offline transcription + language ID.
#
# Usage:
#   ./scripts/asr/serve.sh [extra --flags]                      # published default
#   MODEL_DIR=/path/to/indic-transcribe-hf ./scripts/asr/serve.sh   # or a local checkpoint
# Optional env: NUM_REPLICAS (default = GPU count), PORT (default 8000).
#
# Note on streaming: IndicTranscribe is an attention encoder-decoder model, so
# there is no frame-synchronous output. Each update re-decodes a rolling
# buffer, and the latency floor is --stream_min_decode_s (default 1 s).
# See docs/asr/serving.md.
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — the ASR server requires a GPU node." >&2
    exit 1
fi
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "ERROR: no GPUs visible to nvidia-smi." >&2
    exit 1
fi

export RAY_ADDRESS="${RAY_ADDRESS:-local}"

# HTTP Basic auth. Point ASR_AUTH_FILE at a 0600 file containing "user:password"
# (outside the repo -- the credential must not be committed, and only the path
# travels through Ray). Serving without it takes ASR_AUTH_ALLOW_OPEN=1, said out
# loud, because these endpoints drive every GPU on the box.
if [ -z "${ASR_AUTH_FILE:-}" ] && [ "${ASR_AUTH_ALLOW_OPEN:-}" != "1" ]; then
    echo "ERROR: set ASR_AUTH_FILE=/path/to/credfile ('user:password', chmod 600)," >&2
    echo "       or ASR_AUTH_ALLOW_OPEN=1 to serve unauthenticated on purpose." >&2
    exit 1
fi
export ASR_AUTH_FILE ASR_AUTH_ALLOW_OPEN

exec python -m bodhan_genai.asr.serving.app \
    --model_dir "${MODEL_DIR:-}" \
    --num_replicas "${NUM_REPLICAS:-$NUM_GPUS}" \
    --port "${PORT:-8000}" \
    "$@"
