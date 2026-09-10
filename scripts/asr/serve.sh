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

# Optional HTTP Basic auth, off by default: this is a reference setup, and a server you can
# start with no arguments is the point. Set ASR_AUTH_FILE to a 0600 file holding "user:password"
# and every route except /health requires it. Keep that file outside the repo -- only the path
# travels, so the credential is never committed and never goes through Ray's runtime_env.
#
# Unset means OPEN, and this binds 0.0.0.0. On a shared network put a TLS reverse proxy in
# front, or bind to localhost and tunnel.
export ASR_AUTH_FILE

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — the ASR server requires a GPU node." >&2
    exit 1
fi
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Free-port search, lifted from scripts/mt/serve.sh. Shared clusters squat ports, and Ray Serve
# reports a collision as `RuntimeError: Failed to bind to address` buried in a deduplicated
# controller log on the worker node -- which is a long way from "the port was taken".
PORT="${PORT:-8000}"
port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }
if port_in_use "${PORT}"; then
    echo "Port ${PORT} is in use; searching for a free one ..."
    for candidate in $(seq "$((PORT + 1))" "$((PORT + 200))"); do
        if ! port_in_use "${candidate}"; then PORT="${candidate}"; break; fi
    done
    if port_in_use "${PORT}"; then
        echo "ERROR: no free port found near ${PORT}." >&2
        exit 1
    fi
    echo "Using port ${PORT}."
fi
export PORT

if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "ERROR: no GPUs visible to nvidia-smi." >&2
    exit 1
fi

export RAY_ADDRESS="${RAY_ADDRESS:-local}"

# MODEL_DIR from the environment is honoured ONLY inside a deployment image (see
# scripts/asr/infer.sh for the reasoning). Pass --model_dir to override locally.
if [ "${BODHAN_GENAI_DEPLOYMENT:-}" != "1" ]; then
    unset MODEL_DIR
fi
# Resolved port and PID, as shell-sourceable lines -- the same contract scripts/mt/serve.sh and
# scripts/ocr/serve.sh provide via their INFO_FILE. It matters more now that the port search
# above can move the port: without this a caller cannot tell where the server ended up.
# `exec` below replaces this shell, so $$ is the server's PID.
INFO_FILE="${INFO_FILE:-asr-serve.info}"
printf 'PID=%s\nPORT=%s\nHEALTH=%s\n' "$$" "${PORT}" "http://127.0.0.1:${PORT}/health" > "${INFO_FILE}"
echo "Starting on port ${PORT} (details in ${INFO_FILE})"
echo "Readiness: this runs in the FOREGROUND, so poll ${PORT}/health until it answers 200."
echo "  python -m bodhan_genai.asr.serving.client --mode transcribe --url http://127.0.0.1:${PORT} --paths clip.wav --lang hi"

exec python -m bodhan_genai.asr.serving.app \
    --model_dir "${MODEL_DIR:-}" \
    --num_replicas "${NUM_REPLICAS:-$NUM_GPUS}" \
    --port "${PORT}" \
    "$@"
