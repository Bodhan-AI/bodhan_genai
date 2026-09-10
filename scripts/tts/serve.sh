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

# Optional HTTP Basic auth, off by default: this is a reference setup, and a server you can
# start with no arguments is the point. Set TTS_AUTH_FILE to a 0600 file holding "user:password"
# and every route except /health requires it. Keep that file outside the repo -- only the path
# travels, so the credential is never committed and never goes through Ray's runtime_env.
#
# Unset means OPEN, and this binds 0.0.0.0. On a shared network put a TLS reverse proxy in
# front, or bind to localhost and tunnel.
export TTS_AUTH_FILE

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — the serving stack requires a GPU node." >&2
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
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Checkpoint paths from the environment are honoured ONLY inside a deployment image, where
# they are how mounted weights under /models are addressed. Outside one, an inherited CHECKPOINT / TOKENIZER / SNAC
# silently serving different weights is a correctness bug that presents as a model regression,
# so it is ignored and the published default is used. To point this at your own checkpoint
# locally, pass the flag -- "$@" is forwarded last and argparse takes the later occurrence:
#   ./scripts/tts/serve.sh --checkpoint_path /path/to/ckpt
if [ "${BODHAN_GENAI_DEPLOYMENT:-}" != "1" ]; then
    unset CHECKPOINT TOKENIZER SNAC
fi
CHECKPOINT="${CHECKPOINT:-bodhan-ai/indic-speak}"
# Resolved port and PID, as shell-sourceable lines -- the same contract scripts/mt/serve.sh and
# scripts/ocr/serve.sh provide via their INFO_FILE. It matters more now that the port search
# above can move the port: without this a caller cannot tell where the server ended up.
# `exec` below replaces this shell, so $$ is the server's PID.
INFO_FILE="${INFO_FILE:-tts-serve.info}"
printf 'PID=%s\nPORT=%s\nHEALTH=%s\n' "$$" "${PORT}" "http://127.0.0.1:${PORT}/health" > "${INFO_FILE}"
echo "Starting on port ${PORT} (details in ${INFO_FILE})"
echo "Readiness: this runs in the FOREGROUND, so poll ${PORT}/health until it answers 200."
echo "  python -m bodhan_genai.tts.serving.client --mode stream --url http://127.0.0.1:${PORT} --text 'Hello world' --out out.wav"

exec python -m bodhan_genai.tts.serving.app \
    --checkpoint_path "$CHECKPOINT" \
    --tokenizer_path "${TOKENIZER:-$CHECKPOINT}" \
    --snac_model_path "${SNAC:-hubertsiuzdak/snac_24khz}" \
    --num_replicas "${NUM_REPLICAS:-$NUM_GPUS}" \
    --port "${PORT}" \
    "$@"
