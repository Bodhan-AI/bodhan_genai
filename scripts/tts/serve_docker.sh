#!/usr/bin/env bash
# Build (if needed) and run the dockerized streaming TTS server.
#
# Usage:
#   CHECKPOINT_DIR=/abs/path/to/checkpoint scripts/tts/serve_docker.sh [--build] [extra serving.app flags]
#
# Env:
#   CHECKPOINT_DIR   host dir of the model checkpoint (mounted read-only)   [required*]
#   TOKENIZER_DIR    host dir of the extended tokenizer (default: checkpoint's own)
#   SNAC_DIR         host dir of the SNAC codec (default: hubertsiuzdak/snac_24khz from the Hub)
#   CHECKPOINT       *alternative to CHECKPOINT_DIR: an HF hub id (needs network / HF_TOKEN)
#   PORT             host port to expose (default 8000)
#   GPUS             "all" (default) or a device list like "0" / "0,1"
#   NUM_REPLICAS     replicas inside the container (default: visible GPU count)
#   SHM_SIZE         /dev/shm size for Ray + vLLM (default 8g)
#   IMAGE            image tag (default bodhan-tts-serve)
#
# Examples:
#   CHECKPOINT_DIR=/path/to/training_checkpoints/llama3_new/pretrain/checkpoint-46005 \
#   TOKENIZER_DIR=/path/to/checkpoints/llama-3-audio-tok_trimmed \
#   SNAC_DIR=/path/to/checkpoints/snac/snac_24khz \
#   GPUS=0 scripts/tts/serve_docker.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

IMAGE="${IMAGE:-bodhan-tts-serve}"

if [ "${1:-}" = "--build" ]; then
    shift
    docker build -f docker/tts/Dockerfile.serve -t "$IMAGE" .
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -f docker/tts/Dockerfile.serve -t "$IMAGE" .
fi

RUN_ARGS=(--rm --shm-size "${SHM_SIZE:-8g}" -p "${PORT:-8000}:8000")
if [ -t 0 ]; then RUN_ARGS+=(-it); fi

GPUS="${GPUS:-all}"
if [ "$GPUS" = "all" ]; then
    RUN_ARGS+=(--gpus all)
else
    RUN_ARGS+=(--gpus "device=${GPUS}")
fi

ENV_ARGS=()
MOUNTS=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    MOUNTS+=(-v "$(readlink -f "$CHECKPOINT_DIR"):/models/checkpoint:ro")
    ENV_ARGS+=(-e CHECKPOINT=/models/checkpoint -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
elif [ -n "${CHECKPOINT:-}" ]; then
    ENV_ARGS+=(-e CHECKPOINT="$CHECKPOINT")   # HF hub id — needs network (and HF_TOKEN if gated)
else
    echo "ERROR: set CHECKPOINT_DIR=/abs/path/to/checkpoint (or CHECKPOINT=<hf-id>)" >&2
    exit 1
fi
if [ -n "${TOKENIZER_DIR:-}" ]; then
    MOUNTS+=(-v "$(readlink -f "$TOKENIZER_DIR"):/models/tokenizer:ro")
    ENV_ARGS+=(-e TOKENIZER=/models/tokenizer)
fi
if [ -n "${SNAC_DIR:-}" ]; then
    MOUNTS+=(-v "$(readlink -f "$SNAC_DIR"):/models/snac:ro")
    ENV_ARGS+=(-e SNAC=/models/snac)
fi
# pass through only when set on the host
for var in NUM_REPLICAS HF_TOKEN; do
    if [ -n "${!var:-}" ]; then ENV_ARGS+=(-e "$var=${!var}"); fi
done

exec docker run "${RUN_ARGS[@]}" "${MOUNTS[@]}" "${ENV_ARGS[@]}" "$IMAGE" "$@"
