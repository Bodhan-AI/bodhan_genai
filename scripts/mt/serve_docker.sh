#!/usr/bin/env bash
# Build (if needed) and run the dockerized MT server: stock `vllm serve` behind
# scripts/mt/serve.sh, on vLLM's OpenAI-compatible /v1.
#
# Usage:
#   HF_TOKEN=hf_... scripts/mt/serve_docker.sh [--build] [extra vllm serve flags]
#   CHECKPOINT_DIR=/abs/path/to/merged-ckpt scripts/mt/serve_docker.sh
#
# Env:
#   CHECKPOINT_DIR   host dir of a merged checkpoint (mounted read-only)
#   CHECKPOINT       *alternative to CHECKPOINT_DIR: an HF hub id (needs network).
#                    Unset, serve.sh falls back to bodhan-ai/indic-translate
#   HF_TOKEN         hub credentials. Not optional for that default — it is a PRIVATE
#                    repo, and `hf auth login` is not available inside the container
#   HF_CACHE_DIR     host dir for the hub cache (mounted rw). Without it the 15.9 GB
#                    pull lands in the container layer and --rm discards it
#   PORT             host port to expose (default 8000)
#   GPUS             "all" (default) or a device list like "0" / "0,1"
#   SERVED_NAME      the name clients must ask for (default indic_translate)
#   MAX_MODEL_LEN    KV-cache window (default 8192; 32768 is the validated ceiling)
#   GPU_MEMORY_UTILIZATION   default 0.90
#   TENSOR_PARALLEL_SIZE     default 1; needs that many devices in GPUS
#   SHM_SIZE         /dev/shm size (default 8g)
#   IMAGE            image tag (default bodhan-mt-serve)
#
# Examples:
#   HF_TOKEN=hf_... HF_CACHE_DIR=~/.cache/huggingface GPUS=0 scripts/mt/serve_docker.sh
#   CHECKPOINT_DIR=/path/to/merged-ckpt GPUS=0,1 TENSOR_PARALLEL_SIZE=2 \
#       MAX_MODEL_LEN=32768 scripts/mt/serve_docker.sh
#   scripts/mt/serve_docker.sh --max-num-seqs 64      # extras reach `vllm serve`
set -euo pipefail
cd "$(dirname "$0")/../.."

IMAGE="${IMAGE:-bodhan-mt-serve}"

if [ "${1:-}" = "--build" ]; then
    shift
    docker build -f docker/mt/Dockerfile.serve -t "$IMAGE" .
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -f docker/mt/Dockerfile.serve -t "$IMAGE" .
fi

# Published as PORT:8000, and PORT is deliberately NOT passed inside: serve.sh
# would move off a busy port, and in the container's own netns nothing is busy —
# so the search can only ever relocate the server away from the mapped port.
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
    ENV_ARGS+=(-e CHECKPOINT="$CHECKPOINT")   # HF hub id — needs network (and HF_TOKEN if private)
elif [ -z "${HF_TOKEN:-}" ]; then
    # Neither set means serve.sh's default, which is a private repo. Say so now:
    # the 401 otherwise arrives minutes in, from inside a container that started fine.
    echo "WARNING: no CHECKPOINT_DIR / CHECKPOINT and no HF_TOKEN — the default" >&2
    echo "         A private or gated checkpoint will 401 without HF_TOKEN." >&2
fi
# HF_HOME is /models/hf-cache in the image; mounting it rw is what makes the
# 15.9 GB pull a one-time cost across --rm runs.
if [ -n "${HF_CACHE_DIR:-}" ]; then
    MOUNTS+=(-v "$(readlink -f "$HF_CACHE_DIR"):/models/hf-cache")
fi

# --gpus already selected the host devices and the container renumbers them from 0,
# so serve.sh's GPU knob (which becomes CUDA_VISIBLE_DEVICES) must name
# container-local indices. Left at its default of `0`, a tensor-parallel run would
# see one device and never find its peers.
TP="${TENSOR_PARALLEL_SIZE:-1}"
if [ "$TP" -gt 1 ]; then
    ENV_ARGS+=(-e GPU="$(seq -s, 0 "$((TP - 1))")")
fi

# pass through only when set on the host
for var in SERVED_NAME MAX_MODEL_LEN GPU_MEMORY_UTILIZATION TENSOR_PARALLEL_SIZE HF_TOKEN; do
    if [ -n "${!var:-}" ]; then ENV_ARGS+=(-e "$var=${!var}"); fi
done

exec docker run "${RUN_ARGS[@]}" "${MOUNTS[@]}" "${ENV_ARGS[@]}" "$IMAGE" "$@"
