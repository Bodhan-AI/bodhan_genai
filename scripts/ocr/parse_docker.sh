#!/usr/bin/env bash
# Build (if needed) and run the dockerized IndicOCR over a folder of page images.
#
# Usage:
#   scripts/ocr/parse_docker.sh [--build] IN_DIR OUT_DIR [extra CLI flags]
#
#   scripts/ocr/parse_docker.sh --build pages/ out/
#   scripts/ocr/parse_docker.sh pages/ out/ --save-layout --table-format markdown
#   SUBCOMMAND=layout scripts/ocr/parse_docker.sh pages/ out/     # stage 1 only
#
# Env:
#   SUBCOMMAND       layout | ocr | parse (default parse)
#   LAYOUT_CKPT      host dir of an IndicDocLayout checkpoint (mounted read-only)
#   RECOGNIZER_CKPT  host dir of an IndicBlockOCR checkpoint (mounted read-only)
#                    Either one unset, that stage resolves from the Hub
#   HF_TOKEN         hub credentials. Not needed for the default (bodhan-ai/indic-ocr is
#                    public); required for a gated or private checkpoint
#   BODHAN_OCR_HF_REPO  resolve weights from a different Hub repo
#   HF_CACHE_DIR     host dir for the hub cache; keeps the download across runs
#   FLASHINFER_DIR   host dir for the JIT cache; keeps compiled kernels across runs
#   GPUS             "all" (default) or a device list like "0" / "0,1"
#   SHM_SIZE         /dev/shm size (default 8g)
#   IMAGE            image tag (default bodhan-ocr-parse)
set -euo pipefail

IMAGE="${IMAGE:-bodhan-ocr-parse}"

WANT_BUILD=0
if [ "${1:-}" = "--build" ]; then WANT_BUILD=1; shift; fi

# Resolve the caller's paths before cd-ing to the repo root for the build context,
# so a relative IN_DIR/OUT_DIR means what the caller expects.
IN_DIR="${1:?usage: parse_docker.sh [--build] IN_DIR OUT_DIR [flags]}"; shift
OUT_DIR="${1:?usage: parse_docker.sh [--build] IN_DIR OUT_DIR [flags]}"; shift

[ -d "$IN_DIR" ] || { echo "input directory not found: $IN_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
IN_DIR="$(readlink -f "$IN_DIR")"
OUT_DIR="$(readlink -f "$OUT_DIR")"
for var in LAYOUT_CKPT RECOGNIZER_CKPT HF_CACHE_DIR FLASHINFER_DIR; do
    if [ -n "${!var:-}" ]; then
        [ -d "${!var}" ] || mkdir -p "${!var}"
        printf -v "$var" '%s' "$(readlink -f "${!var}")"
    fi
done

cd "$(dirname "$0")/../.."

if [ "$WANT_BUILD" = 1 ]; then
    docker build -f docker/ocr/Dockerfile.parse -t "$IMAGE" .
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' not found. Re-run with --build (first build takes a while)." >&2
    exit 1
fi

# :ro on the input so a run cannot write back over the pages it was given.
# --user so results are owned by the caller rather than by root.
RUN_ARGS=(--rm --shm-size "${SHM_SIZE:-8g}" --user "$(id -u):$(id -g)")
if [ -t 0 ]; then RUN_ARGS+=(-it); fi

GPUS="${GPUS:-all}"
if [ "$GPUS" = "all" ]; then
    RUN_ARGS+=(--gpus all)
else
    RUN_ARGS+=(--gpus "device=${GPUS}")
fi

MOUNTS=(-v "${IN_DIR}:/in:ro" -v "${OUT_DIR}:/out")
ENV_ARGS=()

# Per-stage checkpoint overrides, as read by engine/checkpoints.py.
if [ -n "${LAYOUT_CKPT:-}" ]; then
    MOUNTS+=(-v "${LAYOUT_CKPT}:/models/layout:ro")
    ENV_ARGS+=(-e BODHAN_OCR_LAYOUT_CKPT=/models/layout)
fi
if [ -n "${RECOGNIZER_CKPT:-}" ]; then
    MOUNTS+=(-v "${RECOGNIZER_CKPT}:/models/recognizer:ro")
    ENV_ARGS+=(-e BODHAN_OCR_RECOGNIZER_CKPT=/models/recognizer)
fi
# Warn per stage, not for both together: with only one checkpoint set, the other still
# resolves from the Hub. Skipped when a different repo is named, which may be public, and
# when offline, where a warm cache is used and no credentials are involved.
if [ -z "${BODHAN_OCR_HF_REPO:-}" ] && [ -z "${HF_TOKEN:-}" ] && [ -z "${HF_HUB_OFFLINE:-}" ] \
   && { [ -z "${LAYOUT_CKPT:-}" ] || [ -z "${RECOGNIZER_CKPT:-}" ]; }; then
    echo "WARNING: no HF_TOKEN, and at least one checkpoint is unset. Those stages resolve" >&2
    echo "         from bodhan-ai/indic-ocr; a gated or private repo needs HF_TOKEN." >&2
    echo "         Set LAYOUT_CKPT and RECOGNIZER_CKPT, or HF_TOKEN, or BODHAN_OCR_HF_REPO." >&2
fi

# Mount these read-write so downloaded weights and compiled kernels survive --rm.
if [ -n "${HF_CACHE_DIR:-}" ]; then
    MOUNTS+=(-v "${HF_CACHE_DIR}:/models/hf-cache")
fi
if [ -n "${FLASHINFER_DIR:-}" ]; then
    MOUNTS+=(-v "${FLASHINFER_DIR}:/models/flashinfer")
fi

for var in HF_TOKEN HF_HUB_OFFLINE BODHAN_OCR_HF_REPO; do
    if [ -n "${!var:-}" ]; then ENV_ARGS+=(-e "$var=${!var}"); fi
done

exec docker run "${RUN_ARGS[@]}" "${MOUNTS[@]}" "${ENV_ARGS[@]}" "$IMAGE" \
    "${SUBCOMMAND:-parse}" /in -o /out "$@"
