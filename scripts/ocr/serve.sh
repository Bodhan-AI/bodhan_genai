#!/usr/bin/env bash
# Serve IndicBlockOCR on stock `vllm serve`.
#
#   scripts/ocr/serve.sh                    # background, waits for readiness
#   scripts/ocr/serve.sh --foreground       # don't background the server
#
# This serves the RECOGNIZER ONLY. Layout runs client-side, so talk to it through
# `python -m bodhan_genai.ocr.serving.client`, not with a bare OpenAI client: the
# endpoint expects a single block crop per request, and sending a whole page
# returns plausible nonsense.
#
# There is no custom server here; this is a wrapper. Flags worth knowing:
#   --limit-mm-per-prompt       one image per request, as the pipeline sends them
#   --max-model-len 8192        matches RecognizerConfig
#   --mm-processor-cache-type   shared-memory cache for preprocessed images, per the
#                               vLLM Qwen3.5 recipe; the crops are all distinct so this
#                               is about transfer cost, not hit rate
#   --no-enable-prefix-caching  document parsing has no multi-turn prefix to reuse, so
#                               the hashing is pure overhead. vLLM's own PaddleOCR-VL
#                               recipe says to turn it off for OCR; the Qwen3.5 recipe
#                               enables it, but that one is written for chat.
#
# ENFORCE_EAGER=1 matches the validated offline configuration and skips ~4 minutes of
# torch.compile. That is the right trade for a batch job and the wrong one for a server,
# which pays it once; the default here is therefore 0.
#
# DATA_PARALLEL_SIZE replicates the model across GPUs. IndicBlockOCR is 0.8B and fits on
# one, so data parallel is the way to use several -- not tensor parallel. Worth setting
# for a benchmark sweep.
#
# Deliberately NOT set, from that same recipe: --enable-expert-parallel (this checkpoint
# is dense, no experts), --speculative-config mtp (no num_nextn_predict_layers),
# --reasoning-parser qwen3 (the chat template already closes <think> in the generation
# prompt, and a parser would move the transcription into reasoning_content).
set -euo pipefail
cd "$(dirname "$0")/../.."

# Optional bearer-token auth, off by default. Stock `vllm serve` enforces a token itself, so
# there is nothing to write or maintain here: set OCR_API_KEY=$(openssl rand -hex 32) and clients
# send it the way every OpenAI SDK already does (api_key=...), which MTClient/OCRClient accept.
#
# Unset means OPEN, and this binds 0.0.0.0. On a shared network put a TLS reverse proxy in
# front, or bind to localhost and tunnel.

# CHECKPOINT from the environment is honoured ONLY inside a deployment image, where it is how
# mounted weights under /models are addressed. Outside one an inherited CHECKPOINT silently
# serving different weights is a correctness bug that presents as a model regression, so it is
# ignored; pass --model to vllm serve, or set CHECKPOINT for one command inside the image.
if [ "${BODHAN_GENAI_DEPLOYMENT:-}" != "1" ]; then
    unset CHECKPOINT
fi

# The weights sit in a weights/ocr subfolder, which `vllm serve` cannot address, so resolve to a
# local path with the package's own resolver. Outside a deployment image that honours an explicit
# CHECKPOINT then the published default; inside one it also consults
# BODHAN_OCR_RECOGNIZER_CKPT and a bundled weights/.
if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(python -c \
        'from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt; print(resolve_ckpt("recognizer"))' \
        2>/dev/null || true)"
fi
if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: could not resolve the recognizer checkpoint." >&2
    echo "Set CHECKPOINT to a local path, or BODHAN_OCR_RECOGNIZER_CKPT, or export HF_TOKEN" >&2
    echo "so it can be pulled from the Hub." >&2
    exit 1
fi

MODEL="${CHECKPOINT}"
SERVED_NAME="${SERVED_NAME:-indic_ocr}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LOG_FILE="${LOG_FILE:-vllm-serve.log}"
# Written once the server is up, as shell-sourceable `PID=`/`PORT=` lines. The port
# matters because we may have moved off a squatted one.
# Per-modality, not a shared "vllm-serve.info": running two servers on one box
# otherwise has the second clobber the first's PID and port.
INFO_FILE="${INFO_FILE:-ocr-serve.info}"

FOREGROUND=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) FOREGROUND=1; shift ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

VLLM_BIN="${VLLM_BIN:-$(command -v vllm 2>/dev/null || true)}"
if [[ -z "${VLLM_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    cat >&2 <<'EOF'
ERROR: no vllm binary found on $PATH.

Build the environment:

    ./install.sh
    source .venv/bin/activate

or point this script at an interpreter's vllm:

    VLLM_BIN=/path/to/env/bin/vllm scripts/ocr/serve.sh
EOF
    exit 1
fi

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

# Same runtime settings the in-process recognizer applies: ninja lives in the venv's
# bin/, and these two kernels are off in the configuration we validated.
export PATH="$(dirname "$(command -v python)"):${PATH}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"

echo "===== serving ${SERVED_NAME} (recognizer only) ====="
echo "  vllm          : ${VLLM_BIN}"
echo "  checkpoint    : ${MODEL}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES}  (tp ${TENSOR_PARALLEL_SIZE}, dp ${DATA_PARALLEL_SIZE})"
echo "  port          : ${PORT}"
echo "  max-model-len : ${MAX_MODEL_LEN}"
echo ""
echo "  Layout runs client-side: python -m bodhan_genai.ocr.serving.client pages/ -o out/"
echo ""

SERVE_ARGS=(
    serve "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --data-parallel-size "${DATA_PARALLEL_SIZE}"
    --limit-mm-per-prompt '{"image":1}'
    --mm-processor-cache-type shm
    --no-enable-prefix-caching
    --trust-remote-code
    --port "${PORT}"
)
[[ "${ENFORCE_EAGER}" == "1" ]] && SERVE_ARGS+=(--enforce-eager)
# Handed over in the ENVIRONMENT, not as --api-key on the command line: a process's
# argv is world-readable (`ps -eo args` from any account on the node), so the flag
# would publish the token to every user on a shared box. /proc/<pid>/environ is
# 0400 owner-only. vLLM reads VLLM_API_KEY and enforces it identically; --api-key
# merely takes precedence when both are set.
[[ -n "${OCR_API_KEY:-}" ]] && export VLLM_API_KEY="${OCR_API_KEY}"
[[ ${#EXTRA[@]} -gt 0 ]] && SERVE_ARGS+=("${EXTRA[@]}")

if [[ "${FOREGROUND}" -eq 1 ]]; then
    exec "${VLLM_BIN}" "${SERVE_ARGS[@]}"
fi

"${VLLM_BIN}" "${SERVE_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
VLLM_PID=$!
echo "vLLM pid ${VLLM_PID}, logs: ${LOG_FILE}"
echo "Waiting for the server to become ready (a first-time pull is ~1.7 GB) ..."

for _ in $(seq 1 180); do
    # Require OUR served-model-name: on a shared box the port could belong to
    # someone else's server, and a bare 200 would be a false positive.
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${SERVED_NAME}\""; then
        printf 'PID=%s\nPORT=%s\nSERVED_NAME=%s\n' \
            "${VLLM_PID}" "${PORT}" "${SERVED_NAME}" > "${INFO_FILE}"
        echo ""
        echo "Ready on http://127.0.0.1:${PORT}/v1  (wrote ${INFO_FILE})"
        echo "Stop with: kill ${VLLM_PID}"
        exit 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup. Last lines of ${LOG_FILE}:" >&2
        tail -20 "${LOG_FILE}" >&2
        exit 1
    fi
    sleep 5
done

echo "ERROR: server did not become ready in 15 minutes. See ${LOG_FILE}." >&2
exit 1
