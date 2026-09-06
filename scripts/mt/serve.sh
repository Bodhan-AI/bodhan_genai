#!/usr/bin/env bash
# Serve IndicTranslate behind vLLM's OpenAI-compatible API.
#
#   scripts/mt/serve.sh                              # defaults, GPU 0, port 8000
#   GPU=3 PORT=8100 scripts/mt/serve.sh
#   CHECKPOINT=/path/to/merged-ckpt scripts/mt/serve.sh
#   MAX_MODEL_LEN=32768 scripts/mt/serve.sh          # full-document workloads
#   scripts/mt/serve.sh --foreground                 # don't background the server
#
# There is no custom server: this is a wrapper around stock `vllm serve`. Talk to
# it with bodhan_genai.mt.serving.MTClient, which owns the prompt contract.
#
# Everything set below is set for a reason:
#   --enforce-eager        skips CUDA-graph capture; the configuration we validated
#   --dtype bfloat16       the checkpoint's native precision, unquantised
#   --trust-remote-code    required for the Gemma 4 processor
#   VLLM_USE_DEEP_GEMM=0   matches the validated serving config
#   free-port search       shared clusters squat ports; binding blindly fails EADDRINUSE
#
# A checkpoint you trained yourself needs the KV-shared k_norm sidecar first:
#   python -m bodhan_genai.mt.tools.vllm_ready <checkpoint>

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${CHECKPOINT:-bodhan-ai/indic-translate}"
SERVED_NAME="${SERVED_NAME:-indic_translate}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_FILE="${LOG_FILE:-vllm-serve.log}"
# Written once the server is up, as shell-sourceable `PID=`/`PORT=` lines. The
# port matters because we may have had to move off a squatted one, so callers
# cannot assume $PORT held.
INFO_FILE="${INFO_FILE:-vllm-serve.info}"

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

Build the environment (one venv covers every modality):

    ./install.sh
    source .venv/bin/activate

or point this script at an interpreter's vllm:

    VLLM_BIN=/path/to/env/bin/vllm scripts/mt/serve.sh
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

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"

echo "===== serving ${MODEL} ====="
echo "  vllm          : ${VLLM_BIN}"
echo "  GPU(s)        : ${CUDA_VISIBLE_DEVICES}  (tensor-parallel ${TENSOR_PARALLEL_SIZE})"
echo "  port          : ${PORT}"
echo "  max-model-len : ${MAX_MODEL_LEN}"
echo "  served name   : ${SERVED_NAME}"
echo ""
echo "  A private or gated repo needs credentials: \`hf auth login\`, or export HF_TOKEN."
echo ""

SERVE_ARGS=(
    serve "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --dtype bfloat16
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --trust-remote-code
    --enforce-eager
    --port "${PORT}"
)
[[ ${#EXTRA[@]} -gt 0 ]] && SERVE_ARGS+=("${EXTRA[@]}")

if [[ "${FOREGROUND}" -eq 1 ]]; then
    exec "${VLLM_BIN}" "${SERVE_ARGS[@]}"
fi

"${VLLM_BIN}" "${SERVE_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
VLLM_PID=$!
echo "vLLM pid ${VLLM_PID}, logs: ${LOG_FILE}"
echo "Waiting for the server to become ready (a first-time pull is ~15.9 GB) ..."

for _ in $(seq 1 180); do
    # Require OUR served-model-name: on a shared box the port could belong to
    # someone else's server, and a bare 200 would be a false positive.
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${SERVED_NAME}\""; then
        printf 'PID=%s\nPORT=%s\nSERVED_NAME=%s\n' \
            "${VLLM_PID}" "${PORT}" "${SERVED_NAME}" > "${INFO_FILE}"
        echo ""
        echo "Ready: http://127.0.0.1:${PORT}/v1   (details in ${INFO_FILE})"
        echo ""
        echo "Try it:"
        echo "  python -m bodhan_genai.mt.serving.client --url http://127.0.0.1:${PORT}/v1 \\"
        echo "      --tgt-lang Hindi --text 'The meeting has been postponed.'"
        echo ""
        echo "  Stop the server with: kill ${VLLM_PID}"
        exit 0
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming ready. Last 40 log lines:" >&2
        tail -40 "${LOG_FILE}" >&2
        exit 1
    fi
    sleep 10
done

echo "ERROR: vLLM did not become ready within 30 minutes. Last 40 log lines:" >&2
tail -40 "${LOG_FILE}" >&2
exit 1
