#!/usr/bin/env bash
# IN22 score replication against a running IndicTranslate server.
#
# Usage:
#   scripts/mt/eval.sh                                              # full run, 45,056 requests
#   scripts/mt/eval.sh --langs hin_Deva --directions en-xx --max-samples 32   # smoke
#
# Start a server first: scripts/mt/serve.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="${OUTPUT_DIR:-in22-results}"
URL="${URL:-http://localhost:8000/v1}"
# serve.sh writes the port it actually bound to; prefer it over the default. Same
# INFO_FILE default as serve.sh -- the two must agree, or a server that moved off a busy
# port is silently missed and the eval runs against whatever answers on :8000.
INFO_FILE="${INFO_FILE:-mt-serve.info}"
if [[ -z "${URL_OVERRIDDEN:-}" && -f "${INFO_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${INFO_FILE}"
    URL="http://127.0.0.1:${PORT}/v1"
fi

echo "server: $URL"
echo "output: $OUTPUT_DIR"

exec python -m bodhan_genai.mt.eval.in22 \
    --output-dir "$OUTPUT_DIR" \
    --url "$URL" \
    "$@"
