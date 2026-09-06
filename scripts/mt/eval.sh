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
# serve.sh writes the port it actually bound to; prefer it over the default.
if [[ -z "${URL_OVERRIDDEN:-}" && -f vllm-serve.info ]]; then
    # shellcheck disable=SC1091
    source vllm-serve.info
    URL="http://127.0.0.1:${PORT}/v1"
fi

echo "server: $URL"
echo "output: $OUTPUT_DIR"

exec python -m bodhan_genai.mt.eval.in22 \
    --output-dir "$OUTPUT_DIR" \
    --url "$URL" \
    "$@"
