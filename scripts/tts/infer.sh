#!/usr/bin/env bash
# Offline batch TTS inference launcher.
#
# Usage:
#   scripts/tts/infer.sh --jsonl-path manifests/eval.jsonl --output_dir out/eval_run [more flags]
#
# Defaults come from configs/tts/infer/offline_vllm.yaml; any flag given here
# overrides the config (e.g. --checkpoint_path <your SFT checkpoint>).
set -euo pipefail

cd "$(dirname "$0")/../.."

# Local single-node Ray unless the caller points at an existing cluster.
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
# Keep each vLLM engine in-process inside its Ray actor (see offline_vllm.py).
export VLLM_ENABLE_V1_MULTIPROCESSING=0

exec python -m bodhan_genai.tts.inference.offline_vllm \
    --config configs/tts/infer/offline_vllm.yaml \
    "$@"
