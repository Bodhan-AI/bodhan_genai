#!/usr/bin/env bash
# bodhan-genai installer — conda-free, uv-first (plain venv+pip fallback).
#
# ONE ENVIRONMENT FOR EVERY MODALITY:
#
#   ./install.sh                        # TTS + MT + OCR -> ./.venv
#   ./install.sh --extras all-mt        # MT only (lean serving image, etc.)
#   ./install.sh --extras all-tts       # TTS only
#   ./install.sh --extras all-ocr       # OCR only
#
# All three modalities share one environment. Their earlier separate environments
# were based on old version pins; the last real conflict was tts-serve's exact
# fastapi==0.141.1 against vLLM 0.26's fastapi[standard]>=0.133,<0.137. Relaxing
# that to a range resolves all three onto vllm 0.26 / torch 2.11.0 / transformers
# 5.13.1.
#
# REQUIRED install order — deviating from it is the number-one cause of a broken
# environment:
#   1. vllm FIRST, pinned, from its per-CUDA wheel index, letting it pull its own
#      matched torch. Pre-pinning torch leaves a half-installed torch behind on
#      0.26. The per-CUDA index matters because PyPI's default is a CUDA 13 build,
#      which imports fine and then reports no GPU on this cluster's 12.2 driver.
#   2. torchaudio/torchvision from the per-CUDA index, matched to that torch (TTS
#      needs them; OCR and MT do not)
#   3. the package + extras under constraints.txt
#   4. flash-attn LAST and unpinned — see the note above the install step
#
# Usage:
#   ./install.sh [--extras NAME] [--no-venv] [--cpu] [--offline DIR] [--no-flash-attn]
#   ./install.sh --no-venv              # install into the CURRENT environment instead
#   ./install.sh --cpu                  # laptops/CI: skip GPU wheels, install .[dev] only
#   ./install.sh --offline DIR          # air-gapped: install from a pre-downloaded wheel dir
#   ./install.sh --no-flash-attn        # skip the flash-attn build (TTS *training* only)
#
# Env overrides: CUDA_TAG (default cu129) if your driver needs another CUDA line.
#
# uv is a single static binary; if missing, install it without root:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="3.12.13"   # keep in sync with .python-version
VENV_DIR=".venv"
CONSTRAINTS="constraints.txt"

# cu129 pairs with this cluster's CUDA 12.2 driver (forward-compatible). Override
# CUDA_TAG if yours needs a different line, e.g. CUDA_TAG=cu126.
CUDA_TAG="${CUDA_TAG:-cu129}"
TORCH_VERSION="2.11.0"
# Pinned exactly rather than given as a floor because PyPI and the per-CUDA wheel
# publish different dependencies for the same version, which a range cannot resolve.
# vLLM pins torch, so 0.26.0 brings torch 2.11.0 — the same torch the previous
# 0.20.2 environment used.
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
CU_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
VLLM_INDEX="https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TAG}"
# flashinfer-cubin is in vLLM 0.26's own requirements/cuda.txt but is NOT published to PyPI, so
# a plain `pip install vllm` silently omits it. Without it flashinfer JIT-compiles kernels at
# engine warm-up, which needs nvcc; with it, ~16k kernels arrive prebuilt (attention, gemm,
# fmha, deep-gemm). It does NOT cover the *sampling* module -- that still JIT-builds, which is
# why every engine here also sets VLLM_USE_FLASHINFER_SAMPLER=0.
FLASHINFER_INDEX="https://flashinfer.ai/whl/"

EXTRAS="all"
MODE="gpu"
OFFLINE_DIR=""
USE_VENV=1
WANT_FLASH_ATTN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --extras) EXTRAS="${2:?--extras requires a name, e.g. all / all-mt / all-tts}"; shift ;;
    --cpu) MODE="cpu" ;;
    --offline) MODE="offline"; OFFLINE_DIR="${2:?--offline requires a wheel directory}"; shift ;;
    --no-venv) USE_VENV=0 ;;
    --no-flash-attn) WANT_FLASH_ATTN=0 ;;
    # Obsolete: every modality shares one venv now. Accepted and ignored so existing
    # scripts do not hard-fail on an argument that used to exist.
    --modality)
      case "${2:?--modality requires a name, e.g. tts / mt / ocr}" in
        tts|mt|ocr)
          echo "NOTE: --modality $2 is obsolete — all modalities share one environment." >&2
          echo "      Use --extras all-$2 for a lean install." >&2 ;;
        *) echo "ERROR: --modality must be 'tts', 'mt' or 'ocr', got '$2'" >&2; exit 2 ;;
      esac
      shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# flash-attn is a TTS *training* dependency and nothing else. A lean MT- or OCR-only
# install has no use for a tens-of-minutes CUDA source build — `--modality ocr` used to
# skip it outright, and collapsing the environments must not silently reintroduce it.
case "$EXTRAS" in
  all|*tts*) ;;
  *) WANT_FLASH_ATTN=0 ;;
esac

# --- environment -----------------------------------------------------------
if [ "$USE_VENV" = 1 ]; then
  if command -v uv >/dev/null 2>&1; then
    # uv downloads a standalone CPython if 3.12.13 isn't on the machine.
    # --allow-existing so re-running after a failed dependency step reuses the
    # venv instead of aborting; the pip steps below are idempotent. Delete the
    # directory by hand for a clean build.
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION" --allow-existing
  elif command -v "python$( echo "$PYTHON_VERSION" | cut -d. -f1-2 )" >/dev/null 2>&1; then
    "python$( echo "$PYTHON_VERSION" | cut -d. -f1-2 )" -m venv "$VENV_DIR"
  else
    echo "ERROR: neither uv nor python3.12 found." >&2
    echo "Install uv (no root needed): curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

PY_VERSION="$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
case "$PY_VERSION" in
  3.12.*) ;;
  *) echo "ERROR: python 3.12.x required (blessed runtime is $PYTHON_VERSION), found $PY_VERSION" >&2; exit 1 ;;
esac

# uv pip is a drop-in, much faster resolver; fall back to pip transparently.
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
  # uv defaults to --index-strategy first-index (a dependency-confusion guard):
  # it only considers the FIRST index that carries a package. That breaks this
  # stack, which legitimately spans three indexes — vLLM's per-CUDA index, the
  # PyTorch CUDA index and PyPI. Two concrete failures without this flag:
  #   * `packaging` resolves to the old copy on the PyTorch index, so
  #     flashinfer-python (needs >=24.2) becomes unsatisfiable;
  #   * `vllm==0.26.0` refuses to match the published `0.26.0+cu129` local version.
  # pip does best-match across indexes by default, which is why the reference
  # environment (plain pip) never hit this.
  UV_INDEX_FLAGS=(--index-strategy unsafe-best-match)
else
  PIP="python -m pip"
  UV_INDEX_FLAGS=()
fi

# --- install ---------------------------------------------------------------
case "$MODE" in
  gpu)
    # vLLM FIRST, letting it pull its own matched torch. Pre-pinning torch leaves a
    # half-installed torch behind on 0.26.
    $PIP install "vllm==${VLLM_VERSION}" \
      --extra-index-url "$VLLM_INDEX" \
      --extra-index-url "$CU_INDEX" \
      "${UV_INDEX_FLAGS[@]}"
    # TTS needs torchaudio (audio I/O) and torchvision. torch is deliberately NOT
    # named here: `torch==2.11.0` does not match the published `2.11.0+cu129` local
    # version under uv, so naming it would fight the build vLLM just installed.
    # --index-url (not --extra-index-url) restricts this to the per-CUDA index, whose
    # torchaudio/torchvision are built against exactly that torch.
    $PIP install torchaudio torchvision --index-url "$CU_INDEX"
    # Prebuilt flashinfer kernels, version-matched to the flashinfer-python vLLM already pulled.
    # Non-fatal: a miss costs JIT compiles at engine warm-up, not correctness.
    FI_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))' 2>/dev/null || true)"
    if [ -n "${FI_VERSION}" ]; then
      $PIP install "flashinfer-cubin==${FI_VERSION}" --extra-index-url "$FLASHINFER_INDEX" \
        "${UV_INDEX_FLAGS[@]}" \
        || echo "WARNING: flashinfer-cubin ${FI_VERSION} unavailable; kernels will JIT-compile (needs nvcc)." >&2
    fi
    # Keep vLLM's transformers pin if it already clears the PPDocLayoutV3 floor OCR needs.
    python -c 'import transformers,sys; v=tuple(int(x) for x in transformers.__version__.split(".")[:2]); sys.exit(0 if v>=(5,7) else 1)' 2>/dev/null \
      || $PIP install "transformers>=5.7" "${UV_INDEX_FLAGS[@]}"
    # The vLLM index is deliberately not repeated here: vllm is already installed and
    # satisfies the extras' floor, and re-exposing the index reopens the version conflict.
    $PIP install -e ".[${EXTRAS},dev]" --constraint "$CONSTRAINTS" \
      --extra-index-url "$CU_INDEX" "${UV_INDEX_FLAGS[@]}"
    # The install order above is load-bearing; a torch that moved underneath vLLM
    # imports fine and then fails at engine start, so fail loudly here instead.
    python -c "
import torch, sys
if not torch.__version__.startswith('${TORCH_VERSION}'):
    sys.exit('ERROR: torch is %s, expected ${TORCH_VERSION} — the resolver moved it.' % torch.__version__)
" || exit 1
    ;;
  cpu)
    # Core + dev only: enough for the CPU test suite and the prompt/template modules.
    $PIP install -e ".[dev]" --constraint "$CONSTRAINTS"
    ;;
  offline)
    # torchaudio/torchvision are TTS-only; demanding them would fail against a wheel
    # directory prepared for a lean MT- or OCR-only install.
    OFFLINE_TORCH=(torch vllm)
    case "$EXTRAS" in
      all|*tts*) OFFLINE_TORCH=(torch torchaudio torchvision vllm) ;;
    esac
    $PIP install --no-index --find-links "$OFFLINE_DIR" "${OFFLINE_TORCH[@]}"
    $PIP install --no-index --find-links "$OFFLINE_DIR" -e ".[${EXTRAS},dev]" --constraint "$CONSTRAINTS"
    ;;
esac

# --- flash-attn (TTS training only) -----------------------------------------
# Deliberately NOT pinned to a prebuilt wheel URL. A pinned wheel is exact to one
# (flash-attn, torch, CUDA, cpython, arch) tuple and silently goes stale the moment
# any of the four moves — which is what happened when the stack went to torch 2.11.
# PyPI ships flash-attn as an sdist, so this compiles: slow (tens of minutes, wants
# ninja and a CUDA toolkit) but always matched to the torch that is actually
# installed.
#
# Only TTS *training* needs it (`attn_implementation="flash_attention_2"`). Data,
# inference, serving and all of MT run without it — vLLM ships its own attention
# kernels and MT runs on sdpa — so a failed build is a warning, not an error.
if [ "$MODE" = "gpu" ] && [ "$WANT_FLASH_ATTN" = 1 ]; then
  echo ""
  echo "Building flash-attn from source (TTS training only; --no-flash-attn skips it)."
  echo "This compiles CUDA kernels and takes a while."
  if ! $PIP install flash-attn --no-build-isolation; then
    echo "" >&2
    echo "WARNING: flash-attn failed to build. Everything except TTS *training* works." >&2
    echo "         TTS training asserts attn_implementation='flash_attention_2' and will" >&2
    echo "         fail until this is installed. Re-run with a CUDA toolkit and ninja" >&2
    echo "         available, or train on another machine." >&2
  fi
fi

# --- smoke check -----------------------------------------------------------
if [ "$MODE" = "cpu" ]; then
  # No GPU extras installed, so only the light entry points can be imported.
  python -c "import bodhan_genai.tts, bodhan_genai.mt, bodhan_genai.ocr; print('core imports OK')" >/dev/null
else
  # One environment, so one check — but scoped to what was actually installed, since
  # --extras can still select a lean single-modality install.
  BODHAN_EXTRAS="$EXTRAS" python - <<'EOF'
import os
import shutil
import sys

import torch
import transformers


def want(modality: str) -> bool:
    extras = os.environ.get("BODHAN_EXTRAS", "all")
    return extras == "all" or modality in extras


problems = []
print(f"  torch        {torch.__version__} (built for CUDA {torch.version.cuda})")
print(f"  transformers {transformers.__version__}")
try:
    import vllm
    print(f"  vllm         {vllm.__version__}")
except ImportError:
    print("  vllm         not installed (fine for a data-only install)")

if want("mt"):
    if tuple(int(p) for p in transformers.__version__.split(".")[:2]) < (5, 12):
        problems.append(f"transformers {transformers.__version__} < 5.12 (Gemma 4 floor)")
    from transformers.models.auto import modeling_auto as ma
    if ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("gemma4") != "Gemma4ForConditionalGeneration":
        problems.append("this transformers build does not register the Gemma 4 architecture")
    else:
        print("  mt arch      Gemma 4 registered")

if want("ocr"):
    # IndicDocLayout is a PPDocLayoutV3 subclass; without this class the layout stage
    # cannot load.
    try:
        from transformers import PPDocLayoutV3ForObjectDetection  # noqa: F401
        print("  layout arch  PPDocLayoutV3 present")
    except ImportError:
        problems.append(
            f"transformers {transformers.__version__} does not ship PPDocLayoutV3 (need >= 5.7)"
        )
    # The recognizer JIT-compiles a kernel on first inference. ninja lives in the venv's
    # bin/, which the package puts on PATH at runtime; mirror that here.
    os.environ["PATH"] = os.path.join(sys.prefix, "bin") + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        problems.append("`ninja` not on PATH (the recognizer's GDN kernel JIT needs it)")
    else:
        print("  ninja        on PATH")

# The check that matters: a wheel/driver CUDA mismatch does NOT raise on import, it
# just leaves you with no GPU. Catch it here instead of 15.9 GB into a model load.
if torch.cuda.is_available():
    print(f"  GPUs visible {torch.cuda.device_count()}")
else:
    problems.append(
        "torch.cuda.is_available() is False. If this machine has GPUs, the installed "
        "wheel is built for a newer CUDA than the driver provides — rerun with CUDA_TAG "
        "matching your driver, e.g. CUDA_TAG=cu126. (On a login node with no GPU "
        "attached this is expected: rerun the check on a GPU node.)"
    )

if problems:
    print("\nFAILED:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\n  all checks passed")
EOF
fi

echo "bodhan-genai installed (environment: $VENV_DIR, extras: $EXTRAS, mode: $MODE)."
[ "$USE_VENV" = 1 ] && echo "Activate with: source $VENV_DIR/bin/activate"