"""Import hygiene: `import bodhan_genai.asr` must stay cheap.

Same contract (and same subprocess technique) as test_tts_lazy_import.py:
a fresh interpreter so this process's already-loaded modules cannot mask a
regression. Callers that only want a version string or a submodule path
should not pay a multi-second torch/transformers import.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY = ("torch", "transformers", "torchaudio", "sentencepiece", "soundfile")


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_importing_asr_package_does_not_pull_heavy_deps():
    proc = _run(
        "import sys\n"
        "import bodhan_genai.asr\n"
        f"for heavy in {HEAVY!r}:\n"
        "    assert heavy not in sys.modules, f'{heavy} imported at package import time'\n"
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_lazy_names_resolve_and_dir_lists_them():
    proc = _run(
        "import bodhan_genai.asr as a\n"
        "names = set(a.__all__)\n"
        "assert 'IndicASREngine' in names, names\n"
        "assert 'IndicTranscribeConfig' in names, names\n"
        "assert names <= set(dir(a))\n"
        "cfg = a.IndicTranscribeConfig()\n"  # forces the lazy import
        "assert cfg.model_type == 'indic_transcribe'\n"
        "assert cfg.vocab_size == 7152 and cfg.d_model == 1024\n"
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_unknown_attribute_raises_attribute_error():
    proc = _run(
        "import bodhan_genai.asr as a\n"
        "try:\n"
        "    a.NoSuchThing\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('expected AttributeError')\n"
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_inference_cli_help_runs_without_a_checkpoint():
    """--help must not require torch, a GPU, or a model directory."""
    proc = _run(
        "import sys\n"
        "sys.argv = ['transcribe', '--help']\n"
        "from bodhan_genai.asr.inference.transcribe import build_parser\n"
        "text = build_parser().format_help()\n"
        "assert '--manifest' in text and '--num-shards' in text and '--chunk-above' in text\n"
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
