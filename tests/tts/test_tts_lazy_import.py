"""Import hygiene for the public API: bodhan_genai.tts and the engine modules
must never drag torch / vllm into sys.modules at import time (PEP 562 lazy
exports + heavy-imports-inside-methods). Each check runs in a fresh subprocess
so this test's own interpreter state cannot mask a regression."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(code: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_package_import_and_lazy_types_leave_heavy_deps_out():
    _run(
        "import sys\n"
        "import bodhan_genai.tts\n"
        "from bodhan_genai.tts import SamplingConfig\n"
        "assert SamplingConfig().temperature == 0.6\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'vllm' not in sys.modules, 'vllm imported'\n"
    )


def test_engine_modules_import_without_heavy_deps():
    _run(
        "import sys\n"
        "import bodhan_genai.tts.engine.offline\n"
        "import bodhan_genai.tts.engine.streaming\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'vllm' not in sys.modules, 'vllm imported'\n"
    )


def test_lazy_engine_class_resolves_without_heavy_deps():
    _run(
        "import sys\n"
        "from bodhan_genai.tts import IndicTTSEngine, IndicStreamingTTSEngine\n"
        "from bodhan_genai.tts import split_sentences, plan_dialogue_chunks\n"
        "assert isinstance(IndicTTSEngine, type)\n"
        "assert isinstance(IndicStreamingTTSEngine, type)\n"
        "assert callable(split_sentences) and callable(plan_dialogue_chunks)\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'vllm' not in sys.modules, 'vllm imported'\n"
    )


def test_dir_lists_lazy_names_and_bogus_attribute_raises():
    _run(
        "import bodhan_genai.tts as t\n"
        "names = dir(t)\n"
        "for n in ('IndicTTSEngine', 'IndicStreamingTTSEngine', 'SamplingConfig', 'TTSResult',\n"
        "          'ChunkedIndicStreamingTTS', 'chunk_text', 'split_sentences',\n"
        "          'plan_dialogue_chunks'):\n"
        "    assert n in names, n\n"
        "try:\n"
        "    t.NoSuchThing\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected AttributeError for bogus attribute')\n"
    )
