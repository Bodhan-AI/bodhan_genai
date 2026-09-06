"""Import hygiene for the MT public API.

``bodhan_genai.mt`` and its light modules must never drag torch / vllm /
transformers / trl into ``sys.modules`` at import time (PEP 562 lazy exports +
heavy-imports-inside-functions). Each check runs in a fresh subprocess so this
test's own interpreter state cannot mask a regression.

This matters beyond startup time: the prompt contract, the render stage and the
config loaders are all usable — and tested — in an environment with none of the GPU
stack installed, which is what keeps CI CPU-only.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY = ("torch", "vllm", "trl", "peft", "transformers", "datasets", "openai", "sacrebleu")


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


_ASSERT_CLEAN = "".join(f"assert {mod!r} not in sys.modules, '{mod} imported'\n" for mod in HEAVY)


def test_package_import_and_prompt_contract_leave_heavy_deps_out():
    _run(
        "import sys\n"
        "import bodhan_genai.mt\n"
        "from bodhan_genai.mt import build_conversation, resolve_language, LANGUAGE_NAMES\n"
        "assert resolve_language('hin_Deva') == 'Hindi'\n"
        "assert len(LANGUAGE_NAMES) == 25\n"
        "assert build_conversation('hi', 'Hindi')[0]['role'] == 'user'\n" + _ASSERT_CLEAN
    )


def test_lazy_types_resolve_without_heavy_deps():
    _run(
        "import sys\n"
        "from bodhan_genai.mt import MTSamplingConfig, MTResult\n"
        "assert MTSamplingConfig().temperature == 0.0\n"
        "assert MTSamplingConfig().greedy\n"
        "assert MTResult().ok\n" + _ASSERT_CLEAN
    )


def test_lazy_engine_class_resolves_without_heavy_deps():
    _run(
        "import sys\n"
        "from bodhan_genai.mt import IndicMTEngine\n"
        "assert isinstance(IndicMTEngine, type)\n" + _ASSERT_CLEAN
    )


def test_engine_and_template_modules_import_clean():
    _run(
        "import sys\n"
        "import bodhan_genai.mt.engine\n"
        "import bodhan_genai.mt.engine.offline\n"
        "import bodhan_genai.mt.templates\n"
        "from bodhan_genai.mt.engine import TranslationBackend\n"
        "assert TranslationBackend is not None\n" + _ASSERT_CLEAN
    )


def test_cli_dispatcher_and_render_import_clean():
    """The `--help` path must not pay for the GPU stack."""
    _run(
        "import sys\n"
        "import bodhan_genai.mt.inference.cli\n"
        "import bodhan_genai.mt.inference.common\n"
        "import bodhan_genai.mt.data.render\n"
        "import bodhan_genai.mt.data.dataset\n"
        "import bodhan_genai.mt.training.config\n"
        "import bodhan_genai.mt.tools.vllm_ready\n" + _ASSERT_CLEAN
    )


def test_dir_lists_lazy_names_and_bogus_attribute_raises():
    _run(
        "import bodhan_genai.mt as m\n"
        "names = dir(m)\n"
        "for n in ('IndicMTEngine', 'MTSamplingConfig', 'MTResult', 'LANGUAGE_NAMES',\n"
        "          'STOP_STRINGS', 'build_conversation', 'build_instruction',\n"
        "          'resolve_language'):\n"
        "    assert n in names, n\n"
        "try:\n"
        "    m.NoSuchThing\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected AttributeError for bogus attribute')\n"
    )


def test_importing_mt_does_not_import_tts():
    """The two modalities are independent; neither should pull the other in."""
    _run(
        "import sys\n"
        "import bodhan_genai.mt\n"
        "assert 'bodhan_genai.tts' not in sys.modules, 'mt imported tts'\n"
    )
