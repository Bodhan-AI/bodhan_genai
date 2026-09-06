"""Import hygiene: no torch / vllm / transformers / PIL at import time.

This is what keeps the contract, the cleanup geometry and ``--help`` usable with none of the GPU
stack installed, and CI CPU-only. Each check runs in a fresh subprocess so this interpreter's
state cannot mask a regression.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("torch", "vllm", "transformers", "PIL", "numpy", "huggingface_hub")

_ASSERT_CLEAN = "".join(f"assert {m!r} not in sys.modules, '{m} imported'\n" for m in HEAVY)


def _run(code: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), HF_HUB_OFFLINE="1")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"


def test_importing_every_module_pulls_in_nothing_heavy():
    """Every module, the public API and the config types -- one process, one assertion set."""
    _run(
        "import sys\n"
        "import bodhan_genai.ocr.engine.blocks, bodhan_genai.ocr.engine.reconstruct\n"
        "import bodhan_genai.ocr.engine.layout, bodhan_genai.ocr.engine.offline\n"
        "import bodhan_genai.ocr.engine.recognizer, bodhan_genai.ocr.engine.recognizer_vllm\n"
        "import bodhan_genai.ocr.engine.checkpoints\n"
        "import bodhan_genai.ocr.inference.cli, bodhan_genai.ocr.layout.labels\n"
        "from bodhan_genai.ocr import IndicOCR, IndicDocLayout, IndicBlockOCR\n"
        "from bodhan_genai.ocr import DedupConfig, CropConfig, RecognizerConfig, TableFormat\n"
        "from bodhan_genai.ocr import map_label, prompt_for\n"
        "assert map_label('Sub-section-title') == 'SectionHeader'\n"
        "assert DedupConfig().mode == 'both' and CropConfig().min_px == 256 ** 2\n"
        "assert RecognizerConfig().table_format is TableFormat.HTML\n"
        "assert all(isinstance(c, type) for c in (IndicOCR, IndicDocLayout, IndicBlockOCR))\n"
        + _ASSERT_CLEAN
    )


def test_removed_aliases_no_longer_resolve():
    """The pre-rename aliases are gone; every modality now exposes one name per class.

    Asserted rather than merely deleted: a lazy __getattr__ that silently resolved
    a removed name would keep old code working and mask the break.
    """
    _run(
        "import bodhan_genai.ocr as m\n"
        "for gone in ('BodhanOCREngine', 'LayoutParser', 'BlockOCR'):\n"
        "    assert gone not in m.__all__, gone\n"
        "    assert gone not in dir(m), gone\n"
        "    try:\n"
        "        getattr(m, gone)\n"
        "    except AttributeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError(f'{gone} still resolves')\n"
    )


def test_canonical_names_resolve_and_a_bogus_attribute_raises():
    _run(
        "import bodhan_genai.ocr as m\n"
        "assert m.IndicOCR.__name__ == 'IndicOCR'\n"
        "assert m.IndicDocLayout.__name__ == 'IndicDocLayout'\n"
        "assert m.IndicBlockOCR.__name__ == 'IndicBlockOCR'\n"
        "assert 'TableFormat' in dir(m)\n"
        "try:\n"
        "    m.NoSuchThing\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected AttributeError')\n"
    )


def test_ocr_does_not_import_its_sibling_modalities():
    _run(
        "import sys, bodhan_genai.ocr\n"
        "assert 'bodhan_genai.tts' not in sys.modules, 'ocr imported tts'\n"
        "assert 'bodhan_genai.mt' not in sys.modules, 'ocr imported mt'\n"
    )


# --------------------------------------------------------------------------- #
# Runtime preparation. Both of these fail deep inside vLLM engine startup with an error that
# does not name its cause, so they are set before vLLM is imported -- and tested here.
# --------------------------------------------------------------------------- #


def test_flashinfer_cache_probe_checks_subdirectories_not_just_the_root(tmp_path):
    """The failure mode seen in practice: a writable root with a root-owned version dir under it."""
    from bodhan_genai.ocr.engine.recognizer_vllm import _flashinfer_cache_is_usable

    assert _flashinfer_cache_is_usable(tmp_path / "does-not-exist-yet")

    root = tmp_path / "flashinfer"
    (root / "0.6.14").mkdir(parents=True)
    assert _flashinfer_cache_is_usable(root)

    (root / "0.6.14").chmod(0o555)  # readable, not writable — as a root-created dir appears
    try:
        assert not _flashinfer_cache_is_usable(root)
    finally:
        (root / "0.6.14").chmod(0o755)


def test_prepare_runtime_sets_the_vllm_flags_and_puts_ninja_on_path():
    _run(
        "import os, sys\n"
        "from bodhan_genai.ocr.engine.recognizer_vllm import _prepare_runtime\n"
        "_prepare_runtime()\n"
        "assert os.environ['VLLM_USE_DEEP_GEMM'] == '0'\n"
        "assert os.environ['VLLM_USE_FLASHINFER_SAMPLER'] == '0'\n"
        "assert os.path.dirname(sys.executable) in os.environ['PATH'].split(os.pathsep)\n"
        "assert 'vllm' not in sys.modules, 'preparing the runtime must not import vllm'\n"
    )


def test_an_explicit_flashinfer_workspace_is_never_overridden():
    _run(
        "import os\n"
        "os.environ['FLASHINFER_WORKSPACE_BASE'] = '/somewhere/of/my/own'\n"
        "from bodhan_genai.ocr.engine.recognizer_vllm import _prepare_runtime\n"
        "_prepare_runtime()\n"
        "assert os.environ['FLASHINFER_WORKSPACE_BASE'] == '/somewhere/of/my/own'\n"
    )
