"""Checkpoint and file resolution for IndicTranscribe.

``IndicTranscribeForConditionalGeneration`` accepts a Hub repo id for free, inheriting the
transformers resolver. The tokenizer and feature extractor read files with ``sentencepiece`` and
``safetensors``, both pure filesystem — so without ``resolve_file`` they accept only directories,
and the three loaders disagree about what an argument means. That asymmetry is what these tests
pin down.

Nothing here touches the network: every case resolves locally or asserts the failure before a
download would be attempted.
"""

from __future__ import annotations

import os

import pytest

from bodhan_genai.asr.checkpoints import (
    DEFAULT_HF_REPO,
    DEPLOYMENT_ENV,
    HF_REPO_ENV,
    resolve_ckpt,
    resolve_file,
)


def test_the_default_repo_is_the_published_one():
    assert DEFAULT_HF_REPO == "bodhan-ai/indic-transcribe-core"


# --- resolve_ckpt: identifier selection, no I/O ------------------------------------------------


def test_explicit_wins_over_everything(monkeypatch):
    monkeypatch.delenv(DEPLOYMENT_ENV, raising=False)
    monkeypatch.setenv(HF_REPO_ENV, "someone-else/other-repo")
    assert resolve_ckpt("/some/path") == "/some/path"


def test_the_environment_is_ignored_outside_a_deployment_image(monkeypatch):
    """The point of the whole mechanism: an inherited variable cannot redirect weights.

    A stale BODHAN_ASR_HF_REPO in a shell silently transcribing with a different checkpoint is a
    correctness bug that presents as a model regression, so it is not consulted at all here.
    """
    monkeypatch.delenv(DEPLOYMENT_ENV, raising=False)
    monkeypatch.setenv(HF_REPO_ENV, "someone-else/other-repo")
    assert resolve_ckpt() == DEFAULT_HF_REPO


def test_the_environment_is_honoured_inside_a_deployment_image(monkeypatch):
    """Inside the image it is how mounted weights are addressed, so it must still work."""
    monkeypatch.setenv(DEPLOYMENT_ENV, "1")
    monkeypatch.setenv(HF_REPO_ENV, "someone-else/other-repo")
    assert resolve_ckpt() == "someone-else/other-repo"


def test_an_explicit_argument_beats_the_environment_even_in_an_image(monkeypatch):
    monkeypatch.setenv(DEPLOYMENT_ENV, "1")
    monkeypatch.setenv(HF_REPO_ENV, "someone-else/other-repo")
    assert resolve_ckpt("/some/path") == "/some/path"


def test_the_default_is_used_when_nothing_else_is_set(monkeypatch):
    monkeypatch.delenv(HF_REPO_ENV, raising=False)
    assert resolve_ckpt() == DEFAULT_HF_REPO


def test_resolve_ckpt_does_not_check_existence(monkeypatch):
    """It selects an identifier. Whether it resolves is resolve_file's problem."""
    monkeypatch.delenv(HF_REPO_ENV, raising=False)
    assert resolve_ckpt("/definitely/not/here") == "/definitely/not/here"


# --- resolve_file: directory or repo id --------------------------------------------------------


def test_a_file_in_a_local_directory_is_returned_as_is(tmp_path):
    (tmp_path / "tokenizer_multilingual.model").write_bytes(b"x")
    got = resolve_file(str(tmp_path), "tokenizer_multilingual.model")
    assert got == str(tmp_path / "tokenizer_multilingual.model")


def test_a_directory_missing_the_file_raises(tmp_path):
    """A half-copied checkpoint must fail, not quietly fall through to a download."""
    with pytest.raises(FileNotFoundError):
        resolve_file(str(tmp_path), "tokenizer_multilingual.model")


def test_local_failures_do_not_touch_the_network(tmp_path):
    """A typo'd path must fail immediately rather than becoming a Hub lookup.

    Timing is the only way to observe this from outside: `cached_file` gives local misses back
    in microseconds and takes hundreds of milliseconds to decide a repo id does not exist. The
    bound is deliberately loose — this is asserting "no network round trip", not a latency SLO.
    """
    import time

    for target in (str(tmp_path), str(tmp_path / "nope" / "still-nope")):
        start = time.monotonic()
        with pytest.raises(FileNotFoundError):
            resolve_file(target, "tokenizer_multilingual.model")
        elapsed = time.monotonic() - start
        assert elapsed < 0.25, f"{target!r} took {elapsed:.2f}s — it probably hit the Hub"


def test_the_error_names_the_likely_cause(tmp_path):
    """The default checkpoint is private, and the Hub reports 404 for that rather than 403."""
    with pytest.raises(FileNotFoundError, match="HF_TOKEN"):
        resolve_file(str(tmp_path), "tokenizer_multilingual.model")


# --- the property the whole module exists for --------------------------------------------------


def test_all_three_loaders_accept_the_same_argument_shape():
    """The point of this module: one identifier works for model, tokenizer and feature extractor.

    Asserted structurally rather than by loading, which would need the private checkpoint. If a
    loader stops routing through resolve_file it goes back to being directory-only, and callers
    get a loader that fails on the id the other two accept.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # tests/asr/x.py -> repo root
    for rel in (
        "src/bodhan_genai/asr/model/tokenization_indic_transcribe.py",
        "src/bodhan_genai/asr/model/feature_extraction_indic_transcribe.py",
    ):
        text = (root / rel).read_text()
        assert "resolve_file(" in text, f"{rel} no longer resolves Hub ids"
        assert "os.path.join(load_dir" not in text, f"{rel} still joins load_dir directly"


def test_the_import_is_cheap(tmp_path):
    """Resolution must not drag in huggingface_hub or torch unless a download is needed."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from bodhan_genai.asr.checkpoints import resolve_ckpt\n"
            f"assert resolve_ckpt({str(tmp_path)!r})\n"
            "assert 'huggingface_hub' not in sys.modules, 'imported without needing a download'\n"
            "assert 'torch' not in sys.modules, 'pulled in torch'\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
