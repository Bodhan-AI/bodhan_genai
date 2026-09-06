"""Version and public-API surface of bodhan_genai.mt."""

from __future__ import annotations

import tomllib
from pathlib import Path

import bodhan_genai.mt as mt

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Everything importable from `bodhan_genai.mt`. Adding to this is a deliberate
#: public-API change; the list exists so it cannot happen by accident.
PUBLIC_API = {
    "IndicMTEngine",
    "LANGUAGE_NAMES",
    "MTResult",
    "MTSamplingConfig",
    "STOP_STRINGS",
    "build_conversation",
    "build_instruction",
    "resolve_language",
}


def test_version_is_exposed():
    assert isinstance(mt.__version__, str)
    assert mt.__version__


def test_version_matches_pyproject_when_installed():
    """Skipped implicitly when running from a source tree without an install,
    where __version__ falls back to the sentinel."""
    if mt.__version__ == "0.0.0+unknown":
        return
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert mt.__version__ == pyproject["project"]["version"]


def test_shared_version_with_tts():
    """One package, one version: the modalities ship together."""
    import bodhan_genai.tts as tts

    assert mt.__version__ == tts.__version__


def test_public_api_surface_is_exactly_as_declared():
    assert set(mt.__all__) == PUBLIC_API | {"__version__"}


def test_every_declared_name_actually_resolves():
    for name in PUBLIC_API:
        assert getattr(mt, name) is not None, name
