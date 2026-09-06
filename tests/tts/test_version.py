"""Packaging smoke test: the tts package imports and reports a version."""

import bodhan_genai.tts


def test_version_present():
    assert isinstance(bodhan_genai.tts.__version__, str)
    assert bodhan_genai.tts.__version__
