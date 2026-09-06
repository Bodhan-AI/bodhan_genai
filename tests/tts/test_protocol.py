"""Tests for bodhan_genai.tts.serving.protocol — wire-frame serialization round-trips.

CPU-only: json + pydantic + numpy; must never import vllm/ray (the serving
package __init__ stays import-light for exactly this reason).
"""

from __future__ import annotations

import json
import sys
from typing import ClassVar

import numpy as np
import pytest

from bodhan_genai.tts.serving.protocol import (
    SAMPLE_RATE,
    SynthesisRequest,
    end_frame,
    error_frame,
    start_frame,
)


def test_sample_rate_is_24k():
    assert SAMPLE_RATE == 24_000


def test_start_frame_roundtrip():
    ev = json.loads(start_frame())
    assert ev == {"event": "start", "sample_rate": 24_000, "encoding": "pcm_s16le"}


def test_end_frame_roundtrip():
    ev = json.loads(end_frame(12.345678, 145))
    assert ev["event"] == "end"
    assert ev["n_frames"] == 145
    assert ev["audio_dur_s"] == round(12.345678, 4)
    # json types survive the trip
    assert isinstance(ev["n_frames"], int) and isinstance(ev["audio_dur_s"], float)


def test_error_frame_roundtrip():
    detail = 'engine died: "CUDA error" <retry>'
    ev = json.loads(error_frame(detail))
    assert ev == {"event": "error", "detail": detail}


def test_synthesis_request_roundtrip():
    msg = {
        "text": "namaste duniya",
        "speaker": "spk7",
        "temperature": 0.55,
        "top_p": 0.9,
        "top_k": 50,
        "max_new_tokens": 512,
    }
    req = SynthesisRequest(**msg)
    dumped = req.model_dump()
    for k, v in msg.items():
        assert dumped[k] == v
    # JSON round-trip (what the WS handler actually does)
    again = SynthesisRequest(**json.loads(req.model_dump_json()))
    assert again == req


def test_synthesis_request_defaults():
    req = SynthesisRequest(text="hi")
    d = req.model_dump()
    assert d["speaker"] == "" and d["language"] == ""
    # unset sampling knobs stay None so the server falls back to ServeConfig
    assert d["temperature"] is None and d["top_p"] is None
    assert d["top_k"] is None and d["max_new_tokens"] is None


def test_pcm_frame_binary_roundtrip():
    """Audio frames are raw int16 LE PCM: one SNAC frame = 2048 samples = 4096 bytes."""
    rng = np.random.default_rng(0)
    frame = rng.integers(-32768, 32768, size=2048, dtype=np.int16)
    payload = frame.tobytes()
    assert len(payload) == 4096
    back = np.frombuffer(payload, dtype="<i2")
    assert np.array_equal(back, frame)


def test_no_heavy_imports_pulled_in():
    """Importing the protocol module must not drag in vllm or ray."""
    assert "vllm" not in sys.modules
    assert "ray" not in sys.modules


class TestChunkedField:
    """The WS handler round-trips requests through SynthesisRequest; the
    chunked flag must survive (review finding: pydantic silently dropped it)."""

    def test_chunked_roundtrip(self):
        from bodhan_genai.tts.serving.protocol import SynthesisRequest

        assert SynthesisRequest(text="x", chunked=True).model_dump()["chunked"] is True
        assert SynthesisRequest(text="x", chunked=False).model_dump()["chunked"] is False

    def test_chunked_defaults_to_none_sentinel(self):
        from bodhan_genai.tts.serving.protocol import SynthesisRequest

        # None = "use the server default" — the replica must branch on None,
        # not treat it as falsy opt-out.
        assert SynthesisRequest(text="x").model_dump()["chunked"] is None


class TestDialogueMessages:
    """Dialogue requests: a ``messages`` list of {"speaker","text"} turns,
    mutually exclusive with ``text`` (XOR-validated)."""

    TURNS: ClassVar[list[dict]] = [{"speaker": "a", "text": "hi"}, {"speaker": "b", "text": "yo"}]

    def test_messages_roundtrip(self):
        req = SynthesisRequest(messages=self.TURNS)
        assert req.model_dump()["messages"] == self.TURNS
        # JSON round-trip (what the WS handler actually does)
        again = SynthesisRequest(**json.loads(req.model_dump_json()))
        assert again == req

    def test_both_text_and_messages_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SynthesisRequest(text="hi", messages=self.TURNS)

    def test_neither_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SynthesisRequest()
        with pytest.raises(ValidationError):
            SynthesisRequest(text="")
        # blank-whitespace text counts as absent
        with pytest.raises(ValidationError):
            SynthesisRequest(text="   ")

    def test_exactly_one_ok(self):
        # messages-only: text falls back to its "" default
        assert SynthesisRequest(messages=self.TURNS).model_dump()["text"] == ""
        # text-only: messages stays None in the dump
        assert SynthesisRequest(text="hi").model_dump()["messages"] is None
