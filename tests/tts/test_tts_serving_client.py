"""The bundled TTS client's HTTP paths.

Nothing tested this module before, and a live run caught the consequence: the
SSE reader called ``write_wav_24k(out, pcm, sample_rate=sr)`` when the writer
takes two positional arguments and normalized float32. The whole endpoint was
correct and the client crashed on the last line, after the audio had arrived.

These tests stub ``urlopen``, so no server, no GPU and no network.
"""

from __future__ import annotations

import json
from base64 import b64encode

import numpy as np
import pytest
import soundfile as sf

from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE
from bodhan_genai.tts.serving import client as tts_client
from bodhan_genai.tts.serving.protocol import audio_frame, end_frame, error_frame, sse, start_frame

FRAME = (np.arange(2048, dtype=np.int16) * 7 % 3000).astype(np.int16).tobytes()
N_FRAMES = 3


class FakeResponse:
    """Just enough of an http.client.HTTPResponse: line iteration and a context."""

    def __init__(self, body: bytes):
        self._lines = body.splitlines(keepends=True)

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse_body(frames=N_FRAMES, error: str | None = None) -> bytes:
    out = [sse("start", start_frame())]
    for i in range(frames):
        out.append(sse("audio", audio_frame(i, FRAME)))
    if error:
        out.append(sse("error", error_frame(error)))
    else:
        total = frames * (len(FRAME) // 2)
        out.append(sse("end", end_frame(total / SNAC_SAMPLE_RATE, total // 2048)))
    return b"".join(out)


@pytest.fixture
def stub_urlopen(monkeypatch):
    """Capture the request and serve a canned body."""
    seen = {}

    def fake(req, *a, **k):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["body"] = json.loads(req.data)
        return FakeResponse(seen.pop("_body", _sse_body()))

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return seen


def test_sse_writes_a_readable_wav_of_the_decoded_frames(tmp_path, stub_urlopen):
    out = tmp_path / "out.wav"
    tts_client.run_sse("http://x/tts/sse", {"text": "hi", "speaker": "Amit"}, str(out))

    audio, sr = sf.read(out, dtype="int16")
    assert sr == SNAC_SAMPLE_RATE
    # float32 round-trip through /32767 is lossy by at most one int16 step
    expected = np.frombuffer(FRAME * N_FRAMES, dtype=np.int16)
    assert np.abs(audio.astype(int) - expected.astype(int)).max() <= 1
    assert stub_urlopen["body"] == {"text": "hi", "speaker": "Amit"}
    assert stub_urlopen["headers"]["accept"] == "text/event-stream"


def test_sse_sends_basic_auth_when_given_a_credential(tmp_path, stub_urlopen):
    tts_client.run_sse("http://x/tts/sse", {"text": "hi"}, str(tmp_path / "o.wav"), "alice:s3cret")
    expected = "Basic " + b64encode(b"alice:s3cret").decode()
    assert stub_urlopen["headers"]["authorization"] == expected


def test_sse_sends_no_auth_header_when_open(tmp_path, stub_urlopen):
    tts_client.run_sse("http://x/tts/sse", {"text": "hi"}, str(tmp_path / "o.wav"))
    assert "authorization" not in stub_urlopen["headers"]


def test_sse_writes_nothing_on_an_error_event(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, *a, **k: FakeResponse(_sse_body(frames=1, error="scripted failure")),
    )
    out = tmp_path / "out.wav"
    tts_client.run_sse("http://x/tts/sse", {"text": "hi"}, str(out))
    # partial audio is discarded rather than written as a truncated utterance
    assert not out.exists()
    assert "scripted failure" in capsys.readouterr().err


class TestCredentialResolution:
    def test_explicit_wins(self, tmp_path, monkeypatch):
        f = tmp_path / "creds"
        f.write_text("bob:from-file")
        monkeypatch.setenv("TTS_AUTH_FILE", str(f))
        assert tts_client._credential("alice:explicit") == "alice:explicit"

    def test_falls_back_to_the_file_the_server_reads(self, tmp_path, monkeypatch):
        f = tmp_path / "creds"
        f.write_text("bob:from-file\n")
        monkeypatch.setenv("TTS_AUTH_FILE", str(f))
        assert tts_client._credential("") == "bob:from-file"

    def test_open_server_needs_nothing(self, monkeypatch):
        monkeypatch.delenv("TTS_AUTH_FILE", raising=False)
        assert tts_client._credential("") == ""

    def test_an_unreadable_file_warns_and_stays_open(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("TTS_AUTH_FILE", str(tmp_path / "missing"))
        assert tts_client._credential("") == ""
        assert "unreadable" in capsys.readouterr().err

    def test_the_credential_is_never_read_from_a_plain_env_var(self, monkeypatch):
        """A credential in the environment ends up in every `ps` listing."""
        monkeypatch.delenv("TTS_AUTH_FILE", raising=False)
        monkeypatch.setenv("TTS_AUTH", "alice:s3cret")
        monkeypatch.setenv("TTS_PASSWORD", "s3cret")
        assert tts_client._credential("") == ""
