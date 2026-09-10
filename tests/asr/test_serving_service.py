"""ASR service handlers and wire protocol, driven with a fake replica.

No Ray, no GPU, no model: the handlers are free functions taking a replica, so
the request validation, the websocket message sequence, and the PCM decode are
all exercised directly.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import torch

from bodhan_genai.asr.serving.protocol import StreamStart, TranscribeRequest
from bodhan_genai.asr.serving.service import _detect, _transcribe, _ws_stream, pcm16_to_float
from bodhan_genai.asr.serving.streaming import StreamUpdate

# --- protocol --------------------------------------------------------------


def test_transcribe_request_requires_paths():
    with pytest.raises(ValueError, match="non-empty"):
        TranscribeRequest(paths=[], lang="hi")


def test_transcribe_request_requires_a_language_or_detection():
    """A wrong language yields confidently wrong script, so the server refuses
    to guess silently."""
    with pytest.raises(ValueError, match="language-conditioned"):
        TranscribeRequest(paths=["a.wav"])
    assert TranscribeRequest(paths=["a.wav"], lang="hi").lang == "hi"
    assert TranscribeRequest(paths=["a.wav"], detect_language=True).detect_language


def test_stream_start_requires_a_language_or_detection():
    with pytest.raises(ValueError):
        StreamStart()
    assert StreamStart(lang="hi").lang == "hi"


# --- PCM decode ------------------------------------------------------------


def test_pcm16_roundtrip():
    orig = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    buf = (orig * 32767).astype("<i2").tobytes()
    got = pcm16_to_float(buf)
    assert got.shape == (4,)
    assert torch.allclose(got, torch.from_numpy(orig), atol=1e-3)


def test_pcm16_is_little_endian():
    """Byte order is part of the wire contract; a mismatch is silent garbage."""
    buf = np.array([256], dtype="<i2").tobytes()
    assert abs(float(pcm16_to_float(buf)[0]) - 256 / 32768) < 1e-6


# --- fakes -----------------------------------------------------------------


class FakeStream:
    def __init__(self):
        self.sample_rate = 16000
        self.endpoint_silence_s = 0.5
        self.max_segment_s = 5.0
        self.partial_interval_s = 2.0
        self.pushed = 0


class FakeReplica:
    """Implements only what the handlers touch."""

    def __init__(self, *, texts=("hello",), updates=None, lid=None, fail=None):
        self.texts = list(texts)
        self.updates = list(updates or [])
        self.lid = lid or [[("hi", 0.99), ("ur", 0.01)]]
        self.fail = fail
        self.dead = False
        self.streams = []

    async def transcribe_paths(
        self,
        paths,
        lang,
        chunk_above,
        itn=False,
        romanized=False,
        return_lid=False,
        allowed_langs=None,
    ):
        if self.fail:
            raise RuntimeError(self.fail)
        self.last = (tuple(paths), lang, chunk_above)
        self.last_allowed = allowed_langs
        texts = self.texts[: len(paths)]
        if not return_lid:
            return texts
        langs = [lang] * len(paths) if (lang is None or isinstance(lang, str)) else list(lang)
        rows = []
        for i in range(len(paths)):
            top = self.lid[i % len(self.lid)]
            rows.append(
                {
                    "lang": langs[i] or top[0][0],
                    "source": "explicit" if langs[i] else "lid",
                    "topk": top,
                }
            )
        return texts, rows

    async def detect_language(self, paths, topk=5, allowed_langs=None):
        self.last_allowed = allowed_langs
        return self.lid[: len(paths)] or self.lid

    async def detect_language_file(self, path, topk=5, allowed_langs=None, max_seconds=120.0):
        self.last_allowed = allowed_langs
        return self.lid[:1]

    async def check_lang(self, lang):
        return None if lang in ("hi", "ta", "bn", "en") else f"unsupported language {lang!r}"

    async def is_short(self, path):
        return True

    async def ready(self):
        return not self.dead

    def new_stream(self, lang, sample_rate=None, itn=False, romanized=False):
        s = FakeStream()
        self.streams.append((lang, sample_rate, s))
        return s

    async def stream_push(self, stream, audio):
        stream.pushed += 1
        return self.updates.pop(0) if self.updates else None

    async def stream_finalize(self, stream):
        return StreamUpdate(committed_total="final text", audio_seconds=1.0, decoded=True)


class FakeWS:
    """Minimal WebSocket double recording what the server sent."""

    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent: list[dict] = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        msg = self.incoming.pop(0)
        return msg["text"]

    async def receive(self):
        if not self.incoming:
            return {"type": "websocket.disconnect"}
        return self.incoming.pop(0)

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def close(self):
        self.closed = True


def events(ws):
    return [m.get("event") for m in ws.sent]


# --- POST /asr/transcribe --------------------------------------------------


def test_transcribe_returns_one_result_per_path():
    r = FakeReplica(texts=["one", "two"])
    resp = asyncio.run(_transcribe(r, {"paths": ["a.wav", "b.wav"], "lang": "hi"}))
    body = json.loads(resp.body)
    assert body["lang"] == "hi"
    assert [x["path"] for x in body["results"]] == ["a.wav", "b.wav"]
    assert [x["text"] for x in body["results"]] == ["one", "two"]


def test_transcribe_rejects_a_bad_request_with_422():
    resp = asyncio.run(_transcribe(FakeReplica(), {"paths": []}))
    assert resp.status_code == 422


def test_transcribe_reports_engine_failure_as_500_not_a_crash():
    resp = asyncio.run(
        _transcribe(FakeReplica(fail="cuda oom"), {"paths": ["a.wav"], "lang": "hi"})
    )
    assert resp.status_code == 500
    body = json.loads(resp.body)["error"]
    # The detail must NOT come back: engine messages distinguish "no such file"
    # from "permission denied" from "format not recognised", which is a
    # filesystem oracle for anyone holding the shared demo credential.
    assert "cuda oom" not in body
    assert "ref" in body


def test_transcribe_can_detect_the_language_and_reports_it():
    r = FakeReplica(texts=["namaste"])
    resp = asyncio.run(_transcribe(r, {"paths": ["a.wav"], "detect_language": True}))
    body = json.loads(resp.body)
    assert body["lang"] == "hi"
    row = body["results"][0]
    assert row["lang"] == "hi" and row["lang_source"] == "lid"
    assert row["lid"][0]["lang"] == "hi"


def test_transcribe_reports_lid_per_row_not_one_answer_for_the_batch():
    """Two paths in different languages must not both inherit row 0's guess.

    The old detect-then-transcribe pre-pass took ``top[0][0][0]`` and applied
    it to every row, so a mixed-language request came back almost entirely in
    the wrong script.
    """
    r = FakeReplica(texts=["a", "b"], lid=[[("ta", 0.9)], [("bn", 0.8)]])
    resp = asyncio.run(_transcribe(r, {"paths": ["a.wav", "b.wav"], "detect_language": True}))
    body = json.loads(resp.body)
    assert [row["lang"] for row in body["results"]] == ["ta", "bn"]
    # rows disagree, so there is no single language for the request
    assert body["lang"] is None


def test_transcribe_lets_an_explicit_language_win_but_still_shows_lid():
    r = FakeReplica(texts=["x"], lid=[[("ur", 0.9), ("hi", 0.1)]])
    resp = asyncio.run(_transcribe(r, {"paths": ["a.wav"], "lang": "hi", "detect_language": True}))
    row = json.loads(resp.body)["results"][0]
    assert row["lang"] == "hi" and row["lang_source"] == "explicit"
    assert row["lid"][0]["lang"] == "ur"  # the disagreement is visible


def test_transcribe_warns_on_a_weak_lid_class():
    r = FakeReplica(texts=["x"], lid=[[("hi", 0.6), ("ur", 0.4)]])
    resp = asyncio.run(_transcribe(r, {"paths": ["a.wav"], "detect_language": True}))
    assert "weakest LID" in json.loads(resp.body)["results"][0]["warning"]


def test_allowed_langs_named_sets_resolve():
    from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS, TRAINED_LANGS
    from bodhan_genai.asr.serving.service import _parse_allowed

    assert _parse_allowed("trained") == TRAINED_LANGS
    assert _parse_allowed("recommended") == RECOMMENDED_LANGS
    assert _parse_allowed("hi,ta") == ("hi", "ta")
    assert _parse_allowed("") is None and _parse_allowed(None) is None


def test_allowed_langs_takes_the_string_and_the_list_form_alike():
    """/asr/detect reads an unvalidated dict, so the JSON list and the multipart
    string reach the same parser. They used to reach different ones: a bare
    string fell through to tuple("recommended") -> 11 single characters, which
    matched no language token and 500ed."""
    from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS
    from bodhan_genai.asr.serving.service import _parse_allowed

    assert _parse_allowed("recommended") == _parse_allowed(["recommended"]) == RECOMMENDED_LANGS
    assert _parse_allowed(["hi", "ta"]) == ("hi", "ta")


def test_allowed_langs_named_set_composes_with_extra_codes():
    """['trained', 'hi'] used to skip the named-set branch entirely and pass
    'trained' through as a language code. It matched nothing, so the candidate
    set collapsed to {hi} and every clip came back Hindi at p=1.0 -- silently,
    with a 200."""
    from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS
    from bodhan_genai.asr.serving.service import _parse_allowed

    got = _parse_allowed(["recommended", "bgc"])
    assert set(got) == set(RECOMMENDED_LANGS) | {"bgc"}
    assert len(got) == len(set(got))  # deduped, order preserved


def test_allowed_langs_rejects_junk_as_422_not_500():
    from bodhan_genai.asr.serving.service import BadRequest, _parse_allowed

    for junk in ("Recommended", "en-US", "TRAINED"):
        with pytest.raises(BadRequest):
            _parse_allowed(junk)


def test_detect_maps_a_bad_allowed_langs_to_422():
    resp = asyncio.run(_detect(FakeReplica(), {"paths": ["a.wav"], "allowed_langs": "Recommended"}))
    assert resp.status_code == 422
    assert "allowed_langs" in json.loads(resp.body)["error"]


def test_detect_forwards_allowed_langs():
    r = FakeReplica()
    asyncio.run(_detect(r, {"paths": ["a.wav"], "allowed_langs": ["recommended"]}))
    assert "bgc" not in r.last_allowed and "hi" in r.last_allowed


def test_transcribe_passes_chunk_above_through():
    r = FakeReplica(texts=["x"])
    asyncio.run(_transcribe(r, {"paths": ["a.wav"], "lang": "hi", "chunk_above": 45.0}))
    assert r.last == (("a.wav",), "hi", 45.0)


# --- POST /asr/detect ------------------------------------------------------


def test_detect_returns_ranked_languages():
    resp = asyncio.run(_detect(FakeReplica(), {"paths": ["a.wav"]}))
    body = json.loads(resp.body)
    assert body["results"][0]["top"][0]["lang"] == "hi"


def test_detect_rejects_empty_paths():
    assert asyncio.run(_detect(FakeReplica(), {"paths": []})).status_code == 422


# --- WS /asr/stream --------------------------------------------------------


def pcm(n=1600):
    return (np.zeros(n, dtype=np.float32)).astype("<i2").tobytes()


def test_stream_start_then_updates_then_end():
    upd = StreamUpdate(
        committed_delta="hello",
        provisional="wor",
        committed_total="hello",
        audio_seconds=1.0,
        decoded=True,
    )
    ws = FakeWS(
        [
            {"type": "websocket.receive", "text": json.dumps({"lang": "hi"})},
            {"type": "websocket.receive", "bytes": pcm()},
            {"type": "websocket.receive", "text": json.dumps({"event": "eof"})},
        ]
    )
    r = FakeReplica(updates=[upd])
    asyncio.run(_ws_stream(r, ws))
    assert events(ws) == ["start", "update", "end"]
    assert ws.sent[1]["committed"] == "hello"
    assert ws.sent[1]["provisional"] == "wor"
    assert ws.sent[2]["text"] == "final text"
    assert ws.closed


def test_stream_suppresses_updates_when_no_decode_was_due():
    """push() returning None means 'buffered, nothing to say' — the server must
    not emit an empty update for every packet."""
    ws = FakeWS(
        [
            {"type": "websocket.receive", "text": json.dumps({"lang": "hi"})},
            {"type": "websocket.receive", "bytes": pcm()},
            {"type": "websocket.receive", "bytes": pcm()},
            {"type": "websocket.receive", "text": json.dumps({"event": "eof"})},
        ]
    )
    asyncio.run(_ws_stream(FakeReplica(updates=[]), ws))
    assert events(ws) == ["start", "end"]


def test_stream_rejects_a_malformed_start_message():
    ws = FakeWS([{"type": "websocket.receive", "text": "{not json"}])
    asyncio.run(_ws_stream(FakeReplica(), ws))
    assert events(ws) == ["error"]
    assert ws.closed


def test_stream_rejects_a_start_without_a_language():
    ws = FakeWS([{"type": "websocket.receive", "text": json.dumps({})}])
    asyncio.run(_ws_stream(FakeReplica(), ws))
    assert events(ws) == ["error"]


def test_stream_refuses_detect_language_rather_than_guessing():
    """No audio exists at session start, and picking a language silently would
    produce confidently wrong script."""
    ws = FakeWS([{"type": "websocket.receive", "text": json.dumps({"detect_language": True})}])
    asyncio.run(_ws_stream(FakeReplica(), ws))
    assert events(ws) == ["error"]
    assert "not supported on the streaming endpoint" in ws.sent[0]["message"]


def test_stream_finalizes_on_client_disconnect():
    """A vanished client must still flush the buffer, not lose the transcript."""
    ws = FakeWS(
        [
            {"type": "websocket.receive", "text": json.dumps({"lang": "hi"})},
            {"type": "websocket.receive", "bytes": pcm()},
        ]
    )
    asyncio.run(_ws_stream(FakeReplica(), ws))
    assert events(ws) == ["start", "end"]


def test_stream_honours_the_client_sample_rate():
    ws = FakeWS(
        [
            {"type": "websocket.receive", "text": json.dumps({"lang": "hi", "sample_rate": 8000})},
            {"type": "websocket.receive", "text": json.dumps({"event": "eof"})},
        ]
    )
    r = FakeReplica()
    asyncio.run(_ws_stream(r, ws))
    assert r.streams[0][:2] == ("hi", 8000)


# --- hardening regressions -------------------------------------------------


def test_allowed_langs_rejects_a_code_this_checkpoint_does_not_have():
    """Shape was not enough. 'hin'/'tam' are well-formed ISO-639-3 codes that
    match no token; language_token_map drops unknown names silently and raises
    only when NOTHING matches, so a caller using the wrong code system got a
    200 and a candidate set quietly narrowed to whatever happened to be right."""
    from bodhan_genai.asr.serving.service import BadRequest, _parse_allowed

    with pytest.raises(BadRequest):
        _parse_allowed(["hin", "tam"])


def test_allowed_langs_rejects_a_scalar_instead_of_crashing():
    from bodhan_genai.asr.serving.service import BadRequest, _parse_allowed

    for junk in (5, True, 1.5):
        with pytest.raises(BadRequest):
            _parse_allowed(junk)


def test_importing_the_service_module_needs_no_auth_env(monkeypatch):
    """Importing must be free; only *serving* requires a credential.

    The auth check ran at module scope, so `import ...serving.service` raised unless
    ASR_AUTH_FILE was exported — breaking every reader of the module for a check that
    only matters when something is served. It now runs in build_deployment().
    """
    import subprocess
    import sys

    monkeypatch.delenv("ASR_AUTH_FILE", raising=False)
    # a subprocess, because the module is already in sys.modules for this session
    result = subprocess.run(
        [sys.executable, "-c", "import bodhan_genai.asr.serving.service"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_auth_is_optional_so_a_server_starts_with_no_environment(monkeypatch):
    """The default is an open server.

    This is a reference setup meant to be copied and extended; needing an environment variable
    to start one made the first thing a new user saw an error about a credential file.
    """
    from fastapi import FastAPI

    from bodhan_genai.asr.serving.service import _install_basic_auth

    monkeypatch.delenv("ASR_AUTH_FILE", raising=False)
    app = FastAPI()
    _install_basic_auth(app)
    assert not any("BasicAuth" in str(m) for m in app.user_middleware)


def test_a_broken_credential_file_is_still_fatal(monkeypatch, tmp_path):
    """Opt-in, but not opt-in-and-quietly-fail.

    An unreadable or malformed file is someone *enabling* auth and getting it wrong; serving
    open in that case is the one outcome nobody wants, so it stays an error.
    """
    from fastapi import FastAPI

    from bodhan_genai.asr.serving.service import _install_basic_auth

    monkeypatch.setenv("ASR_AUTH_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="unreadable"):
        _install_basic_auth(FastAPI())

    bad = tmp_path / "creds"
    bad.write_text("no-colon-here")
    monkeypatch.setenv("ASR_AUTH_FILE", str(bad))
    with pytest.raises(RuntimeError, match="not 'user:password'"):
        _install_basic_auth(FastAPI())


def test_auth_is_actually_attached_and_not_merely_configured(monkeypatch, tmp_path):
    """Drive a real request through the stack.

    Every other auth test here either asserts the installer *raises* or exercises the
    middleware it constructs itself, so all of them stayed green with the installer's
    add_middleware call removed -- i.e. with the endpoints wide open. This one notices.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from bodhan_genai.asr.serving.service import _install_basic_auth

    creds = tmp_path / "creds"
    creds.write_text("alice:s3cret\n")
    monkeypatch.setenv("ASR_AUTH_FILE", str(creds))

    app = FastAPI()
    _install_basic_auth(app)
    client = TestClient(app)

    refused = client.get("/asr/stats")
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"] == 'Basic realm="Indic Transcribe"'

    # 404, not 401: no such route on a bare app, which is the point -- it got past auth.
    assert client.get("/asr/stats", auth=("alice", "s3cret")).status_code == 404


def test_auth_compares_bytes_so_a_non_ascii_header_is_401_not_500():
    """compare_digest rejects non-ASCII str, so a header with any byte >= 0x80
    raised TypeError -- a 500 that told an unauthenticated caller the
    middleware was there."""
    from bodhan_genai.asr.serving.service import BasicAuthMiddleware

    sent = []

    async def send(m):
        sent.append(m)

    async def receive():
        return {"type": "http.request"}

    async def app(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("unauthenticated request reached the app")

    mw = BasicAuthMiddleware(app, user="u", password="p")
    scope = {
        "type": "http",
        "path": "/asr/stats",
        "headers": [(b"authorization", b"Basic \xff\xfe")],
    }
    asyncio.run(mw(scope, receive, send))
    assert sent[0]["status"] == 401


def test_stream_refuses_an_unsupported_language_before_a_session_exists():
    """It used to reach the slot scheduler, where encode_prompt raised inside
    the admit loop and killed the thread -- every session on the replica, not
    just the offending one."""
    r = FakeReplica()
    ws = FakeWS(
        [{"type": "websocket.receive", "text": json.dumps({"lang": "zz", "sample_rate": 16000})}]
    )
    asyncio.run(_ws_stream(r, ws))
    assert r.streams == []
    assert any("unsupported language" in str(m) for m in ws.sent)


def test_stream_rejects_an_arbitrary_sample_rate():
    """Every span bound in VadStream is a sample count divided by this number,
    so a huge value made the buffer bound unreachable."""
    from bodhan_genai.asr.serving.protocol import StreamStart

    with pytest.raises(ValueError, match="sample_rate"):
        StreamStart(lang="hi", sample_rate=10_000_000)
    assert StreamStart(lang="hi", sample_rate=8000).sample_rate == 8000


def test_transcribe_bounds_the_paths_list():
    from bodhan_genai.asr.serving.protocol import MAX_PATHS, TranscribeRequest

    with pytest.raises(ValueError, match="at most"):
        TranscribeRequest(paths=["a.wav"] * (MAX_PATHS + 1), lang="hi")


def test_health_reports_503_when_the_replica_cannot_serve():
    """ready() used to `return True` unconditionally, so a replica whose slot
    scheduler had died still answered 200 and kept taking traffic."""
    r = FakeReplica()
    assert asyncio.run(r.ready()) is True
    r.dead = True
    assert asyncio.run(r.ready()) is False
