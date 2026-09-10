"""The four serving endpoints, driven through the shared handlers with a fake
service — real FastAPI routing/framing, no Ray Serve, no GPU, no vllm."""

from __future__ import annotations

import io
import json
from base64 import b64decode
from typing import ClassVar

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.testclient import TestClient

from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE, pcm16_to_wav_bytes
from bodhan_genai.tts.serving.protocol import SynthesisRequest
from bodhan_genai.tts.serving.service import (
    _offline_synthesis,
    _sse_synthesis,
    _ws_synthesis,
)

FRAME = (np.arange(2048, dtype=np.int16) * 7 % 3000).astype(np.int16).tobytes()
N_FRAMES = 3


class FakeService:
    """Duck-type of TtsService for the handlers: synthesize + recording attrs."""

    def __init__(self, fail_after: int | None = None):
        self._writer = None
        self._record_dir = ""
        self._rec_counter = 0
        self._fail_after = fail_after
        self.seen_req: dict | None = None

    def synthesize(self, req: dict):
        self.seen_req = dict(req)

        async def gen():
            for i in range(N_FRAMES):
                yield FRAME
                if self._fail_after is not None and i + 1 >= self._fail_after:
                    raise RuntimeError("scripted synthesis failure")

        return gen()


def make_app(svc: FakeService) -> FastAPI:
    """Mirror TtsService's thin routes over the shared handler functions."""
    app = FastAPI()

    @app.websocket("/tts")
    async def tts(ws: WebSocket):
        await _ws_synthesis(svc, ws, force_chunked=None)

    @app.websocket("/tts/chunked")
    async def tts_chunked(ws: WebSocket):
        await _ws_synthesis(svc, ws, force_chunked=True)

    @app.post("/tts/sse")
    async def tts_sse(request: SynthesisRequest):
        return StreamingResponse(
            _sse_synthesis(svc, request.model_dump()), media_type="text/event-stream"
        )

    @app.post("/tts/offline")
    async def tts_offline(request: SynthesisRequest):
        req = request.model_dump()
        try:
            pcm, total_samples = await _offline_synthesis(svc, req)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        return Response(
            content=pcm16_to_wav_bytes(pcm),
            media_type="audio/wav",
            headers={"X-Audio-Duration-S": f"{total_samples / SNAC_SAMPLE_RATE:.3f}"},
        )

    return app


def drain_sse(resp) -> tuple[list[dict], list[bytes]]:
    """Parse an SSE body into control events and decoded PCM frames.

    Deliberately checks the ``event:`` line against the ``"event"`` key inside
    the JSON: the two are written separately, and a mismatch would break an
    EventSource listener while every data-only reader stayed happy.
    """
    controls, frames = [], []
    event = None
    for line in resp.text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            msg = json.loads(line[5:].strip())
            assert msg["event"] == event, f"event line {event!r} != data {msg['event']!r}"
            if msg["event"] == "audio":
                frames.append(b64decode(msg["pcm_b64"]))
            else:
                controls.append(msg)
    return controls, frames


def drain_ws(ws) -> tuple[list[dict], list[bytes]]:
    """Read raw WS messages until close; split into JSON control frames + PCM."""
    controls, frames = [], []
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            break
        if msg.get("text") is not None:
            ev = json.loads(msg["text"])
            controls.append(ev)
            if ev.get("event") in ("end", "error"):
                break
        elif msg.get("bytes") is not None:
            frames.append(msg["bytes"])
    return controls, frames


class TestStreamEndpoint:
    def test_framing_and_flag_passthrough(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"text": "hi", "speaker": "Amit"}))
            controls, frames = drain_ws(ws)
        assert controls[0]["event"] == "start"
        assert controls[-1]["event"] == "end"
        assert frames == [FRAME] * N_FRAMES
        assert controls[-1]["n_frames"] == N_FRAMES
        # /tts passes the None sentinel through (server default decides)
        assert svc.seen_req["chunked"] is None
        assert svc.seen_req["speaker"] == "Amit"

    def test_explicit_chunked_flag_survives(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"text": "hi", "chunked": True}))
            drain_ws(ws)
        assert svc.seen_req["chunked"] is True

    def test_invalid_request_gets_error_frame(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"speaker": "no text field"}))
            controls, frames = drain_ws(ws)
        assert controls[-1]["event"] == "error"
        assert frames == []
        assert svc.seen_req is None  # never reached synthesis

    def test_midstream_failure_yields_error_after_partial_audio(self):
        svc = FakeService(fail_after=1)
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"text": "hi"}))
            controls, frames = drain_ws(ws)
        assert controls[0]["event"] == "start"
        assert controls[-1]["event"] == "error"
        assert len(frames) == 1  # partial audio delivered before the error


class TestChunkedEndpoint:
    def test_forces_chunked_routing(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts/chunked") as ws:
            ws.send_text(json.dumps({"text": "long text"}))
            _controls, frames = drain_ws(ws)
        assert svc.seen_req["chunked"] is True
        assert frames == [FRAME] * N_FRAMES

    def test_forced_overrides_client_false(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts/chunked") as ws:
            ws.send_text(json.dumps({"text": "long text", "chunked": False}))
            drain_ws(ws)
        assert svc.seen_req["chunked"] is True  # the endpoint decides


class TestOfflineEndpoint:
    def test_returns_wav_of_full_stream(self):
        import soundfile as sf

        svc = FakeService()
        client = TestClient(make_app(svc))
        resp = client.post("/tts/offline", json={"text": "hi", "speaker": "Amit"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="int16")
        assert sr == SNAC_SAMPLE_RATE
        assert audio.tobytes() == FRAME * N_FRAMES
        expected_dur = (len(FRAME) // 2) * N_FRAMES / SNAC_SAMPLE_RATE
        assert float(resp.headers["X-Audio-Duration-S"]) == pytest.approx(expected_dur, abs=1e-3)
        assert svc.seen_req["chunked"] is None  # server default decides

    def test_synthesis_failure_maps_to_500(self):
        svc = FakeService(fail_after=1)
        client = TestClient(make_app(svc))
        resp = client.post("/tts/offline", json={"text": "hi"})
        assert resp.status_code == 500
        assert "scripted synthesis failure" in resp.json()["error"]

    def test_invalid_body_is_422(self):
        client = TestClient(make_app(FakeService()))
        assert client.post("/tts/offline", json={"speaker": "no text"}).status_code == 422


def test_pcm16_wav_roundtrip():
    import soundfile as sf

    pcm = (np.arange(4096, dtype=np.int16) - 2048).tobytes()
    wav = pcm16_to_wav_bytes(pcm)
    audio, sr = sf.read(io.BytesIO(wav), dtype="int16")
    assert sr == SNAC_SAMPLE_RATE
    assert audio.tobytes() == pcm


class TestDialogueRequests:
    """`messages` dialogue requests over the websocket and offline endpoints."""

    TURNS: ClassVar[list[dict]] = [{"speaker": "a", "text": "hi"}, {"speaker": "b", "text": "yo"}]

    def test_stream_ws_passes_messages_through(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"messages": self.TURNS}))
            controls, frames = drain_ws(ws)
        assert svc.seen_req["messages"] == self.TURNS
        assert controls[0]["event"] == "start"
        assert controls[-1]["event"] == "end"
        assert frames == [FRAME] * N_FRAMES

    def test_chunked_ws_forces_flag_and_keeps_messages(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts/chunked") as ws:
            ws.send_text(json.dumps({"messages": self.TURNS}))
            drain_ws(ws)
        assert svc.seen_req["chunked"] is True
        assert svc.seen_req["messages"] == self.TURNS

    def test_offline_returns_wav_for_dialogue(self):
        import soundfile as sf

        svc = FakeService()
        client = TestClient(make_app(svc))
        resp = client.post("/tts/offline", json={"messages": self.TURNS})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/wav")
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="int16")
        assert sr == SNAC_SAMPLE_RATE
        assert audio.tobytes() == FRAME * N_FRAMES
        assert svc.seen_req["messages"] == self.TURNS

    def test_ws_both_text_and_messages_is_error_frame(self):
        svc = FakeService()
        client = TestClient(make_app(svc))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps({"text": "hi", "messages": self.TURNS}))
            controls, frames = drain_ws(ws)
        assert controls[-1]["event"] == "error"
        assert frames == []
        assert svc.seen_req is None  # never reached synthesis

    def test_offline_both_text_and_messages_is_422(self):
        client = TestClient(make_app(FakeService()))
        resp = client.post("/tts/offline", json={"text": "hi", "messages": self.TURNS})
        assert resp.status_code == 422


class TestSseEndpoint:
    """POST /tts/sse — the websocket stream over plain HTTP."""

    def test_frames_arrive_base64_between_start_and_end(self):
        svc = FakeService()
        resp = TestClient(make_app(svc)).post("/tts/sse", json={"text": "hi", "speaker": "Amit"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        controls, frames = drain_sse(resp)
        assert [c["event"] for c in controls] == ["start", "end"]
        assert frames == [FRAME] * N_FRAMES
        assert controls[-1]["n_frames"] == N_FRAMES
        assert svc.seen_req["speaker"] == "Amit"
        assert svc.seen_req["chunked"] is None  # server default decides, as on /tts

    def test_it_reports_the_same_stream_the_websocket_does(self):
        """One request, both transports: the control frames must agree.

        They share the frame builders precisely so this holds; the test is here
        because sharing them is easy to undo by inlining one 'small' dict.
        """
        payload = {"text": "hi", "speaker": "Amit"}
        client = TestClient(make_app(FakeService()))
        sse_controls, sse_frames = drain_sse(client.post("/tts/sse", json=payload))
        with client.websocket_connect("/tts") as ws:
            ws.send_text(json.dumps(payload))
            ws_controls, ws_frames = drain_ws(ws)
        assert sse_controls == ws_controls
        assert sse_frames == ws_frames

    def test_sequence_numbers_are_dense_and_ordered(self):
        """SSE has no framing of its own past the event boundary, so a client
        reassembling audio has only ``seq`` to detect a dropped frame."""
        resp = TestClient(make_app(FakeService())).post("/tts/sse", json={"text": "hi"})
        seqs = [
            json.loads(line[5:])["seq"]
            for line in resp.text.splitlines()
            if line.startswith("data:") and '"audio"' in line
        ]
        assert seqs == list(range(N_FRAMES))

    def test_midstream_failure_is_an_error_event_not_a_status(self):
        """The response is committed 200 at the first event, so a 500 is not
        available -- and a bare exception would just sever the connection."""
        resp = TestClient(make_app(FakeService(fail_after=1))).post("/tts/sse", json={"text": "hi"})
        assert resp.status_code == 200
        controls, frames = drain_sse(resp)
        assert [c["event"] for c in controls] == ["start", "error"]
        assert "scripted synthesis failure" in controls[-1]["detail"]
        assert len(frames) == 1  # partial audio delivered before the failure

    def test_invalid_body_is_422(self):
        client = TestClient(make_app(FakeService()))
        assert client.post("/tts/sse", json={"speaker": "no text"}).status_code == 422
        both = {"text": "hi", "messages": [{"speaker": "a", "text": "hi"}]}
        assert client.post("/tts/sse", json=both).status_code == 422

    def test_dialogue_passes_through(self):
        svc = FakeService()
        turns = [{"speaker": "a", "text": "hi"}, {"speaker": "b", "text": "yo"}]
        _controls, frames = drain_sse(
            TestClient(make_app(svc)).post("/tts/sse", json={"messages": turns})
        )
        assert svc.seen_req["messages"] == turns
        assert frames == [FRAME] * N_FRAMES


def test_sse_is_incremental_not_buffered():
    """The point of the endpoint is time-to-first-audio.

    TestClient hands back a complete body, so no test above can tell a stream
    from a buffered join. Drive the generator directly instead: ``start`` must
    arrive before synthesis has produced anything, and each frame as it comes.
    """
    import asyncio

    produced: list[str] = []

    class SlowService:
        _writer = None

        def synthesize(self, req):
            async def gen():
                for _ in range(N_FRAMES):
                    produced.append("frame")
                    yield FRAME

            return gen()

    async def drive():
        seen = []
        async for chunk in _sse_synthesis(SlowService(), {"text": "hi"}):
            seen.append((len(produced), chunk.split(b"\n")[0]))
        return seen

    seen = asyncio.run(drive())
    # start ships with nothing synthesised yet; frame N ships after exactly N frames exist
    assert seen[0] == (0, b"event: start")
    assert [n for n, _ in seen[1 : 1 + N_FRAMES]] == list(range(1, N_FRAMES + 1))
    assert seen[-1][1] == b"event: end"
