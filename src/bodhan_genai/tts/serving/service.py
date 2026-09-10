"""Merged streaming TTS deployment: one Ray Serve deployment that is BOTH the
ingress AND the GPU worker (vLLM AsyncLLM + in-process SNAC). Handlers run in
the replica process, so decoded PCM goes engine -> SNAC -> transport with
**no** inter-deployment Serve-streaming hop (the relay funnel that capped
concurrency). Serve's HTTP proxy load-balances / admits incoming connections
across replicas (one per GPU).

One server, three synthesis endpoints sharing ONE engine:

  WS   /tts          — live streaming (per-request ``chunked`` flag honored)
  WS   /tts/chunked  — long-form chunked streaming (chunked routing forced)
  POST /tts/sse      — the same live stream over plain HTTP (Server-Sent
                       Events, base64 audio) for callers that cannot hold a
                       websocket open
  POST /tts/offline  — complete utterance: JSON SynthesisRequest -> audio/wav
                       (server-side accumulation of the same stream; the
                       ``chunked`` flag selects long-form accumulation)

Optional `--record_dir`: full-utterance WAVs are written by a BACKGROUND process
pool, off the streaming hot path — never on the timed path.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from bodhan_genai._serving_auth import install_basic_auth as _install_basic_auth_shared
from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE, pcm16_to_wav_bytes
from bodhan_genai.tts.serving.protocol import (
    SynthesisRequest,
    audio_frame,
    end_frame,
    error_frame,
    sse,
    start_frame,
)
from bodhan_genai.tts.serving.replica import TtsReplica

logger = logging.getLogger("serving.service")
fastapi_app = FastAPI(title="bodhan-genai streaming server")


def _write_wav_job(path: str, pcm: bytes) -> None:
    """Runs in a background writer process (not the replica event loop)."""
    import numpy as np

    from bodhan_genai.tts.inference.audio_io import write_wav_24k

    write_wav_24k(path, np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0)


def _submit_recording(service, pcm: bytes) -> None:
    """Fire-and-forget background WAV recording (off the hot path)."""
    if service._writer is None or not pcm:
        return
    service._rec_counter += 1
    path = str(Path(service._record_dir) / f"utt_{service._rec_counter:08d}.wav")
    try:
        service._writer.submit(_write_wav_job, path, pcm)
    except Exception:
        logger.warning("[TtsService] background record submit failed", exc_info=True)


async def _ws_synthesis(service, ws: WebSocket, force_chunked: bool | None) -> None:
    """Shared websocket handler for /tts (flag-routed) and /tts/chunked
    (``force_chunked=True``). Module-level so tests can drive it with a fake
    service and a plain FastAPI app — no Ray Serve required."""
    await ws.accept()
    try:
        msg = await ws.receive_json()
    except Exception:
        await ws.close(code=1003)
        return
    try:
        req = SynthesisRequest(**msg).model_dump()
    except Exception as e:
        try:
            await ws.send_text(error_frame(f"bad request: {e}"))
            await ws.close(code=1003)
        except Exception:
            pass
        return
    if force_chunked is not None:
        req["chunked"] = force_chunked

    record = service._writer is not None
    chunks: list[bytes] = []
    total_samples = 0
    ok = False
    try:
        await ws.send_text(start_frame())
        async for frame in service.synthesize(req):
            await ws.send_bytes(frame)
            total_samples += len(frame) // 2
            if record:
                chunks.append(frame)
        await ws.send_text(end_frame(total_samples / SNAC_SAMPLE_RATE, total_samples // 2048))
        await ws.close()
        ok = True
    except Exception as e:
        try:
            await ws.send_text(error_frame(str(e)))
            await ws.close(code=1011)
        except Exception:
            pass
    if ok and chunks:
        _submit_recording(service, b"".join(chunks))


async def _sse_synthesis(service, req: dict):
    """The websocket stream, re-framed as Server-Sent Events.

    Module-level and generator-shaped for the same reason as ``_ws_synthesis``:
    tests drive it with a fake service and no Ray Serve. It reuses the same
    control-frame builders, so the two transports cannot drift apart.

    A failure after the first event is an ``error`` event, never a status --
    ``StreamingResponse`` commits 200 as soon as anything ships. Nothing is
    re-raised: a raised exception mid-body just severs the connection, which
    tells the client only that something ended.
    """
    record = service._writer is not None
    chunks: list[bytes] = []
    total_samples = 0
    yield sse("start", start_frame())
    try:
        seq = 0
        async for frame in service.synthesize(req):
            yield sse("audio", audio_frame(seq, frame))
            seq += 1
            total_samples += len(frame) // 2
            if record:
                chunks.append(frame)
    except Exception as e:
        logger.error("[TtsService] sse synthesis failed: %s", e)
        yield sse("error", error_frame(str(e)))
        return
    yield sse("end", end_frame(total_samples / SNAC_SAMPLE_RATE, total_samples // 2048))
    if chunks:
        _submit_recording(service, b"".join(chunks))


async def _offline_synthesis(service, req: dict) -> tuple[bytes, int]:
    """Drain the (possibly chunked) synthesis stream into one PCM buffer.
    Exceptions propagate to the route, which maps them to a JSON 500."""
    chunks: list[bytes] = []
    total_samples = 0
    async for frame in service.synthesize(req):
        chunks.append(frame)
        total_samples += len(frame) // 2
    return b"".join(chunks), total_samples


class TtsService(TtsReplica):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._record_dir = (getattr(cfg, "record_dir", "") or "").strip()
        self._writer = None
        self._rec_counter = 0
        if self._record_dir:
            Path(self._record_dir).mkdir(parents=True, exist_ok=True)
            self._writer = ProcessPoolExecutor(max_workers=2)
            logger.info("[TtsService] background WAV recording -> %s", self._record_dir)

    @fastapi_app.get("/health")
    async def health(self):
        # Served by a ready replica (Serve only routes to initialized replicas).
        return {"status": "ok"}

    @fastapi_app.websocket("/tts")
    async def tts(self, ws: WebSocket):
        await _ws_synthesis(self, ws, force_chunked=None)

    @fastapi_app.websocket("/tts/chunked")
    async def tts_chunked(self, ws: WebSocket):
        await _ws_synthesis(self, ws, force_chunked=True)

    @fastapi_app.post("/tts/sse")
    async def tts_sse(self, request: SynthesisRequest):
        return StreamingResponse(
            _sse_synthesis(self, request.model_dump()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # nginx and friends buffer a proxied response by default, which turns a
                # streaming endpoint into a slow offline one with no error anywhere.
                "X-Accel-Buffering": "no",
                "X-Sample-Rate": str(SNAC_SAMPLE_RATE),
            },
        )

    @fastapi_app.post("/tts/offline")
    async def tts_offline(self, request: SynthesisRequest):
        req = request.model_dump()
        try:
            pcm, total_samples = await _offline_synthesis(self, req)
        except Exception as e:
            logger.error("[TtsService] offline synthesis failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
        # WAV containerization off the event loop (tens of MB for long text).
        wav = await asyncio.get_running_loop().run_in_executor(None, pcm16_to_wav_bytes, pcm)
        _submit_recording(self, pcm)
        return Response(
            content=wav,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration-S": f"{total_samples / SNAC_SAMPLE_RATE:.3f}",
                "X-Sample-Rate": str(SNAC_SAMPLE_RATE),
            },
        )


def _install_basic_auth(app) -> None:
    """Install HTTP Basic auth if TTS_AUTH_FILE is set; open otherwise.

    The websocket endpoints are the reason this is raw-ASGI middleware rather than a FastAPI
    dependency: a BaseHTTPMiddleware subclass only sees ``http`` scopes, so ``WS /tts`` would
    sail straight past it.
    """
    _install_basic_auth_shared(app, prefix="TTS", realm="IndicSpeak")


def build_deployment():
    """Apply the Ray Serve decorators lazily.

    ``from ray import serve`` costs ~9 s and drags the whole ray stack in;
    keeping it out of module scope lets tests and tooling import this module
    (handlers, FastAPI app, TtsService class) instantly and CPU-only. The
    FastAPI routes are already registered on ``fastapi_app`` at class-body
    evaluation — Serve only needs to wrap the class for ingress dispatch.

    Auth is installed here and NOT at module scope: importing this module must stay free, or
    every reader of it breaks for a check that only matters when something is served. It fails
    closed — see :func:`bodhan_genai._serving_auth.install_basic_auth`."""
    from ray import serve

    _install_basic_auth(fastapi_app)
    return serve.deployment(serve.ingress(fastapi_app)(TtsService))
