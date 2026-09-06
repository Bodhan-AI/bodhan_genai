"""Merged streaming TTS deployment: one Ray Serve deployment that is BOTH the
ingress AND the GPU worker (vLLM AsyncLLM + in-process SNAC). Handlers run in
the replica process, so decoded PCM goes engine -> SNAC -> transport with
**no** inter-deployment Serve-streaming hop (the relay funnel that capped
concurrency). Serve's HTTP proxy load-balances / admits incoming connections
across replicas (one per GPU).

One server, three synthesis endpoints sharing ONE engine:

  WS   /tts          — live streaming (per-request ``chunked`` flag honored)
  WS   /tts/chunked  — long-form chunked streaming (chunked routing forced)
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
from fastapi.responses import JSONResponse

from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE, pcm16_to_wav_bytes
from bodhan_genai.tts.serving.protocol import SynthesisRequest, end_frame, error_frame, start_frame
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


def build_deployment():
    """Apply the Ray Serve decorators lazily.

    ``from ray import serve`` costs ~9 s and drags the whole ray stack in;
    keeping it out of module scope lets tests and tooling import this module
    (handlers, FastAPI app, TtsService class) instantly and CPU-only. The
    FastAPI routes are already registered on ``fastapi_app`` at class-body
    evaluation — Serve only needs to wrap the class for ingress dispatch."""
    from ray import serve

    return serve.deployment(serve.ingress(fastapi_app)(TtsService))
