"""TtsReplica: one GPU's worth of streaming TTS = vLLM AsyncLLM (in-process) +
in-process SNAC decoder + SNAC micro-batcher, co-located on a single GPU.

Now a thin adapter over ``bodhan_genai.tts.engine.streaming.IndicStreamingTTSEngine``
(the near-verbatim extraction of this replica's previous internals) — this class
only maps ServeConfig / request dicts onto the engine API.

Plain async class (no Serve decorator) so it can be driven by a local harness for
testing; ``serving/service.py`` wraps it as a Ray Serve deployment (one replica per
GPU). ``synthesize`` is an async generator yielding raw int16 PCM frames.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncGenerator

logger = logging.getLogger("serving.replica")


def _build_engine(cfg):
    """Map a ServeConfig onto the engine ctor. ``engine_kwargs=cfg.engine_kwargs()``
    hands over the FULL AsyncEngineArgs kwargs dict (chunked prefill, quantization,
    kv-cache dtype, ...) exactly as the replica always built them."""
    from bodhan_genai.tts.engine.streaming import IndicStreamingTTSEngine
    from bodhan_genai.tts.engine.types import SamplingConfig

    decoder_factory = None
    if getattr(cfg, "snac_topology", "colocated") == "pooled":

        def decoder_factory():
            # EXPERIMENTAL: decode on the shared SNAC actor pool (dedicated GPU)
            # instead of in-process. The micro-batcher is unchanged; only the
            # decode call goes remote (sync ray.get inside the executor thread).
            import ray

            from bodhan_genai.tts.serving.snac_pool import (
                SNAC_POOL_NAMESPACE,
                RemoteSnacDecoder,
                pool_actor_name,
            )

            idx = os.getpid() % max(1, int(cfg.num_snac_actors))
            handle = ray.get_actor(pool_actor_name(idx), namespace=SNAC_POOL_NAMESPACE)
            logger.info("[TtsReplica] pooled SNAC: bound to %s", pool_actor_name(idx))
            return RemoteSnacDecoder(handle, cfg.snac_cudagraph_batch)

    return IndicStreamingTTSEngine(
        cfg.checkpoint_path,
        tokenizer=(cfg.tokenizer_path or None),
        snac_model_path=cfg.snac_model_path,
        sampling=SamplingConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=getattr(cfg, "repetition_penalty", 1.0),
            max_new_tokens=cfg.max_new_tokens,
        ),
        engine_kwargs=cfg.engine_kwargs(),
        snac_cudagraph_batch=cfg.snac_cudagraph_batch,
        snac_window_frames=cfg.snac_window_frames,
        snac_compile_mode=cfg.snac_compile_mode,
        snac_flush_interval_ms=cfg.snac_flush_interval_ms,
        per_request_queue_max=cfg.per_request_queue_max,
        decoder_factory=decoder_factory,
    )


class TtsReplica:
    def __init__(self, cfg):
        self.cfg = cfg
        self._engine = _build_engine(cfg)
        from bodhan_genai.tts.engine.chunked import ChunkedIndicStreamingTTS

        self._chunked = ChunkedIndicStreamingTTS(
            self._engine,
            min_chunk_chars=int(getattr(cfg, "chunk_min_chars", 16)),
            max_chunk_chars=int(getattr(cfg, "chunk_max_chars", 300)),
            first_chunk_chars=(int(getattr(cfg, "chunk_first_chars", 120)) or None),
            gap_ms=float(getattr(cfg, "chunk_gap_ms", 250.0)),
            target_lufs=float(getattr(cfg, "chunk_target_lufs", -23.0)),
            peak_dbfs=float(getattr(cfg, "chunk_peak_dbfs", -1.0)),
            trim_db=float(getattr(cfg, "chunk_trim_db", 30.0)),
            # Pay librosa/pyloudnorm lazy-init (~20 s first librosa trim) HERE,
            # before the replica reports ready — never on the event loop.
            warmup=True,
        )
        logger.info("[TtsReplica] ready.")

    async def ready(self) -> bool:
        """Liveness probe — only returns once __init__ finished (engine + SNAC
        loaded), so the ingress can gate /health on real replica readiness."""
        return True

    async def check_health(self) -> None:
        """Ray Serve periodic health check. Raising marks the replica unhealthy
        (after REPLICA_HEALTH_CHECK_UNHEALTHY_THRESHOLD=3 consecutive failures,
        hardcoded in Serve) -> Serve tears it down and starts a fresh one (new
        process, new EngineCore). Delegates to the engine, which covers both the
        vLLM EngineCore and the SNAC micro-batcher task."""
        self._engine.check_health()

    async def __del__(self):
        # Serve's graceful teardown awaits an async __del__ before the final
        # ray.kill (SIGKILL). Without this nothing SIGTERMs the EngineCore
        # SUBPROCESS: ray.kill delivers no signal to children, and a surviving
        # EngineCore keeps holding gmu*VRAM, OOM-looping the replacement replica.
        with contextlib.suppress(Exception):
            await self.shutdown()

    def synthesize(self, req: dict) -> AsyncGenerator[bytes, None]:
        """Yield raw int16 PCM frames (2048 samples each) for one request.

        Dialogue requests (``messages`` list of ``{"speaker","text"}`` turns)
        route to the engine's conversation renderers instead of the plain
        text stream.

        Returns the engine's stream generator DIRECTLY (no wrapping generator):
        with a stacked generator, ``aclose()`` on the outer one defers the
        inner finally (drive cancel + vLLM abort) to a GC-scheduled task; the
        direct return keeps that cleanup inline, matching the pre-refactor
        replica. ``async for`` call sites are unaffected.

        Sampling overrides are passed None-defaulted so the engine's
        ``SamplingConfig.merged`` reproduces the old per-field pick() semantics
        (absent/None request field -> server default)."""

        def _opt(name, cast):
            v = req.get(name)
            return None if v is None else cast(v)

        kwargs = dict(
            speaker=req.get("speaker", ""),
            temperature=_opt("temperature", float),
            top_p=_opt("top_p", float),
            top_k=_opt("top_k", int),
            max_new_tokens=_opt("max_new_tokens", int),
            frames_per_message=self.cfg.frames_per_message,
        )
        # Long-form chunked mode: per-request {"chunked": true/false} overrides
        # the server default. None-sentinel, NOT req.get(..., default): the WS
        # handler round-trips through SynthesisRequest, whose model_dump()
        # carries chunked=None when the client omitted it.
        chunked_flag = req.get("chunked")
        use_chunked = (
            bool(getattr(self.cfg, "chunked_default", False))
            if chunked_flag is None
            else bool(chunked_flag)
        )
        msgs = req.get("messages")
        if msgs:
            # Dialogue: speakers ride inline per turn — the top-level speaker
            # kwarg does not apply to conversation rendering.
            conv_kwargs = {k: v for k, v in kwargs.items() if k != "speaker"}
            turns = [dict(m) for m in msgs]
            if use_chunked:
                return self._chunked.stream_conversation_long(turns, **conv_kwargs)
            return self._engine.stream_conversation(turns, **conv_kwargs)
        if use_chunked:
            return self._chunked.stream_long(req.get("text", ""), **kwargs)
        return self._engine.stream(req.get("text", ""), **kwargs)

    async def shutdown(self) -> None:
        await self._engine.shutdown()
