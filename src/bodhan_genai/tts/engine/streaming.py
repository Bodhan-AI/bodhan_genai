"""IndicStreamingTTSEngine: one GPU's worth of streaming TTS = vLLM AsyncLLM
(in-process) + in-process SNAC decoder + SNAC micro-batcher, co-located on a
single GPU.

Extracted near-verbatim from ``serving/replica.py`` so the same hot path serves
both the Ray Serve replica (``TtsReplica`` is now a thin adapter over this
class) and library users: async ``stream`` yields raw int16 PCM frames;
``stream_sync`` is a blocking convenience wrapper for scripts/notebooks.

Light at import time: heavy deps (torch / vllm / transformers / snac — and the
serving modules that import them transitively) stay inside methods, so this
module imports clean on a CPU-only box.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue as _queue
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Any

from bodhan_genai.tts.engine.types import SamplingConfig

logger = logging.getLogger("engine.streaming")

DEFAULT_SNAC_MODEL = "hubertsiuzdak/snac_24khz"


def _default_vllm_request(input_ids: list[int], sc: SamplingConfig, stop_ids: list[int]):
    """Default request seam: map ``(input_ids, SamplingConfig, stop ids)`` onto
    vLLM's ``(TokensPrompt, SamplingParams)``. vllm imports live inside the
    function so tests can inject a ``request_factory`` returning plain tuples
    without ever importing vllm."""
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    return (
        TokensPrompt(prompt_token_ids=list(input_ids)),
        SamplingParams(
            temperature=float(sc.temperature),
            top_p=float(sc.top_p),
            top_k=int(sc.top_k),
            repetition_penalty=float(sc.repetition_penalty),
            max_tokens=int(sc.max_new_tokens),
            stop_token_ids=list(stop_ids),
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
        ),
    )


class _SyncStreamBridge:
    """Blocking facade used by ``stream_sync``: ONE persistent daemon thread
    running a private event loop, created on first use and reused for the
    engine's lifetime. Each call drives one async generator on that loop via a
    pump coroutine that pushes frames into a thread-safe ``queue.Queue``; a
    sentinel tuple carries completion (or the raised exception) back to the
    caller. Closing the returned generator cancels the pump via
    ``call_soon_threadsafe``."""

    _SENTINEL = object()

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The private loop, or None if the bridge was never used / was stopped."""
        if self._thread is not None and self._thread.is_alive():
            return self._loop
        return None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name="bodhan-tts-stream-sync", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def run(self, agen_factory: Callable[[], AsyncGenerator[bytes, None]]) -> Iterator[bytes]:
        """Drive ``agen_factory()`` on the bridge loop; yield its items
        synchronously. One stream at a time (convenience API)."""
        loop = self._ensure_loop()
        out: _queue.Queue = _queue.Queue()
        task_box: dict[str, asyncio.Task] = {}

        async def _pump() -> None:
            err: BaseException | None = None
            try:
                async for item in agen_factory():
                    out.put(item)
            except BaseException as e:
                err = e
            out.put((self._SENTINEL, err))

        def _start() -> None:
            task_box["task"] = loop.create_task(_pump())

        loop.call_soon_threadsafe(_start)
        try:
            while True:
                item = out.get()
                if isinstance(item, tuple) and len(item) == 2 and item[0] is self._SENTINEL:
                    err = item[1]
                    if err is not None and not isinstance(err, asyncio.CancelledError):
                        raise err
                    return
                yield item
        finally:
            # generator.close() / early break: cancel the in-flight pump so the
            # engine's `finally` (drive cancel + abort) runs on the bridge loop.
            task = task_box.get("task")
            if task is not None and not task.done():
                loop.call_soon_threadsafe(task.cancel)

    def stop(self) -> None:
        """Stop and join the bridge loop thread (no-op if never started)."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None:
            return
        if thread is not None and thread.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)
        # RuntimeError: join timed out and the loop is still running
        with contextlib.suppress(RuntimeError):
            loop.close()


class IndicStreamingTTSEngine:
    """Streaming TTS engine: text -> raw int16 PCM frames
    (2048 samples each @ 24 kHz).

    The ``*_factory`` / ``*_loader`` kwargs are dependency-injection seams
    (used by the serving replica for pooled SNAC decode and by tests to run
    the real windower + micro-batcher without vllm / a GPU)."""

    def __init__(
        self,
        model: str = "bodhan-ai/indic-speak",
        *,
        tokenizer: str | None = None,
        snac_model_path: str = DEFAULT_SNAC_MODEL,
        sampling: SamplingConfig | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 256,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        seed: int = 0,
        engine_kwargs: dict | None = None,
        snac_cudagraph_batch: int = 32,
        snac_window_frames: int = 3,
        snac_compile_mode: str = "reduce-overhead",
        snac_flush_interval_ms: float = 4.0,
        per_request_queue_max: int = 64,
        llm_factory: Callable[[dict], Any] | None = None,
        decoder_factory: Callable[[], Any] | None = None,
        request_factory: Callable[[list[int], SamplingConfig, list[int]], tuple] | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        # NOTE: AsyncLLM in vLLM 0.19 ALWAYS runs its EngineCore as a subprocess
        # (VLLM_ENABLE_V1_MULTIPROCESSING only affects the sync LLMEngine), so the
        # LLM and the in-process SNAC are separate CUDA contexts on the same GPU.
        # Per-pid compile caches set by InProcessSnacDecoder.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        self._model = model
        self._snac_window_frames = int(snac_window_frames)
        self._per_request_queue_max = int(per_request_queue_max)
        self._sampling = sampling if sampling is not None else SamplingConfig()
        self._request_factory = request_factory or _default_vllm_request
        self._counter = 0
        self._started = False
        self._engine_dead = False
        self._shutdown_done = False
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._bg_tasks: set[asyncio.Task] = set()  # keeps fire-and-forget aborts alive
        self._bridge = _SyncStreamBridge()

        tok_src = tokenizer if tokenizer is not None else model
        if isinstance(tok_src, (str, os.PathLike)):
            if tokenizer_loader is not None:
                self._tokenizer = tokenizer_loader(str(tok_src))
            else:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(str(tok_src))
        else:
            self._tokenizer = tok_src  # already-instantiated tokenizer object

        from bodhan_genai.tts.inference.prompts import resolve_snac_ids
        from bodhan_genai.tts.templates.chat import get_template_ids

        self._tmpl = get_template_ids(self._tokenizer)
        self._snac_ids = resolve_snac_ids(self._tokenizer)

        # --- vLLM AsyncLLM ---------------------------------------------------
        if engine_kwargs is not None:
            # FULL AsyncEngineArgs kwargs override (the serving replica passes
            # cfg.engine_kwargs() so every server-side engine knob survives).
            kw = dict(engine_kwargs)
        else:
            kw = dict(
                model=model,
                tokenizer=str(tok_src) if isinstance(tok_src, (str, os.PathLike)) else model,
                dtype=dtype,
                gpu_memory_utilization=float(gpu_memory_utilization),
                max_model_len=int(max_model_len),
                max_num_seqs=int(max_num_seqs),
                enforce_eager=bool(enforce_eager),
                disable_log_stats=True,
                trust_remote_code=True,
                seed=int(seed),
            )
        if llm_factory is not None:
            self._llm = llm_factory(kw)
        else:
            # Before the import: vLLM >= 0.26 samples through a flashinfer kernel it
            # JIT-compiles at engine warm-up, which needs nvcc. Nodes with a runtime-only
            # CUDA install have none, and the build failure surfaces as the generic
            # "EngineCore failed to start". Streaming TTS samples on every token
            # (temperature 0.6 / top_p 0.95), so this path always reaches the sampler.
            # Override by exporting VLLM_USE_FLASHINFER_SAMPLER=1 where nvcc exists.
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            # Same reason, different kernel: vLLM probes vllm.third_party.deep_gemm, whose import
            # asserts on _find_cuda_home(). Without nvcc that assertion fails and vLLM logs a
            # twenty-line traceback ending in a bare AssertionError, then carries on -- so a
            # perfectly healthy engine start looks like a crash. IndicOCR's recognizer has always
            # set this; the TTS and MT engines did not, which made the noise look modality-specific.
            os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

            from vllm import AsyncEngineArgs

            try:
                from vllm.v1.engine.async_llm import AsyncLLM
            except ImportError:
                from vllm import AsyncLLM
            logger.info(
                "[IndicStreamingTTSEngine] building AsyncLLM (gmu=%s) ...",
                kw.get("gpu_memory_utilization", "?"),
            )
            self._llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))

        # --- SNAC decoder + micro-batcher -------------------------------------
        from bodhan_genai.tts.serving.snac_streamer import InProcessSnacDecoder, SnacMicroBatcher

        if decoder_factory is not None:
            # Injection seam for alternative decode backends (the serving
            # replica passes the pooled RemoteSnacDecoder through this).
            self._decoder = decoder_factory()
        else:
            logger.info("[IndicStreamingTTSEngine] loading + compiling in-process SNAC ...")
            self._decoder = InProcessSnacDecoder(
                snac_model_path,
                cudagraph_batch=snac_cudagraph_batch,
                compile_mode=(snac_compile_mode or ""),
                device="cuda",
                window_frames=snac_window_frames,
            )
        self._batcher = SnacMicroBatcher(
            self._decoder, flush_interval_s=snac_flush_interval_ms / 1000.0
        )
        logger.info("[IndicStreamingTTSEngine] ready.")

    # -- health / lifecycle ----------------------------------------------------

    @property
    def engine_dead(self) -> bool:
        """True once the vLLM EngineCore has been detected dead (unrecoverable)."""
        return self._engine_dead

    def check_health(self) -> None:
        """Raise RuntimeError when the engine can no longer serve. Raising from
        the replica's periodic health check marks it unhealthy -> Serve tears it
        down and starts a fresh one (new process, new EngineCore). Without this,
        a crashed EngineCore leaves a zombie that keeps receiving + failing
        traffic. Also covers the SNAC batcher task: if it died, every stream
        hangs even though the vLLM engine is healthy."""
        if self._engine_dead or bool(getattr(self._llm, "errored", False)):
            raise RuntimeError("vLLM EngineCore is dead; engine must be restarted")
        if self._started and self._batcher.task_dead():
            raise RuntimeError("SNAC micro-batcher task died; engine must be restarted")

    def _ensure_started(self) -> None:
        # Start the batcher task lazily (needs a running event loop — guaranteed
        # inside stream()). The engine is then bound to that loop for its
        # lifetime: asyncio queues/tasks are not loop-portable.
        loop = asyncio.get_running_loop()
        if not self._started:
            self._batcher.start()
            self._started = True
            self._owner_loop = loop
        elif loop is not self._owner_loop:
            raise RuntimeError(
                "IndicStreamingTTSEngine is bound to another event loop; create one "
                "engine per loop (or use stream_sync, which owns a private loop)"
            )

    async def _shutdown_impl(self) -> None:
        try:
            # Only touch the batcher's asyncio primitives from the loop that owns
            # them; if that loop is already gone the task died with it.
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if self._owner_loop is None or self._owner_loop is running:
                await self._batcher.stop()
            elif not self._owner_loop.is_closed():
                # Cross-loop shutdown: dispatch the stop to the owner loop so the
                # batcher task actually exits (pre-refactor always-stop semantics)
                # without touching foreign-loop primitives from here.
                fut = asyncio.run_coroutine_threadsafe(self._batcher.stop(), self._owner_loop)
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: fut.result(timeout=5)
                )
        except Exception:
            pass
        try:
            sh = getattr(self._llm, "shutdown", None)
            if sh:
                sh()
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Stop the SNAC micro-batcher, the vLLM engine and the stream_sync
        bridge loop. Idempotent (safe to await twice)."""
        if not self._shutdown_done:
            self._shutdown_done = True
            bloop = self._bridge.loop
            if bloop is not None and self._owner_loop is bloop:
                # stream_sync-only usage: the batcher lives on the bridge loop,
                # so run the teardown there and wait from here.
                fut = asyncio.run_coroutine_threadsafe(self._shutdown_impl(), bloop)
                await asyncio.wrap_future(fut)
            else:
                await self._shutdown_impl()
        self._bridge.stop()

    async def __aenter__(self) -> IndicStreamingTTSEngine:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    # -- generation --------------------------------------------------------------

    async def _safe_abort(self, rid: str) -> None:
        with contextlib.suppress(Exception):
            await self._llm.abort(rid)

    async def _drive(self, rid: str, prompt, sp, windower) -> None:
        try:
            async for out in self._llm.generate(prompt, sp, rid):
                for job in windower.push(out.outputs[0].token_ids, finished=out.finished):
                    await self._batcher.submit(job)
                if out.finished:
                    break
            for job in windower.flush_tail():
                await self._batcher.submit(job)
            await self._batcher.finalize(rid, windower.total_emits)
        except asyncio.CancelledError:
            await self._safe_abort(rid)
            raise
        except Exception as e:
            # EngineDeadError = the EngineCore subprocess crashed (e.g. CUDA
            # illegal-memory-access). It never recovers on its own — flag it so
            # check_health fails and Serve restarts this replica with a fresh engine.
            if e.__class__.__name__ == "EngineDeadError" and not self._engine_dead:
                self._engine_dead = True
                logger.error(
                    "[IndicStreamingTTSEngine] ENGINE DEAD — failing health "
                    "checks so the owner restarts this engine"
                )
            logger.exception("[IndicStreamingTTSEngine] generation failed for %s", rid)
            # Mark failed BEFORE finalize: the consumer must report an error frame,
            # not a clean end (a truncated stream counted as success poisons the
            # dataset and is never retried by the client).
            self._batcher.failed.add(rid)
            # flush whatever we have so the consumer still gets END
            for job in windower.flush_tail():
                with contextlib.suppress(Exception):
                    await self._batcher.submit(job)
            await self._batcher.finalize(rid, windower.total_emits)

    async def stream(
        self,
        text: str,
        *,
        speaker: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> AsyncGenerator[bytes, None]:
        """Yield raw int16 PCM frames (2048 samples each) for one utterance.

        Sampling kwargs are None-defaulted overrides merged onto the engine's
        ``SamplingConfig``. The first frame ships alone (low time-to-first-audio),
        then ``frames_per_message`` frames are grouped per yielded message."""
        from bodhan_genai.tts.inference.prompts import build_prompt_ids

        self._check_not_dead()
        if not (text or "").strip():
            return

        input_ids = build_prompt_ids(text, speaker, self._tokenizer, tmpl=self._tmpl)
        if not input_ids:
            return

        async for frame in self._stream_ids(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            frames_per_message=frames_per_message,
        ):
            yield frame

    async def stream_conversation(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a multi-turn conversation as ONE continuous PCM stream.

        ``messages`` is a chat-style list of ``{"speaker": ..., "text": ...}``
        dicts rendered through the conversation chat template
        (``<|speaker>NAME<speaker|>`` tags inline, no metadata prefix). Same
        framing/backpressure semantics as ``stream``."""
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        self._check_not_dead()
        input_ids = build_conversation_prompt_ids(messages, self._tokenizer, tmpl=self._tmpl)
        if not input_ids:
            return

        async for frame in self._stream_ids(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            frames_per_message=frames_per_message,
        ):
            yield frame

    def _check_not_dead(self) -> None:
        if self._engine_dead:
            # Fail fast (client retries land on a healthy engine) instead of
            # queueing into a dead engine while the health check converges.
            raise RuntimeError("engine dead; retry")

    async def _stream_ids(
        self,
        input_ids: list[int],
        *,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repetition_penalty: float | None,
        max_new_tokens: int | None,
        frames_per_message: int,
    ) -> AsyncGenerator[bytes, None]:
        """Shared streaming pipeline: prompt ids -> AsyncLLM DELTA -> windower ->
        SNAC micro-batcher -> grouped int16 PCM frames."""
        from bodhan_genai.tts.serving.snac_streamer import END
        from bodhan_genai.tts.serving.windower import StreamingWindower

        self._ensure_started()
        self._counter += 1
        rid = f"r{self._counter}"

        sc = self._sampling.merged(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        stop_ids = [self._snac_ids["end_of_audio_id"], self._snac_ids["eos_token_id"]]
        prompt, sp = self._request_factory(input_ids, sc, stop_ids)

        out_q: asyncio.Queue = asyncio.Queue(maxsize=self._per_request_queue_max)

        def _on_overflow(r):
            # Keep a reference so the abort task can't be GC'd mid-flight.
            task = asyncio.create_task(self._safe_abort(r))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            with contextlib.suppress(asyncio.QueueFull):
                out_q.put_nowait(END)  # unblock consumer if it has space

        self._batcher.register(rid, out_q, on_overflow=_on_overflow)
        windower = StreamingWindower(rid, self._snac_ids, window_frames=self._snac_window_frames)
        drive = asyncio.create_task(self._drive(rid, prompt, sp, windower))
        group = max(1, int(frames_per_message))
        buf: list[bytes] = []
        sent_first = False
        try:
            while True:
                try:
                    item = await asyncio.wait_for(out_q.get(), timeout=0.25)
                except TimeoutError:
                    # The END sentinel is dropped when the out queue is full at
                    # overflow/completion time — without this escape the consumer
                    # (and its WS handler + admission slot) would hang forever.
                    if self._batcher.is_closed(rid):
                        break
                    continue
                if item is END:
                    if buf:
                        yield b"".join(buf)
                    break
                buf.append(item)
                # First frame ships alone (low time-to-first-audio); then group
                # `frames_per_message` frames per message to cut streaming overhead.
                if not sent_first or len(buf) >= group:
                    yield b"".join(buf)
                    buf = []
                    sent_first = True
            if rid in self._batcher.failed:
                self._batcher.failed.discard(rid)
                raise RuntimeError("stream failed mid-generation (engine/decoder error); retry")
        finally:
            self._batcher.failed.discard(rid)  # never leak entries (e.g. on disconnect)
            if not drive.done():
                drive.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drive
            await self._safe_abort(rid)

    def stream_sync(
        self,
        text: str,
        *,
        speaker: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> Iterator[bytes]:
        """Blocking convenience wrapper around ``stream`` for scripts/notebooks:
        drives the async generator on ONE persistent private event loop (daemon
        thread, created on first use and reused for the engine's lifetime) and
        yields frames synchronously. One stream at a time — for concurrency use
        the async API. ``generator.close()`` cancels the in-flight request."""
        return self._bridge.run(
            lambda: self.stream(
                text,
                speaker=speaker,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                frames_per_message=frames_per_message,
            )
        )

    def stream_conversation_sync(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> Iterator[bytes]:
        """Blocking convenience wrapper around ``stream_conversation`` — same
        bridge-loop semantics as ``stream_sync``."""
        return self._bridge.run(
            lambda: self.stream_conversation(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                frames_per_message=frames_per_message,
            )
        )
