"""In-process SNAC decode for the serving replica.

Two pieces, both living inside one replica process (co-located with the vLLM
engine on the same GPU/CUDA context, so window jobs never cross a process
boundary — no Ray serialization):

- ``InProcessSnacDecoder``: compiled ``snac.decode`` (reduce-overhead CUDA graph
  at a fixed batch), wrapping ``bodhan_genai.tts.codec.snac.decode_window_batch``.
- ``SnacMicroBatcher``: an async background task that collects 4-frame window
  jobs from all in-flight requests (time-or-size flush), decodes one padded
  ndarray batch, and scatters the resulting int16 frames back to each request's
  output queue **in emit_index order**, finishing with an ``END`` sentinel.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from bodhan_genai.tts.codec.snac import SNAC_WINDOW_TOKENS
from bodhan_genai.tts.serving.windower import WindowJob

logger = logging.getLogger("serving.snac_streamer")


class _EndSentinel:
    __slots__ = ()


END = _EndSentinel()  # pushed to a request's out-queue after its last frame


class _Finalize(NamedTuple):
    request_id: str
    total_emits: int


class _Stop:
    __slots__ = ()


_STOP = _Stop()


class InProcessSnacDecoder:
    """Compiled SNAC decode at a fixed CUDA-graph batch ``B``. ``decode`` accepts
    an ``(n, 28)`` int32 batch of raw-code windows (n may be < or > B), pads each
    chunk up to ``B`` so the captured graph is reused, and returns ``(n, spf)``
    int16 (spf = samples per frame, 2048 @ 24 kHz — the window's middle frame)."""

    def __init__(
        self,
        snac_model_path: str,
        cudagraph_batch: int = 32,
        compile_mode: str = "reduce-overhead",
        device: str = "cuda",
        window_frames: int = SNAC_WINDOW_TOKENS // 7,
    ):
        # Per-pid compile caches: concurrent replicas compiling snac.decode into a
        # shared ~/.cache corrupts the inductor cache ("pickle data was truncated").
        pid = os.getpid()
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_snac_{pid}")
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_snac_{pid}")
        import torch

        from bodhan_genai.tts.codec.snac import (
            SNAC_NUM_CODEBOOKS,
            decode_window_batch,
            load_snac_model,
        )

        self._torch = torch
        self._decode_window_batch = decode_window_batch
        self._device = device
        self._B = max(1, int(cudagraph_batch))
        self._WT = max(2, int(window_frames)) * SNAC_NUM_CODEBOOKS  # codes per window (W*7)

        snac = load_snac_model(snac_model_path, device=device, compile_model=False)
        self._eager = snac.decode
        self._decode_fn = snac.decode
        if compile_mode and str(device).startswith("cuda"):
            try:
                self._decode_fn = torch.compile(snac.decode, mode=compile_mode)
            except Exception:
                self._decode_fn = self._eager
        # Warm the fixed-shape graph; fall back to eager if the compiled path errors.
        try:
            for _ in range(3):
                self._decode_window_batch(
                    self._decode_fn, np.zeros((self._B, self._WT), dtype=np.int32), device=device
                )
        except Exception:
            self._decode_fn = self._eager
            self._decode_window_batch(
                self._decode_fn, np.zeros((self._B, self._WT), dtype=np.int32), device=device
            )

    @property
    def batch_size(self) -> int:
        return self._B

    def decode(self, window_codes: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(window_codes, dtype=np.int32)
        n = arr.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.int16)
        chunks = []
        for s in range(0, n, self._B):
            chunk = arr[s : s + self._B]
            m = chunk.shape[0]
            if m < self._B:
                padded = np.zeros((self._B, self._WT), dtype=np.int32)
                padded[:m] = chunk
            else:
                padded = chunk
            samp = self._decode_window_batch(self._decode_fn, padded, device=self._device)
            chunks.append(samp[:m])
        return np.concatenate(chunks, axis=0)


class SnacMicroBatcher:
    """Async window-batcher. One per replica. All in-flight requests submit window
    jobs into a single queue; a background task batches + decodes + scatters."""

    def __init__(
        self,
        decoder,
        flush_interval_s: float = 0.004,
        run_in_executor: bool = True,
        max_consecutive_decode_failures: int = 3,
    ):
        self._decoder = decoder
        self._flush = float(flush_interval_s)
        self._B = decoder.batch_size
        self._run_in_executor = run_in_executor
        self._max_decode_failures = int(max_consecutive_decode_failures)
        self._decode_failures = 0
        self._in: asyncio.Queue = asyncio.Queue()
        self._out: dict[str, asyncio.Queue] = {}
        self._next: dict[str, int] = {}
        self._buf: dict[str, dict[int, bytes]] = {}
        self._total: dict[str, int | None] = {}
        self._overflow: dict[str, Callable[[str], None] | None] = {}
        self._closed: set[str] = set()
        self._task: asyncio.Task | None = None
        # Requests that ended abnormally (decode crash, overflow abort, engine
        # death). The consumer checks + discards after its read loop ends so the
        # WS handler can send an error frame instead of a silent "success".
        self.failed: set[str] = set()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        await self._in.put(_STOP)
        if self._task is not None:
            await self._task
            self._task = None

    def is_closed(self, request_id: str) -> bool:
        """True once the batcher will never deliver to this request again
        (cleanup ran). Lets the consumer escape `out_q.get()` even when the END
        sentinel could not be enqueued (full queue at overflow/completion)."""
        return request_id not in self._out

    def task_dead(self) -> bool:
        """True if the background batch loop has died (e.g. persistent decoder
        failure) — surfaced via the replica's check_health."""
        return self._task is not None and self._task.done()

    def register(
        self,
        request_id: str,
        out_q: asyncio.Queue,
        on_overflow: Callable[[str], None] | None = None,
    ) -> None:
        self._out[request_id] = out_q
        self._next[request_id] = 0
        self._buf[request_id] = {}
        self._total[request_id] = None
        self._overflow[request_id] = on_overflow
        self._closed.discard(request_id)

    async def submit(self, job: WindowJob) -> None:
        await self._in.put(job)

    async def finalize(self, request_id: str, total_emits: int) -> None:
        await self._in.put(_Finalize(request_id, total_emits))

    def _cleanup(self, rid: str) -> None:
        for d in (self._out, self._next, self._buf, self._total, self._overflow):
            d.pop(rid, None)
        self._closed.discard(rid)

    def _drain_out(self, rid: str) -> None:
        """Deliver contiguous decoded frames in emit_index order; push END once
        all `total` frames are out. Bounded out-queue -> overflow policy."""
        if rid in self._closed or rid not in self._out:
            return
        out_q, buf = self._out[rid], self._buf[rid]
        while self._next[rid] in buf:
            frame = buf.pop(self._next[rid])
            try:
                out_q.put_nowait(frame)
            except asyncio.QueueFull:
                # Aborted for backpressure: mark failed so the consumer (which
                # escapes via is_closed even though END can't fit in the full
                # queue) reports an error instead of a truncated "success".
                self.failed.add(rid)
                cb = self._overflow.get(rid)
                if cb is not None:
                    cb(rid)
                self._closed.add(rid)
                self._cleanup(rid)
                return
            self._next[rid] += 1
        total = self._total[rid]
        if total is not None and self._next[rid] >= total:
            with contextlib.suppress(asyncio.QueueFull):
                out_q.put_nowait(END)
            self._cleanup(rid)

    async def _collect_batch(self):
        item = await self._in.get()
        batch, finals, stop = [], [], False
        if isinstance(item, WindowJob):
            batch.append(item)
        elif isinstance(item, _Finalize):
            finals.append(item)
        elif item is _STOP:
            return batch, finals, True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._flush
        while len(batch) < self._B:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(self._in.get(), timeout)
            except TimeoutError:
                break
            if isinstance(nxt, WindowJob):
                batch.append(nxt)
            elif isinstance(nxt, _Finalize):
                finals.append(nxt)
            elif nxt is _STOP:
                stop = True
                break
        return batch, finals, stop

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch, finals, stop = await self._collect_batch()
            if batch:
                # Only decode jobs whose request is still live.
                live = [
                    j
                    for j in batch
                    if j.request_id in self._out and j.request_id not in self._closed
                ]
                if live:
                    arr = np.stack([j.codes for j in live])
                    try:
                        if self._run_in_executor:
                            samples = await loop.run_in_executor(None, self._decoder.decode, arr)
                        else:
                            samples = self._decoder.decode(arr)
                    except Exception:
                        # Fail this batch's requests (consumer reports an error,
                        # client retries) but keep serving others. A persistent
                        # failure (e.g. corrupted CUDA context) kills the task,
                        # which check_health surfaces -> Serve restarts the replica.
                        self._decode_failures += 1
                        logger.exception(
                            "[SnacMicroBatcher] decode failed (%d consecutive)",
                            self._decode_failures,
                        )
                        for j in live:
                            self.failed.add(j.request_id)
                            self._closed.add(j.request_id)
                            self._cleanup(j.request_id)
                        if self._decode_failures >= self._max_decode_failures:
                            raise
                    else:
                        self._decode_failures = 0
                        touched = set()
                        for i, j in enumerate(live):
                            if j.request_id in self._buf:
                                self._buf[j.request_id][j.emit_index] = samples[i].tobytes()
                                touched.add(j.request_id)
                        for rid in touched:
                            self._drain_out(rid)
            for fm in finals:
                if fm.request_id in self._total:
                    self._total[fm.request_id] = fm.total_emits
                    self._drain_out(fm.request_id)
            if stop:
                return
