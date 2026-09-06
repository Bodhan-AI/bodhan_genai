# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Continuous-batching slot pool for STREAMING ASR.

Replaces the fixed-batch streaming path. A fixed batch has two costs that this
removes, both measured on one H100:

  1. **It cannot be CUDA-graphed without waiting.** A graph needs a constant
     shape, so a fixed batch must either pad to a fixed size or wait to fill.
     A slot pool makes the batch dim *permanently* constant — every step runs
     ALL slots and masks the idle ones — so graphs work and admission never
     waits. Measured: 5.8-6.0 ms/step eager vs 2.5-2.7 ms graphed, a **2.3x**
     win that is nearly flat from 8 to 32 slots.
  2. **Ragged waste.** A fixed batch steps every row until the LONGEST hits
     EOS. Across a realistic cohort of streaming buffers, output lengths ran
     22-70 tokens, so ~33% of decoder steps were spent on rows that had
     already finished. A slot pool evicts them and refills. Worth ~1.28x.

The offline ``IndicTranscribeEngine`` already has the graphed all-slots step, the
layer-major KV buffers, bucketing and eviction; this subclasses it so that
gate-verified step math is reused untouched, and replaces only the *scheduler*.
What could not be reused is ``run()``: it drains a FIXED list, duration-sorts
it, and sizes KV buffers from the first batch — so a later, longer utterance is
failed outright. A server admits unknown requests forever.

**The fixed streaming window is what makes this work.** Because a session's
span is bounded (``stream_max_segment_s``), the encoder length and the length
cap are known at startup, so the buffers are allocated ONCE, statically, and
no request can ever exceed them. That was the exact constraint that made the
offline engine unusable for serving.

Threading: the scheduler owns the GPU and runs on its own thread. Sessions
submit from their own worker threads and block on a ``Future``. Graph capture
happens on the scheduler thread, which matters — CUDA graph replay is
thread-affine, and capturing on one thread to replay on another silently
corrupts output.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import torch

from bodhan_genai.asr.engine.continuous_batching import IndicTranscribeEngine, _sync

logger = logging.getLogger("asr.serving.slot_engine")


@dataclass
class _Req:
    wav: torch.Tensor
    lang: str
    future: Future
    # output mode (prompt slots 6/7); carried on the request so prompt BUILD
    # (_admit) and STRIP (_evict) always use the same flags
    itn: bool = False
    romanized: bool = False


@dataclass
class SlotStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    steps: int = 0
    occupancy_sum: float = 0.0
    # NOTE: these phase timers are only meaningful with INDIC_TRANSCRIBE_PROFILE=1,
    # which inserts device syncs. Without it a graphed decode replay is async
    # (its time lands in whatever syncs next, usually the encoder) and the
    # split is an artifact — measured 94.9% "encode" that way, which is wrong.
    encode_s: float = 0.0
    decode_s: float = 0.0
    graphs: int = 0
    replays: int = 0
    eager: int = 0

    @property
    def mean_occupancy(self) -> float:
        return self.occupancy_sum / max(1, self.steps)


@dataclass
class _Slot:
    req: _Req | None = None


class StreamingSlotEngine(IndicTranscribeEngine):
    """Continuous batching for a live stream of bounded-length decode requests.

    Args:
      max_buffer_s: the streaming window. Fixes encoder length and length cap,
                    hence the static allocation. A submitted buffer longer than
                    this is rejected rather than silently truncated.
      slots:        pool size. Graphed step cost is nearly flat in slots, so
                    this is close to free capacity until the GPU saturates.
      admit_batch:  how many pending requests to encode in one pass.
    """

    def __init__(
        self,
        *,
        model,
        feature_extractor,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_buffer_s: float = 10.0,
        slots: int = 32,
        admit_batch: int = 16,
        cuda_graphs: bool = True,
        bucket: int = 64,
        cross_bucket: int = 64,
        max_graphs: int = 48,
        evict_every: int = 1,
        idle_sleep_s: float = 0.002,
        overlap_encode: bool = True,
    ):
        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            slots=slots,
            admit_batch=admit_batch,
            cuda_graphs=cuda_graphs,
            bucket=bucket,
            cross_bucket=cross_bucket,
            max_graphs=max_graphs,
            evict_every=evict_every,
            overlap_prefill=False,  # the scheduler thread IS the producer here
        )
        self.max_buffer_s = max_buffer_s
        self.idle_sleep_s = idle_sleep_s
        # Run the encoder on a PRODUCER thread + side CUDA stream so it does not
        # block decode steps. At the measured ceiling the GPU is only ~72% busy
        # and the phases split 42/58 encode/decode, so average throughput is not
        # the limit -- BURSTS are: when many sessions come due together, an
        # inline encoder stalls every decode behind it. Upstream measured plain
        # stream overlap at only 1.02-1.06x (the phases contend for the same
        # SMs), so this is aimed at burst smoothing, not raw throughput.
        self.overlap_encode = overlap_encode and str(device).startswith("cuda")
        self.stats = SlotStats()

        # Static sizing from the fixed window -- the whole reason this works.
        n_samples = int(max_buffer_s * self.fe.sample_rate)
        mel_frames = int(self.fe.get_seq_len(torch.tensor([n_samples])).item())
        self.t_enc_max = int(
            self.model.model.encoder.pre_encode.calc_lengths(torch.tensor([mel_frames])).item()
        )
        self.l_alloc = (
            min(self.cfg.max_target_positions, self.t_enc_max + self.cfg.max_generation_delta) + 1
        )
        self._max_samples = n_samples

        self._q: queue.Queue[_Req] = queue.Queue()
        # encoded-and-staged, waiting for a free slot
        self._staged: collections.deque = collections.deque()
        self._staged_lock = threading.Lock()
        self._enc_stream = None
        self._producer: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._err: list[BaseException] = []
        self._thread = threading.Thread(target=self._scheduler, name="asr-slots", daemon=True)

    # -- lifecycle -----------------------------------------------------------

    def start(self, timeout: float = 120.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("slot engine failed to become ready")
        if self._err:
            raise self._err[0]

    def close(self) -> None:
        self._stop.set()
        if self._producer is not None and self._producer.is_alive():
            self._producer.join(timeout=30)
        if self._thread.is_alive():
            self._thread.join(timeout=30)

    # -- public API ----------------------------------------------------------

    def submit(
        self, wav: torch.Tensor, lang: str, itn: bool = False, romanized: bool = False
    ) -> Future:
        """Queue one buffer; the Future resolves with its transcript."""
        fut: Future = Future()
        if self._err:
            fut.set_exception(self._err[0])
            return fut
        if wav.numel() > self._max_samples:
            fut.set_exception(
                ValueError(
                    f"buffer {wav.numel() / self.fe.sample_rate:.1f}s exceeds the engine's "
                    f"fixed window {self.max_buffer_s:.1f}s; the pool is statically sized "
                    "for that window and cannot grow"
                )
            )
            return fut
        self.stats.submitted += 1
        self._q.put(_Req(wav, lang, fut, itn=itn, romanized=romanized))
        return fut

    def transcribe(
        self, wav: torch.Tensor, lang: str, itn: bool = False, romanized: bool = False
    ) -> str:
        """Blocking convenience wrapper for a caller already on a worker thread."""
        return self.submit(wav, lang, itn=itn, romanized=romanized).result()

    # -- scheduler -----------------------------------------------------------

    def _alloc(self) -> None:
        """Allocate the slot pool ONCE. Sizes come from the fixed window, not
        from whatever happened to arrive first (the offline engine's rule),
        so nothing admitted later can outgrow them."""
        S, dev = self.n_slots, self.device
        self._self_k = torch.zeros(
            self.n_layers, S, self.h, self.l_alloc + 1, self.d_k, dtype=self.dtype, device=dev
        )
        self._self_v = torch.zeros_like(self._self_k)
        self._cross_k = torch.zeros(
            self.n_layers, S, self.h, self.t_enc_max, self.d_k, dtype=self.dtype, device=dev
        )
        self._cross_v = torch.zeros_like(self._cross_k)
        self._slot_ids = torch.full(
            (S, self.l_alloc + 2), self.cfg.pad_token_id, dtype=torch.long, device=dev
        )
        self._slot_pos = torch.zeros(S, dtype=torch.long, device=dev)
        self._slot_cap = torch.full(
            (S,), self.cfg.max_target_positions, dtype=torch.long, device=dev
        )
        self._slot_enc_len = torch.ones(S, dtype=torch.long, device=dev)
        self._alive_l = torch.zeros(S, dtype=torch.long, device=dev)
        self._alive = torch.zeros(S, dtype=torch.bool, device=dev)
        self._finished = torch.zeros(S, dtype=torch.bool, device=dev)
        self._last_tok = torch.full((S, 1), self.cfg.pad_token_id, dtype=torch.long, device=dev)
        self._buf = dict(
            self_k=self._self_k,
            self_v=self._self_v,
            cross_k=self._cross_k,
            cross_v=self._cross_v,
            slot_ids=self._slot_ids,
            slot_pos=self._slot_pos,
            last_tok=self._last_tok,
            slot_enc_len=self._slot_enc_len,
            alive_l=self._alive_l,
            finished=self._finished,
            slot_cap=self._slot_cap,
            eos=self.cfg.eos_token_id,
            pad=self.cfg.pad_token_id,
        )
        self._slots: list[_Slot] = [_Slot() for _ in range(S)]
        self._pos_mirror = [0] * S
        self._enc_mirror = [1] * S
        self._alive_set: set[int] = set()
        gib = (
            (self._self_k.numel() + self._cross_k.numel()) * 2 * self._self_k.element_size() / 2**30
        )
        logger.info(
            "[slots] %d slots, T_enc_max=%d L_alloc=%d, KV buffers %.1f GiB, graphs=%s",
            S,
            self.t_enc_max,
            self.l_alloc,
            gib,
            self.use_cuda_graphs,
        )

    def _drain(self, max_n: int) -> list[_Req]:
        out: list[_Req] = []
        while len(out) < max_n:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return [r for r in out if not r.future.cancelled()]

    @torch.inference_mode()
    def _encode_batch(self, reqs: list[_Req]) -> None:
        """Encoder + cross-KV projection for a batch, staged for admission.

        Split out of admission so it can run on a producer thread: the encoder
        is ~42% of GPU time and, run inline, stalls every queued decode behind
        it. Staged tensors are per-request copies, so the scheduler can adopt
        them into slots whenever one frees.
        """
        t0 = time.perf_counter()
        wavs = [r.wav for r in reqs]
        lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.int64)
        batch = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
        for i, w in enumerate(wavs):
            batch[i, : w.shape[0]] = w
        feats, feat_lens = self.fe(batch.to(self.device), lens.to(self.device))
        feats = feats.to(self.dtype)
        att = (
            torch.arange(feats.size(2), device=self.device).unsqueeze(0) < feat_lens.unsqueeze(1)
        ).long()
        enc = self.model.model.encoder(feats, attention_mask=att)
        states, enc_lens = enc.last_hidden_state, enc.lengths

        staged = []
        for i, req in enumerate(reqs):
            t = int(enc_lens[i])
            if t > self.t_enc_max:  # cannot happen given the window guard
                req.future.set_exception(
                    RuntimeError(f"encoder length {t} exceeds static pool {self.t_enc_max}")
                )
                self.stats.failed += 1
                continue
            ks, vs = [], []
            for layer in self.layers:
                k, v = layer.second_sub_layer.project_kv(states[i : i + 1])
                ks.append(k[0, :, :t])
                vs.append(v[0, :, :t])
            staged.append((req, t, torch.stack(ks).contiguous(), torch.stack(vs).contiguous()))
        _sync()
        self.stats.encode_s += time.perf_counter() - t0

        ev = None
        if self._enc_stream is not None:
            ev = torch.cuda.Event()
            ev.record(self._enc_stream)
        with self._staged_lock:
            for item in staged:
                self._staged.append((*item, ev))

    def _producer_loop(self) -> None:
        """Encode ahead of the scheduler on a side stream."""
        try:
            dev = torch.device(self.device)
            torch.cuda.set_device(
                dev.index if dev.index is not None else torch.cuda.current_device()
            )
            self._enc_stream = torch.cuda.Stream(device=self.device)
            while not self._stop.is_set():
                with self._staged_lock:
                    backlog = len(self._staged)
                # keep the backlog shallow: staged cross-KV is real memory, and
                # encoding far ahead of admission buys nothing
                if backlog >= 2 * self.admit_batch:
                    time.sleep(self.idle_sleep_s)
                    continue
                reqs = self._drain(self.admit_batch)
                if not reqs:
                    time.sleep(self.idle_sleep_s)
                    continue
                # Capture and the producer must not enqueue concurrently: CUDA
                # graph capture fails if another thread touches the device.
                with self._prefill_lock, torch.cuda.stream(self._enc_stream):
                    self._encode_batch(reqs)
        except BaseException as e:
            logger.exception("[slots] producer died")
            self._err.append(e)

    @torch.inference_mode()
    def _admit(self) -> int:
        """Move staged (already-encoded) requests into free slots."""
        free = [i for i, s in enumerate(self._slots) if s.req is None]
        if not free:
            return 0

        if not self.overlap_encode:
            reqs = self._drain(min(len(free), self.admit_batch))
            if reqs:
                self._encode_batch(reqs)

        with self._staged_lock:
            take = [self._staged.popleft() for _ in range(min(len(free), len(self._staged)))]
        if not take:
            return 0

        admitted = []
        for (req, t, ck, cv, ev), s in zip(take, free, strict=False):
            if ev is not None:
                # the encoder ran on the side stream; order this stream behind it
                ev.wait(torch.cuda.current_stream())
                ck.record_stream(torch.cuda.current_stream())
                cv.record_stream(torch.cuda.current_stream())
            self._cross_k[:, s, :, :t] = ck
            self._cross_v[:, s, :, :t] = cv
            if t < self.t_enc_max:
                self._cross_k[:, s, :, t:] = 0
                self._cross_v[:, s, :, t:] = 0
            self._slots[s].req = req
            self._slot_enc_len[s] = t
            self._slot_cap[s] = (
                min(self.cfg.max_target_positions, t + self.cfg.max_generation_delta) + 1
            )
            self._slot_ids[s].fill_(self.cfg.pad_token_id)
            self._enc_mirror[s] = t
            admitted.append(s)

        if not admitted:
            return 0
        # Build prompts defensively: this used to be one comprehension inside
        # torch.stack, so an unpromptable language raised out of the scheduler
        # loop and killed the thread -- every session on the replica, not just
        # the offending one. A bad row now fails its own future, like _evict.
        rows, kept = [], []
        for sl in admitted:
            req = self._slots[sl].req
            try:
                rows.append(
                    torch.tensor(
                        self.tokenizer.encode_prompt(
                            req.lang, itn=req.itn, romanized=req.romanized
                        ),
                        device=self.device,
                    )
                )
                kept.append(sl)
            except Exception as e:  # one bad row must not stop the pool
                logger.warning("[slots] dropping request: %s", e)
                if not req.future.done():
                    req.future.set_exception(e)
                self.stats.failed += 1
                self._slots[sl].req = None
        if not kept:
            return 0
        admitted = kept
        idx = torch.tensor(admitted, dtype=torch.long, device=self.device)
        prompts = torch.stack(rows)
        plen = self.tokenizer.prompt_len
        self._slot_ids[idx.unsqueeze(1), torch.arange(plen, device=self.device).unsqueeze(0)] = (
            prompts
        )
        self._prompt_step(
            idx,
            prompts,
            self._self_k,
            self._self_v,
            self._cross_k,
            self._cross_v,
            self._slot_enc_len,
            self._slot_ids,
            self._slot_pos,
            self._last_tok,
        )
        self._alive[idx] = True
        self._alive_l[idx] = 1
        for s in admitted:
            self._pos_mirror[s] = plen
            self._alive_set.add(s)
        return len(admitted)

    def _evict(self) -> None:
        done = self._alive & self._finished
        idx = torch.nonzero(done, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        for s in idx.tolist():
            slot = self._slots[s]
            req = slot.req
            try:
                n = int(self._slot_pos[s]) + 1
                ids = self._slot_ids[s, :n].tolist()
                ids = self.tokenizer.strip_prompt_and_trim(
                    ids,
                    self.tokenizer.encode_prompt(req.lang, itn=req.itn, romanized=req.romanized),
                )
                if not req.future.done():
                    req.future.set_result(self.tokenizer.decode(ids))
                self.stats.completed += 1
            except Exception as e:  # one bad row must not stop the pool
                if not req.future.done():
                    req.future.set_exception(e)
                self.stats.failed += 1
            slot.req = None
            self._pos_mirror[s] = 0
            self._alive_set.discard(s)
        self._alive[idx] = False
        self._alive_l[idx] = 0
        self._finished[idx] = False
        self._slot_pos[idx] = 0
        self._slot_enc_len[idx] = 1
        self._slot_cap[idx] = self.cfg.max_target_positions
        self._last_tok[idx, 0] = self.cfg.pad_token_id

    def _scheduler(self) -> None:
        try:
            if str(self.device).startswith("cuda"):
                # CUDA context is thread-local and graph replay is thread-affine,
                # so this thread must own the device it will capture on.
                # set_device wants an index; a bare "cuda" means current device.
                dev = torch.device(self.device)
                torch.cuda.set_device(
                    dev.index if dev.index is not None else torch.cuda.current_device()
                )
            self._alloc()
            if self.overlap_encode:
                self._producer = threading.Thread(
                    target=self._producer_loop, name="asr-encode", daemon=True
                )
                self._producer.start()
            self._ready.set()
            while not self._stop.is_set():
                self._admit()
                if not self._alive_set:
                    # nothing running: wait briefly rather than spinning. With
                    # the producer owning the queue, watch the staged deque.
                    if self.overlap_encode:
                        time.sleep(self.idle_sleep_s)
                    else:
                        with contextlib.suppress(queue.Empty):
                            self._q.put(self._q.get(timeout=self.idle_sleep_s * 10))
                    continue
                key_len = max(self._pos_mirror[s] for s in self._alive_set) + 1
                t_cross = max(self._enc_mirror[s] for s in self._alive_set)
                self.stats.occupancy_sum += len(self._alive_set) / self.n_slots
                t0 = time.perf_counter()
                self._step(key_len, self._buf, log=logger.debug, t_cross_needed=t_cross)
                _sync()
                self.stats.decode_s += time.perf_counter() - t0
                self.stats.steps += 1
                for s in self._alive_set:
                    self._pos_mirror[s] += 1
                if self.stats.steps % self.evict_every == 0:
                    self._evict()
            # drain: fail anything still in flight so no caller hangs on shutdown
            for slot in getattr(self, "_slots", []):
                if slot.req is not None and not slot.req.future.done():
                    slot.req.future.set_exception(RuntimeError("engine shut down"))
        except BaseException as e:
            logger.exception("[slots] scheduler died")
            self._err.append(e)
            self._ready.set()
            while True:
                try:
                    r = self._q.get_nowait()
                except queue.Empty:
                    break
                if not r.future.done():
                    r.future.set_exception(e)
