# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""AsrReplica: one GPU's worth of ASR = one IndicASREngine.

A thin adapter over the public engine, mirroring the TTS replica's shape: a
plain async class (no Serve decorator) so a local harness can drive it in
tests, wrapped as a deployment by ``serving/service.py``.

The GPU work is synchronous torch, so every entry point hops to a thread via
``run_in_executor``. That is not cosmetic: the replica's event loop also
services the websocket, and a multi-second decode running inline would stall
every other connection on the replica — the exact failure the TTS server hit
when a ~20 s librosa call ran on the loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger("asr.serving.replica")

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class AsrReplica:
    def __init__(self, cfg):
        from bodhan_genai.asr.engine import IndicASREngine
        from bodhan_genai.asr.serving.slot_engine import StreamingSlotEngine

        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._engine = IndicASREngine(cfg.model_dir, device=device, dtype=_DTYPES[cfg.dtype])
        # Continuous-batching slot pool for streaming. It owns its own thread
        # and the GPU work on it; sessions submit and block on a Future. This
        # replaced a fixed-batch micro-batcher because a fixed batch cannot be
        # CUDA-graphed without waiting to fill, and pays max-length for every
        # row. Shares this replica's already-loaded weights rather than a
        # second ~5 GiB copy.
        self._slots = StreamingSlotEngine(
            model=self._engine.model,
            feature_extractor=self._engine.fe,
            tokenizer=self._engine.tokenizer,
            device=device,
            dtype=_DTYPES[cfg.dtype],
            # The pool is sized for the WORST-CASE span, not the nominal one.
            # VadStream force-cuts at stream_max_segment_s, but the check runs
            # on packet arrival, so a span can overshoot by one packet; and a
            # forced cut may keep audio after the chosen silence, which starts
            # the next span non-empty. +1 s covers both. Sizing to the nominal
            # window made the static pool reject real traffic.
            max_buffer_s=cfg.stream_max_segment_s + 1.0,
            slots=cfg.stream_slots,
            admit_batch=cfg.stream_admit_batch,
            cuda_graphs=not cfg.stream_no_cuda_graphs,
            overlap_encode=cfg.stream_overlap_encode,
        )
        self._slots.start()
        logger.info(
            "[AsrReplica] ready on %s (dtype=%s, %d slots, span<=%.1fs, graphs=%s)",
            device,
            cfg.dtype,
            cfg.stream_slots,
            cfg.stream_max_segment_s,
            not cfg.stream_no_cuda_graphs,
        )

    async def ready(self) -> bool:
        """Can this replica actually serve right now?

        It used to ``return True`` unconditionally, which made /health a test
        that __init__ had once finished. A replica whose slot scheduler thread
        had died -- which one bad prompt was enough to do -- kept answering 200
        and kept being sent traffic.
        """
        if self._slots._err:
            return False
        return self._slots._thread.is_alive()

    async def check_lang(self, lang: str) -> str | None:
        """``None`` if the model can be prompted in ``lang``, else why not.

        Cheap (a prompt build, no GPU) and authoritative -- it asks the very
        function that would otherwise raise deep inside the scheduler thread.
        """
        try:
            self._engine.tokenizer.encode_prompt(lang)
        except Exception as e:
            return f"unsupported language {lang!r}: {e}"
        return None

    async def is_short(self, path: str) -> bool:
        """Whether this file stays on the batch (non-chunked) path."""
        import soundfile as sf

        thr = self.cfg.chunk_above
        if not thr:
            return True
        try:
            return await self._in_thread(lambda: sf.info(path).duration <= thr)
        except Exception:
            return True

    async def detect_language_file(
        self, path: str, topk: int = 5, allowed_langs=None, max_seconds: float = 120.0
    ):
        """LID for one file, with a bound on how much audio reaches the encoder.

        ``detect_language`` encodes whatever it is given in ONE forward pass, so
        pointing it at an hour of audio is an unbounded allocation on a GPU that
        is also serving live streaming sessions. Long files are probed instead:
        a few windows spread across the recording, voted -- the same shape
        ``transcribe_long`` uses, and for the same reason (an opening window is
        disproportionately likely to be silence, music or a jingle).

        Returns the same ``[[(lang, prob), ...]]`` shape as ``detect_language``.
        """
        import soundfile as sf

        from bodhan_genai.asr.engine.lid import LONG_LID_PROBES, probe_indices

        def work():
            info = sf.info(path)
            if info.duration <= max_seconds:
                return self._engine.detect_language([path], topk=topk, allowed_langs=allowed_langs)
            win = max_seconds / LONG_LID_PROBES
            n_win = max(1, int(info.duration // win))
            wavs = []
            for i in probe_indices(n_win, LONG_LID_PROBES):
                start = int(i * win * info.samplerate)
                block, _sr = sf.read(
                    path,
                    start=start,
                    frames=int(win * info.samplerate),
                    dtype="float32",
                    always_2d=True,
                )
                wavs.append(torch.from_numpy(block.mean(axis=1)))
            tops = self._engine.detect_language(
                wavs, sample_rate=info.samplerate, topk=topk, allowed_langs=allowed_langs
            )
            tally: dict[str, float] = {}
            for top in tops:
                for cand, prob in top:
                    tally[cand] = tally.get(cand, 0.0) + prob
            n = len(tops)
            ranked = sorted(tally.items(), key=lambda kv: -kv[1])[:topk]
            return [[(k, v / n) for k, v in ranked]]

        return await self._in_thread(work)

    @property
    def sample_rate(self) -> int:
        return self._engine.fe.sample_rate

    async def _in_thread(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # -- offline ------------------------------------------------------------

    async def transcribe_paths(
        self,
        paths: list[str],
        lang: str | Sequence[str | None] | None,
        chunk_above: float | None,
        itn: bool = False,
        romanized: bool = False,
        return_lid: bool = False,
        allowed_langs: Sequence[str] | None = None,
    ):
        """Transcribe server-readable audio paths.

        Rows longer than ``chunk_above`` go through the silence-aware chunked
        path; the rest are one batch.

        ``lang`` may be a single language, a per-row sequence, or ``None``. A
        ``None`` row is filled by LID; a supplied one is never overridden. Rows
        are no longer assumed to share a language: they used to, because the
        request carried exactly one, and a detect-then-transcribe pre-pass
        applied row 0's guess to every row -- so a multi-path request in mixed
        languages was transcribed almost entirely in the wrong script.

        With ``return_lid`` the call returns ``(texts, lid_rows)``, one dict per
        row, mirroring the engine's own convention.
        """
        thr = self.cfg.chunk_above if chunk_above is None else chunk_above
        n = len(paths)
        if lang is None or isinstance(lang, str):
            langs: list[str | None] = [lang] * n
        else:
            langs = list(lang)
        if len(langs) != n:
            raise ValueError(f"got {len(langs)} languages for {n} audio paths")

        def batched(sub_paths, sub_langs):
            """One engine call per cfg.batch_size rows.

            The whole `paths` list used to become a single encoder batch, so a
            caller could size one forward pass themselves -- cfg.batch_size is
            documented as the measured throughput knee but the serving path
            never read it.
            """
            size = max(1, int(self.cfg.batch_size))
            texts, rows = [], []
            for i in range(0, len(sub_paths), size):
                res = self._engine.transcribe_batch(
                    sub_paths[i : i + size],
                    sub_langs[i : i + size],
                    itn=itn,
                    romanized=romanized,
                    return_lid=return_lid,
                    allowed_langs=allowed_langs,
                )
                if return_lid:
                    t, r = res
                    texts.extend(t)
                    rows.extend(r)
                else:
                    texts.extend(res)
            return (texts, rows) if return_lid else texts

        def work():
            if not thr:
                return batched(list(paths), langs)
            import soundfile as sf

            out: list[str | None] = [None] * n
            lids: list[dict | None] = [None] * n
            short_idx, long_idx = [], []
            for i, p in enumerate(paths):
                try:
                    (long_idx if sf.info(p).duration > thr else short_idx).append(i)
                except Exception:
                    # unreadable: let the batch path raise a useful error for it
                    short_idx.append(i)
            if short_idx:
                res = batched([paths[i] for i in short_idx], [langs[i] for i in short_idx])
                texts, rows = res if return_lid else (res, None)
                for j, i in enumerate(short_idx):
                    out[i] = texts[j]
                    if rows is not None:
                        lids[i] = rows[j]
            for i in long_idx:
                res = self._engine.transcribe_long(
                    paths[i],
                    langs[i],
                    chunk_above=thr,
                    chunk_min=self.cfg.chunk_min,
                    chunk_max=self.cfg.chunk_max,
                    itn=itn,
                    romanized=romanized,
                    allowed_langs=allowed_langs,
                    return_lang=return_lid,
                )
                if return_lid:
                    out[i], lids[i] = res
                else:
                    out[i] = res
            texts = [t or "" for t in out]
            return (texts, lids) if return_lid else texts

        return await self._in_thread(work)

    async def detect_language(
        self, paths: list[str], topk: int = 5, allowed_langs: Sequence[str] | None = None
    ):
        return await self._in_thread(
            self._engine.detect_language, paths, topk=topk, allowed_langs=allowed_langs
        )

    # -- streaming ----------------------------------------------------------

    def new_stream(
        self,
        lang: str,
        sample_rate: int | None = None,
        itn: bool = False,
        romanized: bool = False,
    ):
        """A VadStream bound to this replica's shared slot pool.

        The stream's ``transcribe`` runs on a worker thread (``push`` is sync)
        and submits the span to the slot pool, which admits it into a free slot
        and steps it alongside every other live session under a CUDA graph. So a
        session blocking here costs nothing: the pool is decoding the others
        concurrently, not waiting for this one.
        """
        from bodhan_genai.asr.serving.streaming import VadStream

        sr = sample_rate or self.cfg.stream_sample_rate

        model_sr = self._engine.fe.sample_rate

        def transcribe(wav: torch.Tensor) -> str:
            # Resample to the MODEL's rate here, not on ingest. The stream
            # buffers at the client's rate, and resampling the whole buffer at
            # decode time is what a file read does -- resampling each arriving
            # packet instead would put a filter boundary at every packet edge
            # and change the samples (see engine/audio_input.read_span_and_slice
            # for the measured version of that mistake).
            #
            # This was missing entirely: a 24 kHz client's audio was handed to
            # a 16 kHz model, so every streaming transcript was decoding
            # time-stretched audio. It only surfaced when the statically sized
            # slot pool started checking lengths.
            if sr != model_sr:
                wav = self._engine.fe.resample(wav, sr)
            # Called on a worker thread (push() is sync). The slot engine's own
            # scheduler thread owns the GPU, so this just queues and waits --
            # meanwhile every other due session is decoding in the same pool.
            return self._slots.transcribe(wav, lang, itn=itn, romanized=romanized)

        return VadStream(
            transcribe=transcribe,
            sample_rate=sr,
            endpoint_silence_s=self.cfg.stream_endpoint_silence_s,
            max_segment_s=self.cfg.stream_max_segment_s,
            # 0 in config means "no interim text"; the stream takes None.
            partial_interval_s=self.cfg.stream_partial_interval_s or None,
            min_segment_s=self.cfg.stream_min_segment_s,
        )

    async def stream_push(self, stream, audio: torch.Tensor):
        return await self._in_thread(stream.push, audio)

    async def stream_finalize(self, stream):
        return await self._in_thread(stream.finalize)

    def slot_stats(self) -> dict:
        """Live scheduler counters — how the GPU time actually splits."""
        st = self._slots.stats
        total = st.encode_s + st.decode_s
        return {
            "submitted": st.submitted,
            "completed": st.completed,
            "failed": st.failed,
            "steps": st.steps,
            "mean_occupancy": round(st.mean_occupancy, 3),
            "encode_s": round(st.encode_s, 2),
            "decode_s": round(st.decode_s, 2),
            "encode_frac": round(st.encode_s / total, 3) if total else 0.0,
            "graphs": len(self._slots._graphs),
        }

    async def shutdown(self) -> None:
        """Stop the slot scheduler; in-flight sessions fail rather than hang."""
        self._slots.close()
