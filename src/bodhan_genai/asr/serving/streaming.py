# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""VAD-endpointed streaming for an attention encoder-decoder ASR model.

**IndicTranscribe cannot stream frame-synchronously, and neither can Whisper.**
A CTC or RNNT model emits a token per encoder frame, so text falls out as
audio arrives — that is why NeMo's cache-aware streaming supports those
architectures and refuses this one outright (``mixins.py`` raises
``NotImplementedError`` for anything that is not ``EncDecCTCModel`` /
``EncDecRNNTModel``). Two independent reasons it cannot work here: Canary-style
AED configs train with ``att_context_size: [-1, -1]`` (unlimited context), and
the Transformer decoder cross-attends over the *whole* encoder output for the
utterance, so there is no per-frame emission point.

So text must be produced from *complete spans* of audio. This module cuts those
spans at pauses found by the energy VAD in ``asr.engine.chunker`` — the same
mechanism long-form chunking uses — and decodes each span exactly once::

    speech ─────────────┐ pause >= endpoint_silence_s
                        └──> decode the span once ──> FINAL text (never revised)

A span that reaches ``max_segment_s`` without a qualifying pause is force-cut at
the best silence available (hard cut only if there is none), so ``max_segment_s``
is a hard bound on time-to-first-text — the latency SLO knob. Between endpoints
the in-progress span is re-decoded every ``partial_interval_s`` to produce
PROVISIONAL text, which the next update replaces wholesale.

**Why not LocalAgreement (``whisper_streaming``), which this replaced.** That
approach re-decodes a growing buffer and commits the longest prefix surviving N
consecutive decodes. It works, but measured on this stack it re-encoded each
second of audio **4.43x** and re-generated about **4.3x** the decoder tokens
actually needed (8757 against ~2000 over 778 s), because every interval redoes
the whole buffer. Endpointing decodes each second once. It also hands the model
complete spans rather than buffers truncated mid-word, which is what the
checkpoint was trained on. Measurements in docs/asr/serving.md.

The cost of the change, stated plainly: text arrives in bursts at pauses rather
than growing continuously, and a long pause-free stretch yields nothing until
``max_segment_s``, whereas LocalAgreement's latency was bounded by its interval
regardless of content. Partials exist to soften the first of those, and they are
not free — each one is a full re-decode of the open span.

This module is pure logic: it holds no model and no torch device state, takes
a ``transcribe`` callable, and is therefore CPU-testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from bodhan_genai.asr.engine.chunker import ChunkConfig, frame_db, silence_runs


@dataclass
class StreamUpdate:
    """One incremental result.

    ``committed_delta`` is the text of a span that just closed — append-only,
    never revised. ``provisional`` is the open span's current best guess, which
    the next update may replace outright. A caption UI renders committed text as
    final and provisional text greyed out.
    """

    committed_delta: str = ""
    provisional: str = ""
    committed_total: str = ""
    audio_seconds: float = 0.0
    decoded: bool = False  # False when the update did no work
    is_final: bool = False  # True when this update closed a span


@dataclass
class VadStream:
    """Silence-endpointed spans, each decoded once, plus periodic partials.

    ``transcribe(wav_1d_tensor) -> str`` is supplied by the caller (a replica,
    or a fake in tests) and is called on one complete span at a time.

    Args:
      sample_rate:        audio rate of everything pushed in.
      endpoint_silence_s: trailing pause that closes a span. Production
                          convention is 0.5-0.8 s; below ~0.35 s ordinary breath
                          pauses start closing spans mid-phrase.
      max_segment_s:      force-cut bound, and therefore the hard ceiling on
                          time-to-first-text.
      partial_interval_s: re-decode the open span this often for interim text.
                          Each partial is a full re-decode, so this trades
                          capacity for responsiveness directly; ``None``
                          disables partials and is the cheapest setting.
      min_segment_s:      never emit a span shorter than this — an AED given a
                          fraction of a second invents a word.
      silence_dbfs:       a span whose loudest frame is quieter than this holds
                          no speech and is never sent to the model. Absolute on
                          purpose: the relative threshold cannot help in a
                          session that has been silent throughout, because it
                          calibrates against the loudest thing it has heard.
    """

    transcribe: Callable[[torch.Tensor], str]
    sample_rate: int = 16000
    endpoint_silence_s: float = 0.5
    max_segment_s: float = 5.0
    partial_interval_s: float | None = 2.0
    min_segment_s: float = 0.6
    silence_dbfs: float = -50.0
    chunk_cfg: ChunkConfig = field(default_factory=ChunkConfig)
    # Frames of past dB kept for the noise-floor estimate. A percentile taken
    # over only the open span is unstable while that span is short: early in a
    # session it can put the floor above the speech level (everything looks
    # silent) or below the true floor (nothing does).
    floor_history_s: float = 20.0

    # -- state ---------------------------------------------------------------
    _seg: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    _committed: list[str] = field(default_factory=list)
    _provisional: str = ""
    _since_partial: int = 0
    _total_samples: int = 0
    _db_hist: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    # -- properties ----------------------------------------------------------

    @property
    def committed_text(self) -> str:
        return " ".join(self._committed)

    @property
    def provisional_text(self) -> str:
        return self._provisional

    @property
    def segment_seconds(self) -> float:
        return self._seg.numel() / self.sample_rate

    @property
    def audio_seconds(self) -> float:
        return self._total_samples / self.sample_rate

    # -- ingest --------------------------------------------------------------

    def push(self, audio: torch.Tensor) -> StreamUpdate | None:
        """Append audio; return an update when a span closed or a partial is due.

        Returns None when the chunk was buffered but nothing was due — the
        common case for small packets, and the reason a caller should treat
        None as "nothing to send" rather than an error.
        """
        audio = audio.detach().to(torch.float32).reshape(-1)
        self._seg = torch.cat([self._seg, audio])
        self._since_partial += audio.numel()
        self._total_samples += audio.numel()
        self._drop_leading_silence()

        # The hard bound outranks the pause search: an over-long span degrades
        # the model and the slot pool is statically sized for a fixed window.
        if self.segment_seconds >= self.max_segment_s:
            return self._close(self._forced_cut())

        cut = self._endpoint_cut()
        if cut is not None:
            return self._close(cut)

        if (
            self.partial_interval_s is not None
            and self._since_partial >= self.partial_interval_s * self.sample_rate
            and self.segment_seconds >= self.min_segment_s
        ):
            self._since_partial = 0
            self._provisional = self.transcribe(self._seg).strip()
            return StreamUpdate(
                provisional=self._provisional,
                committed_total=self.committed_text,
                audio_seconds=self.audio_seconds,
                decoded=True,
            )
        return None

    def finalize(self) -> StreamUpdate:
        """Flush: decode the open span and commit it.

        A sliver shorter than ``min_segment_s`` is dropped rather than decoded.
        """
        if self.segment_seconds < self.min_segment_s:
            self._seg = torch.zeros(0)
            self._provisional = ""
            return StreamUpdate(
                committed_total=self.committed_text,
                audio_seconds=self.audio_seconds,
                is_final=True,
            )
        return self._close((self._seg.numel(), self._seg.numel()))

    # -- core ----------------------------------------------------------------

    def _close(self, cut: tuple[int, int]) -> StreamUpdate:
        """Decode ``_seg[:speech_end]`` as a finished span, drop ``_seg[:keep_from]``.

        ``cut`` is ``(speech_end, keep_from)``: audio between the two is the
        pause that closed the span and belongs to neither side.
        """
        speech_end, keep_from = cut
        span = self._seg[:speech_end]
        self._seg = self._seg[keep_from:]
        self._since_partial = self._seg.numel()
        self._provisional = ""

        decoded = span.numel() > 0 and self._has_speech(span)
        text = self.transcribe(span).strip() if decoded else ""
        if text:
            self._committed.append(text)
        return StreamUpdate(
            committed_delta=text,
            committed_total=self.committed_text,
            audio_seconds=self.audio_seconds,
            decoded=decoded,
            is_final=True,
        )

    def _has_speech(self, span: torch.Tensor) -> bool:
        """Absolute check that a span holds something worth decoding.

        Handing an AED pure silence is the classic way to get a hallucinated
        phrase back. The relative threshold cannot catch this case: it clamps to
        ``peak - 20 dB``, so in a session that has only ever heard silence the
        silence itself becomes the reference and registers as speech.
        """
        return float(frame_db(span, self.sample_rate, self.chunk_cfg).max()) >= self.silence_dbfs

    def _drop_leading_silence(self) -> None:
        """Discard a silent prefix so a span never begins with a pause.

        Two things go wrong without it. The gap between an endpoint and the next
        word rides into the following span and is encoded for nothing (measured
        1.6 s of span for 1.0 s of speech). And a long gap between utterances
        grows a silence-only span until ``max_segment_s`` forces it to be
        DECODED -- see ``_has_speech`` for why that is the failure to avoid.
        """
        if self._seg.numel() < int(0.2 * self.sample_rate):
            return
        db = frame_db(self._seg, self.sample_rate, self.chunk_cfg)
        thr = self._threshold(db)
        if thr is None:
            return
        voiced = (db >= thr).nonzero()
        if voiced.numel() == 0:
            # Nothing but pause so far: keep learning the floor, drop the audio.
            self._db_hist = self._trim_hist(torch.cat([self._db_hist, db]))
            self._seg = torch.zeros(0)
            self._since_partial = 0
            return
        hop = max(1, int(self.sample_rate * self.chunk_cfg.hop_ms / 1000))
        first = int(voiced[0]) * hop
        if first > 0:
            self._seg = self._seg[first:]

    def _threshold(self, db: torch.Tensor) -> float | None:
        """Silence threshold from session history plus the open span.

        Mirrors ``chunker.silence_runs`` so a stream and a batch job agree on
        what counts as a pause. None while there is too little audio for the
        percentile to mean anything.
        """
        hist = torch.cat([self._db_hist, db]) if self._db_hist.numel() else db
        if hist.numel() < int(1000.0 / self.chunk_cfg.hop_ms):  # under 1 s of frames
            return None
        cfg = self.chunk_cfg
        floor = torch.quantile(hist, cfg.floor_pct / 100.0)
        return float(torch.minimum(floor + cfg.floor_margin_db, hist.max() - 20.0))

    def _endpoint_cut(self) -> tuple[int, int] | None:
        """``(speech_end, keep_from)`` if the open span ends in a long enough pause."""
        if self.segment_seconds < self.min_segment_s + self.endpoint_silence_s:
            return None
        db = frame_db(self._seg, self.sample_rate, self.chunk_cfg)
        thr = self._threshold(db)
        if thr is None:
            return None
        hop = max(1, int(self.sample_rate * self.chunk_cfg.hop_ms / 1000))
        need = max(1, int(self.endpoint_silence_s * 1000.0 / self.chunk_cfg.hop_ms))
        if db.numel() < need or not bool((db[-need:] < thr).all()):
            return None
        # Walk back over the WHOLE trailing pause, not just the qualifying tail,
        # so the silence is neither handed to the model nor carried into the
        # next span (where it would delay that span's own endpoint).
        i = db.numel() - need
        while i > 0 and bool(db[i - 1] < thr):
            i -= 1
        speech_end = i * hop
        if speech_end < self.min_segment_s * self.sample_rate:
            return None
        self._db_hist = self._trim_hist(torch.cat([self._db_hist, db]))
        return speech_end, self._seg.numel()

    def _forced_cut(self) -> tuple[int, int]:
        """``(speech_end, keep_from)`` at ``max_segment_s`` with no qualifying pause.

        Cut at the last real silence if there is one — an AED re-transcribing a
        word fragment invents a word — otherwise cut hard, because an over-long
        span is the worse failure.
        """
        db = frame_db(self._seg, self.sample_rate, self.chunk_cfg)
        thr = self._threshold(db)
        self._db_hist = self._trim_hist(torch.cat([self._db_hist, db]))
        hop = max(1, int(self.sample_rate * self.chunk_cfg.hop_ms / 1000))
        floor_samples = int(self.min_segment_s * self.sample_rate)
        if thr is not None:
            for a, b in reversed(silence_runs(db, self.chunk_cfg)):
                if a * hop >= floor_samples:
                    return a * hop, min(b * hop, self._seg.numel())
        return self._seg.numel(), self._seg.numel()

    def _trim_hist(self, hist: torch.Tensor) -> torch.Tensor:
        keep = max(1, int(self.floor_history_s * 1000.0 / self.chunk_cfg.hop_ms))
        return hist[-keep:]
