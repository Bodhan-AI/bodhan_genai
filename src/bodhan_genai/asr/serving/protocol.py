# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Wire types for the ASR server.

Kept in one place (and validated with pydantic) so the websocket handler, the
HTTP handlers, and the example client cannot drift — the TTS server learned
this the hard way when a request field was silently dropped because the model
did not declare it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

#: Per-request ceiling. Rows are batched at cfg.batch_size internally, but an
#: unbounded list still means unbounded work held open on one replica.
MAX_PATHS = 512

#: Rates a client may declare for a raw-PCM stream. This is not a preference:
#: every span bound in VadStream is a sample count DIVIDED by this number, so a
#: client claiming 10_000_000 Hz made the buffer bound unreachable and grew the
#: replica's memory without limit.
STREAM_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)


class TranscribeRequest(BaseModel):
    """POST /asr/transcribe — one or more audio paths readable BY THE SERVER."""

    paths: list[str] = Field(default_factory=list)
    lang: str | None = None
    # Long-form: segment audio longer than this on silences. None = use the
    # server default; 0 disables. Below the threshold audio is decoded whole,
    # because chunking short audio measurably hurts (docs/asr/caveats.md).
    chunk_above: float | None = None
    detect_language: bool = False
    # Narrow the LID candidate set. None = every language token (the library
    # default). A HARD filter: audio in an excluded language is reassigned to
    # the nearest permitted one, never flagged.
    allowed_langs: list[str] | None = None
    # Output mode (prompt slots; default native script). itn=True gives
    # mixed-script/ITN output, romanized=True gives Latin romanization.
    itn: bool = False
    romanized: bool = False

    @model_validator(mode="after")
    def _check(self):
        if not self.paths:
            raise ValueError("paths must be a non-empty list")
        if len(self.paths) > MAX_PATHS:
            raise ValueError(f"at most {MAX_PATHS} paths per request")
        if not self.lang and not self.detect_language:
            raise ValueError(
                "pass lang, or detect_language=true — the model is language-conditioned "
                "and a wrong label yields confidently wrong script, not obvious garbage"
            )
        return self


class StreamStart(BaseModel):
    """First (JSON) message on WS /asr/stream, before any audio."""

    lang: str | None = None
    detect_language: bool = False
    sample_rate: int | None = None  # None = the server's configured rate
    # Output mode for the whole stream (see TranscribeRequest).
    itn: bool = False
    romanized: bool = False

    @model_validator(mode="after")
    def _check(self):
        if not self.lang and not self.detect_language:
            raise ValueError("pass lang, or detect_language=true")
        if self.sample_rate is not None and self.sample_rate not in STREAM_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {STREAM_SAMPLE_RATES}")
        return self
