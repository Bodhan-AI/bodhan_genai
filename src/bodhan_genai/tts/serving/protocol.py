"""Wire protocol for the TTS server (three endpoints, one engine):

  WS   /tts          — live streaming (``chunked`` request flag honored)
  WS   /tts/chunked  — long-form chunked streaming (chunked forced)
  POST /tts/offline  — JSON ``SynthesisRequest`` in -> complete ``audio/wav``
                       out (``X-Audio-Duration-S`` / ``X-Sample-Rate`` headers)

Websocket flow — client -> server: one JSON ``SynthesisRequest``; server ->
client: a JSON ``start`` control frame, then a sequence of BINARY audio frames
(raw int16 little-endian PCM @ 24 kHz), then a JSON ``end`` (or ``error``)
control frame.

Dialogue requests carry a ``messages`` list of ``{"speaker", "text"}`` turns
INSTEAD of ``text`` (exactly one of the two must be provided); in chunked mode
dialogues are split at turn boundaries.

Message sizing: in normal mode one message = ``frames_per_message`` SNAC frames
(one SNAC frame = 2048 samples = 4096 bytes; the first frame ships alone). In
long-form chunked mode (``chunked: true``) messages are variable-length int16
PCM — grouped frames plus inter-chunk silence gaps — and the ``end`` frame's
``n_frames``/``audio_dur_s`` counts INCLUDE the gap silence.

Both WebSocket and HTTP-chunked transports use the same framing: control frames
are UTF-8 JSON text, audio frames are raw bytes.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, model_validator

from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE

SAMPLE_RATE = SNAC_SAMPLE_RATE  # 24_000


class Message(BaseModel):
    speaker: str
    text: str


class SynthesisRequest(BaseModel):
    text: str = ""
    # Dialogue turns; exactly one of text/messages must be provided.
    messages: list[Message] | None = None
    speaker: str = ""
    language: str = ""
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int | None = None
    # Long-form chunked synthesis (ChunkedIndicStreamingTTS). None = use the
    # server's --chunked_default. The field MUST exist here: the WS handler
    # round-trips requests through SynthesisRequest(**msg), and pydantic
    # silently drops unknown keys.
    chunked: bool | None = None

    @model_validator(mode="after")
    def _one_of_text_or_messages(self):
        has_text = bool(self.text.strip())
        has_msgs = bool(self.messages)
        if has_text == has_msgs:
            raise ValueError("provide exactly one of text or messages")
        return self


def start_frame() -> str:
    return json.dumps({"event": "start", "sample_rate": SAMPLE_RATE, "encoding": "pcm_s16le"})


def end_frame(audio_dur_s: float, n_frames: int) -> str:
    return json.dumps(
        {"event": "end", "audio_dur_s": round(float(audio_dur_s), 4), "n_frames": int(n_frames)}
    )


def error_frame(detail: str) -> str:
    return json.dumps({"event": "error", "detail": str(detail)})
