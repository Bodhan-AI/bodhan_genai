"""Wire protocol for the TTS server (four endpoints, one engine):

  WS   /tts          — live streaming (``chunked`` request flag honored)
  WS   /tts/chunked  — long-form chunked streaming (chunked forced)
  POST /tts/sse      — live streaming over plain HTTP (Server-Sent Events)
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

``POST /tts/sse`` streams the same sequence over Server-Sent Events, for
callers that cannot hold a websocket open — a browser ``EventSource``, an
HTTP/2 client, anything behind a proxy that terminates upgrades. SSE is a text
protocol, so audio rides base64 inside the event data rather than as bytes;
that costs 33% on the wire and is the price of not needing a websocket. The
control frames are the *same functions* the websocket path uses, so the two
transports cannot describe a stream differently:

    event: start
    data: {"event": "start", "sample_rate": 24000, "encoding": "pcm_s16le"}

    event: audio
    data: {"event": "audio", "seq": 0, "pcm_b64": "..."}

    event: end
    data: {"event": "end", "audio_dur_s": 1.234, "n_frames": 15}

A failure mid-stream arrives as an ``error`` event, not an HTTP status: the
response is committed 200 the moment the first event ships, which is exactly
why the websocket path closes with 1011 instead of sending one.
"""

from __future__ import annotations

import json
from base64 import b64encode

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


def audio_frame(seq: int, pcm: bytes) -> str:
    """One base64 audio frame, for the SSE transport only.

    The websocket sends PCM as binary and never needs this; SSE is text, so
    there is no alternative to an encoding step here.
    """
    return json.dumps({"event": "audio", "seq": int(seq), "pcm_b64": b64encode(pcm).decode()})


def sse(event: str, data: str) -> bytes:
    """Frame one control/audio payload as an SSE event.

    The ``event:`` line duplicates the ``"event"`` key inside ``data`` on
    purpose: it lets an ``EventSource`` register per-type listeners, while a
    plain line reader can ignore it and parse the JSON.
    """
    return f"event: {event}\ndata: {data}\n\n".encode()
