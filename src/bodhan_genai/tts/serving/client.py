"""Minimal single-session client for all four server endpoints.

    python -m bodhan_genai.tts.serving.client --mode stream  --text "..." --out out.wav
    python -m bodhan_genai.tts.serving.client --mode chunked --text "<long text>" --out long.wav
    python -m bodhan_genai.tts.serving.client --mode sse     --text "..." --out out.wav
    python -m bodhan_genai.tts.serving.client --mode offline --text "..." --out out.wav
    python -m bodhan_genai.tts.serving.client --mode chunked --dialogue-json turns.json --out dlg.wav

``--dialogue-json`` sends a dialogue instead of ``--text``: a JSON file holding
``[{"speaker": ..., "text": ...}, ...]`` turns (speakers ride inline per turn,
so ``--speaker`` is ignored).

``--url`` is the server base (``ws://HOST:8000`` or ``http://HOST:8000``); the
endpoint path and scheme are derived from ``--mode`` (stream -> WS /tts,
chunked -> WS /tts/chunked, sse -> POST /tts/sse, offline -> POST
/tts/offline). A legacy ``--url ws://HOST:8000/tts`` still works — the
trailing path is stripped.

``sse`` is ``stream`` over plain HTTP for callers that cannot hold a websocket
open; the audio arrives base64-encoded, which costs 33% on the wire.

The server is open unless it was started with ``TTS_AUTH_FILE`` set. For one
that was, pass ``--auth user:password``, or let this read the same
``TTS_AUTH_FILE`` the server did if you are on the same box.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from base64 import b64encode

import numpy as np

from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE, write_wav_24k

_WS_PATHS = {"stream": "/tts", "chunked": "/tts/chunked"}


def _credential(explicit: str) -> str:
    """``user:password``, or "" for an open server.

    Reads TTS_AUTH_FILE as a fallback so an operator on the server's own box
    needs no second copy of the credential. Never accepts it from a plain
    environment variable: that would put it in every `ps` listing.
    """
    if explicit:
        return explicit
    path = os.environ.get("TTS_AUTH_FILE", "").strip()
    if not path:
        return ""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError as e:
        print(f"warning: TTS_AUTH_FILE={path!r} is unreadable: {e}", file=sys.stderr)
        return ""


def _basic_header(cred: str) -> dict[str, str]:
    if not cred:
        return {}
    return {"Authorization": "Basic " + b64encode(cred.encode()).decode()}


def _base_url(url: str, scheme: str) -> str:
    """Normalize --url to '<scheme>://host:port' (strip known endpoint paths)."""
    base = url.rstrip("/")
    for suffix in ("/tts/chunked", "/tts/offline", "/tts"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for old in ("ws://", "wss://", "http://", "https://"):
        if base.startswith(old):
            secure = old in ("wss://", "https://")
            host = base[len(old) :]
            return f"{scheme}{'s' if secure else ''}://{host}"
    return f"{scheme}://{base}"


async def run_ws(url: str, payload: dict, out: str, cred: str = "") -> None:
    import websockets  # deferred: [serve] extra, keep module import light

    frames: list[bytes] = []
    sr = SNAC_SAMPLE_RATE
    t0 = time.perf_counter()
    first = None
    async with websockets.connect(url, max_size=None, additional_headers=_basic_header(cred)) as ws:
        await ws.send(json.dumps(payload))
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                if first is None:
                    first = time.perf_counter() - t0
                frames.append(bytes(msg))
            else:
                ev = json.loads(msg)
                if ev.get("event") == "start":
                    sr = ev.get("sample_rate", sr)
                elif ev.get("event") == "end":
                    break
                elif ev.get("event") == "error":
                    print("server error:", ev)
                    return
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0
    write_wav_24k(out, audio)
    print(
        f"TTFP={(first or 0) * 1000:.0f}ms frames={len(frames)} audio={len(audio) / sr:.2f}s -> {out}"
    )


def run_offline(url: str, payload: dict, out: str, cred: str = "") -> None:
    """POST /tts/offline: complete audio/wav response, written straight to disk."""
    import urllib.error
    import urllib.request

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **_basic_header(cred)},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            wav = resp.read()
            dur = resp.headers.get("X-Audio-Duration-S", "?")
    except urllib.error.HTTPError as e:
        print(f"server error ({e.code}):", e.read().decode("utf-8", "replace"))
        return
    with open(out, "wb") as f:
        f.write(wav)
    print(f"offline: {dur}s audio in {time.perf_counter() - t0:.2f}s wall -> {out}")


def run_sse(url: str, payload: dict, out: str, cred: str = "") -> None:
    """POST /tts/sse: read events as they arrive and write the PCM out.

    Parses SSE by hand rather than pulling in a client library: the framing is
    two line prefixes, and the point of this endpoint is to need nothing
    special on the client side.
    """
    import urllib.error
    import urllib.request
    from base64 import b64decode

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **_basic_header(cred),
        },
        method="POST",
    )
    frames: list[bytes] = []
    sr = SNAC_SAMPLE_RATE
    t0 = time.perf_counter()
    first = None
    try:
        with urllib.request.urlopen(req) as resp:
            for raw in resp:  # SSE events are newline-delimited; one data: line each here
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                msg = json.loads(line[5:].strip())
                event = msg.get("event")
                if event == "start":
                    sr = int(msg.get("sample_rate", SNAC_SAMPLE_RATE))
                elif event == "audio":
                    if first is None:
                        first = time.perf_counter() - t0
                    frames.append(b64decode(msg["pcm_b64"]))
                elif event == "error":
                    print("server error:", msg.get("detail"), file=sys.stderr)
                    return
    except urllib.error.HTTPError as e:
        print(f"server error ({e.code}):", e.read().decode("utf-8", "replace"), file=sys.stderr)
        return
    if not frames:
        print("no audio received", file=sys.stderr)
        return
    # Same conversion as run_ws: write_wav_24k takes normalized float32 and is
    # 24 kHz by definition, so `sr` from the start frame only scales the report.
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32767.0
    write_wav_24k(out, audio)
    print(
        f"sse: TTFP={(first or 0) * 1000:.0f}ms events={len(frames)} "
        f"audio={len(audio) / sr:.2f}s -> {out}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://localhost:8000", help="server base URL")
    p.add_argument("--mode", choices=("stream", "chunked", "sse", "offline"), default="stream")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--dialogue-json", help='JSON file: [{"speaker", "text"}, ...]')
    p.add_argument("--speaker", default="")
    p.add_argument("--out", default="client_out.wav")
    p.add_argument(
        "--auth", default="", metavar="USER:PASSWORD", help="or set TTS_AUTH_FILE to the same file"
    )
    # legacy alias for --mode chunked (pre-endpoint flag)
    p.add_argument("--chunked", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    mode = "chunked" if (a.chunked and a.mode == "stream") else a.mode

    if a.dialogue_json:
        with open(a.dialogue_json, encoding="utf-8") as f:
            messages = json.load(f)
        if a.speaker:
            print(
                "warning: --speaker is ignored with --dialogue-json "
                "(dialogue turns carry speakers inline)",
                file=sys.stderr,
            )
        payload = {"messages": messages}
    else:
        payload = {"text": a.text, "speaker": a.speaker}

    cred = _credential(a.auth)
    if mode == "offline":
        run_offline(_base_url(a.url, "http") + "/tts/offline", payload, a.out, cred)
    elif mode == "sse":
        run_sse(_base_url(a.url, "http") + "/tts/sse", payload, a.out, cred)
    else:
        ws_url = _base_url(a.url, "ws") + _WS_PATHS[mode]
        asyncio.run(run_ws(ws_url, payload, a.out, cred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
