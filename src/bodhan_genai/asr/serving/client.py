# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Example ASR client.

# buffered streaming from a file, paced like a live mic
python -m bodhan_genai.asr.serving.client --mode stream --audio a.wav --lang hi

# offline: server reads the paths itself
python -m bodhan_genai.asr.serving.client --mode transcribe --paths a.wav b.wav --lang hi

# language identification only
python -m bodhan_genai.asr.serving.client --mode detect --paths a.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request


def _http(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


async def run_stream(
    url: str,
    audio_path: str,
    lang: str,
    realtime: bool,
    packet_ms: int,
    itn: bool = False,
    romanized: bool = False,
) -> int:
    import soundfile as sf
    import websockets

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    ws_url = f"{ws_url}/asr/stream"

    packet = max(1, int(sr * packet_ms / 1000))
    async with websockets.connect(ws_url, max_size=None, open_timeout=60) as ws:
        await ws.send(
            json.dumps({"lang": lang, "sample_rate": int(sr), "itn": itn, "romanized": romanized})
        )

        async def receive():
            async for msg in ws:
                ev = json.loads(msg)
                kind = ev.get("event")
                if kind == "start":
                    print(
                        f"[start] lang={ev['lang']} sr={ev['sample_rate']} "
                        f"endpoint {ev['endpoint_silence_s']}s, span <={ev['max_segment_s']}s",
                        flush=True,
                    )
                elif kind == "update":
                    # committed text is final; provisional may still change
                    print(
                        f"[{ev['audio_seconds']:6.1f}s] {ev['committed']}"
                        + (f"  ~{ev['provisional']}~" if ev["provisional"] else ""),
                        flush=True,
                    )
                elif kind == "end":
                    print(f"\n=== final ===\n{ev['text']}", flush=True)
                    return 0
                elif kind == "error":
                    print(f"ERROR: {ev['message']}", file=sys.stderr)
                    return 1
            return 0

        recv = asyncio.create_task(receive())
        for i in range(0, len(wav), packet):
            await ws.send((wav[i : i + packet] * 32767).astype("<i2").tobytes())
            if realtime:
                # pace like a live mic so the printed latency is meaningful
                await asyncio.sleep(packet / sr)
        await ws.send(json.dumps({"event": "eof"}))
        return await recv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--mode", choices=("stream", "transcribe", "detect"), default="stream")
    p.add_argument("--audio", help="stream mode: one audio file")
    p.add_argument("--paths", nargs="+", help="transcribe/detect mode: server-readable paths")
    p.add_argument("--lang", default=None)
    p.add_argument("--detect-language", action="store_true")
    p.add_argument("--itn", action="store_true", help="mixed-script/ITN output mode")
    p.add_argument("--romanized", action="store_true", help="Latin romanization output mode")
    p.add_argument("--chunk-above", type=float, default=None)
    p.add_argument(
        "--no-realtime",
        action="store_true",
        help="stream mode: push audio as fast as possible instead of pacing it like a mic",
    )
    p.add_argument("--packet-ms", type=int, default=200, help="stream mode: packet size")
    a = p.parse_args(argv)

    if a.mode == "stream":
        if not a.audio or not a.lang:
            p.error("--mode stream needs --audio and --lang")
        return asyncio.run(
            run_stream(
                a.url,
                a.audio,
                a.lang,
                not a.no_realtime,
                a.packet_ms,
                itn=a.itn,
                romanized=a.romanized,
            )
        )

    if not a.paths:
        p.error(f"--mode {a.mode} needs --paths")
    base = a.url.rstrip("/")
    if a.mode == "detect":
        out = _http(f"{base}/asr/detect", {"paths": a.paths})
        for row in out.get("results", []):
            top = ", ".join(f"{t['lang']}={t['prob']:.3f}" for t in row["top"][:3])
            print(f"{row['path']}\t{top}")
        return 0

    out = _http(
        f"{base}/asr/transcribe",
        {
            "paths": a.paths,
            "lang": a.lang,
            "itn": a.itn,
            "romanized": a.romanized,
            "detect_language": a.detect_language,
            "chunk_above": a.chunk_above,
        },
    )
    if "error" in out:
        print(f"ERROR: {out['error']}", file=sys.stderr)
        return 1
    for row in out["results"]:
        print(f"{row['path']}\t{row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
