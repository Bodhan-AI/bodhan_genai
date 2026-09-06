# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Concurrency load-test for the ASR server.

    python -m bodhan_genai.asr.serving.loadtest --url ws://HOST:8000 \
        --jsonl audio.jsonl --concurrency 1 4 8 16

**The metric that matters for streaming is whether a session keeps up with
real time.** This is an attention encoder-decoder model: every decode interval
re-decodes the WHOLE rolling buffer, and the buffer grows until a commit trims
it. So per-decode cost rises with buffer length, and if a decode ever takes
longer than the interval that produced it, the session falls behind — and
because audio keeps arriving at 1x, it never catches back up. That runaway is
the failure mode to find, and it is invisible to a throughput-only benchmark.

Metrics per streaming session:
  - ttft_s        wall time from first audio byte to the first COMMITTED word.
  - decode_rtf    decode wall time / audio consumed. > 1.0 means the session
                  cannot keep up in real time; the p95 across sessions is the
                  capacity signal.
  - lag_s         at end of stream, how far behind real time the committed
                  transcript is (audio pushed - audio committed, in wall time).
  - drift         whether lag GREW monotonically over the session (the runaway
                  signature) rather than staying bounded.
  - updates       how many incremental updates the client saw.

Sessions are paced at 1x real time by default, because that is the only way
these numbers mean anything: pushing audio as fast as possible measures batch
throughput, not streaming behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[max(0, min(len(s) - 1, round(p / 100.0 * (len(s) - 1))))])


def load_rows(jsonl: str, n: int, seed: int = 0):
    import random

    rows = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("audio_path"):
                    rows.append(r)
    random.Random(seed).shuffle(rows)
    return [rows[i % len(rows)] for i in range(n)]


async def stream_session(url: str, row: dict, packet_ms: int, realtime: bool) -> dict:
    """One websocket streaming session, paced like a live microphone."""
    import soundfile as sf
    import websockets

    try:
        wav, sr = sf.read(row["audio_path"], dtype="float32", always_2d=False)
    except Exception as e:
        return {"ok": False, "err": f"read: {repr(e)[:120]}"}
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    audio_s = len(wav) / sr
    packet = max(1, int(sr * packet_ms / 1000))

    t0 = time.perf_counter()
    ttft = None
    updates = 0
    lag_samples = []  # (wall_elapsed, committed_audio_estimate)
    committed_words = 0
    last_committed = ""

    try:
        async with websockets.connect(
            f"{url.rstrip('/')}/asr/stream", max_size=None, open_timeout=60, close_timeout=10
        ) as ws:
            await ws.send(json.dumps({"lang": row["language"], "sample_rate": int(sr)}))

            done = asyncio.Event()
            result: dict = {}

            async def reader():
                nonlocal ttft, updates, committed_words, last_committed
                try:
                    async for msg in ws:
                        ev = json.loads(msg)
                        kind = ev.get("event")
                        if kind == "update":
                            updates += 1
                            committed = ev.get("committed", "")
                            if committed and ttft is None:
                                ttft = time.perf_counter() - t0
                            if committed != last_committed:
                                last_committed = committed
                                committed_words = len(committed.split())
                            # audio_seconds is what the SERVER has ingested;
                            # elapsed wall time is what the client has sent.
                            lag_samples.append(
                                (time.perf_counter() - t0, float(ev.get("audio_seconds", 0.0)))
                            )
                        elif kind == "end":
                            result["text"] = ev.get("text", "")
                            result["end_audio_s"] = float(ev.get("audio_seconds", 0.0))
                            done.set()
                            return
                        elif kind == "error":
                            result["err"] = ev.get("message", "")[:160]
                            done.set()
                            return
                except Exception as e:
                    result["err"] = f"reader: {repr(e)[:120]}"
                finally:
                    done.set()

            task = asyncio.create_task(reader())
            send_start = time.perf_counter()
            for i in range(0, len(wav), packet):
                await ws.send((wav[i : i + packet] * 32767).astype("<i2").tobytes())
                if realtime:
                    # pace to wall clock so lateness accumulates realistically
                    target = send_start + (i + packet) / sr
                    delay = target - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
            push_done = time.perf_counter() - t0
            await ws.send(json.dumps({"event": "eof"}))
            await asyncio.wait_for(done.wait(), timeout=600)
            await task
    except Exception as e:
        return {"ok": False, "err": repr(e)[:160]}

    if result.get("err"):
        return {"ok": False, "err": result["err"]}

    total_wall = time.perf_counter() - t0
    # Finalization happens after the last byte; how long the caller waits at
    # the end for the transcript to close out.
    tail_s = total_wall - push_done

    # Runaway detection: is the gap between wall time and server-ingested audio
    # growing? A healthy session holds it roughly flat.
    drift = 0.0
    if len(lag_samples) >= 4:
        gaps = [w - a for w, a in lag_samples]
        half = len(gaps) // 2
        drift = statistics.fmean(gaps[half:]) - statistics.fmean(gaps[:half])

    return {
        "ok": True,
        "audio_s": audio_s,
        "wall_s": total_wall,
        "ttft_s": ttft if ttft is not None else float("nan"),
        "tail_s": tail_s,
        "rtf": total_wall / audio_s if audio_s > 0 else float("inf"),
        "drift_s": drift,
        "updates": updates,
        "words": len(result.get("text", "").split()),
        "empty": not result.get("text", "").strip(),
    }


async def run_stream_point(url, rows, packet_ms, realtime, ramp_s):
    async def launch(i, row):
        if ramp_s:
            await asyncio.sleep(ramp_s * i / max(1, len(rows)))
        return await stream_session(url, row, packet_ms, realtime)

    return await asyncio.gather(*[launch(i, r) for i, r in enumerate(rows)])


async def offline_session(url: str, rows: list[dict]) -> dict:
    """One POST /asr/transcribe call with a batch of paths (single language)."""
    import aiohttp

    t0 = time.perf_counter()
    payload = {"paths": [r["audio_path"] for r in rows], "lang": rows[0]["language"]}
    try:
        async with (
            aiohttp.ClientSession() as sess,
            sess.post(
                f"{url.rstrip('/')}/asr/transcribe",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=900),
            ) as resp,
        ):
            body = await resp.json()
    except Exception as e:
        return {"ok": False, "err": repr(e)[:160]}
    if "error" in body:
        return {"ok": False, "err": str(body["error"])[:160]}
    wall = time.perf_counter() - t0
    audio_s = sum(r.get("duration", 0.0) for r in rows)
    return {
        "ok": True,
        "wall_s": wall,
        "audio_s": audio_s,
        "n": len(rows),
        "rtfx": audio_s / wall if wall > 0 else 0.0,
        "empty": sum(1 for x in body.get("results", []) if not x.get("text", "").strip()),
    }


def report_stream(c: int, res: list[dict], wall: float) -> None:
    ok = [r for r in res if r.get("ok")]
    fail = len(res) - len(ok)
    if not ok:
        print(f"{c:>4} {'-':>5} {fail:>5}   all sessions failed: {res[0].get('err', '')[:60]}")
        return
    rtf = [r["rtf"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] == r["ttft_s"]]  # drop NaN
    drift = [r["drift_s"] for r in ok]
    # A session is REALTIME-OK if it never fell behind: rtf <= 1 (it finished in
    # about the audio duration) and its lag did not grow through the session.
    rt_ok = sum(1 for r in ok if r["rtf"] <= 1.15 and r["drift_s"] < 1.0)
    audio = sum(r["audio_s"] for r in ok)
    print(
        f"{c:>4} {len(ok):>5} {fail:>5} {pct(ttft, 50):>8.2f} {pct(ttft, 95):>8.2f} "
        f"{pct(rtf, 50):>8.2f} {pct(rtf, 95):>8.2f} {pct(drift, 50):>8.2f} "
        f"{pct(drift, 95):>8.2f} {pct([r['tail_s'] for r in ok], 95):>7.2f} "
        f"{statistics.fmean([r['updates'] for r in ok]):>6.1f} "
        f"{audio / wall:>7.1f} {100.0 * rt_ok / len(ok):>6.0f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--url", default="ws://localhost:8000")
    p.add_argument("--jsonl", required=True, help="{'audio_path','language','duration'} per line")
    p.add_argument("--mode", choices=("stream", "offline"), default="stream")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--packet-ms", type=int, default=200)
    p.add_argument(
        "--no-realtime",
        action="store_true",
        help="push audio as fast as possible. Measures batch throughput, NOT streaming "
        "behaviour — the realtime pacing is what makes lag/drift meaningful.",
    )
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=8, help="offline mode: paths per request")
    a = p.parse_args(argv)

    if a.mode == "offline":
        base = a.url.replace("ws://", "http://").replace("wss://", "https://")
        print(f"{'C':>4} {'ok':>5} {'fail':>5} {'wall_s':>8} {'RTFx':>9} {'empty':>6}")
        for c in a.concurrency:
            rows = load_rows(a.jsonl, c * a.batch)
            groups = [rows[i * a.batch : (i + 1) * a.batch] for i in range(c)]
            # each request must be single-language (the prompt encodes it)
            groups = [[r for r in g if r["language"] == g[0]["language"]] for g in groups]

            async def _all(groups=groups):
                return await asyncio.gather(*[offline_session(base, g) for g in groups])

            t0 = time.perf_counter()
            res = asyncio.run(_all())
            wall = time.perf_counter() - t0
            ok = [r for r in res if r.get("ok")]
            audio = sum(r["audio_s"] for r in ok)
            print(
                f"{c:>4} {len(ok):>5} {len(res) - len(ok):>5} {wall:>8.1f} "
                f"{audio / wall if wall else 0:>9.1f} {sum(r['empty'] for r in ok):>6}"
            )
        return 0

    print(
        f"{a.jsonl} | realtime pacing={'off' if a.no_realtime else 'on'} | packet={a.packet_ms}ms"
    )
    print(
        "rt_ok% = sessions that kept up (rtf<=1.15 and lag did not grow). "
        "drift>0 means the session fell progressively behind."
    )
    print(
        f"{'C':>4} {'ok':>5} {'fail':>5} {'ttft50':>8} {'ttft95':>8} {'rtf50':>8} {'rtf95':>8} "
        f"{'drift50':>8} {'drift95':>8} {'tail95':>7} {'upd':>6} {'xRT':>7} {'rt_ok%':>6}"
    )
    for c in a.concurrency:
        rows = load_rows(a.jsonl, c)
        t0 = time.perf_counter()
        res = asyncio.run(run_stream_point(a.url, rows, a.packet_ms, not a.no_realtime, a.ramp_s))
        report_stream(c, res, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
