"""Chunk-plan harness: dry-run inspector + GPU smoke modes for long-form TTS.

The default mode is a CPU-only DRY RUN: it shows exactly how ``chunk_text`` /
``plan_dialogue_chunks`` would split an input, with per-chunk serialized cost,
boundary kind (sentence / clause / hard), duration and token estimates, and
WARN lines for hard cuts and token-ceiling risk. Exit code 0 = clean plan,
1 = plan has warnings, 2 = usage error.

  python -m bodhan_genai.tts.engine.chunk_harness --text "..."
  python -m bodhan_genai.tts.engine.chunk_harness --text-file article.txt --max-chars 240
  python -m bodhan_genai.tts.engine.chunk_harness --dialogue-json examples/tts/dialogue_sample.json
  python -m bodhan_genai.tts.engine.chunk_harness --text "..." --json

GPU smoke modes (heavy imports happen only inside these branches):

  python -m bodhan_genai.tts.engine.chunk_harness --text-file article.txt \\
      --synthesize --model /path/to/ckpt --out-dir out/harness
  python -m bodhan_genai.tts.engine.chunk_harness --dialogue-json d.json \\
      --stream --model /path/to/ckpt
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from bodhan_genai.tts.engine.chunked import (
    _CLOSER_RUN_RE,
    _SEP,
    _TERM_RUN_RE,
    ChunkedIndicStreamingTTS,
    _turn_cost,
    chunk_text,
    plan_dialogue_chunks,
)

# Rule-of-thumb rates for Hindi/English speech at 24 kHz SNAC framing.
_CHARS_PER_SECOND = 14.0
_TOKENS_PER_SECOND = 82.0
_TOKEN_CEILING = 2048
# Clause marks, lock-step with chunked._CLAUSE_RE: "," ";" ":" plus the
# fullwidth comma/semicolon/colon and the ideographic comma.
_CLAUSE_MARKS = frozenset(",;:，；：、")  # noqa: RUF001


# --- plan computation ----------------------------------------------------------


def _classify(text: str) -> str:
    """Boundary kind of a chunk from its final character, using chunked.py's
    own character classes: terminator or trailing closer -> "sentence", clause
    mark -> "clause", anything else -> "hard"."""
    tail = text.rstrip()
    if not tail:
        return "hard"
    ch = tail[-1]
    if _TERM_RUN_RE.fullmatch(ch) or _CLOSER_RUN_RE.fullmatch(ch):
        return "sentence"
    if ch in _CLAUSE_MARKS:
        return "clause"
    return "hard"


def _estimates(chars: int) -> tuple[float, int]:
    """(est_seconds, est_tokens) for ``chars`` characters of speech text."""
    est_seconds = chars / _CHARS_PER_SECOND
    return round(est_seconds, 2), round(est_seconds * _TOKENS_PER_SECOND)


def _plan_text(text: str, args: argparse.Namespace) -> dict:
    chunks = chunk_text(text, min_chars=args.min_chars, max_chars=args.max_chars)
    rows = []
    for i, c in enumerate(chunks):
        est_s, est_tok = _estimates(len(c))
        rows.append(
            {
                "index": i,
                "chars": len(c),
                "cost": len(c),
                "kind": _classify(c),
                "est_seconds": est_s,
                "est_tokens": est_tok,
            }
        )
    return {"mode": "text", "n_chunks": len(chunks), "chunks": rows}


def _nonws(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _turn_end_flags(messages: list[dict], chunks: list[list[dict]]) -> list[bool]:
    """Per-chunk flag: does the chunk's last segment end an ORIGINAL turn?

    Exploded segments consume their turn's text in order; strip/re-join only
    touches whitespace, so a cursor over non-whitespace character counts is
    exact: a chunk is turn-end iff its last segment exhausts the current
    original (sanitized, non-empty) turn."""
    counts = [
        _nonws(str(m.get("text", "") or "").strip())
        for m in messages
        if str(m.get("text", "") or "").strip()
    ]
    flags: list[bool] = []
    ti = 0
    consumed = 0
    for chunk in chunks:
        end = False
        for seg in chunk:
            consumed += _nonws(seg["text"])
            end = False
            if ti < len(counts) and consumed >= counts[ti]:
                consumed = 0
                ti += 1
                end = True
        flags.append(end)
    return flags


def _plan_dialogue(messages: list[dict], args: argparse.Namespace) -> dict:
    chunks = plan_dialogue_chunks(
        messages, max_chars=args.max_chars, long_turn_chars=args.long_turn_chars
    )
    flags = _turn_end_flags(messages, chunks)
    rows = []
    for i, (chunk, turn_end) in enumerate(zip(chunks, flags, strict=True)):
        chars = sum(len(m["text"]) for m in chunk)
        # SAME arithmetic as the packer: len(format_messages(chunk)).
        cost = sum(_turn_cost(m) for m in chunk) + _SEP * (len(chunk) - 1)
        est_s, est_tok = _estimates(chars)
        rows.append(
            {
                "index": i,
                "chars": chars,
                "cost": cost,
                "kind": _classify(chunk[-1]["text"]),
                "turn_end": turn_end,
                "speakers": list(dict.fromkeys(m["speaker"] for m in chunk)),
                "est_seconds": est_s,
                "est_tokens": est_tok,
            }
        )
    return {"mode": "dialogue", "n_chunks": len(chunks), "chunks": rows}


def _collect_warnings(rows: list[dict], max_chunks: int) -> list[str]:
    warnings: list[str] = []
    for r in rows:
        if r["kind"] == "hard":
            warnings.append(
                f"WARN chunk {r['index']}: hard cut "
                "(chunk does not end at a sentence or clause boundary)"
            )
        if r["est_tokens"] > _TOKEN_CEILING:
            warnings.append(
                f"WARN chunk {r['index']}: est_tokens {r['est_tokens']} > {_TOKEN_CEILING} "
                "(may hit the generation ceiling)"
            )
    if len(rows) > max_chunks:
        warnings.append(f"WARN plan: {len(rows)} chunks exceed --max-chunks {max_chunks}")
    return warnings


def _print_table(plan: dict, warnings: list[str]) -> None:
    print(
        f"{'idx':>4}  {'chars':>5}  {'cost':>5}  {'kind':<18}  "
        f"{'speakers':<16}  {'est_s':>6}  {'est_tok':>7}"
    )
    for r in plan["chunks"]:
        kind = r["kind"] + ("+turn-end" if r.get("turn_end") else "")
        speakers = ",".join(r.get("speakers", [])) or "-"
        print(
            f"{r['index']:>4}  {r['chars']:>5}  {r['cost']:>5}  {kind:<18}  "
            f"{speakers:<16}  {r['est_seconds']:>6.2f}  {r['est_tokens']:>7}"
        )
    for w in warnings:
        print(w)
    total_chars = sum(r["chars"] for r in plan["chunks"])
    total_s = sum(r["est_seconds"] for r in plan["chunks"])
    print(
        f"total: {plan['n_chunks']} chunks, {total_chars} chars, "
        f"est {total_s:.1f} s, {len(warnings)} warnings"
    )


# --- GPU branches (heavy imports stay inside) ------------------------------------


def _sampling_kwargs(args: argparse.Namespace) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }


def _engine_kwargs(args: argparse.Namespace) -> dict:
    kw: dict = {}
    if args.tokenizer:
        kw["tokenizer"] = args.tokenizer
    if args.snac:
        kw["snac_model_path"] = args.snac
    return kw


def _wrap_chunked(engine, args: argparse.Namespace) -> ChunkedIndicStreamingTTS:
    return ChunkedIndicStreamingTTS(
        engine,
        min_chunk_chars=args.min_chars,
        max_chunk_chars=args.max_chars,
        max_chunks=args.max_chunks,
        gap_ms=args.gap_ms,
    )


def _resolve_method(name: str, *candidates):
    """First callable ``name`` across candidates. The long-form methods land
    incrementally (wrapper first, possibly engine-level later), so resolution
    is getattr-tolerant instead of hard-binding one home."""
    for obj in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn
    return None


def _lufs(y, sample_rate: int) -> float | None:
    try:
        import pyloudnorm
    except ImportError:
        return None
    try:
        return float(pyloudnorm.Meter(sample_rate).integrated_loudness(y.astype("float64")))
    except Exception:  # too-short chunk etc. — report n/a, don't kill the run
        return None


def _run_synthesize(args: argparse.Namespace, text: str | None, messages: list[dict] | None) -> int:
    import math

    from bodhan_genai.tts.engine.loudness import active_rms, rms
    from bodhan_genai.tts.engine.offline import IndicTTSEngine
    from bodhan_genai.tts.inference.audio_io import write_wav_24k

    engine = IndicTTSEngine(args.model, **_engine_kwargs(args))
    wrapper = _wrap_chunked(engine, args)
    sampling = _sampling_kwargs(args)

    if messages is None:
        fn = _resolve_method("synthesize_long", wrapper, engine)
        if fn is None:
            print("error: synthesize_long is not available on this engine", file=sys.stderr)
            return 2
        merged, results = fn(text, speaker=args.speaker, return_chunks=True, **sampling)
    else:
        fn = _resolve_method("synthesize_conversation_long", wrapper, engine)
        if fn is None:
            print(
                "error: synthesize_conversation_long is not available on this engine",
                file=sys.stderr,
            )
            return 2
        merged, results = fn(messages, return_chunks=True, **sampling)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sr = merged.sample_rate
    actives: list[float] = []
    print(
        f"{'idx':>4}  {'dur_s':>6}  {'gen_s':>6}  {'dec_s':>6}  {'rtf':>5}  "
        f"{'rms':>7}  {'act_rms':>7}  {'lufs':>7}"
    )
    for i, r in enumerate(results):
        write_wav_24k(args.out_dir / f"chunk_{i:03d}.wav", r.audio)
        dur = r.duration_s
        rtf = (r.gen_time_s + r.decode_time_s) / dur if dur > 0 else float("inf")
        active = active_rms(r.audio, sample_rate=sr)
        actives.append(active)
        lufs = _lufs(r.audio, sr)
        lufs_str = f"{lufs:>7.1f}" if lufs is not None else f"{'n/a':>7}"
        print(
            f"{i:>4}  {dur:>6.2f}  {r.gen_time_s:>6.2f}  {r.decode_time_s:>6.2f}  "
            f"{rtf:>5.2f}  {rms(r.audio):>7.4f}  {active:>7.4f}  {lufs_str}"
        )
    write_wav_24k(args.out_dir / "combined.wav", merged.audio)

    voiced = [a for a in actives if a > 0]
    if len(voiced) >= 2:
        spread_db = 20.0 * math.log10(max(voiced) / min(voiced))
        verdict = "consistent" if spread_db <= 3.0 else "INCONSISTENT"
        print(
            f"volume consistency: active-RMS spread {spread_db:.2f} dB "
            f"across {len(voiced)} voiced chunks ({verdict})"
        )
    print(
        f"wrote {len(results)} chunk wavs + combined.wav ({merged.duration_s:.2f} s) to {args.out_dir}"
    )
    return 0


def _run_stream(args: argparse.Namespace, text: str | None, messages: list[dict] | None) -> int:
    import numpy as np

    from bodhan_genai.tts.engine.streaming import IndicStreamingTTSEngine
    from bodhan_genai.tts.inference.audio_io import write_wav_24k

    engine = IndicStreamingTTSEngine(args.model, **_engine_kwargs(args))
    wrapper = _wrap_chunked(engine, args)
    sampling = _sampling_kwargs(args)

    if messages is None:
        name = "stream_long_sync"
        fn = _resolve_method(name, wrapper, engine)
        run = (lambda: fn(text, speaker=args.speaker, **sampling)) if fn else None
    else:
        name = "stream_conversation_long_sync"
        fn = _resolve_method(name, wrapper, engine)
        run = (lambda: fn(messages, **sampling)) if fn else None
    if run is None:
        print(f"error: {name} is not available on this engine", file=sys.stderr)
        return 2

    sr = int(getattr(engine, "sample_rate", 24_000))
    parts: list[bytes] = []
    stamps: list[float] = []
    t0 = time.perf_counter()
    for msg in run():
        stamps.append(time.perf_counter())
        parts.append(msg)
    wall = time.perf_counter() - t0

    total_bytes = sum(len(p) for p in parts)
    audio_s = total_bytes / 2 / sr
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    print(f"messages: {len(parts)}")
    if stamps:
        print(f"ttfa_s: {stamps[0] - t0:.3f}")
    print(
        f"wall_s: {wall:.3f}  audio_s: {audio_s:.3f}  "
        f"xrt: {audio_s / wall if wall > 0 else 0.0:.2f}"
    )
    if gaps:
        print(f"mean_inter_message_gap_s: {sum(gaps) / len(gaps):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    y = np.frombuffer(b"".join(parts), dtype=np.int16).astype(np.float32) / 32767.0
    write_wav_24k(args.out_dir / "combined.wav", y)
    print(f"wrote combined.wav ({audio_s:.2f} s) to {args.out_dir}")
    return 0


# --- CLI -------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.tts.engine.chunk_harness",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="inline text to plan")
    src.add_argument("--text-file", type=Path, help="UTF-8 text file to plan")
    src.add_argument(
        "--dialogue-json",
        type=Path,
        help='JSON file: list of {"speaker": ..., "text": ...} turns',
    )
    p.add_argument("--min-chars", type=int, default=16, help="min chunk chars (best-effort)")
    p.add_argument("--max-chars", type=int, default=300, help="strict max chunk chars")
    p.add_argument(
        "--long-turn-chars",
        type=int,
        default=None,
        help="keep-turns-intact escape hatch for dialogue (see plan_dialogue_chunks)",
    )
    p.add_argument("--gap-ms", type=float, default=250.0, help="inter-chunk silence gap")
    p.add_argument("--max-chunks", type=int, default=256, help="chunk budget (dry-run: warn only)")
    p.add_argument("--json", action="store_true", dest="as_json", help="emit the plan as JSON")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--synthesize", action="store_true", help="GPU: offline long-form synthesis + wav report"
    )
    mode.add_argument(
        "--stream", action="store_true", help="GPU: streaming long-form synthesis + TTFA report"
    )
    p.add_argument("--model", help="model checkpoint (required for --synthesize/--stream)")
    p.add_argument("--tokenizer", default=None, help="tokenizer path (default: model path)")
    p.add_argument("--snac", default=None, help="SNAC codec model path")
    p.add_argument("--speaker", default="", help="speaker tag for plain-text synthesis")
    p.add_argument("--out-dir", type=Path, default=Path("out/harness"), help="wav output dir")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    text: str | None = None
    messages: list[dict] | None = None
    if args.dialogue_json is not None:
        try:
            messages = json.loads(args.dialogue_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            parser.error(f"--dialogue-json {args.dialogue_json}: {e}")
        if not isinstance(messages, list):
            parser.error(f"--dialogue-json {args.dialogue_json}: expected a JSON list of turns")
    elif args.text_file is not None:
        try:
            text = args.text_file.read_text(encoding="utf-8")
        except OSError as e:
            parser.error(f"--text-file {args.text_file}: {e}")
    else:
        text = args.text

    if args.synthesize or args.stream:
        if not args.model:
            parser.error("--model is required with --synthesize/--stream")
        if args.synthesize:
            return _run_synthesize(args, text, messages)
        return _run_stream(args, text, messages)

    plan = _plan_text(text, args) if messages is None else _plan_dialogue(messages, args)
    warnings = _collect_warnings(plan["chunks"], args.max_chunks)
    if args.as_json:
        print(json.dumps({**plan, "warnings": warnings}, ensure_ascii=False, indent=2))
    else:
        _print_table(plan, warnings)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
