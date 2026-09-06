#!/usr/bin/env python3
"""Long-form ASR: transcribe a long recording with silence-aware chunking.

    PYTHONPATH=src python examples/asr/long_form_asr.py \
        --model /path/to/indic-transcribe-hf --lang hi interview.wav

Whole-file decoding collapses past ~60 s (the checkpoint trains at 30 s and the
decoder emits EOS early), so long audio is split at pauses and the chunk
transcripts joined. Below --chunk-above the file is transcribed whole, because
chunking short audio measurably hurts — see docs/asr/caveats.md.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="long audio file")
    p.add_argument("--model", required=True, help="converted IndicTranscribe HF checkpoint dir")
    p.add_argument("--lang", required=True, help="language code, e.g. hi")
    p.add_argument(
        "--chunk-above",
        type=float,
        default=45.0,
        help="segment audio longer than this many seconds (measured knee: 45)",
    )
    p.add_argument("--chunk-min", type=float, default=15.0)
    p.add_argument("--chunk-max", type=float, default=25.0)
    p.add_argument("--batch-size", type=int, default=64, help="chunks per batch")
    p.add_argument("--srt", default=None, help="also write chunk timings to this .srt file")
    args = p.parse_args()

    import torch

    from bodhan_genai.asr import IndicASREngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = IndicASREngine(args.model, device=device, dtype=torch.bfloat16)

    text, chunks = engine.transcribe_long(
        args.audio,
        args.lang,
        chunk_above=args.chunk_above,
        chunk_min=args.chunk_min,
        chunk_max=args.chunk_max,
        batch_size=args.batch_size,
        return_chunks=True,
    )

    print(f"{len(chunks)} chunk(s)\n")
    for start_s, end_s, chunk_text in chunks:
        print(f"[{start_s:7.2f} - {end_s:7.2f}] {chunk_text}")
    print(f"\n--- joined ---\n{text}")

    if args.srt:

        def stamp(t: float) -> str:
            h, rem = divmod(t, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

        with open(args.srt, "w", encoding="utf-8") as f:
            for i, (start_s, end_s, chunk_text) in enumerate(chunks, 1):
                f.write(f"{i}\n{stamp(start_s)} --> {stamp(end_s)}\n{chunk_text}\n\n")
        print(f"\nwrote {args.srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
