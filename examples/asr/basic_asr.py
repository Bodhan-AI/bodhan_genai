#!/usr/bin/env python3
"""Minimal ASR: transcribe one or more audio files.

    PYTHONPATH=src python examples/asr/basic_asr.py \
        --model /path/to/indic-transcribe-hf --lang hi audio1.wav audio2.wav

The model is language-conditioned: a wrong --lang produces confidently wrong
script rather than obvious garbage. Use examples/asr/detect_language.py (or
--detect here) if you do not know it.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", nargs="+", help="audio file(s) to transcribe")
    p.add_argument("--model", required=True, help="converted IndicTranscribe HF checkpoint dir")
    p.add_argument("--lang", default=None, help="language code, e.g. hi (see --detect)")
    p.add_argument(
        "--detect",
        action="store_true",
        help="identify the language from the first file instead of passing --lang",
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    p.add_argument(
        "--long",
        action="store_true",
        help="use the chunked long-form path (needed past ~60 s of audio)",
    )
    args = p.parse_args()

    if not args.lang and not args.detect:
        p.error("pass --lang, or --detect to identify it from the audio")

    import torch

    from bodhan_genai.asr import IndicASREngine

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = IndicASREngine(args.model, device=device, dtype=dtype)

    lang = args.lang
    if args.detect:
        top = engine.detect_language([args.audio[0]])[0]
        lang, prob = top[0]
        print(f"detected language: {lang} (p={prob:.3f})")
        if lang in ("hi", "ur") and prob < 0.95:
            print("  note: hi/ur are the pair this model cannot reliably separate")

    if args.long:
        for path in args.audio:
            text = engine.transcribe_long(path, lang)
            print(f"{path}\t{text}")
    else:
        # One batch, so every file must share a language (the prompt encodes it).
        for path, text in zip(args.audio, engine.transcribe_batch(args.audio, lang), strict=True):
            print(f"{path}\t{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
