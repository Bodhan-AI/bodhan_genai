"""Minimal single-prompt TTS example (HF generate path, one GPU).

Usage:
    PYTHONPATH=src python examples/tts/basic_tts.py \
        --model <sft_checkpoint> --text "Namaste, kaise hain aap?" --out hello.wav
"""

from __future__ import annotations

import argparse

# Direct module import — always works; `from bodhan_genai.tts import IndicTTSEngine`
# is the equivalent lazy top-level export.
from bodhan_genai.tts.engine.offline import IndicTTSEngine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="SFT checkpoint dir or hub id.")
    p.add_argument("--text", required=True, help="Text to synthesize.")
    p.add_argument("--speaker", default="", help="Optional speaker id.")
    p.add_argument("--out", default="basic_tts.wav")
    args = p.parse_args()

    # Engine defaults: SamplingConfig (temperature 0.6, top_p 0.95,
    # repetition_penalty 1.1, max_new_tokens 2048) and SNAC decode at 24 kHz.
    with IndicTTSEngine(args.model, backend="hf") as engine:
        result = engine.synthesize(args.text, speaker=args.speaker)
        out = result.save(args.out)
    print(f"Wrote {out} ({result.duration_s:.2f}s, rtf {result.rtf:.2f})")


if __name__ == "__main__":
    main()
