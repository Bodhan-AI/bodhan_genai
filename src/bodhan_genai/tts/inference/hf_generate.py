"""Single-GPU sample-quality TTS generation via plain HF ``model.generate``.

Slow path for listening tests and debugging — one prompt, one WAV. Use
``bodhan_genai.tts.inference.offline_vllm`` for anything batch-sized. Supports
optional PEFT adapters (``--adapter_dir``).

Thin CLI over :class:`bodhan_genai.tts.engine.offline.IndicTTSEngine` with
``backend="hf"``. All heavy imports (torch / transformers / snac / peft) stay
inside the engine, so ``--help`` works on a CPU-only box.
"""

from __future__ import annotations

import argparse
import logging

from bodhan_genai.tts.engine.types import SamplingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULTS = SamplingConfig()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="bodhan-ai/indic-speak", help="HF checkpoint dir or hub id.")
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer dir; defaults to --model. Point at the extended audio "
        "tokenizer when step-checkpoints lack tokenizer files.",
    )
    p.add_argument("--adapter_dir", default=None, help="Optional PEFT/LoRA adapter dir.")
    p.add_argument("--snac_model_path", default="hubertsiuzdak/snac_24khz")
    p.add_argument("--text", required=True, help="Text to synthesize.")
    p.add_argument("--speaker", default="", help="Optional speaker id for the metadata prefix.")
    p.add_argument("--out", default="out.wav")
    # sampling (defaults come from the shared SamplingConfig)
    p.add_argument("--temperature", type=float, default=_DEFAULTS.temperature)
    p.add_argument("--top_p", type=float, default=_DEFAULTS.top_p)
    p.add_argument("--repetition_penalty", type=float, default=_DEFAULTS.repetition_penalty)
    p.add_argument("--max_new_tokens", type=int, default=_DEFAULTS.max_new_tokens)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from bodhan_genai.tts.engine.offline import IndicTTSEngine

    with IndicTTSEngine(
        args.model,
        backend="hf",
        tokenizer=args.tokenizer,
        snac_model_path=args.snac_model_path,
        adapter_dir=args.adapter_dir,
    ) as engine:
        result = engine.synthesize(
            args.text,
            speaker=args.speaker,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        out = result.save(args.out)
    logger.info("Wrote %s (%.2fs of audio)", out, result.duration_s)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
