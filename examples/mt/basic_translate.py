#!/usr/bin/env python3
"""One-shot translation without a server — the smallest useful IndicTranslate program.

    python examples/mt/basic_translate.py --text "The committee approved the proposal."
    python examples/mt/basic_translate.py --tgt-lang Tamil --backend hf
    python examples/mt/basic_translate.py --model /path/to/merged-ckpt --tgt-lang mar_Deva

The whole API is `translate(text, tgt_lang=...)`. Note what you never pass: a
source language. The prompt names only the target and the model infers the source,
which is why the same call handles English->Hindi and Hindi->English.
"""

from __future__ import annotations

import argparse

from bodhan_genai.mt import IndicMTEngine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="bodhan-ai/indic-translate")
    p.add_argument("--tgt-lang", default="Hindi", help="name or FLORES code, e.g. hin_Deva")
    p.add_argument("--text", default="The committee approved the proposal after a long debate.")
    p.add_argument(
        "--backend",
        default="vllm",
        choices=("vllm", "hf"),
        help="vllm for throughput, hf for a dependency-light single run",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    # The engine is a context manager; one engine per process, closed on exit.
    with IndicMTEngine(args.model, backend=args.backend) as engine:
        result = engine.translate(
            args.text, tgt_lang=args.tgt_lang, max_new_tokens=args.max_new_tokens
        )
        if not result.ok:
            raise SystemExit(f"translation failed: {result.error}")

        print(f"source ({'auto-detected'}): {result.source}")
        print(f"target ({result.tgt_lang}): {result.text}")
        print(f"[{result.gen_time_s:.2f}s]")

        # Round-tripping back to English uses the identical call.
        back = engine.translate(result.text, tgt_lang="English")
        print(f"back-translation      : {back.text}")


if __name__ == "__main__":
    main()
