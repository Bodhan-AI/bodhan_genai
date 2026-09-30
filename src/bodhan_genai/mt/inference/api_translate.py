"""Translate over the hosted Bodhan API — no GPU, no checkpoint, no extras.

    python -m bodhan_genai.mt.inference.api_translate --tgt-lang kn \\
        --text "The meeting has been postponed."

    BODHAN_API_KEY=sk-... python -m bodhan_genai.mt.inference.api_translate \\
        --tgt-lang Tamil --input-file segments.txt --output-file out.jsonl

The key comes from ``BODHAN_API_KEY`` or ``BODHAN_API_TOKEN``, or ``--api-key``.

**Translation only.** The hosted route serves native-script translation; Roman
output, transliteration and code-mix are not reachable over it and are refused
rather than silently returning native script. Use the ``vllm`` or ``hf`` paths
for those — see ``docs/mt/api.md``.
"""

from __future__ import annotations

import argparse
import sys

from bodhan_genai.mt.inference.common import (
    add_io_args,
    add_language_args,
    add_sampling_args,
    print_languages,
    read_inputs,
    sampling_from_args,
    write_results,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.inference.api_translate",
        description="Translate through the hosted Bodhan API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_language_args(p)
    add_io_args(p)
    add_sampling_args(p)
    p.add_argument(
        "--api-key",
        default=None,
        help="defaults to $BODHAN_API_KEY, then $BODHAN_API_TOKEN",
    )
    p.add_argument("--api-base-url", default=None, help="override the service base URL")
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel in-flight requests. The gateway rate-limits per key, and a "
        "retry counts against it, so keep this modest on long runs.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_languages:
        print_languages()
        return 0
    if not args.tgt_lang:
        print("--tgt-lang is required (or use --list-languages)", file=sys.stderr)
        return 2

    from bodhan_genai.mt.engine.backends import find_api_key
    from bodhan_genai.mt.engine.offline import IndicMTEngine

    if args.api_key is None and find_api_key() is None:
        print(
            "no API key: export BODHAN_API_KEY (or BODHAN_API_TOKEN), or pass --api-key.\n"
            "To run locally instead: python -m bodhan_genai.mt.inference.cli vllm --help",
            file=sys.stderr,
        )
        return 2

    engine = IndicMTEngine(
        backend="api",
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        sampling=sampling_from_args(args),
    )
    engine._backend.num_workers = max(1, args.concurrency)

    try:
        texts = read_inputs(args)
        print(f"[run] {len(texts)} request(s) -> hosted API", file=sys.stderr)
        results = engine.translate_batch(texts, tgt_lang=args.tgt_lang, src_lang=args.src_lang)
        write_results(results, args.output_file)
        return 0 if all(r.ok for r in results) else 1
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
