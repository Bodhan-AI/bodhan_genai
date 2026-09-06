"""``python -m bodhan_genai.mt.inference.hf_generate`` — reference inference with Transformers.

The dependency-light path: no vLLM, one GPU, and the only path that can run a PEFT
adapter without merging it first. Use ``offline_vllm`` for corpus-scale throughput.

Examples
--------
    # single segment
    python -m bodhan_genai.mt.inference.hf_generate \\
        --model bodhan-ai/indic-translate \\
        --tgt-lang Hindi \\
        --text "The committee approved the proposal after a long debate."

    # a file of segments, one per line, written out as JSONL
    python -m bodhan_genai.mt.inference.hf_generate --model $CKPT --tgt-lang mar_Deva \\
        --input-file segments.txt --output-file out.jsonl --batch-size 8

    # a whole document in one request (structure is preserved)
    python -m bodhan_genai.mt.inference.hf_generate --model $CKPT --tgt-lang Tamil \\
        --document --input-file article.txt --max-new-tokens 8192

    # a LoRA adapter, unmerged
    python -m bodhan_genai.mt.inference.hf_generate --model $BASE --adapter-dir $ADAPTER \\
        --tgt-lang Hindi --text "Hello world"
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
from bodhan_genai.mt.templates.prompt import resolve_language


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.inference.hf_generate",
        description="Translate text with IndicTranslate via HuggingFace generate().",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="bodhan-ai/indic-translate",
        help="local path or Hugging Face repo id",
    )
    p.add_argument(
        "--adapter-dir",
        default=None,
        help="PEFT adapter to attach on top of --model (HF backend only)",
    )
    add_language_args(p)
    add_io_args(p)
    p.add_argument("--interactive", action="store_true", help="read segments from a prompt loop")
    add_sampling_args(p)
    p.add_argument("--batch-size", type=int, default=4)

    runtime = p.add_argument_group("runtime")
    runtime.add_argument(
        "--device", default="auto", help="device_map value, e.g. auto / cuda:0 / cpu"
    )
    runtime.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    runtime.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "eager", "flash_attention_2"),
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

    # Fail before the 16 GB load if the language is wrong.
    tgt_name = resolve_language(args.tgt_lang)
    print(f"[run] target language: {tgt_name}", file=sys.stderr)

    from bodhan_genai.mt.engine.offline import IndicMTEngine

    engine = IndicMTEngine(
        args.model,
        backend="hf",
        adapter_dir=args.adapter_dir,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        sampling=sampling_from_args(args),
    )

    try:
        if args.interactive:
            print(
                f"Interactive mode -> {tgt_name}. Ctrl-D or an empty line quits.",
                file=sys.stderr,
            )
            while True:
                try:
                    line = input("src> ").strip()
                except EOFError:
                    break
                if not line:
                    break
                result = engine.translate(line, tgt_lang=args.tgt_lang, src_lang=args.src_lang)
                print("tgt>", result.text if result.ok else f"[error] {result.error}")
            return 0

        texts = read_inputs(args)
        print(
            f"[run] {len(texts)} request(s), batch size {args.batch_size}",
            file=sys.stderr,
        )

        results = []
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            results.extend(
                engine.translate_batch(batch, tgt_lang=args.tgt_lang, src_lang=args.src_lang)
            )
            done = min(start + args.batch_size, len(texts))
            print(f"[run] {done}/{len(texts)}", file=sys.stderr)

        write_results(results, args.output_file)
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
