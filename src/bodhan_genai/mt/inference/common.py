"""Shared argparse groups and IO for the two MT inference entry points.

Keeps ``hf_generate`` and ``offline_vllm`` presenting the same flags for the same
concepts, so a command written against one transfers to the other. Pure stdlib —
no torch / vllm / transformers here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from bodhan_genai.mt.templates.prompt import LANGUAGE_NAMES

if TYPE_CHECKING:
    from bodhan_genai.mt.engine.types import MTResult


def add_io_args(p: argparse.ArgumentParser) -> None:
    """Input selection and where results go."""
    src = p.add_argument_group("input")
    src.add_argument("--text", help="a single segment to translate")
    src.add_argument(
        "--input-file",
        help="file of segments, one per line (or one document with --document)",
    )
    src.add_argument(
        "--document",
        action="store_true",
        help="treat the whole input as one document instead of line-per-segment",
    )
    p.add_argument("--output-file", help="write JSONL results here instead of plain stdout")


def add_language_args(p: argparse.ArgumentParser) -> None:
    """Target language, plus the source language for bookkeeping only."""
    lang = p.add_argument_group("language")
    lang.add_argument(
        "--tgt-lang",
        help="target language name or FLORES-style code, e.g. Hindi / hin_Deva",
    )
    lang.add_argument(
        "--src-lang",
        default=None,
        help="recorded in the JSONL output only -- the prompt never names the "
        "source language, the model infers it",
    )
    p.add_argument(
        "--list-languages",
        action="store_true",
        help="print the supported target languages and exit",
    )


def add_sampling_args(p: argparse.ArgumentParser) -> None:
    """Decoding knobs. Greedy by default: reproducible, and best for translation."""
    gen = p.add_argument_group("generation")
    gen.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="512 suits sentences; ~2048 a paragraph, ~8192 a document",
    )
    gen.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy (recommended for translation)",
    )
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--repetition-penalty", type=float, default=1.0)


def print_languages() -> None:
    """Print the supported target languages, code and prompt name."""
    width = max(len(code) for code in LANGUAGE_NAMES)
    for code, name in LANGUAGE_NAMES.items():
        print(f"  {code:<{width}}  {name}")


def read_inputs(args: argparse.Namespace) -> list[str]:
    """Resolve --text / --input-file / stdin into a list of segments."""
    if args.text:
        return [args.text]
    if args.input_file:
        raw = Path(args.input_file).read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            sys.exit("no input: pass --text, --input-file, or pipe to stdin")
        raw = sys.stdin.read()

    if args.document:
        return [raw.strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def write_results(results: list[MTResult], output_file: str | None) -> None:
    """Write JSONL to ``output_file``, or bare translations to stdout."""
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.as_record(), ensure_ascii=False) + "\n")
        print(f"[run] wrote {len(results)} records to {output_file}", file=sys.stderr)
        failed = [r for r in results if not r.ok]
        if failed:
            print(f"[run] {len(failed)} request(s) failed", file=sys.stderr)
        return

    for r in results:
        if r.ok:
            print(r.text)
        else:
            print(f"[error] {r.error}", file=sys.stderr)


def sampling_from_args(args: argparse.Namespace):
    """Build an ``MTSamplingConfig`` from parsed args (import deferred)."""
    from bodhan_genai.mt.engine.types import MTSamplingConfig

    return MTSamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
