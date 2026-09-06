"""``python -m bodhan_genai.mt.inference.offline_vllm`` — high-throughput batch inference.

Stock vLLM >= 0.20 registers ``Gemma4ForConditionalGeneration``, so the published
checkpoint runs without conversion and without patching vLLM: it ships zeroed
``k_norm`` tensors for its 18 KV-shared layers, which is what vLLM's weight loader
expects. A checkpoint you trained yourself needs
``python -m bodhan_genai.mt.tools.vllm_ready`` first.

Defaults come from ``configs/mt/infer/offline_vllm.yaml`` when ``--config`` is
given; explicit CLI flags always win.

Examples
--------
    # batch translate a file, one segment per line
    python -m bodhan_genai.mt.inference.offline_vllm \\
        --model bodhan-ai/indic-translate --tgt-lang Hindi \\
        --input-file segments.txt --output-file out.jsonl

    # long-context document translation on 2 GPUs
    python -m bodhan_genai.mt.inference.offline_vllm --model $CKPT --tgt-lang Kannada \\
        --document --input-file article.txt \\
        --tensor-parallel-size 2 --max-model-len 32768 --max-new-tokens 8192

For a persistent OpenAI-compatible server instead, use ``scripts/mt/serve.sh``.
"""

from __future__ import annotations

import argparse
import sys

import yaml

from bodhan_genai.mt.engine.offline import DEFAULT_MAX_MODEL_LEN
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

#: Nested YAML sections that are flattened one level onto argparse dests.
_CONFIG_SECTIONS = ("engine", "sampling", "io")


def _apply_yaml_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load a YAML config and install its values as argparse defaults.

    Nested sections (engine/sampling/io) are flattened one level onto the argparse
    dests, so explicit CLI flags always override config values. Unknown keys fail
    loudly to keep the YAML and the CLI in sync.
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    flat: dict[str, object] = {}
    for key, value in raw.items():
        if key in _CONFIG_SECTIONS and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value

    known = {a.dest for a in parser._actions}
    unknown = sorted(set(flat) - known)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown config key(s) {', '.join(unknown)}; "
            f"keys must match the flags in {parser.prog}"
        )
    parser.set_defaults(**flat)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.inference.offline_vllm",
        description="Batch translation with IndicTranslate on vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="YAML config supplying defaults for these flags")
    p.add_argument(
        "--model",
        default="bodhan-ai/indic-translate",
        help="local path or Hugging Face repo id",
    )
    add_language_args(p)
    add_io_args(p)
    add_sampling_args(p)
    p.add_argument("--seed", type=int, default=0, help="only used when --temperature > 0")

    engine = p.add_argument_group("engine")
    engine.add_argument("--tensor-parallel-size", type=int, default=1)
    engine.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="KV-cache window; the architecture allows up to 131072, but only 32768 is validated",
    )
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    engine.add_argument("--dtype", default="bfloat16")
    engine.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip CUDA-graph capture (matches the validated serving config)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Two-pass: read --config with a throwaway parse, install its values as
    # defaults, then parse for real so CLI flags override the file.
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        _apply_yaml_config_defaults(parser, pre.config)
    args = parser.parse_args(argv)

    if args.list_languages:
        print_languages()
        return 0
    if not args.tgt_lang:
        print("--tgt-lang is required (or use --list-languages)", file=sys.stderr)
        return 2

    # Fail before the engine spins up if the language is wrong.
    tgt_name = resolve_language(args.tgt_lang)

    texts = read_inputs(args)
    print(f"[run] {len(texts)} request(s) -> {tgt_name}", file=sys.stderr)

    from bodhan_genai.mt.engine.offline import IndicMTEngine

    sampling = sampling_from_args(args)
    if not sampling.greedy:
        sampling = sampling.merged(seed=args.seed)

    engine = IndicMTEngine(
        args.model,
        backend="vllm",
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        sampling=sampling,
    )

    try:
        # vLLM schedules the whole list as one continuous batch — no manual chunking.
        results = engine.translate_batch(texts, tgt_lang=args.tgt_lang, src_lang=args.src_lang)
        write_results(results, args.output_file)
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
