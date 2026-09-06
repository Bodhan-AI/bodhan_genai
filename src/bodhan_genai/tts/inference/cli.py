"""``python -m bodhan_genai.tts.inference.cli`` — dispatcher for the two inference paths.

  python -m bodhan_genai.tts.inference.cli vllm ...   -> bodhan_genai.tts.inference.offline_vllm (batch, multi-GPU Ray)
  python -m bodhan_genai.tts.inference.cli hf ...     -> bodhan_genai.tts.inference.hf_generate (single prompt, one GPU)

All remaining argv is forwarded verbatim to the chosen entry point, so
``python -m bodhan_genai.tts.inference.cli vllm --help`` shows that path's full flag set. No heavy imports
happen until the subcommand actually runs.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.tts.inference.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "vllm",
        add_help=False,
        help="Offline batch generation (vLLM engines + batched SNAC decode via Ray).",
    )
    sub.add_parser(
        "hf", add_help=False, help="Single-prompt sample-quality generation via HF model.generate."
    )

    args, rest = p.parse_known_args(argv)
    if args.command == "vllm":
        from bodhan_genai.tts.inference.offline_vllm import main as vllm_main

        return vllm_main(rest)
    if args.command == "hf":
        from bodhan_genai.tts.inference.hf_generate import main as hf_main

        return hf_main(rest)
    p.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
