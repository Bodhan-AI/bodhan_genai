"""``python -m bodhan_genai.mt.training.merge`` — LoRA adapter -> standalone checkpoint.

A PEFT adapter is ~300 MB and needs its base model at load time; vLLM wants one
self-contained directory. This folds ``W + (alpha/r) * B @ A`` back into the base
weights and writes a checkpoint that serves on its own.

Two things a naive ``merge_and_unload`` + ``save_pretrained`` leaves out, both of
which make vLLM refuse the result:

*   **The processor.** A merged text model saves no ``processor_config.json``, but
    the architecture is ``Gemma4ForConditionalGeneration`` and vLLM loads its
    processor. Staged here from the base model.
*   **The KV-shared ``k_norm`` tensors.** Handled separately by
    ``python -m bodhan_genai.mt.tools.vllm_ready``, which this prints as the next
    step (and runs for you with ``--vllm-ready``).

Usage
-----
    python -m bodhan_genai.mt.training.merge \\
        --adapter-path training_output/mt-lora/checkpoint-4400 \\
        --output-dir  training_output/mt-lora/merged-4400 --vllm-ready
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("mt.training.merge")

#: Small files that must travel with the merged weights for the result to be
#: loadable on its own.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "generation_config.json",
)


def read_base_model_name(adapter_path: Path, override: str | None = None) -> str:
    """Resolve the base model an adapter was trained against.

    Read from ``adapter_config.json`` rather than asked for, so a merge cannot be
    silently pointed at the wrong base — which produces a model that loads fine
    and translates badly.
    """
    if override:
        return override
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found — is {adapter_path} a PEFT adapter directory? "
            f"(pass --base-model to override)"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"{cfg_path} has no base_model_name_or_path; pass --base-model explicitly")
    return base


def stage_support_files(base_model: str, output_dir: Path, *, skip_processor: bool) -> None:
    """Copy tokenizer + processor artefacts from the base model into the merge."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("staged tokenizer from %s", base_model)

    if skip_processor:
        logger.info("skipping processor (--skip-processor)")
        return

    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        processor.save_pretrained(output_dir)
        logger.info("staged processor from %s", base_model)
    except Exception as exc:
        logger.warning(
            "could not save the processor (%s). vLLM needs processor_config.json for "
            "Gemma4ForConditionalGeneration — copy it from the base checkpoint by hand, "
            "or pass --skip-processor to silence this.",
            exc,
        )


def merge(
    adapter_path: Path,
    output_dir: Path,
    *,
    base_model: str | None = None,
    dtype: str = "bfloat16",
    skip_processor: bool = False,
) -> Path:
    """Merge ``adapter_path`` into its base and write a standalone checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = read_base_model_name(adapter_path, base_model)
    logger.info("base model : %s", base)
    logger.info("adapter    : %s", adapter_path)
    logger.info("output     : %s", output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        base, trust_remote_code=True, dtype=getattr(torch, dtype)
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    logger.info("merging adapter into base weights ...")
    model = model.merge_and_unload()

    # The checkpoint is only useful for inference; re-enable the cache the trainer
    # turned off, so a served model does not recompute the forward pass per token.
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        model.generation_config.do_sample = False

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info("wrote merged weights")

    stage_support_files(base, output_dir, skip_processor=skip_processor)

    # Anything the tokenizer/processor save did not cover but the base ships.
    base_dir = Path(base)
    if base_dir.is_dir():
        for name in _TOKENIZER_FILES:
            src, dst = base_dir / name, output_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.info("copied %s", name)

    return output_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.training.merge",
        description="Merge a LoRA adapter into its base model for standalone serving.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adapter-path", required=True, help="PEFT adapter / checkpoint directory")
    p.add_argument("--output-dir", required=True, help="where the merged checkpoint goes")
    p.add_argument(
        "--base-model",
        default=None,
        help="override the base recorded in adapter_config.json",
    )
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument(
        "--skip-processor",
        action="store_true",
        help="do not stage processor_config.json (vLLM needs it; only for HF-only use)",
    )
    p.add_argument(
        "--vllm-ready",
        action="store_true",
        help="also add the KV-shared k_norm sidecar so stock vLLM loads the result",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    out = merge(
        Path(args.adapter_path),
        Path(args.output_dir),
        base_model=args.base_model,
        dtype=args.dtype,
        skip_processor=args.skip_processor,
    )

    if args.vllm_ready:
        from bodhan_genai.mt.tools.vllm_ready import make_vllm_ready

        make_vllm_ready(out)
    else:
        print(
            f"\nNext, to serve on stock vLLM:\n  python -m bodhan_genai.mt.tools.vllm_ready {out}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
