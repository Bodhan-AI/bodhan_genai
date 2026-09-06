"""``python -m bodhan_genai.mt.training.train`` — LoRA finetuning for IndicTranslate.

PROVISIONAL. This is a working default, not a qualified recipe — see
``docs/mt/training.md``. Nothing outside ``bodhan_genai.mt.training`` imports from
here; the contract with the rest of the package is only:

    in   rendered `messages` JSONL   (bodhan_genai.mt.data.render)
    out  a PEFT adapter directory    (-> training.merge -> tools.vllm_ready)

so replacing this module with a different trainer costs this file, its config, its
launcher and its extra — and nothing else.

Launch through ``accelerate`` so the distributed setup is the launcher's job::

    scripts/mt/train_lora.sh configs/mt/train/lora.yaml

    # or explicitly
    accelerate launch --config_file configs/mt/accelerate/single_node.yaml \\
        -m bodhan_genai.mt.training.train configs/mt/train/lora.yaml

Re-running with the same ``training.output_dir`` resumes from the last checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from bodhan_genai.mt.training.config import TrainConfig, load_config

logger = logging.getLogger("mt.training.train")

#: Keys the recipe owns. A config that sets these is overriding a decision the
#: recipe makes on purpose, so we warn rather than silently honour or ignore them.
_RECIPE_OWNED = ("max_length", "packing", "assistant_only_loss")

#: Recipe-level knobs that live in the `training:` block for readability but are
#: NOT trainer-config fields. Popped before construction, or SFTConfig raises.
_NOT_SFT_CONFIG_FIELDS = ("early_stopping_patience",)


def build_sft_config(
    training: dict[str, Any], *, max_seq_length: int, steps_per_eval: int | None
) -> tuple[Any, dict[str, Any]]:
    """Build the trainer config from the YAML ``training`` block.

    Returns ``(sft_config, extras)`` where ``extras`` holds the recipe-level keys
    that are not trainer-config fields.

    Kept separate from :func:`run` so tests can construct a real config object on
    CPU and catch a library field rename at test time rather than on the cluster.
    """
    from trl import SFTConfig

    kwargs = dict(training)

    extras = {k: kwargs.pop(k) for k in _NOT_SFT_CONFIG_FIELDS if k in kwargs}

    for key in _RECIPE_OWNED:
        if key in kwargs:
            logger.warning(
                "training.%s is set by the recipe; the value in the config is ignored", key
            )
            kwargs.pop(key)

    # Sequence handling. `packing=False` keeps one example per sequence, so the
    # assistant-only loss mask lines up with exactly one translation.
    kwargs["max_length"] = max_seq_length
    kwargs["packing"] = False
    # Loss on the assistant span only. Requires the GEMMA4_TRL_TEMPLATE, which
    # carries the {% generation %} markers this reads.
    kwargs["assistant_only_loss"] = True

    if steps_per_eval is not None:
        kwargs.setdefault("eval_strategy", "steps")
        kwargs.setdefault("save_strategy", "steps")
        kwargs.setdefault("eval_steps", steps_per_eval)
        kwargs.setdefault("save_steps", steps_per_eval)

    return SFTConfig(**kwargs), extras


def resolve_eval_steps(n_train_rows: int, training: dict[str, Any], eval_fraction: float) -> int:
    """Steps between evals, as a fraction of one epoch.

    Expressed relative to the dataset so a config transfers between corpora. An
    explicit ``eval_steps`` in the YAML wins.
    """
    if training.get("eval_steps"):
        return int(training["eval_steps"])

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    per_device = int(training.get("per_device_train_batch_size", 1))
    accum = int(training.get("gradient_accumulation_steps", 1))
    effective_batch = max(1, per_device * accum * world_size)

    steps_per_epoch = max(1, math.ceil(n_train_rows / effective_batch))
    return max(1, int(steps_per_epoch * eval_fraction))


def _last_checkpoint(output_dir: str) -> str | None:
    """Latest ``checkpoint-N`` under ``output_dir``, or None if there is none."""
    from transformers.trainer_utils import get_last_checkpoint

    if not Path(output_dir).is_dir():
        return None
    return get_last_checkpoint(output_dir)


def run(cfg: TrainConfig) -> int:
    """Run one finetune. Returns a process exit code."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
    from trl import SFTTrainer

    from bodhan_genai.mt.data.dataset import load_training_dataset
    from bodhan_genai.mt.templates.trl_chat import GEMMA4_TRL_TEMPLATE

    rank = int(os.environ.get("RANK", 0))
    output_dir = cfg.training["output_dir"]

    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -- tokenizer ------------------------------------------------------------
    tokenizer_path = cfg.model.tokenizer_path or cfg.model.model_path
    logger.info("[rank %d] loading tokenizer: %s", rank, tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding for training (left is an inference-time concern).
    tokenizer.padding_side = "right"
    # The training template. Renders byte-identically to the checkpoint's shipped
    # chat template, but adds the {% generation %} markers assistant_only_loss
    # needs. tests/mt/test_trl_template_parity.py holds the two in lock step.
    tokenizer.chat_template = GEMMA4_TRL_TEMPLATE

    # -- data -----------------------------------------------------------------
    # load_training_dataset takes a file lock internally, so every rank can call
    # this: one builds the cache, the rest block and then read it.
    train_ds = load_training_dataset(
        tokenizer,
        cfg.data.train_file,
        cfg.data.cache_dir,
        "train",
        cfg.model.max_seq_length,
        num_proc=cfg.data.num_proc,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    dev_ds = load_training_dataset(
        tokenizer,
        cfg.data.dev_file,
        cfg.data.cache_dir,
        "dev",
        cfg.model.max_seq_length,
        num_proc=min(cfg.data.num_proc, 4),
        shuffle_seed=cfg.data.shuffle_seed,
    )
    logger.info("[rank %d] train=%d dev=%d", rank, len(train_ds), len(dev_ds))

    # -- model ----------------------------------------------------------------
    logger.info("[rank %d] loading model: %s", rank, cfg.model.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
    )
    # Incompatible with gradient checkpointing, and meaningless during training.
    model.config.use_cache = False

    peft_config = None
    if cfg.model.adapter_path:
        # Continue from an existing adapter as INITIALISATION: weights are loaded
        # and unfrozen, but the optimizer/scheduler/LR all start fresh.
        from peft import PeftModel

        logger.info("[rank %d] seeding from adapter: %s", rank, cfg.model.adapter_path)
        model = PeftModel.from_pretrained(model, cfg.model.adapter_path, is_trainable=True)
    else:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            target_modules=cfg.lora.target_modules,
            exclude_modules=cfg.lora.exclude_modules or None,
            modules_to_save=cfg.lora.modules_to_save or None,
            use_rslora=cfg.lora.use_rslora,
            use_dora=cfg.lora.use_dora,
        )

    # -- trainer --------------------------------------------------------------
    steps_per_eval = resolve_eval_steps(len(train_ds), cfg.training, cfg.eval_fraction)
    logger.info("[rank %d] eval/save every %d steps", rank, steps_per_eval)

    training = dict(cfg.training)
    training.setdefault("gradient_checkpointing", cfg.model.gradient_checkpointing)
    training.setdefault("report_to", cfg.logging_cfg.report_to)
    if cfg.logging_cfg.wandb_run_name:
        training.setdefault("run_name", cfg.logging_cfg.wandb_run_name)

    args, extras = build_sft_config(
        training, max_seq_length=cfg.model.max_seq_length, steps_per_eval=steps_per_eval
    )

    callbacks = []
    patience = int(extras.get("early_stopping_patience", 5))
    if patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.0)
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    if rank == 0:
        trainer.model.print_trainable_parameters()
        # Record what actually ran, next to the checkpoints.
        (Path(output_dir) / "bodhan_mt_run.json").write_text(
            json.dumps(
                {
                    "model_path": cfg.model.model_path,
                    "max_seq_length": cfg.model.max_seq_length,
                    "lora": vars(cfg.lora),
                    "training": cfg.training,
                    "resolved_eval_steps": steps_per_eval,
                    "train_rows": len(train_ds),
                    "dev_rows": len(dev_ds),
                },
                indent=2,
                default=str,
            )
        )

    # -- run ------------------------------------------------------------------
    resume_from = _last_checkpoint(output_dir) if cfg.resume else None
    if resume_from:
        logger.info("resuming from %s", resume_from)
    elif cfg.model.adapter_path:
        logger.info("fresh run seeded from %s", cfg.model.adapter_path)
    else:
        logger.info("fresh run")

    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(output_dir)
    if rank == 0:
        tokenizer.save_pretrained(output_dir)
        logger.info("adapter saved to %s", output_dir)
        logger.info(
            "next: python -m bodhan_genai.mt.training.merge --adapter-path %s "
            "--output-dir <merged>",
            output_dir,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: python -m bodhan_genai.mt.training.train <config.yaml>")
        return 0 if argv and argv[0] in ("-h", "--help") else 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(load_config(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main())
