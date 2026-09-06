"""
LoRA SFT entry point for bodhan_genai.

Sibling to training/train.py — kept separate (rather than gating train.py with
a flag) so existing full-fine-tuning behavior is untouched. See the planning
doc and README for the full rationale.

Launch with accelerate:
  accelerate launch --config_file configs/tts/accelerate/single_node_fsdp.yaml \\
      -m bodhan_genai.tts.training.train_lora configs/tts/train/lora.yaml

Differences from train.py:
  1. Loads PEFT and wraps the model with `get_peft_model(...)` after the base
     model is materialized but before the Trainer is constructed.
  2. Hard-disables `torch.compile` because torch.compile + PEFT module
     injection + FSDP2 reliably triggers Inductor recompile storms. Full FT
     keeps compile on; LoRA pays the un-compiled cost (acceptable since LoRA
     runs are short).
  3. Adds a `PeftAdapterSaveCallback` so each `save_steps` boundary writes the
     adapter weights only (~MBs) rather than the full base + adapter via FSDP
     FULL_STATE_DICT (~GBs). Optimizer/scheduler state is still persisted by
     HF Trainer through its normal path so `--resume_from_checkpoint` works.

Requires the YAML config to have a top-level `lora:` block (see
training/config.py:LoRAConfig). If the block is missing, this script errors
out — use train.py instead for full fine-tuning.
"""

from __future__ import annotations

import logging
import os
import sys

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from bodhan_genai.tts.training.callbacks import (
    BestAndLastCheckpointKeeper,
    EpochSamplerCallback,
    PeftAdapterSaveCallback,
    TrainingMetricsCallback,
)
from bodhan_genai.tts.training.config import TrainConfig, load_config
from bodhan_genai.tts.training.dataset import build_mixed_dataset
from bodhan_genai.tts.training.sampler import require_fast_packing, require_numba_packing
from bodhan_genai.tts.training.trainer import PackingTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MIN_TRANSFORMERS_VERSION = "4.44.0"
torch.set_float32_matmul_precision("high")


def _check_requirements() -> None:
    from packaging.version import Version

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f"transformers >= {MIN_TRANSFORMERS_VERSION} required for FA2 document masking "
            f"via position_ids. Got {transformers.__version__}."
        )


# Reuse the formatting helpers from train.py via import. Avoids duplicating
# logic and keeps wandb run-name format consistent across full and LoRA runs.
from bodhan_genai.tts.training.train import (  # noqa: E402  (import after logging setup is intentional)
    _auto_run_name,
    _enable_activation_checkpointing,
    _packing_uses_numba,
    _packing_uses_sortedcontainers,
    build_training_arguments,
)


def main(config_path: str) -> None:
    _check_requirements()

    config: TrainConfig = load_config(config_path)
    if _packing_uses_sortedcontainers(config):
        require_fast_packing("training")
    if _packing_uses_numba(config):
        require_numba_packing("training")

    if config.lora is None:
        raise ValueError(
            "train_lora.py requires a top-level `lora:` section in the config "
            f"({config_path}). For full fine-tuning, use bodhan_genai.tts.training.train instead."
        )

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    seed = config.training.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer_path = config.model.tokenizer_path or config.model.model_path
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        logger.warning(f"pad_token_id not set, falling back to eos_token_id ({pad_token_id})")

    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------
    logger.info(f"Loading base model from {config.model.model_path}")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        attn_implementation=config.model.attn_implementation,
        dtype=torch_dtype,
    )

    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing token embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Verify FA2 is active. transformers v5 renamed the private
    # `_attn_implementation` attribute; tolerate both spellings.
    _attn_impl = getattr(
        model.config,
        "_attn_implementation",
        getattr(model.config, "attn_implementation", None),
    )
    assert _attn_impl == "flash_attention_2", (
        "Flash Attention 2 not active. Ensure the model supports it and "
        "flash-attn is installed (pip install flash-attn --no-build-isolation)."
    )

    if config.model.activation_checkpointing:
        logger.info("Enabling HF-level activation checkpointing")
        _enable_activation_checkpointing(model)

    # ------------------------------------------------------------------
    # PEFT / LoRA wrap
    # ------------------------------------------------------------------
    # peft is imported lazily so that `--help` / CPU unit tests don't require it.
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    lora_cfg = config.lora
    try:
        task_type = TaskType[lora_cfg.task_type]
    except KeyError as e:
        valid = sorted(t.name for t in TaskType)
        raise ValueError(
            f"lora.task_type={lora_cfg.task_type!r} is not a valid PEFT TaskType. "
            f"Valid options: {valid}"
        ) from e

    peft_cfg = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=task_type,
        target_modules=list(lora_cfg.target_modules),
        modules_to_save=list(lora_cfg.modules_to_save) or None,
        init_lora_weights=lora_cfg.init_lora_weights,
    )
    logger.info(
        "Wrapping model with LoRA (r=%d, alpha=%d, dropout=%g, target=%s, modules_to_save=%s)",
        lora_cfg.r,
        lora_cfg.lora_alpha,
        lora_cfg.lora_dropout,
        lora_cfg.target_modules,
        lora_cfg.modules_to_save or "[]",
    )
    model = get_peft_model(model, peft_cfg)
    # Surfaces "trainable params: X || all params: Y || trainable%: Z" so the
    # rank/scope choice is visible in the launch log every time.
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Compile is intentionally disabled for LoRA — see module docstring.
    # ------------------------------------------------------------------
    if config.model.compile:
        logger.warning(
            "model.compile is True in the config but is being force-disabled "
            "for LoRA. torch.compile + PEFT + FSDP2 triggers Inductor "
            "recompile storms; full-FT keeps compile on, LoRA does not."
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Building datasets ...")
    train_dataset = build_mixed_dataset(config.data.train)
    val_dataset = build_mixed_dataset(config.data.val)

    if train_dataset is None:
        raise ValueError("No training dataset configured.")

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    if config.logging_cfg.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", config.logging_cfg.wandb_project)
    if config.logging_cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", config.logging_cfg.wandb_entity)
    run_name = config.logging_cfg.wandb_run_name or _auto_run_name(config, model)
    if not config.logging_cfg.wandb_run_name:
        logger.info(f"wandb_run_name not set; using auto-generated name: {run_name}")
    os.environ.setdefault("WANDB_RUN_NAME", run_name)

    training_args, training_kwargs, dataloader_prefetch_factor = build_training_arguments(
        config.training
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    # MFU/perplexity logging works the same for LoRA — `num_params` here counts
    # all params (frozen + trainable). The MFU number ends up under-reporting
    # because most params don't get gradients, but train/perplexity is
    # accurate.
    num_params = sum(p.numel() for p in model.parameters())
    grad_accum = training_kwargs.get("gradient_accumulation_steps", 1)
    callbacks = [
        TrainingMetricsCallback(
            num_params=num_params,
            max_seq_len=config.model.max_seq_len,
            gradient_accumulation_steps=grad_accum,
            peak_tflops_per_gpu=config.logging_cfg.peak_tflops_per_gpu,
        )
    ]

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = PackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_len=config.model.max_seq_len,
        pad_token_id=pad_token_id,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        train_mixed_dataset=train_dataset,
        eval_mixed_dataset=val_dataset,
        packing_config=config.data.train.packing,
        callbacks=callbacks,
    )

    trainer.add_callback(EpochSamplerCallback(trainer))
    trainer.add_callback(PeftAdapterSaveCallback(trainer))
    # Retention: union of best-K-by-metric and last-N-by-step (see
    # CheckpointRetentionConfig). Requires save_total_limit=null in the YAML
    # so HF's _rotate_checkpoints doesn't preempt this callback.
    trainer.add_callback(
        BestAndLastCheckpointKeeper(
            last_n=config.checkpoint_retention.last_n,
            best_k=config.checkpoint_retention.best_k,
            metric=config.checkpoint_retention.metric,
        )
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    output_dir = training_kwargs.get("output_dir", "checkpoints")
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("Starting LoRA training ...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("LoRA training complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
