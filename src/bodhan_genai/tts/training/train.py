"""
Main training entry point for bodhan_genai.

Launch with accelerate:
  accelerate launch --config_file configs/tts/accelerate/single_node_fsdp.yaml \\
      -m bodhan_genai.tts.training.train configs/tts/train/pretrain.yaml

Features:
  - Architecture-agnostic: works with any AutoModelForCausalLM backbone
  - Flash Attention 2 via attn_implementation="flash_attention_2"
  - torch.compile with max-autotune for static-shape graphs
  - Sequence packing with First-Fit Decreasing sampler
  - Multi-dataset mixing with configurable ratios
  - DDP + FSDP via accelerate
  - CE loss + perplexity logging for train and val
  - Fault-tolerant: resumes from last checkpoint automatically
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from bodhan_genai.tts.training.callbacks import (
    BestAndLastCheckpointKeeper,
    EpochSamplerCallback,
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
# Stop dynamo from graph-breaking on `max_seqlen_q.item()` inside the FA2
# varlen path — without this each new max_seqlen value triggers a fresh
# recompile. scripts/tts/train.sh also exports TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1;
# this is the belt-and-suspenders for direct `python train.py` invocations.
torch._dynamo.config.capture_scalar_outputs = True


def _check_requirements() -> None:
    """Assert environment meets minimum requirements."""
    from packaging.version import Version

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f"transformers >= {MIN_TRANSFORMERS_VERSION} required for FA2 document masking "
            f"via position_ids. Got {transformers.__version__}."
        )


def _packing_uses_sortedcontainers(config: TrainConfig) -> bool:
    return config.data.train.packing.backend == "sortedlist"


def _packing_uses_numba(config: TrainConfig) -> bool:
    return config.data.train.packing.backend == "numba_bucket"


def _format_lr(lr: float) -> str:
    """Format LR compactly: 2.0e-5 → "2e-5", 0.0003 → "3e-4", 0.001 → "1e-3"."""
    if lr <= 0:
        return str(lr)
    exp = 0
    m = float(lr)
    while m < 1.0:
        m *= 10
        exp -= 1
    while m >= 10.0:
        m /= 10
        exp += 1
    # Trim mantissa to at most 3 significant digits, no trailing zeros
    mantissa = f"{m:.3g}".rstrip("0").rstrip(".")
    if exp == 0:
        return mantissa
    return f"{mantissa}e{exp}"


def _format_size(n: int) -> str:
    """Human-readable size: 24000 → "24k", 1_500_000 → "1.5M"."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "k"
    return str(n)


def _param_count_tag(num_params: int) -> str:
    """Compact param-count tag: 3_100_000_000 → "3B", 500_000_000 → "500M"."""
    if num_params >= 1_000_000_000:
        v = num_params / 1_000_000_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "B"
    if num_params >= 1_000_000:
        v = num_params / 1_000_000
        return f"{v:.0f}M"
    return str(num_params)


def _model_tag(model) -> str:
    """Architecture-derived tag: "llama-3B", "llama-7B", "qwen2-0.5B"."""
    arch = getattr(getattr(model, "config", None), "model_type", None) or "model"
    n = sum(p.numel() for p in model.parameters())
    return f"{arch}-{_param_count_tag(n)}"


def _language_tag(config: TrainConfig) -> str | None:
    """Derive a language identifier from the train dataset paths.

    For SFT runs the user wants to spot which language a run was for at a
    glance in wandb. We use the basename of each dataset path:
      - All paths share a basename  → return that basename ("hi", "te", …)
      - Paths have multiple basenames → return "multilingual"
      - No datasets configured       → return None
    """
    datasets = getattr(getattr(config.data, "train", None), "datasets", None) or []
    basenames = {Path(d.path).name for d in datasets if getattr(d, "path", None)}
    if not basenames:
        return None
    if len(basenames) == 1:
        return next(iter(basenames))
    return "multilingual"


def _auto_run_name(config: TrainConfig, model) -> str:
    """Build a meaningful wandb run name from the most important hyperparameters.

    Format: "{arch}-{size}-{stage}[-{lang}]-lr={lr},seq={max_seq},warmup={warmup},sched={scheduler},{epochs|steps}"
    The {lang} tag is appended for SFT runs only — derived from the train
    dataset basenames (e.g. "hi", "te", or "multilingual" when several langs
    are mixed). CPT runs keep the prior format unchanged.
    """
    t = config.training
    head = _model_tag(model)
    stage = (config.training_stage or "").strip()
    if stage:
        head = f"{head}-{stage}"
    if stage == "sft":
        lang = _language_tag(config)
        if lang:
            head = f"{head}-{lang}"
    parts: list[str] = [head]
    if "learning_rate" in t:
        parts.append(f"lr={_format_lr(float(t['learning_rate']))}")
    parts.append(f"seq={_format_size(int(config.model.max_seq_len))}")
    if "warmup_ratio" in t:
        parts.append(f"warmup={t['warmup_ratio']:g}")
    elif "warmup_steps" in t:
        parts.append(f"warmup={_format_size(int(t['warmup_steps']))}steps")
    scheduler = t.get("lr_scheduler_type")
    if scheduler:
        parts.append(f"sched={scheduler}")
    if t.get("max_steps", -1) and int(t.get("max_steps", -1)) > 0:
        parts.append(f"steps={_format_size(int(t['max_steps']))}")
    elif "num_train_epochs" in t:
        ep = float(t["num_train_epochs"])
        parts.append(f"ep={ep:g}")
    ga = int(t.get("gradient_accumulation_steps", 1) or 1)
    if ga > 1:
        parts.append(f"ga={ga}")
    return parts[0] + "-" + ",".join(parts[1:]) if len(parts) > 1 else parts[0]


def build_training_arguments(
    training_dict: dict,
) -> tuple[TrainingArguments, dict, int]:
    """Construct TrainingArguments from the YAML ``training:`` block.

    Force batch size to 1 — packing handles effective batching. Non-
    TrainingArguments keys (``prefetch_factor``) are popped before
    construction. Returns (args, final_kwargs, dataloader_prefetch_factor).
    """
    training_kwargs = dict(training_dict)
    dataloader_prefetch_factor = int(training_kwargs.pop("prefetch_factor", 4))
    training_kwargs["per_device_train_batch_size"] = 1
    training_kwargs["per_device_eval_batch_size"] = 1
    training_kwargs["report_to"] = "wandb"
    # HF Trainer auto-detects label_names via find_labels(model.__class__). When
    # `model.compile=true`, we wrap with torch.compile before Trainer init so the
    # class is OptimizedModule (forward signature: (*input,)) and find_labels
    # returns []. That makes has_labels=False in prediction_step → no eval_loss
    # in the metrics dict → no eval_loss in wandb. Set explicitly so eval works
    # regardless of compile state.
    training_kwargs.setdefault("label_names", ["labels"])
    return TrainingArguments(**training_kwargs), training_kwargs, dataloader_prefetch_factor


def _enable_activation_checkpointing(model) -> None:
    """Enable HF gradient/activation checkpointing when supported."""
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(
            "activation_checkpointing=true, but this model does not expose "
            "gradient_checkpointing_enable()."
        )
    if getattr(model.config, "use_cache", False):
        logger.info("Disabling use_cache because activation checkpointing is enabled")
        model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def main(config_path: str) -> None:
    _check_requirements()

    config: TrainConfig = load_config(config_path)
    if _packing_uses_sortedcontainers(config):
        require_fast_packing("training")
    if _packing_uses_numba(config):
        require_numba_packing("training")

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
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading model from {config.model.model_path}")
    torch_dtype = getattr(torch, config.model.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        attn_implementation=config.model.attn_implementation,
        dtype=torch_dtype,
    )

    # Resize embeddings if tokenizer is larger than model vocab
    # (happens when using tiktoken extension mode, or when model hasn't been
    # initialized with the extended tokenizer yet)
    # if len(tokenizer) != model.config.vocab_size:
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
        logger.warning(
            "HF-level activation_checkpointing is ON while FSDP activation "
            "checkpointing is also typically ON — double recompute is wasteful. "
            "Prefer setting model.activation_checkpointing=false and relying on "
            "fsdp_activation_checkpointing in the accelerate config."
        )
        _enable_activation_checkpointing(model)

    # torch.compile for static-shape optimization
    if config.model.compile:
        # Cap Dynamo recompile storms: the packing sampler produces fixed-shape
        # [1, max_seq_len] batches, but cu_seqlens / position_id discontinuities
        # can briefly drift on outlier packs. A generous cache limit keeps a few
        # variants warm without allowing unbounded recompiles.
        # NOTE: `import torch._dynamo` (without `as`) would rebind `torch` as a
        # function-local name — Python's static analysis would then treat every
        # `torch` reference in main() as local, including ones earlier in the
        # function, producing an UnboundLocalError. Aliasing keeps the
        # module-level `torch` accessible.
        import torch._dynamo as _torch_dynamo

        _torch_dynamo.config.cache_size_limit = 64
        try:
            import torch._inductor.config as _inductor_cfg

            _inductor_cfg.triton.cudagraphs = False
        except Exception as _err:
            logger.debug(f"Could not set inductor.triton.cudagraphs: {_err}")

        logger.info(f"Compiling model with mode={config.model.compile_mode!r}")
        model = torch.compile(
            model,
            mode=config.model.compile_mode,
            backend="inductor",
            fullgraph=False,
            # Let Dynamo start specialized, then generalize if it observes
            # shape dynamism instead of forcing maximal specialization.
            dynamic=None,
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
    # Set wandb env vars if configured
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

    # EpochSamplerCallback is added after trainer is created (needs sampler ref)

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

    # EpochSamplerCallback resolves the active sampler lazily via the trainer
    # reference — avoids the double-creation bug where priming get_train_dataloader()
    # here creates sampler #1 but trainer.train() later creates #2, orphaning the
    # callback's captured reference (so set_epoch was silently a no-op).
    trainer.add_callback(EpochSamplerCallback(trainer))
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
    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Training complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
