"""Finetuning configuration: dataclass schema + YAML loader.

PROVISIONAL — see the module docstring of :mod:`bodhan_genai.mt.training`. The
trainer behind this schema may change; the schema deliberately uses neutral names
(``max_seq_length``, ``eval_fraction``) rather than TRL's own field names wherever
a neutral one reads as well, so a swap does not force every config to be rewritten.

The ``training`` section is passed through to the trainer's config object as
**kwargs, mirroring how ``bodhan_genai.tts.training.config`` hands its ``training``
block to ``transformers.TrainingArguments``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """What to finetune, and how to load it.

    ``model_path`` is loaded with ``AutoModelForCausalLM`` — the text-only view of
    the multimodal checkpoint. The vision and audio towers are unused for
    translation and stay untouched (LoRA excludes them; see :class:`LoRAConfig`).
    """

    model_path: str
    tokenizer_path: str = ""  # empty = use model_path
    max_seq_length: int = 8192
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    #: Trades compute for memory. On at 8k context, where activations dominate.
    gradient_checkpointing: bool = True
    #: Existing PEFT adapter to load as INITIALISATION (fresh optimizer/LR).
    #: Not the same as resuming — see `resume` in TrainConfig.
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        if self.max_seq_length <= 0:
            raise ValueError(f"model.max_seq_length must be > 0, got {self.max_seq_length}")


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #


@dataclass
class LoRAConfig:
    """PEFT LoRA hyperparameters.

    ``target_modules: "all-linear"`` is deliberate: Gemma 4 carries
    ``per_layer_input_gate`` / ``per_layer_projection`` alongside the usual
    attention and MLP projections, and an explicit list silently misses them.
    ``exclude_modules`` keeps the adapter off the multimodal towers, which
    translation never touches.
    """

    r: int = 32
    lora_alpha: int = 64  # conventional 2 x r; effective scale = alpha/r
    lora_dropout: float = 0.05
    bias: str = "none"  # none | all | lora_only
    task_type: str = "CAUSAL_LM"
    target_modules: str | list[str] = "all-linear"
    exclude_modules: list[str] = field(
        default_factory=lambda: [
            "*vision_tower*",
            "*visual*",
            "*image_encoder*",
            "*audio_tower*",
            "*embed_vision*",
            "*embed_audio*",
            "*lm_head*",
        ]
    )
    modules_to_save: list[str] = field(default_factory=list)
    #: alpha/sqrt(r) instead of alpha/r — steadier at higher ranks.
    use_rslora: bool = False
    use_dora: bool = False

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"lora.r must be > 0, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora.lora_alpha must be > 0, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora.lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(
                f"lora.bias must be one of ('none', 'all', 'lora_only'), got {self.bias!r}"
            )
        if not self.target_modules:
            raise ValueError("lora.target_modules must be non-empty")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Rendered chat JSONL in, cached HF dataset out.

    Both files come from ``python -m bodhan_genai.mt.data.render``. A dev set is
    required: checkpoint selection is by ``eval_loss``, and early stopping has
    nothing to watch without one.
    """

    train_file: str
    dev_file: str
    cache_dir: str
    num_proc: int = 16
    shuffle_seed: int = 42


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


@dataclass
class LoggingConfig:
    wandb_project: str = "bodhan-genai-mt"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    #: "wandb" | "none" | "tensorboard" — passed through to the trainer.
    report_to: str = "wandb"


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    lora: LoRAConfig
    training: dict  # passed to the trainer's config object as **kwargs
    logging_cfg: LoggingConfig = field(default_factory=LoggingConfig)
    #: Restore optimizer/scheduler/LR/global-step from the last checkpoint in
    #: `training.output_dir`. Distinct from `model.adapter_path`, which only seeds
    #: the weights; setting both is a contradiction and raises.
    resume: bool = True
    #: eval/save cadence as a fraction of one epoch. Resolved against the real
    #: dataset size at launch, so a config transfers between corpora instead of
    #: carrying a step count tuned for one of them.
    eval_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.model.adapter_path and self.resume:
            raise ValueError(
                "model.adapter_path and resume are mutually exclusive: adapter_path "
                "seeds weights for a FRESH run (new optimizer/scheduler/LR), while "
                "resume continues an interrupted run from training.output_dir. "
                "Set `resume: false` to start fresh from an adapter."
            )
        if not 0.0 < self.eval_fraction <= 1.0:
            raise ValueError(f"eval_fraction must be in (0, 1], got {self.eval_fraction}")


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

_TOP_LEVEL_KEYS = {
    "model",
    "data",
    "lora",
    "training",
    "logging",
    "resume",
    "eval_fraction",
}


def load_config(config_path: str) -> TrainConfig:
    """Load a finetuning YAML and return a :class:`TrainConfig`."""
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown top-level key(s): {', '.join(unknown)}; "
            f"expected {', '.join(sorted(_TOP_LEVEL_KEYS))}"
        )
    for required in ("model", "data", "training"):
        if required not in raw:
            raise ValueError(f"{config_path}: `{required}:` block is required")

    log_raw = raw.get("logging") or {}
    return TrainConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoRAConfig(**(raw.get("lora") or {})),
        training=dict(raw["training"]),
        logging_cfg=LoggingConfig(**log_raw),
        resume=bool(raw.get("resume", True)),
        eval_fraction=float(raw.get("eval_fraction", 0.1)),
    )
