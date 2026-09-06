"""
Training configuration: dataclass schema + YAML loader.

The YAML structure maps directly to TrainConfig. The `training` section
is passed as **kwargs to HuggingFace TrainingArguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

# ---------------------------------------------------------------------------
# Data config
# ---------------------------------------------------------------------------


@dataclass
class DatasetEntry:
    path: str
    ratio: float = 1.0
    name: str = ""


@dataclass
class DataSplitConfig:
    datasets: list[DatasetEntry]


VALID_PACK_BACKENDS = ("auto", "sortedlist", "bucket", "numba_bucket", "linear")


@dataclass
class PackingConfig:
    """Runtime controls for training-time sequence packing."""

    backend: str = "auto"
    rank_local: bool = False
    equalize_rank_bins: bool = True

    def __post_init__(self):
        if self.backend not in VALID_PACK_BACKENDS:
            raise ValueError(
                f"data.train.packing.backend must be one of {VALID_PACK_BACKENDS}, "
                f"got {self.backend!r}"
            )


@dataclass
class TrainDataSplitConfig(DataSplitConfig):
    packing: PackingConfig = field(default_factory=PackingConfig)


@dataclass
class DataConfig:
    train: TrainDataSplitConfig
    val: DataSplitConfig | None = None


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    model_path: str
    tokenizer_path: str = ""  # If empty, uses model_path
    max_seq_len: int = 8192
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    activation_checkpointing: bool = False
    compile: bool = True
    compile_mode: str = "max-autotune"
    snac_model_path: str = "hubertsiuzdak/snac_24khz"  # For audio decoding in eval


# ---------------------------------------------------------------------------
# LoRA config (consumed only by training/train_lora.py)
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """PEFT LoRA hyperparameters. Optional top-level `lora:` block in YAML.

    When present, `train_lora.py` wraps the loaded model with `get_peft_model`.
    `train.py` ignores this block, so existing full-FT runs are unaffected.
    """

    r: int = 32
    lora_alpha: int = 64  # Conventional 2 x r; effective scale = alpha/r
    lora_dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = field(default_factory=list)
    init_lora_weights: bool | str = True  # True | "gaussian" | "kaiming" (PEFT default)

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"lora.r must be > 0, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora.lora_alpha must be > 0, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora.lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(
                f"lora.bias must be one of ('none', 'all', 'lora_only'), got {self.bias!r}"
            )
        if not self.target_modules:
            raise ValueError("lora.target_modules must be non-empty")


# ---------------------------------------------------------------------------
# Logging config
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    wandb_project: str = "bodhan-genai"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    peak_tflops_per_gpu: float | None = None  # Override for MFU; auto-detected if None
    log_sample_metrics: list[str] = field(default_factory=lambda: ["wer", "mos"])


@dataclass
class CheckpointRetentionConfig:
    """Drives BestAndLastCheckpointKeeper. Retains the union of:
      • top ``best_k`` checkpoints by ``metric`` (lower-is-better, e.g. eval_loss)
      • last ``last_n`` checkpoints by step
    Requires HF's ``save_total_limit`` to be null so its _rotate_checkpoints
    doesn't preempt the keeper.
    """

    last_n: int = 3
    best_k: int = 3
    metric: str = "eval_loss"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    model: ModelConfig
    data: DataConfig
    training: dict  # Passed directly to HuggingFace TrainingArguments
    logging_cfg: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint_retention: CheckpointRetentionConfig = field(
        default_factory=CheckpointRetentionConfig
    )
    training_stage: str = (
        "pt"  # pt | cpt | sft | ft | dpo | rlhf — used in auto-generated wandb run name
    )
    lora: LoRAConfig | None = None  # set by `lora:` block; consumed only by train_lora.py


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _parse_dataset_entry(d: dict) -> DatasetEntry:
    return DatasetEntry(
        path=d["path"],
        ratio=float(d.get("ratio", 1.0)),
        name=d.get("name", ""),
    )


def _parse_data_split(d: dict | None) -> DataSplitConfig | None:
    if d is None:
        return None
    datasets = [_parse_dataset_entry(e) for e in d.get("datasets", [])]
    return DataSplitConfig(datasets=datasets)


def _parse_packing(d: dict | None) -> PackingConfig:
    if not d:
        return PackingConfig()
    return PackingConfig(
        backend=str(d.get("backend", "auto")),
        rank_local=bool(d.get("rank_local", False)),
        equalize_rank_bins=bool(d.get("equalize_rank_bins", True)),
    )


def _parse_train_data_split(d: dict | None) -> TrainDataSplitConfig:
    """Parse the ``data.train`` block: a flat ``datasets:`` list with per-entry
    ratios plus optional ``packing:`` runtime controls."""
    if d is None:
        return TrainDataSplitConfig(datasets=[])

    if d.get("curriculum") is not None:
        raise ValueError(
            "data.train.curriculum is no longer supported in bodhan_genai — "
            "curriculum training was removed; use a flat data.train.datasets list."
        )

    packing = _parse_packing(d.get("packing"))
    datasets = [_parse_dataset_entry(e) for e in d.get("datasets") or []]
    return TrainDataSplitConfig(datasets=datasets, packing=packing)


def _parse_logging(d: dict | None) -> LoggingConfig:
    if not d:
        return LoggingConfig()
    return LoggingConfig(
        wandb_project=d.get("wandb_project", "bodhan-genai"),
        wandb_entity=d.get("wandb_entity"),
        wandb_run_name=d.get("wandb_run_name"),
        peak_tflops_per_gpu=d.get("peak_tflops_per_gpu"),
        log_sample_metrics=d.get("log_sample_metrics", ["wer", "mos"]),
    )


def load_config(config_path: str) -> TrainConfig:
    """Load a YAML training config and return a TrainConfig dataclass."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Features removed in the bodhan_genai port — fail loudly instead of
    # silently ignoring the config block.
    _removed = {
        "slurm_eval": "Slurm-triggered remote eval (slurm_eval:) was removed",
        "remote_eval": "Slurm-triggered remote eval (remote_eval:) was removed",
        "async_checkpoint": "Async DCP checkpointing (async_checkpoint:) was removed",
        "curriculum": "Curriculum training (curriculum:) was removed",
    }
    for key, msg in _removed.items():
        if key in raw:
            raise ValueError(f"{msg}; delete the '{key}:' block from {config_path}")
    if (
        isinstance(raw.get("data"), dict)
        and isinstance(raw["data"].get("train"), dict)
        and "curriculum" in raw["data"]["train"]
    ):
        raise ValueError(
            f"Curriculum training (data.train.curriculum:) was removed; "
            f"delete the block from {config_path}"
        )

    model_raw = raw["model"]
    model = ModelConfig(
        model_path=model_raw["model_path"],
        tokenizer_path=model_raw.get("tokenizer_path", ""),
        max_seq_len=int(model_raw.get("max_seq_len", 8192)),
        attn_implementation=model_raw.get("attn_implementation", "flash_attention_2"),
        torch_dtype=model_raw.get("torch_dtype", "bfloat16"),
        activation_checkpointing=bool(model_raw.get("activation_checkpointing", False)),
        compile=bool(model_raw.get("compile", True)),
        compile_mode=model_raw.get("compile_mode", "max-autotune"),
        snac_model_path=model_raw.get("snac_model_path", "hubertsiuzdak/snac_24khz"),
    )

    data_raw = raw["data"]
    data = DataConfig(
        train=_parse_train_data_split(data_raw.get("train")),
        val=_parse_data_split(data_raw.get("val")),
    )

    training = raw.get("training", {})

    logging_cfg = _parse_logging(raw.get("logging"))

    keep_raw = raw.get("checkpoint_retention") or {}
    checkpoint_retention = CheckpointRetentionConfig(
        last_n=int(keep_raw.get("last_n", 3)),
        best_k=int(keep_raw.get("best_k", 3)),
        metric=str(keep_raw.get("metric", "eval_loss")),
    )

    lora_raw = raw.get("lora")
    lora = LoRAConfig(**lora_raw) if lora_raw else None

    return TrainConfig(
        model=model,
        data=data,
        training=training,
        logging_cfg=logging_cfg,
        checkpoint_retention=checkpoint_retention,
        training_stage=str(raw.get("training_stage", "pt")),
        lora=lora,
    )
