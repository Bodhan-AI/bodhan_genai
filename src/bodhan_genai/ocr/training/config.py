"""Training configuration: one frozen dataclass, loaded from YAML.

An unknown key raises rather than being ignored. A typo'd ``learing_rate`` that silently
falls back to the default is the kind of thing you discover three days into a run, from
the loss curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayoutTrainConfig:
    """Everything a layout run needs. Paths are resolved by the caller."""

    # data
    cache_prefix: str
    val_manifest: str | None = None
    image_size: int = 1024
    source_weights: dict[str, float] = field(default_factory=dict)

    # model
    checkpoint: str = "PaddlePaddle/PP-DocLayoutV3"
    lambda_order: float = 5.0

    # optimisation
    epochs: int = 12
    batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 0.1
    seed: int = 0

    # ema
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup: int = 2000

    # runtime
    output_dir: str = "runs/layout"
    num_workers: int = 8
    log_every: int = 50
    save_every_epochs: int = 1
    eval_every_epochs: int = 1
    max_steps: int | None = None
    bf16: bool = True
    wandb_project: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if self.source_weights and min(self.source_weights.values()) < 0:
            raise ValueError("source_weights must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> LayoutTrainConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{path}: unknown config key(s) {sorted(unknown)}. Known keys: {sorted(known)}"
            )
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
