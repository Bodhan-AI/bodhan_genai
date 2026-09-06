"""Training for IndicDocLayout: detection and reading order, jointly.

    python -m bodhan_genai.ocr.training.train --config configs/ocr/train/layout.yaml

Nothing outside this subpackage imports from it — the same rule the TTS and MT training
packages follow, so a trainer can be swapped without touching inference or serving. The
one exception is the taxonomy, which lives in ``bodhan_genai.ocr.data`` precisely so both
sides can share it.

Only the **layout** model is trained here. IndicBlockOCR (the recognizer) ships as a
released checkpoint; this repo has no recipe for it, and inventing a plausible one would
be worse than saying so.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhan_genai.ocr.training.config import LayoutTrainConfig
    from bodhan_genai.ocr.training.dataset import (
        BlobLayoutDataset,
        DiskLayoutDataset,
        MixedSourceSampler,
        make_collate,
    )
    from bodhan_genai.ocr.training.ema import ModelEma
    from bodhan_genai.ocr.training.modeling import build_model, trainable_class
    from bodhan_genai.ocr.training.order_loss import decode_order, locality_gce, pairwise_scores
    from bodhan_genai.ocr.training.train import train

__all__ = [
    "BlobLayoutDataset",
    "DiskLayoutDataset",
    "LayoutTrainConfig",
    "MixedSourceSampler",
    "ModelEma",
    "build_model",
    "decode_order",
    "locality_gce",
    "make_collate",
    "pairwise_scores",
    "train",
    "trainable_class",
]

_SOURCES = {
    "config": ("LayoutTrainConfig",),
    "dataset": ("BlobLayoutDataset", "DiskLayoutDataset", "MixedSourceSampler", "make_collate"),
    "ema": ("ModelEma",),
    "modeling": ("build_model", "trainable_class"),
    "order_loss": ("decode_order", "locality_gce", "pairwise_scores"),
    "train": ("train",),
}


def __getattr__(name: str):
    for module, exports in _SOURCES.items():
        if name in exports:
            import importlib

            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
