"""Evaluation for the OCR stack: layout quality, and end-to-end transcription.

    python -m bodhan_genai.ocr.eval.layout  --ckpt ... --manifest layout_test.json
    python -m bodhan_genai.ocr.eval.olmocr  --pages bench/ --out runs/olmocr

Two different questions, deliberately kept apart. The layout eval asks whether the
detector finds and orders blocks; the olmOCR eval asks whether the whole pipeline turns a
page into the right Markdown. A regression in the first shows up in the second only
diluted, and a recognizer regression does not show up in the first at all.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhan_genai.ocr.eval.layout import OrderAccumulator, evaluate
    from bodhan_genai.ocr.eval.metrics import (
        DetectionAccumulator,
        average_precision,
        dense_ranks,
        iou_matrix,
        kendall_tau,
        normalized_edit_distance,
        pairwise_accuracy,
        raster_order,
    )
    from bodhan_genai.ocr.eval.olmocr import BenchSettings, predict, score

__all__ = [
    "BenchSettings",
    "DetectionAccumulator",
    "OrderAccumulator",
    "average_precision",
    "dense_ranks",
    "evaluate",
    "iou_matrix",
    "kendall_tau",
    "normalized_edit_distance",
    "pairwise_accuracy",
    "predict",
    "raster_order",
    "score",
]

_SOURCES = {
    "layout": ("OrderAccumulator", "evaluate"),
    "metrics": (
        "DetectionAccumulator", "average_precision", "dense_ranks", "iou_matrix",
        "kendall_tau", "normalized_edit_distance", "pairwise_accuracy", "raster_order",
    ),
    "olmocr": ("BenchSettings", "predict", "score"),
}  # fmt: skip


def __getattr__(name: str):
    for module, exports in _SOURCES.items():
        if name in exports:
            import importlib

            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
