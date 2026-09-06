"""bodhan_genai.ocr.engine -- plain-data types, configuration, and the pipeline stages.

Light at import time: the stage modules keep their heavy dependencies (torch / vllm /
transformers) inside methods, so re-exporting the types here is safe. Importing this package is
an explicit opt-in; the top-level ``bodhan_genai.ocr`` exposes the same names lazily (PEP 562).
"""

from bodhan_genai.ocr.engine.types import (
    DEDUP_MODES,
    Block,
    CropConfig,
    DedupConfig,
    LayoutConfig,
    PageResult,
    RecognizerConfig,
)

__all__ = [
    "DEDUP_MODES",
    "Block",
    "CropConfig",
    "DedupConfig",
    "LayoutConfig",
    "PageResult",
    "RecognizerConfig",
]
