"""bodhan_genai.mt.engine — engine-agnostic MT types and the engine implementation.

Light at import time: the engine module keeps its heavy deps (torch / vllm /
transformers) inside methods, so re-exporting them here is safe. Importing
``bodhan_genai.mt.engine`` is an explicit opt-in; the top-level
``bodhan_genai.mt`` package exposes the same names lazily (PEP 562).
"""

from bodhan_genai.mt.engine.offline import (
    DEFAULT_MAX_MODEL_LEN,
    IndicMTEngine,
    TranslationBackend,
)
from bodhan_genai.mt.engine.types import MTResult, MTSamplingConfig

__all__ = [
    "DEFAULT_MAX_MODEL_LEN",
    "IndicMTEngine",
    "MTResult",
    "MTSamplingConfig",
    "TranslationBackend",
]
