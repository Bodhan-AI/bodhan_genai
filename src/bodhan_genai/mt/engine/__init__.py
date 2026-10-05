"""bodhan_genai.mt.engine — engine-agnostic MT types and the engine implementation.

Light at import time: the engine module keeps its heavy deps (torch / vllm /
transformers) inside methods, so re-exporting them here is safe. Importing
``bodhan_genai.mt.engine`` is an explicit opt-in; the top-level
``bodhan_genai.mt`` package exposes the same names lazily (PEP 562).
"""

from bodhan_genai.mt.engine.backends import (
    BACKEND_NAMES,
    resolve_backend_name,
)
from bodhan_genai.mt.engine.offline import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL_REPO,
    ConversationBackend,
    IndicMTEngine,
    PromptedBackend,
    TranslationBackend,
)
from bodhan_genai.mt.engine.types import BackendOutput, MTRequest, MTResult, MTSamplingConfig

__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_MAX_MODEL_LEN",
    "DEFAULT_MODEL_REPO",
    "BackendOutput",
    "ConversationBackend",
    "IndicMTEngine",
    "MTRequest",
    "MTResult",
    "MTSamplingConfig",
    "PromptedBackend",
    "TranslationBackend",
    "resolve_backend_name",
]
