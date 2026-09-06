"""bodhan_genai.mt.templates — the prompt contract and its training-time variants.

``prompt`` is the frozen served contract: one instruction, target language only,
one user turn. ``variants`` holds the phrasing bank used to render training
corpora. ``trl_chat`` holds the training chat template carrying ``{% generation %}``
loss markers.

stdlib-only throughout — safe to import anywhere.
"""

from bodhan_genai.mt.templates.prompt import (
    DEFAULT_SCRIPT,
    LANGUAGE_NAMES,
    STOP_STRINGS,
    TEMPLATE,
    build_conversation,
    build_instruction,
    resolve_language,
)
from bodhan_genai.mt.templates.trl_chat import GEMMA4_TRL_TEMPLATE

__all__ = [
    "DEFAULT_SCRIPT",
    "GEMMA4_TRL_TEMPLATE",
    "LANGUAGE_NAMES",
    "STOP_STRINGS",
    "TEMPLATE",
    "build_conversation",
    "build_instruction",
    "resolve_language",
]
