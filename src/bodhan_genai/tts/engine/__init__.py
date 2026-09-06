"""bodhan_genai.tts.engine — engine-agnostic TTS types and engine implementations.

Light at import time: the engine modules keep their heavy deps (torch / vllm /
transformers / snac) inside methods, so re-exporting them here is safe.
Importing ``bodhan_genai.tts.engine`` is an explicit opt-in; the top-level
``bodhan_genai.tts`` package exposes the same four names lazily (PEP 562).
"""

from bodhan_genai.tts.engine.chunked import (
    ChunkedIndicStreamingTTS,
    chunk_text,
    estimate_speech_seconds,
    plan_dialogue_chunks,
    split_sentences,
)
from bodhan_genai.tts.engine.offline import IndicTTSEngine
from bodhan_genai.tts.engine.streaming import IndicStreamingTTSEngine
from bodhan_genai.tts.engine.types import SamplingConfig, TTSResult

__all__ = [
    "ChunkedIndicStreamingTTS",
    "IndicStreamingTTSEngine",
    "IndicTTSEngine",
    "SamplingConfig",
    "TTSResult",
    "chunk_text",
    "estimate_speech_seconds",
    "plan_dialogue_chunks",
    "split_sentences",
]
