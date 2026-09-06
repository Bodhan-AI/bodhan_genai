"""bodhan-genai: Orpheus-style TTS — Llama-3.2-3B backbone generating SNAC audio tokens.

Subpackages
-----------
- ``codec``: SNAC audio codec token encode/decode (7 tokens/frame, offset math).
- ``templates``: Llama chat-template builders (basic TTS, conversation).
- ``data``: two-stage data pipeline (audio -> SNAC parquet -> training parquet).
- ``training``: sequence-packing trainer (FSDP2 via ``accelerate launch``).
- ``inference``: offline batch (vLLM) and simple HF ``generate()`` paths.
- ``serving``: Ray Serve streaming websocket server (vLLM AsyncLLM + in-process SNAC).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bodhan-genai")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

# PEP 562 lazy exports: importing bodhan_genai.tts (or pulling SamplingConfig /
# TTSResult from it) must never import the engine modules' dependency chains
# eagerly; each name resolves its module on first attribute access.
_LAZY = {
    "IndicTTSEngine": "bodhan_genai.tts.engine.offline",
    "IndicStreamingTTSEngine": "bodhan_genai.tts.engine.streaming",
    "ChunkedIndicStreamingTTS": "bodhan_genai.tts.engine.chunked",
    "chunk_text": "bodhan_genai.tts.engine.chunked",
    "estimate_speech_seconds": "bodhan_genai.tts.engine.chunked",
    "split_sentences": "bodhan_genai.tts.engine.chunked",
    "plan_dialogue_chunks": "bodhan_genai.tts.engine.chunked",
    "SamplingConfig": "bodhan_genai.tts.engine.types",
    "TTSResult": "bodhan_genai.tts.engine.types",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
