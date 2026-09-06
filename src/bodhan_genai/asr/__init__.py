"""bodhan-genai ASR: IndicTranscribe — a NeMo-independent, HuggingFace-style port
of the Canary-2 AED architecture (FastConformer encoder + Transformer
decoder) for Indic-language speech recognition.

Subpackages
-----------
- ``model``: HF-style model components (``IndicTranscribeConfig``,
  ``IndicTranscribeForConditionalGeneration``, feature extractor, tokenizer).
- ``engine``: public batch-transcription API (``IndicASREngine``).
- ``inference``: offline batch inference CLI.

See ``docs/asr/`` for the model card, caveats, and usage.
"""

# PEP 562 lazy exports, mirroring bodhan_genai.tts: importing bodhan_genai.asr
# must not drag torch / transformers / torchaudio in eagerly; each name
# resolves its module on first attribute access.
_LAZY = {
    "IndicASREngine": "bodhan_genai.asr.engine.engine",
    "IndicTranscribeConfig": "bodhan_genai.asr.model.configuration_indic_transcribe",
    "IndicTranscribeFeatureExtractor": "bodhan_genai.asr.model.feature_extraction_indic_transcribe",
    "IndicTranscribeForConditionalGeneration": "bodhan_genai.asr.model.modeling_indic_transcribe",
    "IndicTranscribeTokenizer": "bodhan_genai.asr.model.tokenization_indic_transcribe",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
