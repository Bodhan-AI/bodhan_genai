"""IndicOCR -- block-level document parsing for English and 22 Indian languages.

Page image in; reading-ordered Markdown and per-block JSON out. Two models, one plain-JSON
handoff, so either half can be inspected, corrected or replaced:

    IndicDocLayout   page image   -> blocks (box, label, reading order)   torch, ~33M
    IndicBlockOCR    image+layout -> one transcription per block          vLLM, ~0.8B

``IndicOCR`` runs both. Math comes back as LaTeX, tables as HTML by default
(``templates.contract.TableFormat`` selects markdown instead).

Needs ``transformers>=5.7`` (ships PPDocLayoutV3) and ``vllm>=0.26`` (the recognizer's GDN
kernels); that vllm floor is what the shared environment installs -- see ``./install.sh``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bodhan-genai")
except PackageNotFoundError:  # source tree, not installed
    __version__ = "0.0.0+unknown"

# PEP 562 lazy exports: importing this package must never pull in torch / vllm / transformers /
# PIL. Asserted by tests/ocr/test_ocr_lazy_import.py.
_LAZY = {
    "IndicOCR": "bodhan_genai.ocr.engine.offline",
    "IndicDocLayout": "bodhan_genai.ocr.engine.offline",
    "IndicBlockOCR": "bodhan_genai.ocr.engine.offline",
    "LayoutBackend": "bodhan_genai.ocr.engine.layout",
    "IndicDocLayoutBackend": "bodhan_genai.ocr.engine.layout",
    "JsonLayoutBackend": "bodhan_genai.ocr.engine.layout",
    "RecognizerBackend": "bodhan_genai.ocr.engine.recognizer",
    "VllmRecognizer": "bodhan_genai.ocr.engine.recognizer_vllm",
    "HfRecognizer": "bodhan_genai.ocr.engine.recognizer",
    "Block": "bodhan_genai.ocr.engine.types",
    "PageResult": "bodhan_genai.ocr.engine.types",
    "LayoutConfig": "bodhan_genai.ocr.engine.types",
    "CropConfig": "bodhan_genai.ocr.engine.types",
    "DedupConfig": "bodhan_genai.ocr.engine.types",
    "RecognizerConfig": "bodhan_genai.ocr.engine.types",
    "TableFormat": "bodhan_genai.ocr.templates.contract",
    "prompt_for": "bodhan_genai.ocr.templates.contract",
    "map_label": "bodhan_genai.ocr.templates.contract",
    "is_transcribed": "bodhan_genai.ocr.templates.contract",
    "KEPT_BLOCK_TYPES": "bodhan_genai.ocr.templates.contract",
    "OCR_SKIP_LABELS": "bodhan_genai.ocr.templates.contract",
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
