"""Data pipeline for IndicDocLayout: manifests, the packed page cache, corpus counts.

Stage order, and what each step writes:

    configs/ocr/data/*.yaml  --splits-->  layout_{train,val,test}.json
    layout_train.json        --pack---->  <cache>_shard*.blob + _meta.npz + _stems.pkl
    layout_*.json            --summarize->  label / source / domain counts

Nothing here imports torch. ``taxonomy`` in particular is pure Python so the label set
and the page parser can be used from a laptop, from a test, or from the recognizer side
of the pipeline without dragging in the training stack.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhan_genai.ocr.data.blob import BlobCache, PackStats, pack
    from bodhan_genai.ocr.data.splits import SourceSpec, build_manifests, load_sources
    from bodhan_genai.ocr.data.summarize import CorpusSummary, summarize
    from bodhan_genai.ocr.data.taxonomy import (
        CLASS_WEIGHTS,
        CLASSES,
        ID2LABEL,
        LABEL2ID,
        NUM_CLASSES,
        labels_from_doc,
    )

__all__ = [
    "CLASSES",
    "CLASS_WEIGHTS",
    "ID2LABEL",
    "LABEL2ID",
    "NUM_CLASSES",
    "BlobCache",
    "CorpusSummary",
    "PackStats",
    "SourceSpec",
    "build_manifests",
    "labels_from_doc",
    "load_sources",
    "pack",
    "summarize",
]

_SOURCES = {
    "blob": ("BlobCache", "PackStats", "pack"),
    "splits": ("SourceSpec", "build_manifests", "load_sources"),
    "summarize": ("CorpusSummary", "summarize"),
    "taxonomy": (
        "CLASSES",
        "CLASS_WEIGHTS",
        "ID2LABEL",
        "LABEL2ID",
        "NUM_CLASSES",
        "labels_from_doc",
    ),
}


def __getattr__(name: str):
    """Import the owning submodule on first use.

    Keeps ``import bodhan_genai.ocr.data`` free, and keeps ``python -m
    bodhan_genai.ocr.data.blob`` from importing that module twice (once as the package
    attribute, once as ``__main__``), which is what produces the RuntimeWarning.
    """
    for module, exports in _SOURCES.items():
        if name in exports:
            import importlib

            return getattr(importlib.import_module(f"{__name__}.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
