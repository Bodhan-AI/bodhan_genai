"""bodhan-genai MT: IndicTranslate — Gemma-4-E4B instruction-tuned for Indic translation.

A decoder-only multimodal LLM (``Gemma4ForConditionalGeneration``) fine-tuned to
translate between English and 22 Eighth-Schedule Indian languages — 25
language-script combinations, 44 directions. There are no language tokens and no
``forced_bos_token_id``: the target language is an English name interpolated into
an instruction sentence, and the source language is never named.

Subpackages
-----------
- ``templates``: the prompt contract (target-language-only, one user turn) plus
  the phrasing variants and the training-time ``{% generation %}`` chat template.
- ``engine``: PUBLIC API — ``IndicMTEngine``, ``MTSamplingConfig``, ``MTResult``.
- ``inference``: batch (vLLM) and reference (HF ``generate``) CLIs.
- ``data``: bitext -> instruction JSONL rendering and the training-dataset loader.
- ``training``: LoRA finetuning (provisional; see ``docs/mt/training.md``).
- ``serving``: typed client for a stock ``vllm serve`` OpenAI-compatible endpoint.
- ``eval``: IN22 score replication (BLEU / chrF++).
- ``tools``: checkpoint surgery — the KV-shared ``k_norm`` sidecar vLLM requires.

Requires ``transformers>=5.12`` and ``vllm>=0.20`` — the releases that know the
Gemma 4 architecture. Both modalities share one environment on those versions;
see ``./install.sh``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bodhan-genai")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

# PEP 562 lazy exports: importing bodhan_genai.mt (or pulling the prompt contract
# from it) must never import torch / vllm / transformers / trl eagerly; each name
# resolves its module on first attribute access.
_LAZY = {
    "IndicMTEngine": "bodhan_genai.mt.engine.offline",
    "MTSamplingConfig": "bodhan_genai.mt.engine.types",
    "MTResult": "bodhan_genai.mt.engine.types",
    "LANGUAGE_NAMES": "bodhan_genai.mt.templates.prompt",
    "STOP_STRINGS": "bodhan_genai.mt.templates.prompt",
    "build_conversation": "bodhan_genai.mt.templates.prompt",
    "build_instruction": "bodhan_genai.mt.templates.prompt",
    "resolve_language": "bodhan_genai.mt.templates.prompt",
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
