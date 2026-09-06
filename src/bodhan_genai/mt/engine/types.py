"""Engine-facing plain-data types shared by every MT backend.

``MTSamplingConfig`` is the backend-agnostic sampling knob set (each backend maps
it onto its own generate/SamplingParams API); ``MTResult`` is the uniform output
record every backend returns.

Defaults are **greedy** (``temperature=0.0``). That is the recommended setting for
translation and the only one reproducible run to run — a sampled translation is a
different translation every time, which makes regressions unmeasurable.

stdlib-only module imports — safe to import without torch / vllm / transformers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class MTSamplingConfig:
    """Backend-agnostic sampling parameters (frozen; use ``merged`` to derive).

    ``max_new_tokens`` sizing, from the model card: 512 suits sentences and short
    segments, 2048 a paragraph, 8192 a full document. Indic targets need roughly
    1.5-2x the English source token count, so budget generously.

    ``top_p`` and ``seed`` are only meaningful when ``temperature > 0``; the
    backends drop them under greedy decoding rather than passing contradictory
    parameters down.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        """True when decoding is deterministic (``temperature`` at or below 0)."""
        return not self.temperature or self.temperature <= 0.0

    def merged(self, **overrides) -> MTSamplingConfig:
        """Return a new config with non-None ``overrides`` applied.

        ``None`` values are ignored (so per-request kwargs can be passed straight
        through without filtering). Unknown keys raise ``TypeError``.
        """
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"Unknown MTSamplingConfig field(s): {', '.join(unknown)}")
        updates = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **updates)


@dataclass
class MTResult:
    """One translated segment plus accounting.

    ``error`` is set (and ``text`` left empty) when generation failed for this
    request — batch callers can keep going and inspect the failures afterwards
    instead of losing the whole batch to one bad row.
    """

    text: str = ""
    source: str = ""
    src_lang: str | None = None
    tgt_lang: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    gen_time_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when this request produced a translation."""
        return self.error is None

    def as_record(self) -> dict[str, object]:
        """JSONL-friendly record, matching the shipped CLIs' output schema.

        ``src_lang`` is carried through for bookkeeping only — the prompt never
        names the source language.
        """
        record: dict[str, object] = {
            "source": self.source,
            "translation": self.text,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
        }
        if self.error is not None:
            record["error"] = self.error
        return record
