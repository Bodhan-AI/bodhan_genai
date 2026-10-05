"""Engine-facing plain-data types shared by every MT backend.

``MTSamplingConfig`` is the backend-agnostic sampling knob set (each backend maps
it onto its own generate/SamplingParams API); ``MTRequest`` is the uniform input
a backend receives; ``BackendOutput`` is what it hands back; ``MTResult`` is the
uniform output record the engine returns to the caller.

``MTRequest`` carries the target language **as given**, not resolved to a prompt
name, because the two backend families need different things from it: the local
ones render a prompt, while the hosted API puts the language *code* on the wire
and builds the prompt itself.

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

    ``max_new_tokens`` sizing: 512 suits native-script sentences and short
    segments, 2048 a paragraph, 8192 a full document. Indic targets need roughly
    1.5-2x the English source token count, so budget generously.

    **A bigger budget is not always better, and on ``script="roman"`` it can be
    worse.** Measured on IN22-Gen, 64 samples per language: a few ``en2roman``
    rows are degenerate — the model loops instead of terminating — and they hit
    whatever budget they are given. Raising 1024 to 2048 left the truncated count
    identical (2/64 into Hindi) and *lowered* pooled chrF++, because the runaway
    rows simply got longer. Median output on the same set is ~130-170 characters.

    So: check ``MTResult.truncated`` (a truncated translation comes back fluent
    and simply stops), but treat a stable truncation count across two budgets as
    degeneration rather than a sizing problem — see
    :func:`bodhan_genai.mt.eval.metrics.is_degenerate`.

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


@dataclass(frozen=True)
class MTRequest:
    """One translation or transliteration request, before any prompt exists.

    This is the unit every backend receives. Deliberately pre-prompt: the hosted
    API never sees a prompt (it builds one server-side from ``text``,
    ``tgt_lang`` and ``script``), so a conversation-level request type would have
    to be un-rendered to reach the wire.

    ``script`` selects the output rendering -- ``native``, ``roman`` or
    ``code_mixed``. ``task`` is ``translate`` or ``transliterate``. ``src_lang``
    is bookkeeping only: the prompt never names the source language and it is
    never sent.
    """

    text: str
    tgt_lang: str
    script: str = "native"
    task: str = "translate"
    src_lang: str | None = None


@dataclass(frozen=True)
class BackendOutput:
    """What a backend returns for one request.

    Not a bare ``str``, because ``truncated`` has to survive the trip. A truncated
    translation comes back fluent and simply stops -- there is no other signal, so
    dropping the flag here loses it for good.

    ``gen_time_s`` is ``None`` when the backend cannot attribute time to a single
    request (a batched local backend times the whole batch); the engine then
    splits the batch elapsed evenly rather than inventing a per-request figure.
    """

    text: str
    truncated: bool = False
    prompt_tokens: int = 0
    generated_tokens: int = 0
    gen_time_s: float | None = None


@dataclass
class MTResult:
    """One translated segment plus accounting.

    ``error`` is set (and ``text`` left empty) when generation failed for this
    request — batch callers can keep going and inspect the failures afterwards
    instead of losing the whole batch to one bad row.

    ``truncated`` means the output hit the token budget and stopped mid-thought.
    It is deliberately NOT folded into ``ok``: a truncated segment is a real
    partial result that must still be scored. Demoting it to a failure would
    silently drop rows from an eval and move the numbers. Use ``complete`` for
    the strict view.
    """

    text: str = ""
    source: str = ""
    src_lang: str | None = None
    tgt_lang: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    gen_time_s: float = 0.0
    error: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        """True when this request produced a translation.

        A truncated result is still ``ok`` — see ``complete``.
        """
        return self.error is None

    @property
    def complete(self) -> bool:
        """True when this request produced a translation that was not cut short."""
        return self.error is None and not self.truncated

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
        if self.truncated:
            # Emitted only when true, mirroring ``error``, so the healthy-case
            # JSONL schema is unchanged and a truncation is a greppable key.
            record["truncated"] = True
        return record
