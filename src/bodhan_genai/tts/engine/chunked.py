"""ChunkedIndicStreamingTTS — long-form synthesis over either engine.

Single-shot generation is capped at ``max_new_tokens`` (2048 ≈ ~25 s of audio
at ~82 SNAC tokens/s) and long inputs degrade reliability, so long text is
segmented into sentences by a rule-based scanner (``split_sentences``:
terminator-run classification with abbreviation/initial/decimal/ellipsis
guards over Latin, Devanagari and CJK punctuation), oversized sentences are
laddered down at clause marks → newlines → whitespace → hard cuts, and the
pieces are greedily packed into chunks of at most ``max_chars``. Each chunk is
synthesized independently (same speaker → consistent voice), and the results
are delivered as one utterance with consistent volume:

- offline (``IndicTTSEngine``): all chunks go through ONE ``synthesize_batch``
  call, then per-chunk silence-trim + LUFS-normalize → silence-gap concat →
  whole-utterance peak CAP (rescale only when headroom is exceeded, so the
  -23 LUFS program level survives).
- streaming (``IndicStreamingTTSEngine``): chunks are prefetched ahead of
  playback; a chunk whose generation finished before its turn gets the exact
  per-chunk treatment off the event loop (typical for paced playback clients —
  unthrottled drains mostly take the live path), while still-generating chunks
  stream through a causal
  :class:`~bodhan_genai.tts.engine.loudness.StreamingLoudnessNormalizer`
  targeting the SAME silence-gated level, for low time-to-first-audio with no
  level step at the hybrid seam.

Conversations are turn-structured already, so turn-boundary chunking is
supported end-to-end: ``chunk_dialogue`` plans a message list into chunks
whose SERIALIZED form (see ``templates.conversation.format_messages``) fits
``max_chars`` (via ``plan_dialogue_chunks``, exploding over-long turns at
sentence boundaries into same-speaker segments),
``synthesize_conversation_long`` runs the plan through ONE
``synthesize_conversation_batch`` call and the offline combine above, and
``stream_conversation_long`` / ``stream_conversation_long_sync`` stream it
through the same prefetch / exact-vs-live / gap machinery as ``stream_long``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncGenerator, Iterator
from typing import Any

import numpy as np

from bodhan_genai.tts.engine.loudness import (
    ACTIVE_FLOOR_DBFS,
    DEFAULT_PEAK_DBFS,
    DEFAULT_TARGET_LUFS,
    DEFAULT_TRIM_DB,
    StreamingLoudnessNormalizer,
    _db_to_lin,
    normalize_loudness,
    rms,
    trim_silence,
)
from bodhan_genai.tts.engine.types import TTSResult

logger = logging.getLogger(__name__)

# --- sentence segmentation ---------------------------------------------------
#
# A maximal run of terminator marks ends a sentence unless a guard says
# otherwise. Devanagari danda/double-danda and CJK terminators are unambiguous
# sentence enders and split even when glued to the next character (real Hindi
# and spaceless CJK text often omit the space); Latin ./!/? need following
# whitespace and survive decimals ("3.14"), abbreviations ("Dr.", "e.g."),
# initials ("J. K. Rowling"), ellipses ("...", "…", ". . .") and quote
# attribution ('"Stop!" she yelled.'). Terminators + closing quotes/brackets
# stay attached to the ending sentence (prosody).
# Terminator runs: . ! ? … । ॥ 。 ！ ？ (maximal runs)  # noqa: RUF003
_TERM_RUN_RE = re.compile(r"[.!?…।॥。！？]+")  # noqa: RUF001
_UNAMBIG = frozenset("।॥。！？")  # noqa: RUF001
_ELLIPSIS = "…"
# Closing quotes/brackets that may trail a terminator run:
# " ' ” ’ » ) ] } 」 』 ） 】 》 〉  # noqa: RUF003
_CLOSER_RUN_RE = re.compile(r"[\"'”’»)\]}」』）】》〉]*")  # noqa: RUF001
# Clause marks (ladder rung 1): Latin need following whitespace; fullwidth
# ， ； ： 、 split even when glued.  # noqa: RUF003
_CLAUSE_RE = re.compile(r"[,;:](?=\s)|[，；：、]")  # noqa: RUF001
_PARA_RE = re.compile(r"\n\s*\n")  # blank line = paragraph boundary; single \n is NOT
_WS_RE = re.compile(r"\s")
# Latin + Devanagari letters, EXCLUDING danda U+0964/65 and digits U+0966-096F.
# \w is deliberately avoided: its Devanagari matra coverage is unreliable.
_LETTER = r"[A-Za-zऀ-ॣ॰-ॿ]"
_TOKEN_RE = re.compile(rf"({_LETTER}+(?:\.{_LETTER}+)*)\Z")  # maximal dotted token before run
# Single uppercase Latin letter preceded by start-of-text or space/dot/opening
# quote/bracket/dash — the dot in the class makes "U.S.A." work.
_INITIAL_RE = re.compile(r"(?:\A|[\s.\"'“‘«(\[{—–-])[A-Z]\Z")  # noqa: RUF001
_ABBREV = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "smt",
        "etc",
        "vs",
        "e.g",
        "i.e",
        "cf",
        "viz",
        "approx",
        "डॉ",
        "प्रो",
    }
)
# Suppressed ONLY when followed by optional whitespace + ASCII/Devanagari digit.
_NUMERIC_ABBREV = frozenset({"no", "rs"})
_DIGIT_AFTER_RE = re.compile(r"\s*[0-9०-९]")  # noqa: RUF001
_SPACED_DOT_AHEAD_RE = re.compile(r"\s*\.")
_SPACED_DOT_BEHIND_RE = re.compile(r"\.\s+\Z")
_LOWER_AHEAD_RE = re.compile(r"\s*[a-z]")


def _ws_or_eot(text: str, i: int) -> bool:
    """True if position ``i`` is end-of-text or whitespace."""
    return i >= len(text) or text[i].isspace()


def _period_is_boundary(text: str, s: int, e: int, e2: int) -> bool:
    """Guards for a bare "." run at ``[s, e)`` with closers consumed up to ``e2``."""
    # G1: whitespace/end must follow (kills "3.14", "e.g" internal dot, glued text).
    if not _ws_or_eot(text, e2):
        return False
    # G2: spaced ellipsis "Wait . . . what" — a dot ahead or a dot+space just behind.
    if _SPACED_DOT_AHEAD_RE.match(text, e2) or _SPACED_DOT_BEHIND_RE.search(text, 0, s):
        return False
    # G3: single-letter initial ("J. K. Rowling", "U.S.A.").
    if _INITIAL_RE.search(text, 0, s):
        return False
    # G4: known abbreviations; numeric ones only when a digit follows ("No. 5").
    m = _TOKEN_RE.search(text, 0, s)
    if m:
        token = m.group(1).lower()
        if token in _ABBREV:
            return False
        if token in _NUMERIC_ABBREV and _DIGIT_AFTER_RE.match(text, e2):
            return False
    # G5: closer consumed and lowercase continuation: 'He said "go home." and left.'
    return not (e2 > e and _LOWER_AHEAD_RE.match(text, e2))


def _split_para(para: str) -> list[str]:
    """Scan one paragraph for sentence boundaries (see ``split_sentences``)."""
    pieces: list[str] = []
    prev = 0
    for m in _TERM_RUN_RE.finditer(para):
        s, e = m.span()
        run = m.group()
        e2 = _CLOSER_RUN_RE.match(para, e).end()  # closers attach to the sentence
        if any(ch in _UNAMBIG for ch in run):
            boundary = True  # R1: danda/CJK end the sentence unconditionally
        elif "?" in run or "!" in run:
            # R2: ?/! runs (?!, !!!, ?..) end at whitespace/end-of-text; R5:
            # closer + lowercase continuation is attribution, not a boundary.
            boundary = _ws_or_eot(para, e2)
            if boundary and e2 > e and _LOWER_AHEAD_RE.match(para, e2):
                boundary = False
        elif len(run) > 1 or _ELLIPSIS in run:
            boundary = False  # R3: ellipsis is a pause, never a boundary
        else:
            boundary = _period_is_boundary(para, s, e, e2)  # R4: run == "."
        if boundary:
            piece = para[prev:e2].strip()
            if piece:
                pieces.append(piece)
            prev = e2
    tail = para[prev:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences (rule-based, no models, Latin + Devanagari
    + CJK aware).

    Paragraphs (blank-line separated) are split first — a single newline is
    NOT a boundary. Within a paragraph, maximal terminator runs are classified
    per the R1-R5/G1-G5 rules above. Normalization: CRLF → LF, zero-width
    space → space; ZWJ/ZWNJ are preserved (they are meaningful in Devanagari).

    Known quirk (accepted): an unambiguous terminator inside quotes splits
    unconditionally — 'उसने कहा, "ठीक है।" और चल दिया।' yields two sentences,
    breaking after ``।"``."""
    text = (text or "").replace("\r\n", "\n").replace("\u200b", " ")
    pieces: list[str] = []
    for para in _PARA_RE.split(text):
        pieces.extend(_split_para(para))
    return pieces


# --- laddering + packing -------------------------------------------------------


def _ladder_cut(s: str, max_chars: int) -> int:
    """Best cut index for a single sentence longer than ``max_chars``.

    Rungs: clause mark whose end lands in [max_chars//4, max_chars] → rightmost
    newline → rightmost whitespace of any kind → hard cut at ``max_chars``."""
    floor = max(1, max_chars // 4)
    best = -1
    for m in _CLAUSE_RE.finditer(s):
        if m.end() > max_chars:
            break
        if m.end() >= floor:
            best = m.end()
    if best != -1:
        return best
    nl = s.rfind("\n", 1, max_chars + 1)
    if nl != -1:
        return nl
    cut = -1
    for m in _WS_RE.finditer(s, 1, max_chars + 1):
        cut = m.start()
    if cut > 0:
        return cut
    return max_chars  # one unbroken token — cut hard


# --- speech-duration estimation ------------------------------------------------

# Rough speaking rates in characters/second by script. Heuristics, deliberately
# coarse — used for chunk budgeting and degenerate-output detection, not
# timing-accurate synthesis planning. Calibrate from real synthesis logs (the
# bench harness reports actual vs estimated duration per run).
_CPS_LATIN = 14.0
_CPS_DEVANAGARI = 12.0
_CPS_CJK = 5.0
_CPS_DEFAULT = 12.0


def _char_rate(ch: str) -> float:
    cp = ord(ch)
    if cp < 0x0300:  # Latin + digits + ASCII punctuation/space
        return _CPS_LATIN
    if 0x0900 <= cp <= 0x097F:  # Devanagari
        return _CPS_DEVANAGARI
    if 0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF:  # CJK
        return _CPS_CJK
    return _CPS_DEFAULT


def estimate_speech_seconds(text: str) -> float:
    """Estimate spoken duration of ``text`` in seconds (script-aware heuristic).

    Sums per-character time by Unicode block (Latin ≈ 14 chars/s, Devanagari
    ≈ 12, CJK ≈ 5). Used to size chunks in seconds and to flag degenerate
    generations (bench harness); NOT a substitute for measured duration."""
    return float(sum(1.0 / _char_rate(ch) for ch in text or ""))


def _ladder_pieces(sentences: list[str], max_chars: int) -> list[str]:
    """Ladder any sentence longer than ``max_chars`` down into fitting pieces."""
    pieces: list[str] = []
    for sent in sentences:
        while len(sent) > max_chars:
            cut = _ladder_cut(sent, max_chars)
            head = sent[:cut].rstrip()
            if head:
                pieces.append(head)
            sent = sent[cut:].lstrip()
        if sent:
            pieces.append(sent)
    return pieces


def _greedy_pack(pieces: list[str], max_chars: int) -> list[str]:
    """Pack consecutive pieces with a " " joiner while staying <= ``max_chars``."""
    packed: list[str] = []
    for p in pieces:
        if packed and len(packed[-1]) + 1 + len(p) <= max_chars:
            packed[-1] = packed[-1] + " " + p
        else:
            packed.append(p)
    return packed


def chunk_text(
    text: str,
    *,
    min_chars: int = 16,
    max_chars: int = 300,
    first_chunk_chars: int | None = None,
) -> list[str]:
    """Split ``text`` into synthesis chunks at sentence boundaries.

    Sentences come from ``split_sentences``; any sentence longer than
    ``max_chars`` is laddered down (clause mark → newline → whitespace → hard
    cut, never mid-word unless the token itself exceeds the limit); consecutive
    pieces are then greedily packed with a " " joiner.

    ``max_chars`` is a STRICT upper bound on every returned chunk — it keeps
    each chunk safely under the generation ceiling (2048 tokens ≈ 25 s ≈ 300+
    chars of Hindi/English speech). ``min_chars`` is best-effort only: the
    greedy pack merges a short fragment into its neighbour whenever the merge
    fits ``max_chars`` (fit-gated), but a fragment whose neighbours are full
    stays short rather than overflow the bound.

    ``first_chunk_chars`` (< ``max_chars``) ramps the schedule: chunk 0 packs
    to the smaller budget so streaming time-to-first-audio is dominated by a
    short generation, while later chunks keep the full budget for prosody."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = split_sentences(text) or [text]
    pieces = _ladder_pieces(sentences, max_chars)
    if not first_chunk_chars or first_chunk_chars >= max_chars or not pieces:
        return _greedy_pack(pieces, max_chars)
    first_budget = max(1, int(first_chunk_chars))
    if len(pieces[0]) > first_budget:
        # First sentence exceeds the ramp budget: ladder IT to the ramp size.
        head_pieces = _ladder_pieces([pieces[0]], first_budget)
        first, rest = head_pieces[0], head_pieces[1:] + pieces[1:]
    else:
        first_parts = [pieces[0]]
        i = 1
        length = len(pieces[0])
        while i < len(pieces) and length + 1 + len(pieces[i]) <= first_budget:
            length += 1 + len(pieces[i])
            first_parts.append(pieces[i])
            i += 1
        first, rest = " ".join(first_parts), pieces[i:]
    return [first, *_greedy_pack(rest, max_chars)]


# --- dialogue packing ----------------------------------------------------------

# Serialization cost model — MUST stay lock-step with
# templates/conversation.format_messages: "<|speaker>NAME<speaker|>\ntext",
# turns joined with "\n\n".
_TAG_OVERHEAD = len("<|speaker>") + len("<speaker|>") + 1  # tags + "\n" == 21
_SEP = 2  # "\n\n" between serialized turns


def _turn_cost(message: dict) -> int:
    """Serialized length of one turn under ``format_messages``."""
    return _TAG_OVERHEAD + len(message["speaker"]) + len(message["text"])


def plan_dialogue_chunks(
    messages: list[dict],
    *,
    max_chars: int,
    long_turn_chars: int | None = None,
    first_chunk_chars: int | None = None,
) -> list[list[dict]]:
    """Plan a multi-turn dialogue into chunks whose SERIALIZED form fits
    ``max_chars``.

    Each message is ``{"speaker": ..., "text": ...}`` (extra keys are carried
    through). Empty-text turns are dropped; a blank speaker with non-empty text
    raises ``ValueError``. A turn whose serialized cost exceeds the per-turn
    budget is exploded at sentence boundaries (``split_sentences`` + ladder +
    greedy regroup) into several same-speaker messages; whole turns are then
    greedily packed into chunks.

    Invariant: ``len(format_messages(chunk)) <= max_chars`` for every chunk
    whenever ``long_turn_chars`` (default: ``max_chars``) does not exceed
    ``max_chars``. Passing ``long_turn_chars > max_chars`` is the documented
    keep-turns-intact escape hatch: turns up to that size are NOT split, and
    may land in oversized single-turn chunks."""
    # 1) sanitize: strip, drop empty-text turns, reject blank speakers.
    turns: list[dict] = []
    for i, m in enumerate(messages):
        speaker = str(m.get("speaker", "") or "").strip()
        text = str(m.get("text", "") or "").strip()
        if not text:
            continue
        if not speaker:
            raise ValueError(f"messages[{i}] has no speaker")
        turns.append({**m, "speaker": speaker, "text": text})

    # 2) explode over-long turns at sentence boundaries.
    ltc = long_turn_chars or max_chars
    segments: list[dict] = []
    for t in turns:
        budget = ltc - (_TAG_OVERHEAD + len(t["speaker"]))
        if budget < 8:
            raise ValueError(
                f"speaker {t['speaker']!r} leaves a per-turn text budget of {budget} chars "
                f"(< 8) under a {ltc}-char turn limit — raise max_chars/long_turn_chars"
            )
        if len(t["text"]) <= budget:
            segments.append(t)
            continue
        sentences = split_sentences(t["text"]) or [t["text"]]
        groups = _greedy_pack(_ladder_pieces(sentences, budget), budget)
        segments.extend({**t, "text": g} for g in groups)

    # 3) pack whole (possibly synthesized) turns greedily. Chunk 0 may use the
    # smaller ramp budget (streaming TTFA); whole turns are never split just
    # for the ramp, so a single oversized-vs-ramp turn still opens chunk 0.
    first_limit = min(int(first_chunk_chars), max_chars) if first_chunk_chars else max_chars
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_cost = 0
    for t in segments:
        limit = first_limit if not chunks else max_chars
        cost = _turn_cost(t)
        if cur and cur_cost + _SEP + cost > limit:
            chunks.append(cur)
            cur, cur_cost = [], 0
        cur_cost += cost + (_SEP if cur else 0)
        cur.append(t)
    if cur:
        chunks.append(cur)
    return chunks


class ChunkedIndicStreamingTTS:
    """Long-form chunked synthesis wrapper over an EXISTING engine instance.

    Pass a :class:`IndicTTSEngine` for offline ``synthesize_long`` or a
    :class:`IndicStreamingTTSEngine` for ``stream_long`` /
    ``stream_long_sync``. The wrapper owns no GPU state of its own — closing /
    shutting down the underlying engine remains the caller's job."""

    def __init__(
        self,
        engine: Any,
        *,
        min_chunk_chars: int = 16,
        max_chunk_chars: int = 300,
        long_turn_chars: int | None = None,
        first_chunk_chars: int | None = None,
        max_chunk_seconds: float | None = None,
        first_chunk_seconds: float | None = None,
        max_chunks: int = 256,
        gap_ms: float = 250.0,
        target_lufs: float = DEFAULT_TARGET_LUFS,
        peak_dbfs: float = DEFAULT_PEAK_DBFS,
        trim_db: float = DEFAULT_TRIM_DB,
        normalize: bool = True,
        prefetch: int = 1,
        warmup: bool = False,
    ) -> None:
        self._engine = engine
        self._min_chars = int(min_chunk_chars)
        self._max_chars = int(max_chunk_chars)
        self._long_turn_chars = None if long_turn_chars is None else int(long_turn_chars)
        # Ramp / duration budgets: seconds knobs convert per call via the
        # text's own script mix (see _char_budgets); chars knobs win when the
        # seconds knobs are unset. Ramp applies to STREAMING paths only.
        self._first_chars = None if first_chunk_chars is None else int(first_chunk_chars)
        self._max_seconds = None if max_chunk_seconds is None else float(max_chunk_seconds)
        self._first_seconds = None if first_chunk_seconds is None else float(first_chunk_seconds)
        self.last_stream_stats: dict | None = None
        self._max_chunks = int(max_chunks)
        self._gap_ms = float(gap_ms)
        self._target_lufs = float(target_lufs)
        self._peak_dbfs = float(peak_dbfs)
        self._trim_db = float(trim_db)
        self._normalize = bool(normalize)
        self._prefetch = max(0, int(prefetch))
        self._bridge = None  # created lazily for stream_long_sync
        if warmup and self._normalize:
            # Pay librosa's ~20 s first-trim lazy init at construction (e.g.
            # replica startup, before it reports ready) — never on the event
            # loop of a live stream.
            from bodhan_genai.tts.engine.loudness import warm_dsp

            warm_dsp(self._sample_rate)

    def _check_chunk_budget(self, chunks: list) -> None:
        if len(chunks) > self._max_chunks:
            raise ValueError(
                f"text splits into {len(chunks)} chunks, exceeding max_chunks="
                f"{self._max_chunks} — synthesize in smaller pieces or raise the limit"
            )

    # -- shared ---------------------------------------------------------------

    def _char_budgets(self, sample_text: str, *, ramp: bool) -> tuple[int, int | None]:
        """Resolve (max_chars, first_chunk_chars) for a call.

        Seconds knobs convert to chars ONCE per call using the sample text's
        own script mix (chars-per-second from ``estimate_speech_seconds``), so
        a Devanagari paragraph gets a smaller char budget than a Latin one for
        the same target duration. ``ramp=False`` (offline) drops the first-
        chunk budget — batch synthesis has no time-to-first-audio concern."""
        max_chars = self._max_chars
        first = self._first_chars if ramp else None
        if self._max_seconds or (ramp and self._first_seconds):
            est = estimate_speech_seconds(sample_text)
            if est > 0:
                rate = len(sample_text) / est  # chars/second for THIS text
                if self._max_seconds:
                    max_chars = max(16, int(rate * self._max_seconds))
                if ramp and self._first_seconds:
                    first = max(8, int(rate * self._first_seconds))
        return max_chars, first

    def chunk(self, text: str, *, ramp: bool = False) -> list[str]:
        """The chunking this wrapper will use for ``text`` (for inspection).
        ``ramp=True`` previews the streaming schedule (small first chunk)."""
        max_chars, first = self._char_budgets(text, ramp=ramp)
        return chunk_text(
            text, min_chars=self._min_chars, max_chars=max_chars, first_chunk_chars=first
        )

    def chunk_dialogue(self, messages: list[dict], *, ramp: bool = False) -> list[list[dict]]:
        """The dialogue chunk plan this wrapper will use for ``messages`` (for
        inspection): turn-boundary packing via ``plan_dialogue_chunks``."""
        sample = " ".join(str(m.get("text", "")) for m in messages)
        max_chars, first = self._char_budgets(sample, ramp=ramp)
        return plan_dialogue_chunks(
            messages,
            max_chars=max_chars,
            long_turn_chars=self._long_turn_chars,
            first_chunk_chars=first,
        )

    @staticmethod
    def _dialogue_labels(plan: list[list[dict]]) -> list[str]:
        """Short per-chunk tags for logs/errors: first turn's speaker + text."""
        return [f"{c[0]['speaker']}: {c[0]['text']}"[:40] for c in plan]

    @property
    def _sample_rate(self) -> int:
        return int(getattr(self._engine, "sample_rate", 24_000))

    def _gap_samples(self) -> int:
        return int(self._gap_ms / 1000.0 * self._sample_rate)

    def _finalize_chunk(self, y: np.ndarray, domain: str = "lufs") -> np.ndarray:
        """Trim + loudness-normalize one chunk.

        ``domain`` picks the level-measurement domain: "lufs" for the offline
        combine (the documented BS.1770 recipe) and "rms" for the STREAMING
        exact path — silence-gated RMS, the same domain the causal
        ``StreamingLoudnessNormalizer`` targets, so prefetched (exact) and live
        chunks land at ONE level with no audible step at the hybrid seam."""
        if not self._normalize:
            return y.astype(np.float32)
        y = trim_silence(y, top_db=self._trim_db, sample_rate=self._sample_rate)
        if domain == "rms":
            from bodhan_genai.tts.engine.loudness import normalize_active_rms

            return normalize_active_rms(
                y, target_db=self._target_lufs, sample_rate=self._sample_rate
            )
        return normalize_loudness(y, target_lufs=self._target_lufs, sample_rate=self._sample_rate)

    # -- offline ---------------------------------------------------------------

    def synthesize_long(
        self,
        text: str,
        *,
        speaker: str = "",
        return_chunks: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> TTSResult | tuple[TTSResult, list[TTSResult]]:
        """Synthesize arbitrarily long text as ONE consistent-volume utterance.

        All chunks run through a single ``synthesize_batch`` call; any failed
        chunk raises (partial audio is never silently glued together)."""
        chunks = self.chunk(text)
        if not chunks:
            raise ValueError("empty text (nothing to synthesize)")
        self._check_chunk_budget(chunks)
        logger.info(
            "synthesize_long: %d chunks (chars %s), gap %.0f ms",
            len(chunks),
            [len(c) for c in chunks],
            self._gap_ms,
        )

        results: list[TTSResult] = self._engine.synthesize_batch(
            chunks,
            speakers=speaker,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        failed = [(i, r.error) for i, r in enumerate(results) if r.error]
        if failed:
            detail = "; ".join(f"chunk {i} ({chunks[i][:40]!r}): {err}" for i, err in failed)
            logger.error(
                "synthesize_long: %d/%d chunks failed: %s", len(failed), len(chunks), detail
            )
            raise RuntimeError(f"{len(failed)}/{len(chunks)} chunks failed: {detail}")

        merged = self._combine_offline_results(results)
        return (merged, results) if return_chunks else merged

    def _combine_offline_results(self, results: list[TTSResult]) -> TTSResult:
        """Combine per-chunk offline results into ONE ``TTSResult``: per-chunk
        trim + LUFS-normalize → silence-gap concat → whole-utterance peak CAP,
        with summed token/time accounting."""
        gap = np.zeros(self._gap_samples(), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for j, r in enumerate(results):
            y = self._finalize_chunk(r.audio, domain="lufs")
            if j > 0 and gap.size:
                pieces.append(gap)
            pieces.append(y)
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        if self._normalize:
            # Peak CAP only (never boost): boosting to -1 dBFS would override
            # the -23 LUFS program level and make output loudness depend on
            # crest factor. Rescale only when headroom is exceeded.
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            limit = _db_to_lin(self._peak_dbfs)
            if peak > limit:
                audio = (audio * (limit / peak)).astype(np.float32)

        return TTSResult(
            audio=audio.astype(np.float32),
            sample_rate=self._sample_rate,
            prompt_tokens=sum(r.prompt_tokens for r in results),
            generated_tokens=sum(r.generated_tokens for r in results),
            audio_tokens=sum(r.audio_tokens for r in results),
            gen_time_s=sum(r.gen_time_s for r in results),
            decode_time_s=sum(r.decode_time_s for r in results),
        )

    def synthesize_conversation_long(
        self,
        messages: list[dict],
        *,
        return_chunks: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> TTSResult | tuple[TTSResult, list[TTSResult]]:
        """Synthesize a long multi-turn conversation as ONE consistent-volume
        utterance.

        The dialogue is planned into turn-boundary chunks (``chunk_dialogue``);
        all chunks run through a single ``synthesize_conversation_batch`` call
        and the same combine as ``synthesize_long``; any failed chunk raises
        (partial audio is never silently glued together)."""
        plan = self.chunk_dialogue(messages)
        if not plan:
            raise ValueError("empty conversation (nothing to synthesize)")
        self._check_chunk_budget(plan)
        logger.info(
            "synthesize_conversation_long: %d chunks (turns %s), gap %.0f ms",
            len(plan),
            [len(c) for c in plan],
            self._gap_ms,
        )

        results: list[TTSResult] = self._engine.synthesize_conversation_batch(
            plan,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        failed = [(i, r.error) for i, r in enumerate(results) if r.error]
        if failed:
            labels = self._dialogue_labels(plan)
            detail = "; ".join(f"chunk {i} ({labels[i]!r}): {err}" for i, err in failed)
            logger.error(
                "synthesize_conversation_long: %d/%d chunks failed: %s",
                len(failed),
                len(plan),
                detail,
            )
            raise RuntimeError(f"{len(failed)}/{len(plan)} chunks failed: {detail}")
        merged = self._combine_offline_results(results)
        return (merged, results) if return_chunks else merged

    # -- streaming ---------------------------------------------------------------

    async def stream_long(
        self,
        text: str,
        *,
        speaker: str = "",
        frames_per_message: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream long text chunk by chunk as raw int16 PCM.

        Up to ``prefetch`` chunks generate ahead of the one being emitted. A
        chunk whose generation ALREADY finished when its turn comes (typical
        for paced playback clients, where audio plays slower than it
        generates; unthrottled drains mostly take the live path) gets the
        exact per-chunk trim + silence-gated-RMS normalization off the event
        loop; a still-generating chunk streams live — first frame ships alone
        for low time-to-first-audio, edges are causally trimmed, and one
        shared causal loudness normalizer keeps the level in the SAME domain
        as the exact path. A ``gap_ms`` silence message separates chunks."""
        chunks = self.chunk(text, ramp=True)
        if not chunks:
            return
        self._check_chunk_budget(chunks)
        logger.info(
            "stream_long: %d chunks (chars %s), gap %.0f ms, prefetch %d",
            len(chunks),
            [len(c) for c in chunks],
            self._gap_ms,
            self._prefetch,
        )

        sampling = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        jobs = [
            lambda k=k: self._engine.stream(
                chunks[k], speaker=speaker, frames_per_message=1, **sampling
            )
            for k in range(len(chunks))
        ]
        labels = [c[:40] for c in chunks]
        async for msg in self._stream_chunk_jobs(
            jobs, labels, frames_per_message=frames_per_message
        ):
            yield msg

    async def stream_conversation_long(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> AsyncGenerator[bytes, None]:
        """Stream a long multi-turn conversation chunk by chunk as raw int16
        PCM.

        Chunks come from ``chunk_dialogue`` (turn-boundary packing); each is
        streamed through the engine's ``stream_conversation`` and delivered
        with the same prefetch / exact-vs-live loudness / gap machinery as
        ``stream_long``."""
        plan = self.chunk_dialogue(messages, ramp=True)
        if not plan:
            return
        self._check_chunk_budget(plan)
        logger.info(
            "stream_conversation_long: %d chunks (turns %s), gap %.0f ms, prefetch %d",
            len(plan),
            [len(c) for c in plan],
            self._gap_ms,
            self._prefetch,
        )

        sampling = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        jobs = [
            lambda k=k: self._engine.stream_conversation(plan[k], frames_per_message=1, **sampling)
            for k in range(len(plan))
        ]
        labels = self._dialogue_labels(plan)
        async for msg in self._stream_chunk_jobs(
            jobs, labels, frames_per_message=frames_per_message
        ):
            yield msg

    async def _stream_chunk_jobs(
        self,
        jobs: list,
        labels: list[str],
        *,
        frames_per_message: int,
    ) -> AsyncGenerator[bytes, None]:
        """Prefetch/emit machinery shared by ``stream_long`` and
        ``stream_conversation_long``: ``jobs[k]`` is a zero-arg factory
        returning the engine async frame generator for chunk k, ``labels[k]``
        tags chunk k in logs and errors. Exact-vs-live hybrid semantics as
        documented on ``stream_long``."""
        # exact/live path counts, exposed for prod logging + the bench harness.
        stats = {"chunks": len(jobs), "exact": 0, "live": 0, "gap_ms": self._gap_ms}
        gap_bytes = bytes(self._gap_samples() * 2)  # int16 zeros
        live_norm = StreamingLoudnessNormalizer(
            target_lufs=self._target_lufs, sample_rate=self._sample_rate
        )
        silence_floor = _db_to_lin(ACTIVE_FLOOR_DBFS)

        # Producers: buffers[k] fills as chunk k generates; done[k] marks EOF;
        # data_evt[k] wakes the consumer on every append (no busy-polling).
        buffers: list[list[bytes]] = [[] for _ in jobs]
        done: list[asyncio.Event] = [asyncio.Event() for _ in jobs]
        data_evt: list[asyncio.Event] = [asyncio.Event() for _ in jobs]
        errors: list[BaseException | None] = [None] * len(jobs)

        async def _produce(k: int) -> None:
            try:
                async for frame in jobs[k]():
                    buffers[k].append(frame)
                    data_evt[k].set()
            except BaseException as e:  # surfaced by the consumer, chunk-tagged
                errors[k] = e
            finally:
                done[k].set()
                data_evt[k].set()

        tasks: list[asyncio.Task] = []

        def _ensure_producers(upto: int) -> None:
            while len(tasks) < min(len(jobs), upto + 1):
                tasks.append(asyncio.create_task(_produce(len(tasks))))

        def _frame_is_voiced(frame: bytes) -> bool:
            y = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767.0
            return rms(y) > silence_floor

        loop = asyncio.get_running_loop()
        group = max(1, int(frames_per_message))
        try:
            _ensure_producers(self._prefetch)  # head + lookahead
            for k in range(len(jobs)):
                _ensure_producers(k + self._prefetch)
                # Boundary tick: the exact path has no awaits, so give pending
                # producers one loop turn before deciding live vs exact.
                await asyncio.sleep(0)
                if done[k].is_set() and errors[k] is not None:
                    # Fail BEFORE the gap: the client must not receive trailing
                    # junk silence ahead of the error frame.
                    self._raise_chunk_error(k, labels, errors[k])
                if k > 0 and gap_bytes:
                    yield gap_bytes

                if done[k].is_set() and errors[k] is None and self._normalize:
                    # Exact path: full chunk available before emission. The DSP
                    # (trim + normalize) runs OFF the event loop — with vLLM,
                    # SNAC micro-batching and every concurrent stream sharing
                    # this loop, a synchronous librosa/pyloudnorm call here
                    # would stall them all.
                    frames, group_ = buffers[k], group
                    msgs = await loop.run_in_executor(
                        None, lambda f=frames, g=group_: list(self._exact_chunk_messages(f, g))
                    )
                    stats["exact"] += 1
                    logger.debug(
                        "stream_long: chunk %d/%d exact path, %d frames",
                        k,
                        len(jobs),
                        len(frames),
                    )
                    for msg in msgs:
                        yield msg
                else:
                    # Live path: emit as frames land — first frame alone (low
                    # time-to-first-audio), causal normalization, causal edge
                    # trim (leading silence dropped, trailing silence held back
                    # and dropped at chunk end).
                    stats["live"] += 1
                    logger.debug("stream_long: chunk %d/%d live path", k, len(jobs))
                    idx = 0
                    buf: list[bytes] = []
                    pending_silence: list[bytes] = []
                    voiced_seen = False
                    sent_first = False
                    while True:
                        if idx < len(buffers[k]):
                            frame = buffers[k][idx]
                            idx += 1
                            if self._normalize:
                                if not _frame_is_voiced(frame):
                                    if not voiced_seen:
                                        continue  # leading silence: drop
                                    pending_silence.append(frame)  # maybe trailing
                                    continue
                                voiced_seen = True
                                for held in pending_silence:  # internal pause: flush
                                    buf.append(live_norm.process(held))
                                pending_silence = []
                                buf.append(live_norm.process(frame))
                            else:
                                buf.append(frame)
                            if buf and (not sent_first or len(buf) >= group):
                                yield b"".join(buf)
                                buf = []
                                sent_first = True
                            continue
                        if done[k].is_set() and idx >= len(buffers[k]):
                            break
                        # Event-driven wakeup (0.25 s safety timeout mirrors the
                        # engine's own consumer escape hatch).
                        data_evt[k].clear()
                        if idx < len(buffers[k]) or done[k].is_set():
                            continue
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(data_evt[k].wait(), timeout=0.25)
                    if buf:
                        yield b"".join(buf)
                    # pending_silence at chunk end = trailing silence: dropped.
                if errors[k] is not None:
                    self._raise_chunk_error(k, labels, errors[k])
                buffers[k] = []  # free emitted PCM: residency = current + prefetch
        finally:
            self.last_stream_stats = stats
            logger.info(
                "stream_long: done — %d chunks (%d exact, %d live)",
                stats["chunks"],
                stats["exact"],
                stats["live"],
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                if not t.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

    def _raise_chunk_error(self, k: int, labels: list[str], err: BaseException) -> None:
        logger.error("stream_long: chunk %d/%d failed (%r): %s", k, len(labels), labels[k], err)
        raise RuntimeError(f"chunk {k}/{len(labels)} failed ({labels[k]!r})") from err

    def _exact_chunk_messages(self, frames: list[bytes], group: int) -> Iterator[bytes]:
        """Assemble a fully-buffered chunk, apply the streaming exact recipe
        (trim + silence-gated-RMS normalize + per-chunk peak cap), re-slice to
        message-sized bytes. Runs in an executor thread — pure numpy/DSP, no
        event-loop state."""
        raw = b"".join(frames)
        if not raw:
            return
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        # "rms" domain: same silence-gated target the causal normalizer tracks,
        # so exact and live chunks sit at one level (no hybrid-seam step).
        y = self._finalize_chunk(y, domain="rms")
        # Whole-utterance peak-norm is impossible online; cap this chunk's
        # peak at the same headroom target instead.
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        limit = _db_to_lin(self._peak_dbfs)
        if peak > limit:
            y = y * (limit / peak)
        i16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        frame_bytes = len(frames[0]) if frames else 4096
        step = max(frame_bytes, frame_bytes * group)
        for off in range(0, len(i16), step):
            yield i16[off : off + step]

    def _get_bridge(self):
        """Bridge loop for the sync wrappers.

        Reuses the ENGINE's bridge loop when available so mixing
        ``engine.stream_sync`` and the wrapper's sync streams on one engine
        never trips the loop-ownership guard (both drive the same private
        loop)."""
        bridge = getattr(self._engine, "_bridge", None)
        if bridge is None:
            from bodhan_genai.tts.engine.streaming import _SyncStreamBridge

            if self._bridge is None:
                self._bridge = _SyncStreamBridge()
            bridge = self._bridge
        return bridge

    def stream_long_sync(
        self,
        text: str,
        *,
        speaker: str = "",
        frames_per_message: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[bytes]:
        """Blocking convenience wrapper around ``stream_long`` (same semantics
        as the engine's ``stream_sync``; bridge reuse per ``_get_bridge``)."""
        return self._get_bridge().run(
            lambda: self.stream_long(
                text,
                speaker=speaker,
                frames_per_message=frames_per_message,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
        )

    def stream_conversation_long_sync(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        frames_per_message: int = 1,
    ) -> Iterator[bytes]:
        """Blocking convenience wrapper around ``stream_conversation_long``
        (same semantics as ``stream_long_sync``; bridge reuse per
        ``_get_bridge``)."""
        return self._get_bridge().run(
            lambda: self.stream_conversation_long(
                messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                frames_per_message=frames_per_message,
            )
        )
