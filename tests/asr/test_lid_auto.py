# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the LID candidate set and scoring.

These run against fakes, so they need neither the checkpoint nor a GPU: what is
under test is the *selection* logic (which vocabulary entries count as languages,
how ``allowed_langs`` narrows them, how top-k is read back), not the acoustics.
"""

from __future__ import annotations

import pytest
import torch

from bodhan_genai.asr.engine.lid import (
    LID_PREFIX_LEN,
    TRAINED_LANGS,
    language_token_map,
    lid_from_encoder_states,
    lid_prefix_ids,
)

# A vocabulary shaped like the real spl block: languages, plus the control tokens
# whose names are also 2-3 lowercase letters and so look like languages.
PIECES = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<|startoftranscript|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|startofcontext|>",
    "<|itn|>",
    "<|noitn|>",
    "<|romanized|>",
    "<|noromanized|>",
    "<|nospeech|>",
    "<|notimestamp|>",
    "<|unklang|>",
    "<|nodiarize|>",
    "<|emo:undefined|>",
    "<|hi|>",
    "<|ur|>",
    "<|bgc|>",
    "<|ta|>",
    "<|en|>",
    "<|sat|>",
    "<|zzz|>",
]
LANG_PIECES = {"<|hi|>", "<|ur|>", "<|bgc|>", "<|ta|>", "<|en|>", "<|sat|>", "<|zzz|>"}


class FakeSpl:
    def id_to_piece(self, tid: int) -> str:
        return PIECES[tid]

    def piece_to_id(self, piece: str) -> int:
        return PIECES.index(piece)


class FakeTokenizer:
    spl_size = len(PIECES)
    spl = FakeSpl()

    def encode_prompt(self, lang: str) -> list[int]:
        # the frozen canary2 prompt; only the first three matter to LID
        lid = PIECES.index(f"<|{lang}|>")
        return [7, 4, 16, lid, lid, 5, 9, 11, 13, 15]


class FakeDecoderStack:
    def __call__(self, prefix, enc_states, cross_mask, past_key_values=None, start_pos=0):
        # (B, T, H); content is irrelevant, the fake head ignores it
        return torch.zeros(prefix.size(0), prefix.size(1), 4)


class FakeInner:
    decoder = FakeDecoderStack()


class FakeModel:
    """Emits a fixed logit vector per row so the expected ranking is known exactly."""

    model = FakeInner()

    def __init__(self, logits_per_row: torch.Tensor):
        self._logits = logits_per_row  # (B, V)

    def _cross_mask_from_lengths(self, lengths, size):
        return torch.ones(lengths.size(0), size)

    def lm_head(self, hidden):
        return self._logits


def _logits(rows: list[dict[str, float]]) -> torch.Tensor:
    """Build (B, V) logits from {piece_name: logit} maps; unnamed entries get -20."""
    out = torch.full((len(rows), len(PIECES)), -20.0)
    for r, mapping in enumerate(rows):
        for piece, val in mapping.items():
            out[r, PIECES.index(piece)] = val
    return out


def _run(model, topk=5, allowed_langs=None, batch=1):
    return lid_from_encoder_states(
        model,
        torch.zeros(batch, 7, 4),
        torch.tensor([7] * batch),
        tokenizer=FakeTokenizer(),
        topk=topk,
        allowed_langs=allowed_langs,
    )


# --------------------------------------------------------------------------- #
# candidate set
# --------------------------------------------------------------------------- #
def test_language_map_picks_up_every_language_token():
    got = set(language_token_map(FakeTokenizer()).values())
    assert got == {p.strip("<|>") for p in LANG_PIECES}


@pytest.mark.parametrize("control", ["itn", "pnc"])
def test_control_tokens_are_not_languages(control):
    """<|itn|>/<|pnc|> match the language regex; they must not be candidates.

    Regression: before the fix these entered the candidate set and could be
    returned as a predicted "language".
    """
    assert control not in language_token_map(FakeTokenizer()).values()


def test_control_token_cannot_win_even_when_it_dominates():
    model = FakeModel(_logits([{"<|itn|>": 20.0, "<|hi|>": 1.0}]))
    (top,) = _run(model)
    assert top[0][0] == "hi"
    assert all(lang not in ("itn", "pnc") for lang, _ in top)


def test_allowed_langs_narrows_the_candidate_set():
    got = set(language_token_map(FakeTokenizer(), ["hi", "ta"]).values())
    assert got == {"hi", "ta"}


def test_allowed_langs_ignores_names_absent_from_the_vocab():
    got = set(language_token_map(FakeTokenizer(), ["hi", "notalang"]).values())
    assert got == {"hi"}


def test_allowed_langs_matching_nothing_raises_rather_than_silently_emptying():
    with pytest.raises(ValueError, match="matched no language token"):
        language_token_map(FakeTokenizer(), ["nope"])


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def test_prefix_is_three_tokens_and_stops_before_the_language_slot():
    prefix = lid_prefix_ids(FakeTokenizer())
    assert len(prefix) == LID_PREFIX_LEN == 3
    full = FakeTokenizer().encode_prompt("hi")
    assert prefix == full[:3]
    # position 3 is the source_lang slot, i.e. exactly what LID is predicting
    assert full[3] == PIECES.index("<|hi|>")


def test_topk_is_ranked_and_probabilities_are_the_softmax_over_full_vocab():
    model = FakeModel(_logits([{"<|hi|>": 3.0, "<|ur|>": 2.0, "<|ta|>": 1.0}]))
    (top,) = _run(model, topk=3)
    assert [lang for lang, _ in top] == ["hi", "ur", "ta"]
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    # softmax is over the whole vocabulary, so the language probs sum to < 1
    assert 0.0 < sum(probs) < 1.0
    expected = torch.softmax(_logits([{"<|hi|>": 3.0, "<|ur|>": 2.0, "<|ta|>": 1.0}])[0], -1)
    assert probs[0] == pytest.approx(float(expected[PIECES.index("<|hi|>")]), rel=1e-5)


def test_topk_is_clamped_to_the_number_of_candidates():
    model = FakeModel(_logits([{"<|hi|>": 3.0, "<|ta|>": 1.0}]))
    (top,) = _run(model, topk=10, allowed_langs=["hi", "ta"])
    assert len(top) == 2


def test_narrowing_can_change_the_answer():
    """The point of allowed_langs, and the reason it is a deliberate opt-in."""
    row = [{"<|bgc|>": 5.0, "<|hi|>": 4.0}]
    (wide,) = _run(FakeModel(_logits(row)))
    (narrow,) = _run(FakeModel(_logits(row)), allowed_langs=["hi", "ta"])
    assert wide[0][0] == "bgc"
    assert narrow[0][0] == "hi"  # bgc is excluded, so its mass is unreachable


def test_batch_rows_are_scored_independently():
    model = FakeModel(_logits([{"<|hi|>": 5.0}, {"<|ta|>": 5.0}]))
    out = _run(model, batch=2)
    assert [row[0][0] for row in out] == ["hi", "ta"]


def test_probabilities_are_fp32_under_a_bf16_model():
    """Callers threshold this number, so the softmax must not stay in bf16.

    bf16 keeps ~8 mantissa bits (~2-3 decimal digits). The check is that the
    returned probability matches the fp32 softmax of the same logits and is
    *not* a value bf16 could represent — i.e. precision was added by the cast in
    lid.py, not merely carried through.
    """
    raw = _logits([{"<|hi|>": 3.0, "<|ur|>": 2.0, "<|ta|>": 1.0}]).bfloat16()
    (top,) = _run(FakeModel(raw), topk=2)

    expected = float(torch.softmax(raw[0].float(), -1)[PIECES.index("<|hi|>")])
    assert top[0][1] == pytest.approx(expected, rel=1e-6)
    # precondition: this probability genuinely needs more than bf16 to express
    assert float(torch.tensor(expected).bfloat16()) != expected
    assert top[0][1] != float(torch.tensor(expected).bfloat16())


# --------------------------------------------------------------------------- #
# the declared language set
# --------------------------------------------------------------------------- #
def test_trained_langs_is_27_unique_sorted_codes():
    assert len(TRAINED_LANGS) == 27
    assert len(set(TRAINED_LANGS)) == 27
    assert list(TRAINED_LANGS) == sorted(TRAINED_LANGS)


def test_trained_langs_contains_no_control_tokens():
    assert not {"itn", "pnc"} & set(TRAINED_LANGS)


# --------------------------------------------------------------------------- #
# chunk probing for long-form audio
# --------------------------------------------------------------------------- #
from bodhan_genai.asr.engine.lid import LONG_LID_PROBES, probe_indices  # noqa: E402


@pytest.mark.parametrize("n", range(1, 4))
def test_probe_indices_returns_every_chunk_when_there_are_few(n):
    assert probe_indices(n, 3) == list(range(n))


def test_probe_indices_never_starts_at_zero_when_it_can_avoid_it():
    """The opening chunk is the one most likely to be silence or a jingle."""
    assert 0 not in probe_indices(20, 3)


@pytest.mark.parametrize("n", [4, 7, 20, 101, 1000])
def test_probe_indices_are_valid_unique_and_spread(n):
    got = probe_indices(n, 3)
    assert got == sorted(got)
    assert len(got) == len(set(got))
    assert all(0 <= i < n for i in got)
    assert len(got) == 3
    # spread across the recording, not clustered at one end
    assert got[0] < n / 2 < got[-1]


def test_probe_indices_stays_in_range_at_the_boundary():
    assert max(probe_indices(4, 3)) <= 3


def test_long_lid_probes_is_small_and_odd():
    """Odd so a simple vote cannot tie; small because each probe is an encoder pass."""
    assert LONG_LID_PROBES % 2 == 1
    assert 1 <= LONG_LID_PROBES <= 5


def test_recommended_langs_is_trained_minus_the_measured_sinks():
    """bgc/bhb are never a ground-truth label on either benchmark yet bgc alone
    absorbs 16,429 predictions (5,681 of them Punjabi). See lid.py for the data."""
    from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS

    assert set(RECOMMENDED_LANGS) == set(TRAINED_LANGS) - {"bgc", "bhb"}
    assert len(RECOMMENDED_LANGS) == 25
    assert list(RECOMMENDED_LANGS) == sorted(RECOMMENDED_LANGS)


def test_recommended_is_not_the_default():
    """Narrowing is a hard filter, so it must be opt-in: the default candidate
    set is every language token, not the recommended subset."""
    m = language_token_map(FakeTokenizer())
    assert "bgc" in m.values()
