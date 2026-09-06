"""Language identification: prompt-prefix derivation and language-token filtering.

The scoring itself needs a real checkpoint, so these cover the parts that are
pure logic and that would fail silently if wrong: which prompt prefix is fed,
and which vocabulary entries count as languages.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from bodhan_genai.asr.engine.lid import (
    LID_PREFIX_LEN,
    language_token_map,
    lid_from_encoder_states,
    lid_prefix_ids,
)


class _StubSPL:
    PIECES: ClassVar[dict[int, str]] = {
        20: "<|hi|>",
        21: "<|bn|>",
        22: "<|ur|>",
        23: "<|mai|>",  # 3-letter code must be recognised
        24: "<|nospeech|>",  # NOT a language
        25: "<|emo:undefined|>",  # NOT a language
        26: "<|unklang|>",  # NOT a language (too long for the 2-3 char class)
        27: "<|startoftranscript|>",
    }

    @staticmethod
    def id_to_piece(i):
        return _StubSPL.PIECES.get(i, f"<tok{i}>")


class _StubTokenizer:
    spl_size = 40
    spl = _StubSPL()

    def encode_prompt(self, lang):
        return [7, 4, 18, 20, 20, 5, 9, 11, 13, 15]


def test_prefix_is_the_language_independent_head():
    """Deliberately derived from encode_prompt rather than hardcoded, so it
    cannot drift from the prompt the transcription path uses."""
    tok = _StubTokenizer()
    assert lid_prefix_ids(tok) == [7, 4, 18]
    assert len(lid_prefix_ids(tok)) == LID_PREFIX_LEN
    assert lid_prefix_ids(tok) == tok.encode_prompt("hi")[:3]


def test_language_map_accepts_two_and_three_letter_codes():
    m = language_token_map(_StubTokenizer())
    assert m == {20: "hi", 21: "bn", 22: "ur", 23: "mai"}


def test_language_map_excludes_non_language_specials():
    """The unrestricted argmax can land on <|nospeech|>/<|emo:*|>/<|unklang|>,
    which are not languages — they must never appear as an LID answer."""
    m = language_token_map(_StubTokenizer())
    assert set(m.values()).isdisjoint({"nospeech", "emo:undefined", "unklang", "startoftranscript"})


def test_lid_scores_only_language_tokens_and_is_sorted():
    """Drive the real scoring path with a stub model: the top-k must come from
    the language subset (softmaxed over the FULL vocab, then restricted) and
    come back in descending probability order."""
    vocab = 40
    tok = _StubTokenizer()

    class _StubDecoder:
        def __call__(self, prefix, enc, mask, past_key_values=None, start_pos=0):
            return torch.zeros(prefix.size(0), prefix.size(1), 8)

    class _StubModel:
        model = type("M", (), {"decoder": _StubDecoder()})()

        @staticmethod
        def _cross_mask_from_lengths(lengths, t_enc):
            return torch.zeros(lengths.size(0), 1, 1, t_enc)

        @staticmethod
        def lm_head(hidden):
            # favour <|bn|>(21) over <|hi|>(20); non-language 24 gets the
            # highest raw logit and must still be excluded from the answer.
            logits = torch.full((hidden.size(0), vocab), -10.0)
            logits[:, 24] = 50.0
            logits[:, 21] = 5.0
            logits[:, 20] = 3.0
            return logits

    out = lid_from_encoder_states(
        _StubModel(),
        torch.zeros(2, 7, 8),
        torch.tensor([7, 7]),
        tokenizer=tok,
        topk=3,
    )
    assert len(out) == 2
    for row in out:
        langs = [name for name, _ in row]
        assert langs[0] == "bn" and langs[1] == "hi"
        assert "nospeech" not in langs
        probs = [p for _, p in row]
        assert probs == sorted(probs, reverse=True)


def test_topk_is_clamped_to_available_languages():
    tok = _StubTokenizer()

    class _StubModel:
        model = type(
            "M",
            (),
            {"decoder": lambda *a, **k: torch.zeros(1, 3, 8)},
        )()

        @staticmethod
        def _cross_mask_from_lengths(lengths, t_enc):
            return torch.zeros(lengths.size(0), 1, 1, t_enc)

        @staticmethod
        def lm_head(hidden):
            return torch.zeros(hidden.size(0), 40)

    out = lid_from_encoder_states(
        _StubModel(), torch.zeros(1, 7, 8), torch.tensor([7]), tokenizer=tok, topk=99
    )
    assert len(out[0]) == 4  # only 4 language tokens exist in the stub vocab
