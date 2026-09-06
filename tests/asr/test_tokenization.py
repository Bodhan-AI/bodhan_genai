"""Tokenizer contract: 10-token prompt (mode slots 6/7), id layout, parity decode.

The real tokenizer needs SentencePiece model files that only ship inside a
converted checkpoint, so these drive the logic through a stub that mimics the
two SentencePieceProcessor objects. That covers everything except SPM itself:
the id layout, the frozen prompt, the piece-join decode, and the prompt-strip.
"""

from __future__ import annotations

import pytest

from bodhan_genai.asr.model.tokenization_indic_transcribe import IndicTranscribeTokenizer

# Frozen canary2 prompt slots (ids fixed by the checkpoint's spl vocab).
SLOTS = {
    "<|startofcontext|>": 7,
    "<|startoftranscript|>": 4,
    "<|emo:undefined|>": 18,
    "<|pnc|>": 5,
    "<|itn|>": 8,
    "<|noitn|>": 9,
    "<|romanized|>": 10,
    "<|noromanized|>": 11,
    "<|notimestamp|>": 13,
    "<|nodiarize|>": 15,
    "<|nospeech|>": 1,
    "<pad>": 2,
    "<|endoftext|>": 3,
    "<|hi|>": 20,
    "<|bn|>": 21,
    "<|ur|>": 22,
}
SPL_SIZE = 40
MULTI_SIZE = 10


class _StubSPM:
    """Enough SentencePieceProcessor surface for the tokenizer's needs."""

    def __init__(self, size, pieces, offset_pieces=None):
        self._size = size
        self._pieces = pieces
        self._offset_pieces = offset_pieces or {}

    def get_piece_size(self):
        return self._size

    def piece_to_id(self, piece):
        return self._pieces.get(piece, 0)  # 0 == unk, as SPM does

    def id_to_piece(self, i):
        if i in self._offset_pieces:
            return self._offset_pieces[i]
        for piece, pid in self._pieces.items():
            if pid == i:
                return piece
        return f"<tok{i}>"


@pytest.fixture
def tok(monkeypatch):
    spl = _StubSPM(SPL_SIZE, SLOTS, {30: "<|0|>", 31: "<|1|>"})
    multi = _StubSPM(MULTI_SIZE, {}, {0: "▁hello", 1: "▁world", 2: "s", 3: "▁"})
    monkeypatch.setattr(
        IndicTranscribeTokenizer,
        "__init__",
        lambda self, *a, **k: _init_stub(self, spl, multi),
    )
    return IndicTranscribeTokenizer("ignored", "ignored")


def _init_stub(self, spl, multi):
    """Mirror the real __init__ against stub SPMs (same code path, no files)."""
    import re

    self.spl, self.multi = spl, multi
    self.spl_size = spl.get_piece_size()
    self.multi_size = multi.get_piece_size()
    self.vocab_size = self.spl_size + self.multi_size
    self.unk_id = 0
    self.nospeech_id = spl.piece_to_id("<|nospeech|>")
    self.pad_id = spl.piece_to_id("<pad>")
    self.eos_id = spl.piece_to_id("<|endoftext|>")
    self.bos_id = spl.piece_to_id("<|startoftranscript|>")
    ts = re.compile(r"^<\|\d+\|>$")
    self._timestamp_ids = frozenset(i for i in range(self.spl_size) if ts.match(spl.id_to_piece(i)))
    slot_ids = [
        spl.piece_to_id(p)
        for p in (
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|pnc|>",
            "<|noitn|>",
            "<|noromanized|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        )
    ]
    boc, bos, emo, pnc, noitn, norom, nots, nodia = slot_ids
    self.noitn_id = noitn
    self.noromanized_id = norom
    self.itn_id = spl.piece_to_id("<|itn|>")
    self.romanized_id = spl.piece_to_id("<|romanized|>")
    self._prompt_table = {}
    for lang in ("hi", "bn", "ur"):
        lid = spl.piece_to_id(f"<|{lang}|>")
        self._prompt_table[lang] = [boc, bos, emo, lid, lid, pnc, noitn, norom, nots, nodia]


def test_special_ids_match_the_frozen_layout(tok):
    assert (tok.unk_id, tok.nospeech_id, tok.pad_id, tok.eos_id, tok.bos_id) == (0, 1, 2, 3, 4)
    assert tok.vocab_size == SPL_SIZE + MULTI_SIZE


def test_prompt_is_the_frozen_ten_token_sequence(tok):
    """[7, 4, 18, L, L, 5, 9, 11, 13, 15] — the language appears TWICE
    (source_lang and target_lang)."""
    assert tok.encode_prompt("hi") == [7, 4, 18, 20, 20, 5, 9, 11, 13, 15]
    assert tok.encode_prompt("bn") == [7, 4, 18, 21, 21, 5, 9, 11, 13, 15]
    assert tok.prompt_len == 10


def test_prompt_is_copied_not_aliased(tok):
    """A caller mutating a returned prompt must not corrupt the cache."""
    p = tok.encode_prompt("hi")
    p[3] = 999
    assert tok.encode_prompt("hi")[3] == 20


def test_unsupported_language_raises(tok):
    with pytest.raises(ValueError, match="unsupported language"):
        tok.encode_prompt("zz")


def test_ids_to_pieces_spans_both_vocab_tiers(tok):
    """ids < spl_size come from spl; ids >= spl_size index the multilingual
    model AFTER subtracting the offset — an off-by-one here silently decodes
    the wrong token."""
    assert tok.ids_to_pieces([7]) == ["<|startofcontext|>"]
    assert tok.ids_to_pieces([SPL_SIZE]) == ["▁hello"]
    assert tok.ids_to_pieces([SPL_SIZE + 1]) == ["▁world"]


def test_out_of_range_id_raises(tok):
    with pytest.raises(ValueError, match="out of range"):
        tok.ids_to_pieces([tok.vocab_size])


def test_decode_is_piece_join_not_spm_decode(tok):
    """Production joins pieces and maps '▁'->' ', then strips."""
    assert tok.decode([SPL_SIZE, SPL_SIZE + 1]) == "hello world"
    assert tok.decode([SPL_SIZE, SPL_SIZE + 2]) == "hellos"


def test_decode_strip_can_be_disabled(tok):
    assert tok.decode([SPL_SIZE + 3], strip=False) == " "
    assert tok.decode([SPL_SIZE + 3]) == ""


def test_decode_warns_on_multiple_timestamp_tokens(tok, caplog):
    """Documented deviation: production rewrites via timestamp_utils, we warn."""
    with caplog.at_level("WARNING"):
        tok.decode([30, 31])
    assert "timestamp tokens" in caplog.text


def test_strip_prompt_and_trim_removes_prompt_and_trailing_pad_eos(tok):
    prompt = tok.encode_prompt("hi")
    ids = [*prompt, SPL_SIZE, SPL_SIZE + 1, tok.eos_id, tok.pad_id, tok.pad_id]
    assert tok.strip_prompt_and_trim(ids, prompt) == [SPL_SIZE, SPL_SIZE + 1]


def test_strip_prompt_keeps_interior_pad_eos(tok):
    """Only the TRAILING run is trimmed; an interior eos is real output."""
    prompt = tok.encode_prompt("hi")
    ids = [*prompt, SPL_SIZE, tok.eos_id, SPL_SIZE + 1, tok.eos_id]
    assert tok.strip_prompt_and_trim(ids, prompt) == [SPL_SIZE, tok.eos_id, SPL_SIZE + 1]


def test_strip_prompt_rejects_mismatched_prefix(tok):
    """Guards against decoding a row generated under a different language."""
    prompt = tok.encode_prompt("hi")
    with pytest.raises(ValueError, match="prompt prefix not found"):
        tok.strip_prompt_and_trim([*tok.encode_prompt("bn"), SPL_SIZE], prompt)


# ---- output-mode slots (itn / romanized) ----------------------------------


def test_prompt_mode_slots_swap_only_slots_6_and_7(tok):
    native = tok.encode_prompt("hi")
    itn = tok.encode_prompt("hi", itn=True)
    rom = tok.encode_prompt("hi", romanized=True)
    both = tok.encode_prompt("hi", itn=True, romanized=True)
    assert native == [7, 4, 18, 20, 20, 5, 9, 11, 13, 15]
    assert itn == [7, 4, 18, 20, 20, 5, 8, 11, 13, 15]
    assert rom == [7, 4, 18, 20, 20, 5, 9, 10, 13, 15]
    assert both == [7, 4, 18, 20, 20, 5, 8, 10, 13, 15]
    # every mode keeps the prompt exactly prompt_len tokens (batching invariant)
    assert {len(native), len(itn), len(rom), len(both)} == {tok.prompt_len}


def test_mode_call_does_not_pollute_the_prompt_cache(tok):
    before = tok.encode_prompt("hi")
    tok.encode_prompt("hi", itn=True, romanized=True)
    assert tok.encode_prompt("hi") == before


def test_strip_rejects_prompt_of_a_different_mode(tok):
    itn_prompt = tok.encode_prompt("hi", itn=True)
    seq = [*itn_prompt, 45, 46, tok.eos_id]
    assert tok.strip_prompt_and_trim(seq, itn_prompt) == [45, 46]
    with pytest.raises(ValueError):
        tok.strip_prompt_and_trim(seq, tok.encode_prompt("hi"))
