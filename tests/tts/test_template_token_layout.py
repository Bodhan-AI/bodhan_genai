"""Regression tests pinning the Llama template token layout to the frozen vocab contract.

Uses the shared FrozenLayoutTokenizer mock (tests/conftest.py) that implements the
frozen special-token ids (<|start_of_speech|>=128257 … style wrappers
156938-156941, bos=128000) so any accidental reordering / re-bracketing of the
SFT builders fails loudly. Runtime code always resolves ids from the
tokenizer — these literals exist only to pin the *structure*, not to be consumed
by src/ code.
"""

from __future__ import annotations

import pytest

from bodhan_genai.tts.templates.chat import (
    _build_llama_sft_tts,
    _build_llama_sft_tts_conversation,
    _build_metadata_prefix_ids,
    build_sequence,
    get_template_ids,
)

# Frozen token contract (see docs) --------------------------------------------
BOS = 128000
EOT_ID = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
SPEAKER_START = 156938
SPEAKER_END = 156939
STYLE_START = 156940
STYLE_END = 156941
NEWLINE = 198  # llama3 tokenizer's "\n" id


@pytest.fixture
def tok(frozen_tokenizer):
    return frozen_tokenizer


@pytest.fixture
def tmpl(tok):
    return get_template_ids(tok)


AUDIO = [128266, 128266 + 4096, 128266 + 2 * 4096]  # fake 3-token audio target


def test_get_template_ids_matches_frozen_contract(tmpl):
    assert tmpl["start_of_human"] == [START_OF_HUMAN]
    assert tmpl["end_of_human"] == [END_OF_HUMAN]
    assert tmpl["start_of_ai"] == [START_OF_AI]
    assert tmpl["end_of_ai"] == [END_OF_AI]
    assert tmpl["start_of_speech"] == [START_OF_SPEECH]
    assert tmpl["end_of_speech"] == [END_OF_SPEECH]
    assert tmpl["speaker_start"] == [SPEAKER_START]
    assert tmpl["speaker_end"] == [SPEAKER_END]
    assert tmpl["style_start"] == [STYLE_START]
    assert tmpl["style_end"] == [STYLE_END]
    assert tmpl["newline"] == [NEWLINE]
    assert tmpl["end_of_text"] == [EOT_ID]


# ---------------------------------------------------------------------------
# Basic TTS builder
# ---------------------------------------------------------------------------


def test_basic_tts_exact_layout_no_metadata(tok, tmpl):
    text = "hello world"
    ids, prompt_end = _build_llama_sft_tts(
        text=text,
        audio_token_ids=AUDIO,
        tmpl=tmpl,
        tokenizer=tok,
    )
    expected = [
        START_OF_HUMAN,
        BOS,
        *tok.encode(text, add_special_tokens=False),
        EOT_ID,
        END_OF_HUMAN,
        START_OF_AI,
        START_OF_SPEECH,
        *AUDIO,
        END_OF_SPEECH,
        END_OF_AI,
    ]
    assert ids == expected
    # prompt_end lands exactly after <|start_of_ai|>: the model's first
    # emitted token must be <|start_of_speech|>.
    assert ids[prompt_end - 1] == START_OF_AI
    assert ids[prompt_end] == START_OF_SPEECH


def test_basic_tts_exact_layout_with_metadata(tok, tmpl):
    text = "hello"
    ids, prompt_end = _build_llama_sft_tts(
        text=text,
        audio_token_ids=AUDIO,
        tmpl=tmpl,
        tokenizer=tok,
        speaker_id="alice",
        style="happy",
        accent="tamil",
    )
    meta = [
        SPEAKER_START,
        *tok.encode("alice", add_special_tokens=False),
        SPEAKER_END,
        NEWLINE,
        STYLE_START,
        *tok.encode("happy", add_special_tokens=False),
        STYLE_END,
        NEWLINE,
        STYLE_START,
        *tok.encode("tamil", add_special_tokens=False),
        STYLE_END,
        NEWLINE,
    ]
    expected = [
        START_OF_HUMAN,
        BOS,
        *meta,
        *tok.encode(text, add_special_tokens=False),
        EOT_ID,
        END_OF_HUMAN,
        START_OF_AI,
        START_OF_SPEECH,
        *AUDIO,
        END_OF_SPEECH,
        END_OF_AI,
    ]
    assert ids == expected
    assert ids[prompt_end - 1] == START_OF_AI
    assert ids[prompt_end] == START_OF_SPEECH


def test_metadata_prefix_order_speaker_style_accent(tok, tmpl):
    """Order is always speaker → style → accent, newline-joined, trailing newline."""
    meta = _build_metadata_prefix_ids(
        tmpl,
        tok,
        speaker_id="alice",
        style="happy",
        accent="tamil",
    )
    assert meta[0] == SPEAKER_START
    assert meta[-1] == NEWLINE
    # Blocks are newline-joined in speaker → style → accent order.
    assert meta.index(SPEAKER_END) < meta.index(STYLE_START)
    first_style_close = meta.index(STYLE_END)
    assert meta[meta.index(SPEAKER_END) + 1] == NEWLINE
    assert meta[first_style_close + 1] == NEWLINE
    second_style_open = meta.index(STYLE_START, first_style_close + 1)
    assert meta[second_style_open + 1 : meta.index(STYLE_END, second_style_open)] == tok.encode(
        "tamil", add_special_tokens=False
    )
    # No metadata → empty prefix (no stray newline).
    assert _build_metadata_prefix_ids(tmpl, tok) == []


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------


def test_conversation_exact_layout_and_no_metadata_prefix(tok, tmpl):
    conv_text = "<|speaker>Charon<speaker|> hello there"
    ids, prompt_end = _build_llama_sft_tts_conversation(
        conversation_text=conv_text,
        audio_token_ids=AUDIO,
        tmpl=tmpl,
        tokenizer=tok,
    )
    expected = [
        START_OF_HUMAN,
        BOS,
        *tok.encode(conv_text, add_special_tokens=False),
        EOT_ID,
        END_OF_HUMAN,
        START_OF_AI,
        START_OF_SPEECH,
        *AUDIO,
        END_OF_SPEECH,
        END_OF_AI,
    ]
    assert ids == expected
    # No structural metadata-wrapper ids: speaker tags live in the *text*,
    # the conversation builder never emits a metadata prefix.
    assert SPEAKER_START not in ids and STYLE_START not in ids
    assert ids[prompt_end - 1] == START_OF_AI
    assert ids[prompt_end] == START_OF_SPEECH


# ---------------------------------------------------------------------------
# build_sequence dispatch + labels
# ---------------------------------------------------------------------------


def test_build_sequence_labels_equal_input_ids_all_dispatches(tok, tmpl):
    entries = [
        {"text": "plain tts", "token_ids": AUDIO},
        {"text": "turn one", "token_ids": AUDIO, "is_conversation": True},
    ]
    for entry in entries:
        out = build_sequence(entry, tok, tmpl=tmpl, return_prompt_end=True)
        assert out is not None
        assert out["labels"] == out["input_ids"]  # full-sequence loss
        assert out["input_ids"][out["prompt_end"] - 1] == START_OF_AI
        assert out["input_ids"][out["prompt_end"]] == START_OF_SPEECH


def test_build_sequence_dispatch_selection(tok, tmpl):
    # is_conversation → conversation template (single speech bracket, no metadata).
    out = build_sequence(
        {"text": "x", "token_ids": AUDIO, "is_conversation": True, "speaker": "alice"},
        tok,
        tmpl=tmpl,
    )
    assert out["input_ids"].count(START_OF_SPEECH) == 1
    assert SPEAKER_START not in out["input_ids"]
    # Basic TTS honours speaker metadata.
    out = build_sequence(
        {"text": "x", "token_ids": AUDIO, "speaker": "alice"},
        tok,
        tmpl=tmpl,
    )
    assert SPEAKER_START in out["input_ids"]


def test_build_sequence_invalid_entries_return_none(tok, tmpl):
    assert build_sequence({"text": "", "token_ids": AUDIO}, tok, tmpl=tmpl) is None
    assert build_sequence({"text": "hi", "token_ids": []}, tok, tmpl=tmpl) is None
    assert build_sequence({}, tok, tmpl=tmpl) is None
