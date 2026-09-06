"""Tests for bodhan_genai.tts.inference.prompts — audio-token extraction, tokenizer
id resolution against the frozen vocab contract, and prompt building.

Uses the shared FrozenLayoutTokenizer mock from tests/conftest.py: runtime code
always resolves ids from the tokenizer; the literals here only pin the contract.
"""

from __future__ import annotations

import json

import pytest

from bodhan_genai.tts.inference.prompts import (
    build_prompt_ids,
    extract_audio_tokens,
    load_prompts_jsonl,
    resolve_snac_ids,
)
from bodhan_genai.tts.templates.chat import get_template_ids

# Frozen token contract (see docs) --------------------------------------------
BOS = 128000
EOT_ID = 128009
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
SNAC_BASE = 128266
NEWLINE = 198


@pytest.fixture
def tok(frozen_tokenizer):
    return frozen_tokenizer


# ---------------------------------------------------------------------------
# extract_audio_tokens
# ---------------------------------------------------------------------------


def test_extract_no_start_marker_returns_empty():
    ids = [1, 2, 3, END_OF_SPEECH, 4]
    assert extract_audio_tokens(ids, START_OF_SPEECH, END_OF_SPEECH) == []


def test_extract_missing_end_returns_tail_slice():
    ids = [1, START_OF_SPEECH, 10, 11, 12]
    assert extract_audio_tokens(ids, START_OF_SPEECH, END_OF_SPEECH) == [10, 11, 12]


def test_extract_between_markers():
    ids = [1, START_OF_SPEECH, 10, 11, END_OF_SPEECH, 99]
    assert extract_audio_tokens(ids, START_OF_SPEECH, END_OF_SPEECH) == [10, 11]


def test_extract_last_start_wins():
    # Two speech segments — only the last one is the generated target.
    ids = [
        START_OF_SPEECH,
        10,
        11,
        END_OF_SPEECH,
        5,
        START_OF_SPEECH,
        20,
        21,
        22,
        END_OF_SPEECH,
    ]
    assert extract_audio_tokens(ids, START_OF_SPEECH, END_OF_SPEECH) == [20, 21, 22]


def test_extract_empty_input():
    assert extract_audio_tokens([], START_OF_SPEECH, END_OF_SPEECH) == []


# ---------------------------------------------------------------------------
# resolve_snac_ids
# ---------------------------------------------------------------------------


def test_resolve_snac_ids_matches_frozen_contract(tok):
    ids = resolve_snac_ids(tok)
    assert ids == {
        "start_of_audio_id": START_OF_SPEECH,
        "end_of_audio_id": END_OF_SPEECH,
        "audio_token_base_id": SNAC_BASE,
        "eos_token_id": EOT_ID,
    }


def test_resolve_snac_ids_base_comes_from_tokenizer_lookup(tok):
    resolve_snac_ids(tok)
    # The base id must be resolved via convert_tokens_to_ids("<|snac_0|>"),
    # never a hardcoded literal.
    assert "<|snac_0|>" in tok.convert_calls


def test_resolve_snac_ids_eos_falls_back_to_end_of_speech(tok):
    tok.eos_token_id = None
    ids = resolve_snac_ids(tok)
    assert ids["eos_token_id"] == END_OF_SPEECH


# ---------------------------------------------------------------------------
# build_prompt_ids
# ---------------------------------------------------------------------------


def test_build_prompt_ids_empty_text_returns_empty(tok):
    assert build_prompt_ids("", "spk", tok) == []
    assert build_prompt_ids("   ", "spk", tok) == []


def test_build_prompt_ids_ends_after_start_of_ai(tok):
    ids = build_prompt_ids("hello world", "", tok)
    assert ids[0] == START_OF_HUMAN
    assert ids[1] == BOS
    # Prompt ends exactly after <|start_of_ai|>: the model's first emitted
    # token must be <|start_of_speech|>.
    assert ids[-1] == START_OF_AI
    assert ids[-2] == END_OF_HUMAN
    assert START_OF_SPEECH not in ids


def test_build_prompt_ids_matches_template_layout(tok):
    text = "hello world"
    tmpl = get_template_ids(tok)
    text_ids = tok.encode(text, add_special_tokens=False)
    expected = [START_OF_HUMAN, BOS, *text_ids, EOT_ID, END_OF_HUMAN, START_OF_AI]
    assert build_prompt_ids(text, "", tok, tmpl=tmpl) == expected


def test_build_prompt_ids_speaker_adds_metadata_prefix(tok):
    plain = build_prompt_ids("hello", "", tok)
    with_spk = build_prompt_ids("hello", "spk_1", tok)
    assert len(with_spk) > len(plain)
    assert 156938 in with_spk and 156939 in with_spk  # <|speaker> ... <speaker|>


# ---------------------------------------------------------------------------
# load_prompts_jsonl
# ---------------------------------------------------------------------------


def test_load_prompts_jsonl_happy_path(tok, tmp_path):
    rows = [
        {
            "text": "hello world",
            "audio_filepath": "/data/a.wav",
            "language": "english",
            "speaker_id": "spk_1",
        },
        {"text": "", "audio_filepath": "/data/skip.wav"},  # skipped: empty text
        {"text": "namaste duniya", "speaker": "spk_2", "language": "hindi"},
    ]
    path = tmp_path / "prompts.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    items = load_prompts_jsonl(str(path), tok)
    assert len(items) == 2
    assert [it["_row_idx"] for it in items] == [0, 1]

    first = items[0]
    assert first["text"] == "hello world"
    assert first["audio_filepath"] == "/data/a.wav"
    assert first["language"] == "english"
    assert first["speaker_id"] == "spk_1"
    # Prompt half only: same slice contract as build_prompt_ids.
    assert first["input_ids"] == build_prompt_ids("hello world", "spk_1", tok)
    assert first["input_ids"][-1] == START_OF_AI

    # "speaker" key is accepted as a fallback for "speaker_id".
    assert items[1]["speaker_id"] == "spk_2"
    assert items[1]["input_ids"] == build_prompt_ids("namaste duniya", "spk_2", tok)


def test_load_prompts_jsonl_max_rows(tok, tmp_path):
    path = tmp_path / "prompts.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"row {i}"}) + "\n")
    items = load_prompts_jsonl(str(path), tok, max_rows=2)
    assert len(items) == 2


class TestBuildConversationPromptIds:
    """Conversation prompts: tagged turns inline, no metadata prefix, same
    structural bracket as the basic prompt."""

    def test_layout_matches_conversation_template(self, frozen_tokenizer):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids
        from bodhan_genai.tts.templates.conversation import format_messages

        tok = frozen_tokenizer
        msgs = [
            {"speaker": "S1", "text": "hello there"},
            {"speaker": "S2", "text": "hi back"},
        ]
        ids = build_conversation_prompt_ids(msgs, tok)
        conv_text = format_messages(msgs)
        expected = [
            tok.convert_tokens_to_ids("<|start_of_human|>"),
            tok.bos_token_id,
            *tok.encode(conv_text, add_special_tokens=False),
            tok.convert_tokens_to_ids("<|eot_id|>"),
            tok.convert_tokens_to_ids("<|end_of_human|>"),
            tok.convert_tokens_to_ids("<|start_of_ai|>"),
        ]
        assert ids == expected

    def test_differs_from_basic_prompt(self, frozen_tokenizer):
        from bodhan_genai.tts.inference.prompts import (
            build_conversation_prompt_ids,
            build_prompt_ids,
        )

        msgs = [{"speaker": "S1", "text": "hello there"}]
        conv = build_conversation_prompt_ids(msgs, frozen_tokenizer)
        basic = build_prompt_ids("hello there", "S1", frozen_tokenizer)
        assert conv != basic  # speaker tag is inline text, not a metadata prefix

    def test_empty_messages_returns_empty(self, frozen_tokenizer):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        assert build_conversation_prompt_ids([], frozen_tokenizer) == []

    def test_malformed_turn_raises(self, frozen_tokenizer):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        with pytest.raises(ValueError, match=r"messages\[0\] has no speaker"):
            build_conversation_prompt_ids([{"text": "no speaker"}], frozen_tokenizer)
