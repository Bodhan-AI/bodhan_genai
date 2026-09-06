"""Tests for bodhan_genai.tts.templates.conversation."""

from bodhan_genai.tts.templates.conversation import (
    expand_conversation_turns,
    format_conversation_text,
)

_RECORD = {
    "sample_id": "test_001",
    "target_language": "ta",
    "target_language_name": "Tamil",
    "tts_generation": {
        "speaker_voices": {"Teacher": "Charon", "Student": "Kore"},
    },
    "turns": [
        {"role": "teacher", "text": "Hello students."},
        {"role": "student", "text": "Hello teacher."},
        {"role": "teacher", "text": "Let us begin."},
    ],
}


def test_format_conversation_text_basic():
    result = format_conversation_text(_RECORD)
    expected = (
        "<|speaker>Charon<speaker|>\nHello students."
        "\n\n"
        "<|speaker>Kore<speaker|>\nHello teacher."
        "\n\n"
        "<|speaker>Charon<speaker|>\nLet us begin."
    )
    assert result == expected


def test_format_conversation_text_fallback_to_role():
    record = {
        "turns": [
            {"role": "narrator", "text": "Once upon a time."},
        ],
    }
    result = format_conversation_text(record)
    assert result == "<|speaker>narrator<speaker|>\nOnce upon a time."


def test_format_conversation_text_missing_tts_generation():
    record = {
        "turns": [
            {"role": "teacher", "text": "A"},
            {"role": "student", "text": "B"},
        ],
    }
    result = format_conversation_text(record)
    assert result == "<|speaker>teacher<speaker|>\nA\n\n<|speaker>student<speaker|>\nB"


def test_format_conversation_text_empty():
    assert format_conversation_text({}) == ""
    assert format_conversation_text({"turns": []}) == ""


def test_expand_conversation_turns_basic():
    rows = expand_conversation_turns(_RECORD)
    assert len(rows) == 3

    assert rows[0] == {
        "text": "Hello students.",
        "speaker": "Charon",
        "role": "teacher",
        "turn_index": 0,
        "sample_id": "test_001",
        "language": "ta",
        "target_language_name": "Tamil",
    }
    assert rows[1]["speaker"] == "Kore"
    assert rows[1]["turn_index"] == 1
    assert rows[2]["speaker"] == "Charon"
    assert rows[2]["turn_index"] == 2


def test_expand_conversation_turns_fallback_to_role():
    record = {"turns": [{"role": "teacher", "text": "Hi"}]}
    rows = expand_conversation_turns(record)
    assert rows[0]["speaker"] == "teacher"


def test_expand_conversation_turns_empty():
    assert expand_conversation_turns({}) == []
    assert expand_conversation_turns({"turns": []}) == []


class TestFormatMessages:
    """format_messages: the chat-style [{"speaker", "text"}] serializer."""

    def test_two_turns(self):
        from bodhan_genai.tts.templates.conversation import format_messages

        out = format_messages(
            [
                {"speaker": "S1", "text": "Hello there."},
                {"speaker": "S2", "text": "Hi back!"},
            ]
        )
        assert out == "<|speaker>S1<speaker|>\nHello there.\n\n<|speaker>S2<speaker|>\nHi back!"

    def test_empty_list_returns_empty_string(self):
        from bodhan_genai.tts.templates.conversation import format_messages

        assert format_messages([]) == ""

    def test_blank_speaker_raises(self):
        import pytest

        from bodhan_genai.tts.templates.conversation import format_messages

        with pytest.raises(ValueError, match=r"messages\[1\] has no speaker"):
            format_messages([{"speaker": "S1", "text": "hi"}, {"speaker": " ", "text": "yo"}])

    def test_blank_text_raises(self):
        import pytest

        from bodhan_genai.tts.templates.conversation import format_messages

        with pytest.raises(ValueError, match=r"messages\[0\] has no text"):
            format_messages([{"speaker": "S1"}])

    def test_values_stripped(self):
        from bodhan_genai.tts.templates.conversation import format_messages

        out = format_messages([{"speaker": "  S1 ", "text": "  padded  "}])
        assert out == "<|speaker>S1<speaker|>\npadded"
