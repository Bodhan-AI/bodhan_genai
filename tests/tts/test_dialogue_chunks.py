"""plan_dialogue_chunks: packer vectors + the serialized-size invariant."""

from __future__ import annotations

import pytest

from bodhan_genai.tts.engine.chunked import plan_dialogue_chunks
from bodhan_genai.tts.templates.conversation import format_messages

A_HELLO = {"speaker": "A", "text": "Hello."}  # serialized cost 21 + 1 + 6 == 28
B_HI = {"speaker": "B", "text": "Hi."}  # serialized cost 21 + 1 + 3 == 25


def make_monologue(n: int = 30) -> str:
    """n 64-char sentences joined by single spaces (n=30 -> 1949 chars)."""
    return " ".join(
        f"Sentence number {i:02d} of this long winded monologue keeps going on." for i in range(n)
    )


def make_b_text() -> str:
    """Eight 74-char sentences joined by single spaces -> 599 chars."""
    sentence = "word " * 13 + "lastword."
    assert len(sentence) == 74
    return " ".join([sentence] * 8)


class TestPackerVectors:
    def test_p1_empty_inputs(self):
        assert plan_dialogue_chunks([], max_chars=300) == []
        assert plan_dialogue_chunks([{"speaker": "A", "text": "  "}], max_chars=300) == []

    def test_p2_two_turns_pack_together(self):
        assert plan_dialogue_chunks([A_HELLO, B_HI], max_chars=100) == [[A_HELLO, B_HI]]

    def test_p3_boundary_costs(self):
        # 28 + 2 + 25 == 55: fits exactly at 55, splits at 54.
        assert plan_dialogue_chunks([A_HELLO, B_HI], max_chars=55) == [[A_HELLO, B_HI]]
        assert plan_dialogue_chunks([A_HELLO, B_HI], max_chars=54) == [[A_HELLO], [B_HI]]

    def test_p4_alternating_speakers_greedy(self):
        msgs = [{"speaker": "AB"[i % 2], "text": "hi"} for i in range(12)]  # cost 24 each
        chunks = plan_dialogue_chunks(msgs, max_chars=300)
        assert [len(c) for c in chunks] == [11, 1]
        assert [m for c in chunks for m in c] == msgs  # order + content preserved

    def test_p5_long_monologue_explodes_at_sentence_ends(self):
        mono = make_monologue()
        chunks = plan_dialogue_chunks([{"speaker": "Narrator", "text": mono}], max_chars=300)
        assert len(chunks) == 8
        assert all(len(c) == 1 for c in chunks)
        assert all(c[0]["speaker"] == "Narrator" for c in chunks)
        assert all(len(format_messages(c)) <= 300 for c in chunks)
        # every internal boundary is a sentence end...
        assert all(c[0]["text"].endswith(".") for c in chunks)
        # ...and the segments rejoin to the monologue (modulo whitespace).
        assert " ".join(c[0]["text"] for c in chunks) == mono

    def test_p6_long_middle_turn_with_tiny_neighbours(self):
        b_text = make_b_text()
        msgs = [
            {"speaker": "A", "text": "Short intro."},
            {"speaker": "B", "text": b_text},
            {"speaker": "A", "text": "Ok."},
        ]
        chunks = plan_dialogue_chunks(msgs, max_chars=300)
        assert all(len(format_messages(c)) <= 300 for c in chunks)
        b_segments = [m for c in chunks for m in c if m["speaker"] == "B"]
        assert len(b_segments) == 3  # 599-char turn explodes into 3 segments
        # boundaries only at turn/sentence ends; segments rejoin to the turn
        assert all(m["text"].endswith(".") for m in b_segments)
        assert " ".join(m["text"] for m in b_segments) == b_text
        # the trailing tiny turn packs with the LAST segment
        assert chunks[-1][-1] == {"speaker": "A", "text": "Ok."}
        assert chunks[-1][-2]["speaker"] == "B"


class TestSerializedSizeInvariant:
    @pytest.mark.parametrize("max_chars", [54, 55, 60, 100, 300])
    def test_short_turns(self, max_chars):
        chunks = plan_dialogue_chunks([A_HELLO, B_HI], max_chars=max_chars)
        assert all(len(format_messages(c)) <= max_chars for c in chunks)

    @pytest.mark.parametrize("max_chars", [60, 120, 300])
    def test_monologue(self, max_chars):
        msgs = [{"speaker": "Narrator", "text": make_monologue()}]
        chunks = plan_dialogue_chunks(msgs, max_chars=max_chars)
        assert all(len(format_messages(c)) <= max_chars for c in chunks)

    @pytest.mark.parametrize("max_chars", [100, 300])
    def test_mixed_dialogue(self, max_chars):
        msgs = [
            {"speaker": "A", "text": "Short intro."},
            {"speaker": "B", "text": make_b_text()},
            {"speaker": "A", "text": "Ok."},
        ]
        chunks = plan_dialogue_chunks(msgs, max_chars=max_chars)
        assert all(len(format_messages(c)) <= max_chars for c in chunks)


class TestSanitation:
    def test_extra_keys_survive_packing(self):
        msgs = [{"speaker": "A", "text": "Hello there.", "lang": "hi", "idx": 7}]
        chunks = plan_dialogue_chunks(msgs, max_chars=300)
        assert chunks == [[{"speaker": "A", "text": "Hello there.", "lang": "hi", "idx": 7}]]

    def test_extra_keys_survive_exploding(self):
        msgs = [{"speaker": "B", "text": make_b_text(), "role": "teacher"}]
        chunks = plan_dialogue_chunks(msgs, max_chars=300)
        segments = [m for c in chunks for m in c]
        assert len(segments) == 3
        assert all(m["role"] == "teacher" for m in segments)

    def test_blank_speaker_with_text_raises_with_index(self):
        msgs = [{"speaker": "A", "text": "hi"}, {"speaker": "  ", "text": "there"}]
        with pytest.raises(ValueError, match=r"messages\[1\] has no speaker"):
            plan_dialogue_chunks(msgs, max_chars=300)

    def test_blank_speaker_with_empty_text_is_dropped(self):
        # empty-text turns drop BEFORE the speaker check
        assert plan_dialogue_chunks([{"speaker": "", "text": " "}], max_chars=300) == []


class TestLimits:
    def test_long_turn_chars_escape_hatch_keeps_turn_intact(self):
        long_text = ("word " * 80).strip()  # 399 chars, no sentence enders
        msgs = [{"speaker": "A", "text": long_text}]
        chunks = plan_dialogue_chunks(msgs, max_chars=100, long_turn_chars=1000)
        # oversized singleton chunk: the turn is NOT split
        assert chunks == [[{"speaker": "A", "text": long_text}]]
        assert len(format_messages(chunks[0])) > 100

    def test_without_escape_hatch_the_same_turn_splits(self):
        long_text = ("word " * 80).strip()
        chunks = plan_dialogue_chunks([{"speaker": "A", "text": long_text}], max_chars=100)
        assert len(chunks) > 1
        assert all(len(format_messages(c)) <= 100 for c in chunks)

    def test_tiny_text_budget_raises(self):
        # budget = 30 - 21 - len("ABCDE") = 4 < 8
        with pytest.raises(ValueError, match="budget"):
            plan_dialogue_chunks([{"speaker": "ABCDE", "text": "hi"}], max_chars=30)


class TestDialogueRamp:
    def test_first_chunk_uses_smaller_budget(self):
        msgs = [{"speaker": "A", "text": f"Turn number {i} says something."} for i in range(8)]
        plain = plan_dialogue_chunks(msgs, max_chars=300)
        ramp = plan_dialogue_chunks(msgs, max_chars=300, first_chunk_chars=80)
        # ramped chunk 0 holds fewer turns; steady-state unchanged rules
        assert len(ramp[0]) < len(plain[0])
        # all content preserved in order
        flat = [m["text"] for c in ramp for m in c]
        assert flat == [m["text"] for m in msgs]

    def test_oversized_single_turn_still_opens_chunk_zero(self):
        msgs = [{"speaker": "A", "text": "word " * 40}]  # one long turn
        ramp = plan_dialogue_chunks(msgs, max_chars=400, first_chunk_chars=50)
        assert ramp[0]  # never split a whole turn just for the ramp
