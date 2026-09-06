"""split_sentences: the 20 spec vectors, edge inputs, and the ladder vectors."""

from __future__ import annotations

import pytest

from bodhan_genai.tts.engine.chunked import chunk_text, split_sentences

# --- vectors with the exact sentence list pinned -------------------------------

EXACT_VECTORS = [
    (
        "It was raining. The streets were empty. We stayed home.",
        ["It was raining.", "The streets were empty.", "We stayed home."],
    ),
    (
        "पहला वाक्य।दूसरा वाक्य॥तीसरा वाक्य।",
        ["पहला वाक्य।", "दूसरा वाक्य॥", "तीसरा वाक्य।"],
    ),
    (
        "She scored 3.5. Then she left.",
        ["She scored 3.5.", "Then she left."],
    ),
    (
        "Dr. Rao went home. He slept.",
        ["Dr. Rao went home.", "He slept."],
    ),
    (
        "Really?! You can't be serious.",
        ["Really?!", "You can't be serious."],
    ),
    (
        "What?!.. Nothing.",
        ["What?!..", "Nothing."],
    ),
    (
        '"Stop!" she yelled. "Go home." Then she left.',
        ['"Stop!" she yelled.', '"Go home."', "Then she left."],
    ),
    (
        "Fruits grow here, e.g. apples. Mangoes too.",
        ["Fruits grow here, e.g. apples.", "Mangoes too."],
    ),
    (
        "मकान No. 5 में Dr. शर्मा रहते हैं। वे अच्छे हैं।",
        ["मकान No. 5 में Dr. शर्मा रहते हैं।", "वे अच्छे हैं।"],
    ),
    (
        "Rs. 500 is the price. Pay now.",
        ["Rs. 500 is the price.", "Pay now."],
    ),
    (
        "今天天气很好。我们去公园吧！你来吗？",  # noqa: RUF001
        ["今天天气很好。", "我们去公园吧！", "你来吗？"],  # noqa: RUF001
    ),
]

# --- vectors that must come back as ONE sentence (the whole text) --------------

SINGLE_SENTENCE_VECTORS = [
    "The value of pi is 3.14 which is well known.",
    "J. K. Rowling wrote it.",
    "Wait... what?",
    "Wait . . . what happened?",
    "उसने सोचा… और फिर वह चल दिया।",
    "And then silence…",  # tail flush
    'He said "go home." and left.',  # G5 quote attribution
    "e.g. apples, mangoes etc. are common here.",
    "भारत में mangoes, apples, और bananas मिलते हैं, लेकिन सबसे अच्छा आम है।",
]


class TestSpecVectors:
    @pytest.mark.parametrize(("text", "expected"), EXACT_VECTORS)
    def test_exact_split(self, text, expected):
        assert split_sentences(text) == expected

    @pytest.mark.parametrize("text", SINGLE_SENTENCE_VECTORS)
    def test_single_sentence(self, text):
        assert split_sentences(text) == [text]


class TestEdges:
    def test_empty(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   \n  ") == []

    def test_single_newline_is_not_a_boundary(self):
        assert split_sentences("Chapter 1\nThe Beginning") == ["Chapter 1\nThe Beginning"]

    def test_blank_line_is_a_paragraph_boundary(self):
        assert split_sentences("Chapter 1\n\nIt was raining.") == ["Chapter 1", "It was raining."]

    def test_unambiguous_terminator_inside_quotes_quirk(self):
        # Documented R1 quirk: danda splits unconditionally, even inside quotes.
        assert split_sentences('उसने कहा, "ठीक है।" और चल दिया।') == [
            'उसने कहा, "ठीक है।"',
            "और चल दिया।",
        ]

    def test_crlf_normalized_before_paragraph_split(self):
        assert split_sentences("First one.\r\n\r\nSecond one.") == ["First one.", "Second one."]

    def test_zero_width_space_becomes_space(self):
        assert split_sentences("One done.\u200bTwo done.") == ["One done.", "Two done."]


class TestLadderVectors:
    def test_clause_ladder_cuts_at_commas(self):
        s = ("alpha beta gamma, " * 22).rstrip(", ") + "."
        chunks = chunk_text(s, max_chars=120)
        assert len(chunks) == 4
        assert all(len(c) <= 120 for c in chunks)
        assert all(c.endswith(",") for c in chunks[:-1])
        assert chunks[-1].endswith(".")
        assert " ".join(chunks) == s

    def test_unbroken_token_hard_cuts(self):
        assert chunk_text("x" * 120, max_chars=50) == ["x" * 50, "x" * 50, "x" * 20]
