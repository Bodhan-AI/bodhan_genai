"""chunk_text: sentence splitting, laddering, strict max-chars packing."""

from __future__ import annotations

import pytest

from bodhan_genai.tts.engine.chunked import chunk_text


class TestSplitting:
    def test_latin_and_danda_boundaries(self):
        text = "पहला वाक्य। दूसरा वाक्य यहाँ है। And an English sentence. Then another one."
        chunks = chunk_text(text, min_chars=4, max_chars=30)
        joined = " ".join(chunks)
        assert joined == text
        # danda and period boundaries both produced splits
        assert any(c.endswith("।") for c in chunks)
        assert any(c.endswith(".") for c in chunks)

    def test_cjk_fullwidth_punctuation(self):
        chunks = chunk_text("第一句话。 第二句话！ 第三句话？", min_chars=2, max_chars=8)  # noqa: RUF001
        assert chunks == ["第一句话。", "第二句话！", "第三句话？"]  # noqa: RUF001

    def test_decimals_and_abbreviations_not_split(self):
        # a period needs following whitespace to end a sentence: 3.14 survives
        chunks = chunk_text("The value is 3.14 exactly", min_chars=4, max_chars=300)
        assert chunks == ["The value is 3.14 exactly"]
        # "Dr." is a known abbreviation — no sentence boundary at all now
        chunks = chunk_text("Dr. Rao spoke", min_chars=2, max_chars=300)
        assert chunks == ["Dr. Rao spoke"]

    def test_punctuation_stays_attached(self):
        chunks = chunk_text("Hello there! How are you?", min_chars=4, max_chars=15)
        assert chunks[0].endswith("!")

    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []


class TestMinCharsMerge:
    def test_short_tail_merges_backward(self):
        chunks = chunk_text(
            "A significantly longer first clause here. Ok.", min_chars=8, max_chars=44
        )
        assert chunks[-1].endswith("Ok.")
        assert len(chunks) == 2 or "Ok." in chunks[-1]

    def test_short_first_chunk_folds_forward(self):
        chunks = chunk_text(
            "Hi. This second clause is much longer than the first.", min_chars=8, max_chars=30
        )
        assert chunks[0].startswith("Hi. This")


class TestMaxCharsPacking:
    def test_small_sentences_pack_together(self):
        text = "One. Two. Three. Four. Five. Six."
        chunks = chunk_text(text, min_chars=2, max_chars=300)
        assert chunks == [text]

    def test_packing_respects_max(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        chunks = chunk_text(text, min_chars=4, max_chars=40)
        assert all(len(c) <= 40 for c in chunks)
        assert " ".join(chunks) == text

    def test_oversized_clause_hard_splits_at_whitespace(self):
        text = "word " * 40  # one clause, no punctuation, ~200 chars
        chunks = chunk_text(text, min_chars=4, max_chars=50)
        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)
        # never mid-word
        assert all(c.startswith("word") and c.endswith("word") for c in chunks)

    def test_unbroken_token_hard_cut(self):
        text = "x" * 120
        chunks = chunk_text(text, min_chars=4, max_chars=50)
        assert "".join(chunks) == text
        assert all(len(c) <= 50 for c in chunks)


class TestGluedPunctuation:
    """Danda/CJK terminators split even with NO following whitespace —
    realistic Hindi and spaceless CJK (review finding)."""

    def test_danda_glued_to_next_word_splits(self):
        chunks = chunk_text("पहला वाक्य।दूसरा वाक्य।तीसरा", min_chars=2, max_chars=12)
        assert chunks == ["पहला वाक्य।", "दूसरा वाक्य।", "तीसरा"]

    def test_cjk_no_space_splits(self):
        chunks = chunk_text("第一句话。第二句话！第三句话？", min_chars=2, max_chars=8)  # noqa: RUF001
        assert chunks == ["第一句话。", "第二句话！", "第三句话？"]  # noqa: RUF001

    def test_latin_period_still_needs_whitespace(self):
        # the decimal guard must survive the new alternation
        assert chunk_text("pi is 3.14159 always", min_chars=2, max_chars=300) == [
            "pi is 3.14159 always"
        ]

    def test_newline_counts_as_hard_split_whitespace(self):
        text = ("word " * 9).strip() + "\n" + ("word " * 9).strip()  # no punctuation
        chunks = chunk_text(text, min_chars=2, max_chars=50)
        assert all(len(c) <= 50 for c in chunks)
        assert all(c.split()[0] == "word" and c.split()[-1] == "word" for c in chunks)


class TestSentenceGuards:
    """chunk_text inherits split_sentences' guards: ellipses pause, clause
    marks and abbreviations never end a sentence."""

    def test_ascii_ellipsis_does_not_split(self):
        assert chunk_text("Wait... what?", min_chars=2, max_chars=300) == ["Wait... what?"]

    def test_unicode_ellipsis_does_not_split(self):
        text = "उसने सोचा… और फिर वह चल दिया।"
        assert chunk_text(text, min_chars=2, max_chars=300) == [text]

    def test_comma_semicolon_colon_do_not_split(self):
        text = "one, two; three: four"
        assert chunk_text(text, min_chars=2, max_chars=300) == [text]

    def test_clause_marks_stay_inside_their_sentence(self):
        chunks = chunk_text("He came, he saw; he left. Then peace.", min_chars=2, max_chars=30)
        assert chunks == ["He came, he saw; he left.", "Then peace."]

    def test_dr_abbreviation(self):
        chunks = chunk_text("Dr. Rao went home. He slept.", min_chars=2, max_chars=20)
        assert chunks == ["Dr. Rao went home.", "He slept."]

    def test_eg_abbreviation(self):
        chunks = chunk_text(
            "Fruits grow here, e.g. apples. Mangoes too.", min_chars=2, max_chars=32
        )
        assert chunks == ["Fruits grow here, e.g. apples.", "Mangoes too."]

    def test_numeric_abbreviation_no_5(self):
        chunks = chunk_text("मकान No. 5 में Dr. शर्मा रहते हैं। वे अच्छे हैं।", min_chars=2, max_chars=40)
        assert chunks == ["मकान No. 5 में Dr. शर्मा रहते हैं।", "वे अच्छे हैं।"]

    def test_numeric_abbreviation_rs_500(self):
        chunks = chunk_text("Rs. 500 is the price. Pay now.", min_chars=2, max_chars=22)
        assert chunks == ["Rs. 500 is the price.", "Pay now."]


STRICT_MAX_INPUTS = [
    (
        "This is the first sentence of a long paragraph. "
        "Here comes a second sentence with more words in it. "
        "And finally a third sentence to close the paragraph."
    ),
    ("alpha beta gamma, " * 22).rstrip(", ") + ".",
    "x" * 120,
    "पहला वाक्य।दूसरा वाक्य॥तीसरा वाक्य।",
    "今天天气很好。我们去公园吧！你来吗？",  # noqa: RUF001
    "one, two; three: four " * 10,
]


class TestStrictMaxInvariant:
    @pytest.mark.parametrize("text", STRICT_MAX_INPUTS)
    @pytest.mark.parametrize("max_chars", [30, 50, 120])
    def test_no_chunk_ever_exceeds_max(self, text, max_chars):
        assert all(len(c) <= max_chars for c in chunk_text(text, max_chars=max_chars))


class TestJoinReconstruction:
    """For single-space-separated input, chunking loses no characters."""

    @pytest.mark.parametrize(
        "text",
        [
            (
                "This is the first sentence of a long paragraph. "
                "Here comes a second sentence with more words in it. "
                "And finally a third sentence to close the paragraph."
            ),
            ("alpha beta gamma, " * 22).rstrip(", ") + ".",
            ("word " * 40).strip(),
            "Dr. Rao went home. He slept. Rs. 500 is the price. Pay now.",
        ],
    )
    @pytest.mark.parametrize("max_chars", [40, 120, 300])
    def test_space_join_restores_text(self, text, max_chars):
        assert " ".join(chunk_text(text, max_chars=max_chars)) == text


class TestFirstChunkRamp:
    TEXT = ("One two three four five six seven. " * 20).strip()

    def test_ramp_shrinks_first_chunk_only(self):
        plain = chunk_text(self.TEXT, max_chars=200)
        ramp = chunk_text(self.TEXT, max_chars=200, first_chunk_chars=60)
        assert len(ramp[0]) <= 60
        assert len(ramp[0]) < len(plain[0])
        assert all(len(c) <= 200 for c in ramp)
        assert " ".join(ramp) == " ".join(plain)  # same content, different schedule

    def test_first_sentence_over_ramp_budget_is_laddered(self):
        text = "word " * 30 + "end. Second sentence here."  # first sentence ~155 chars
        ramp = chunk_text(text.strip(), max_chars=300, first_chunk_chars=40)
        assert len(ramp[0]) <= 40

    def test_ramp_noop_when_ge_max(self):
        assert chunk_text(self.TEXT, max_chars=200, first_chunk_chars=200) == chunk_text(
            self.TEXT, max_chars=200
        )


class TestEstimateSpeechSeconds:
    def test_script_rates(self):
        from bodhan_genai.tts.engine.chunked import estimate_speech_seconds as est

        assert est("a" * 140) == pytest.approx(10.0)
        assert est("क" * 120) == pytest.approx(10.0)
        assert est("好" * 50) == pytest.approx(10.0)
        assert est("") == 0.0
        # per-char: CJK is slower than Latin
        assert est("好") > est("a")
