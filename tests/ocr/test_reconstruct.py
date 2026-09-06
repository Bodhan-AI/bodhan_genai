"""Markdown assembly: reading order, de-hyphenation, math delimiters, furniture exclusion."""

from __future__ import annotations

from bodhan_genai.ocr.engine.reconstruct import dehyphenate, reconstruct
from bodhan_genai.ocr.engine.types import Block


def blk(order, type_, text, label=None) -> Block:
    return Block(
        order=order,
        label=label or type_,
        type=type_,
        bbox_xyxy=[0.0, 0.0, 10.0, 10.0],
        conf=1.0,
        text=text,
    )


# --------------------------------------------------------------------------- #
# De-hyphenation
# --------------------------------------------------------------------------- #


def test_hyphenated_line_break_is_rejoined():
    assert dehyphenate("hyphen-\nation") == "hyphenation"


def test_leading_whitespace_on_the_continuation_line_is_consumed():
    assert dehyphenate("hyphen-\n   ation") == "hyphenation"


def test_every_dash_variant_a_scan_may_emit_is_handled():
    for dash in ("-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        assert dehyphenate(f"co{dash}\noperate") == "cooperate", f"failed on U+{ord(dash):04X}"


def test_dehyphenation_iterates_to_a_fixpoint():
    """A single pass is not enough.

    ``re.sub`` finds non-overlapping matches, so in ``a-\\nb-\\nc`` the first pass consumes the
    ``b`` that the second break needs and leaves ``ab-\\nc`` behind. Iterating repairs it.
    """
    assert dehyphenate("a-\nb-\nc") == "abc"


def test_hyphen_without_a_line_break_is_left_alone():
    assert dehyphenate("well-known") == "well-known"
    assert dehyphenate("end-\n\nstart") == "end-\n\nstart", "a blank line is a paragraph break"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def test_blocks_are_joined_in_reading_order_not_list_order():
    md = reconstruct([blk(2, "Text", "third"), blk(0, "Text", "first"), blk(1, "Text", "second")])
    assert md == "first\n\nsecond\n\nthird"


def test_blocks_are_separated_by_a_blank_line():
    assert reconstruct([blk(0, "Text", "a"), blk(1, "Text", "b")]) == "a\n\nb"


def test_whitespace_around_a_transcription_is_stripped():
    assert reconstruct([blk(0, "Text", "  padded  \n")]) == "padded"


def test_bare_equation_is_wrapped_as_display_math():
    assert reconstruct([blk(0, "Equation", "x^2 + y^2 = z^2")]) == "$$x^2 + y^2 = z^2$$"


def test_equation_that_already_carries_delimiters_is_left_untouched():
    for already in ("$$a=b$$", "$a=b$", "\\[a=b\\]", "\\(a=b\\)"):
        assert reconstruct([blk(0, "Equation", already)]) == already


def test_math_inside_a_text_block_is_not_wrapped():
    assert reconstruct([blk(0, "Text", "where $x$ is the mean")]) == "where $x$ is the mean"


def test_figures_and_pictures_are_excluded_from_the_markdown():
    md = reconstruct(
        [
            blk(0, "Text", "before"),
            blk(1, "Figure", "hallucinated caption", label="Diagram"),
            blk(2, "Picture", "hallucinated", label="Image"),
            blk(3, "Text", "after"),
        ]
    )
    assert md == "before\n\nafter"


def test_untranscribed_blocks_contribute_nothing_but_are_not_required_to_be_removed():
    blocks = [
        blk(0, "PageHeader", "", label="Header"),
        blk(1, "Text", "body"),
        blk(2, "Text", None),
    ]
    assert reconstruct(blocks) == "body"
    assert len(blocks) == 3, "reconstruct must not mutate the caller's list"


def test_page_furniture_that_was_transcribed_is_included():
    md = reconstruct([blk(0, "Text", "body"), blk(1, "PageNumber", "42", label="Page-number")])
    assert md == "body\n\n42"


def test_dehyphenation_is_applied_across_the_assembled_page():
    assert reconstruct([blk(0, "Text", "hyphen-\nated word")]) == "hyphenated word"


def test_empty_page_yields_empty_markdown():
    assert reconstruct([]) == ""
    assert reconstruct([blk(0, "Text", "")]) == ""


# --------------------------------------------------------------------------- #
# Math repair. The recognizer emits Indic script inside math delimiters without \text{}, which
# no LaTeX engine renders -- 34 of 204 expressions on the sample pages.
# --------------------------------------------------------------------------- #

from bodhan_genai.ocr.engine.reconstruct import repair_math  # noqa: E402


def test_indic_run_in_math_is_wrapped_in_text():
    assert repair_math("$$প্রোটন = 9$$") == r"$$\text{প্রোটন} = 9$$"


def test_devanagari_and_arabic_are_wrapped_too():
    assert repair_math("$संख्या = 5$") == r"$\text{संख्या} = 5$"
    assert repair_math("$عدد = 5$") == r"$\text{عدد} = 5$"


def test_an_already_wrapped_run_is_not_double_wrapped():
    assert repair_math(r"$$\text{প্রাচীন} = 111$$") == r"$$\text{প্রাচীন} = 111$$"


def test_a_multi_word_run_stays_one_text_group():
    """Interior spaces and the ZWJ/ZWNJ joiners Indic shaping needs must not split the run."""
    assert repair_math("$$মোট সংখ্যা = 7$$") == r"$$\text{মোট সংখ্যা} = 7$$"


def test_latin_math_is_left_completely_alone():
    for tex in (r"$$\frac{1}{2n}$$", r"$\theta$", r"$$\int_{0}^{\infty} x\,dx$$", "$x^2 + y^2$"):
        assert repair_math(tex) == tex


def test_prose_outside_math_is_untouched():
    md = "এটি একটি বাক্য।\n\nAnd this is prose."
    assert repair_math(md) == md


def test_newlines_in_display_math_become_row_separators():
    """A bare newline inside $$...$$ is a syntax error; \\\\ is the row separator."""
    assert repair_math("$$a = 1\nb = 2$$") == r"$$a = 1 \\ b = 2$$"


def test_the_real_failing_expression_from_the_sample_pages():
    got = repair_math("$$=> প্রোটন = 9\nনিউট্রন = 19 - 8 = 10$$")
    assert got == r"$$=> \text{প্রোটন} = 9 \\ \text{নিউট্রন} = 19 - 8 = 10$$"


def test_reconstruct_applies_the_repair_by_default_and_can_skip_it():
    blocks = [blk(0, "Equation", "প্রোটন = 9")]
    assert reconstruct(blocks) == r"$$\text{প্রোটন} = 9$$"
    assert reconstruct(blocks, repair=False) == "$$প্রোটন = 9$$"


def test_an_equation_block_that_already_carries_delimiters_is_not_rewrapped():
    """The recognizer often returns a prose prefix plus already-delimited math for an Equation
    block. Wrapping that again yields "$$...$$$", which no parser accepts."""
    got = reconstruct([blk(0, "Equation", r"বা, $\frac{\sqrt{3}}{1} = \frac{180}{60}$")])
    assert got.count("$$") == 0
    # The prefix is now prose OUTSIDE the delimiters, so it needs no \text{} -- repair only
    # touches what is inside them.
    assert got == r"বা, $\frac{\sqrt{3}}{1} = \frac{180}{60}$"


def test_an_undelimited_equation_block_is_still_wrapped():
    assert reconstruct([blk(0, "Equation", "x^2 + y^2 = z^2")]) == "$$x^2 + y^2 = z^2$$"


def test_a_bracket_delimited_equation_is_left_alone():
    assert reconstruct([blk(0, "Equation", r"\[a = b\]")]) == r"\[a = b\]"
