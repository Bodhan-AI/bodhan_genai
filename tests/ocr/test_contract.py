"""The pipeline contract: taxonomy closure, prompt coverage, and the label/type vocabularies.

The two vocabularies in play -- IndicDocLayout *labels* and pipeline *types* -- are easy to mix
up, and doing so fails silently (a block quietly becomes ``Text``). These tests pin the
relationships between them, including the case-sensitivity difference between the label sets.
"""

from __future__ import annotations

import pytest

#: The layout model's own class list -- the ground truth for every label set here.
from bodhan_genai.ocr.layout.labels import CLASSES
from bodhan_genai.ocr.templates.contract import (
    DROP_TYPES,
    HEAD_FOOT,
    KEPT_BLOCK_TYPES,
    LABEL_TO_TYPE,
    MARGINALIA,
    OCR_SKIP_LABELS,
    TABLE_PROMPTS,
    TableFormat,
    is_transcribed,
    map_label,
    prompt_for,
)

REACHABLE_TYPES = set(LABEL_TO_TYPE.values()) | {"Text"}  # Text is the fall-through


# --------------------------------------------------------------------------- #
# Taxonomy closure
# --------------------------------------------------------------------------- #


def test_kept_block_types_are_exactly_the_reachable_non_dropped_types():
    assert set(KEPT_BLOCK_TYPES) == REACHABLE_TYPES - set(DROP_TYPES)


def test_dropped_types_are_reachable():
    assert set(DROP_TYPES) <= REACHABLE_TYPES, "dropping a type nothing maps to is a no-op"


def test_every_mapped_label_is_a_real_layout_class():
    known = {c.strip().lower() for c in CLASSES}
    unknown = sorted(set(LABEL_TO_TYPE) - known)
    assert not unknown, f"LABEL_TO_TYPE keys absent from layout.labels.CLASSES: {unknown}"


def test_unmapped_labels_fall_through_to_text():
    # These carry prose and are deliberately absent from LABEL_TO_TYPE.
    for name in ("Question", "Paragraph", "Answer", "List", "MCQ", "Code", "Reference"):
        assert map_label(name) == "Text"


def test_label_lookup_is_case_and_whitespace_insensitive():
    assert map_label("  SUB-SECTION-TITLE  ") == "SectionHeader"
    assert map_label("table") == map_label("Table") == "Table"


def test_unknown_label_is_text_rather_than_an_error():
    assert map_label("Some-Future-Class") == "Text"
    assert map_label(None) == "Text"


# --------------------------------------------------------------------------- #
# Label sets
# --------------------------------------------------------------------------- #


def test_marginalia_and_head_foot_use_exact_class_spellings():
    assert set(CLASSES) >= MARGINALIA
    assert set(HEAD_FOOT) <= set(CLASSES)


def test_ocr_skip_labels_are_real_classes_and_lowercased():
    known = {c.strip().lower() for c in CLASSES}
    assert known >= OCR_SKIP_LABELS
    assert all(label == label.lower() for label in OCR_SKIP_LABELS)


def test_skip_lookup_is_case_insensitive_unlike_the_marginalia_sets():
    assert not is_transcribed("Header")
    assert not is_transcribed("  HEADER ")
    assert is_transcribed("Paragraph")


def test_page_numbers_and_folios_are_transcribed():
    assert is_transcribed("Page-number")
    assert is_transcribed("Folio")


def test_pictorial_labels_are_skipped_and_map_to_dropped_types():
    for label in ("Diagram", "Image", "Chart"):
        assert not is_transcribed(label)
        assert map_label(label) in DROP_TYPES


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_every_kept_type_resolves_to_a_non_empty_prompt():
    for fmt in TableFormat:
        for block_type in KEPT_BLOCK_TYPES:
            assert prompt_for(block_type, fmt).strip()


def test_table_and_equation_are_the_only_types_with_a_special_prompt():
    default = prompt_for("Text")
    special = {t for t in KEPT_BLOCK_TYPES if prompt_for(t) != default}
    assert special == {"Table", "Equation"}


def test_table_format_selects_the_prompt():
    assert "HTML" in prompt_for("Table", TableFormat.HTML)
    assert "markdown" in prompt_for("Table", TableFormat.MARKDOWN)
    assert prompt_for("Table") == prompt_for("Table", TableFormat.HTML), "HTML is the default"


def test_table_format_accepts_its_string_value():
    assert prompt_for("Table", "markdown") == TABLE_PROMPTS[TableFormat.MARKDOWN]


def test_unknown_table_format_is_rejected():
    with pytest.raises(ValueError):
        prompt_for("Table", "latex")


def test_text_prompt_asks_for_latex_math():
    prompt = prompt_for("Text")
    assert "$...$" in prompt and "$$...$$" in prompt


def test_html_table_prompt_asks_for_merged_cell_structure():
    prompt = TABLE_PROMPTS[TableFormat.HTML]
    assert "colspan" in prompt and "rowspan" in prompt
