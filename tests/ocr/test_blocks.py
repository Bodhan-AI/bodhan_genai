"""Layout cleanup: the three ``clean_layout`` rules and nested-equation resolution.

These are the pipeline's most consequential pure functions -- they decide what is transcribed at
all -- and every rule here encodes a specific failure that was observed on real pages. Pure
geometry, so the whole file runs with no GPU stack and no PIL installed.
"""

from __future__ import annotations

import pytest

from bodhan_genai.ocr.engine.blocks import (
    area,
    clamp_to_page,
    clean_layout,
    contained_frac,
    resolve_nested_equations,
)
from bodhan_genai.ocr.engine.types import Block, DedupConfig
from bodhan_genai.ocr.templates.contract import map_label


def blk(order, label, bbox, type_=None) -> Block:
    return Block(
        order=order,
        label=label,
        type=type_ or map_label(label),
        bbox_xyxy=list(bbox),
        conf=1.0,
    )


def labels(blocks):
    return [b.label for b in blocks]


def orders(blocks):
    return [b.order for b in blocks]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_area_of_degenerate_and_inverted_boxes_is_zero():
    assert area([10, 10, 10, 50]) == 0.0  # zero width
    assert area([10, 10, 50, 10]) == 0.0  # zero height
    assert area([50, 50, 10, 10]) == 0.0  # inverted, not negative


def test_contained_frac_is_relative_to_the_inner_box():
    # Half of the inner box lies inside the outer one.
    assert contained_frac([0, 0, 10, 10], [5, 0, 100, 100]) == pytest.approx(0.5)
    assert contained_frac([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert contained_frac([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0  # degenerate, no ZeroDivision


def test_clamp_to_page_clips_both_directions():
    assert clamp_to_page([-5, -5, 120, 220], 100, 200) == [0.0, 0.0, 100.0, 200.0]


# --------------------------------------------------------------------------- #
# Rule 1 -- nested duplicates
# --------------------------------------------------------------------------- #


def test_nested_duplicate_is_dropped_in_favour_of_the_larger_box():
    kept = clean_layout(
        [
            blk(0, "Paragraph", [0, 0, 100, 100]),
            blk(1, "Paragraph", [10, 10, 90, 90]),  # ~64% -- fully inside the larger
        ]
    )
    assert orders(kept) == [0]


def test_partially_overlapping_boxes_both_survive():
    kept = clean_layout(
        [
            blk(0, "Paragraph", [0, 0, 100, 100]),
            blk(1, "Paragraph", [80, 0, 180, 100]),  # only 20% inside
        ]
    )
    assert orders(kept) == [0, 1]


def test_survivors_keep_input_order():
    kept = clean_layout(
        [
            blk(0, "Paragraph", [0, 0, 10, 10]),
            blk(1, "Paragraph", [0, 500, 800, 900]),  # much larger, considered first internally
            blk(2, "Paragraph", [0, 20, 10, 30]),
        ]
    )
    assert orders(kept) == [0, 1, 2], "output must be in input order, not area order"


def test_text_inside_a_figure_is_never_absorbed():
    """The transcribability guard.

    A Diagram is never sent to the recognizer. Absorbing a caption into one would drop the
    caption in favour of a container that is then skipped -- losing the text entirely.
    """
    kept = clean_layout(
        [
            blk(0, "Diagram", [0, 0, 100, 100]),
            blk(1, "Image-caption", [10, 10, 90, 90]),
        ]
    )
    assert "Image-caption" in labels(kept), "caption must survive inside a non-transcribed box"


def test_non_transcribed_box_may_be_absorbed_by_anything():
    kept = clean_layout(
        [
            blk(0, "Paragraph", [0, 0, 100, 100]),
            blk(1, "Diagram", [10, 10, 90, 90]),
        ]
    )
    assert labels(kept) == ["Paragraph"]


def test_marginalia_and_body_are_cleaned_as_separate_groups():
    kept = clean_layout(
        [
            blk(0, "Paragraph", [0, 0, 1000, 1000]),
            blk(1, "Page-number", [900, 950, 990, 990]),
        ]
    )
    assert orders(kept) == [0, 1]


def test_duplicate_page_numbers_are_still_deduplicated_within_their_group():
    kept = clean_layout(
        [
            blk(0, "Page-number", [900, 940, 995, 995]),
            blk(1, "Page-number", [910, 950, 990, 990]),
        ]
    )
    assert orders(kept) == [0]


# --------------------------------------------------------------------------- #
# Rule 2 -- one header, one footer
# --------------------------------------------------------------------------- #


def test_only_the_largest_header_survives():
    kept = clean_layout(
        [
            blk(0, "Header", [0, 0, 1000, 60]),  # largest
            blk(1, "Header", [0, 5, 400, 50]),
            blk(2, "Header", [0, 10, 200, 40]),
            blk(3, "Paragraph", [10, 10, 300, 45]),  # occupies the header, so it is kept
        ]
    )
    assert labels(kept) == ["Header", "Paragraph"]
    assert orders(kept)[0] == 0


def test_header_and_footer_are_reduced_independently():
    kept = clean_layout(
        [
            blk(0, "Header", [0, 0, 1000, 60]),
            blk(1, "Header", [0, 5, 400, 50]),
            blk(2, "Footer", [0, 940, 1000, 1000]),
            blk(3, "Footer", [0, 950, 400, 990]),
            blk(4, "Paragraph", [10, 10, 300, 45]),
            blk(5, "Paragraph", [10, 950, 300, 985]),
        ]
    )
    assert labels(kept) == ["Header", "Footer", "Paragraph", "Paragraph"]


# --------------------------------------------------------------------------- #
# Rule 3 -- empty frames
# --------------------------------------------------------------------------- #


def test_header_wrapping_nothing_is_dropped():
    kept = clean_layout(
        [
            blk(0, "Header", [0, 0, 1000, 60]),
            blk(1, "Paragraph", [0, 400, 1000, 600]),  # nowhere near the header
        ]
    )
    assert labels(kept) == ["Paragraph"]


def test_occupied_header_survives():
    kept = clean_layout(
        [
            blk(0, "Header", [0, 0, 1000, 60]),
            blk(1, "Page-number", [900, 10, 980, 50]),
        ]
    )
    assert labels(kept) == ["Header", "Page-number"]


def test_wrapping_is_measured_against_raw_input_not_survivors():
    """The ordering subtlety that is easiest to lose in a rewrite.

    The header's only occupant is a page number that rule 1 already dropped as a nested
    duplicate. Measuring "does this header wrap anything?" against the *survivors* would find
    nothing and delete a real, occupied header. It must be measured against the raw input.
    """
    blocks = [
        blk(0, "Header", [0, 0, 1000, 60]),
        blk(1, "Page-number", [890, 5, 990, 55]),  # larger of the pair -- survives rule 1
        blk(2, "Page-number", [900, 10, 980, 50]),  # nested duplicate -- dropped by rule 1
    ]
    kept = clean_layout(blocks)
    assert "Header" in labels(kept), "header must survive: it does wrap a raw page-number box"
    assert orders(kept) == [0, 1]


def test_empty_layout_is_handled():
    assert clean_layout([]) == []


# --------------------------------------------------------------------------- #
# Nested equations
# --------------------------------------------------------------------------- #

PARA = [0, 0, 500, 200]
INLINE_EQ = [50, 80, 200, 120]  # small box well inside PARA


def test_inline_equation_is_folded_into_its_paragraph_by_default():
    kept = resolve_nested_equations([blk(0, "Paragraph", PARA), blk(1, "Equation", INLINE_EQ)])
    assert labels(kept) == ["Paragraph"]


def test_eq_only_mode_leaves_inline_math_in_place():
    kept = resolve_nested_equations(
        [blk(0, "Paragraph", PARA), blk(1, "Equation", INLINE_EQ)],
        DedupConfig(mode="eq_only"),
    )
    assert labels(kept) == ["Paragraph", "Equation"]


def test_display_array_rows_collapse_into_the_outer_equation():
    kept = resolve_nested_equations(
        [
            blk(0, "Equation", [0, 0, 400, 300]),
            blk(1, "Equation", [10, 10, 390, 90]),
            blk(2, "Equation", [10, 110, 390, 190]),
        ]
    )
    assert orders(kept) == [0]


def test_text_only_mode_keeps_array_rows_separate():
    kept = resolve_nested_equations(
        [
            blk(0, "Equation", [0, 0, 400, 300]),
            blk(1, "Equation", [10, 10, 390, 90]),
        ],
        DedupConfig(mode="text_only"),
    )
    assert orders(kept) == [0, 1]


def test_equation_inside_a_table_is_kept():
    kept = resolve_nested_equations(
        [blk(0, "Table", [0, 0, 500, 200]), blk(1, "Equation", INLINE_EQ)]
    )
    assert orders(kept) == [0, 1]


def test_coincident_equations_do_not_annihilate_each_other():
    kept = resolve_nested_equations(
        [blk(0, "Equation", [0, 0, 100, 100]), blk(1, "Equation", [0, 0, 100, 100])]
    )
    assert len(kept) == 2


def test_nest_disabled_is_a_passthrough():
    blocks = [blk(0, "Paragraph", PARA), blk(1, "Equation", INLINE_EQ)]
    assert resolve_nested_equations(blocks, DedupConfig(nest=False)) == blocks


def test_non_equation_blocks_are_never_dropped_by_this_rule():
    kept = resolve_nested_equations([blk(0, "Paragraph", PARA), blk(1, "Paragraph", INLINE_EQ)])
    assert orders(kept) == [0, 1]
