"""The 37-class taxonomy and the canonical page parser.

The taxonomy tests guard a property that no runtime check can: class ids are baked into
trained checkpoints, so reordering the list silently relabels every model ever trained.
These assertions fail loudly when someone inserts rather than appends.
"""

from __future__ import annotations

import pytest

from bodhan_genai.ocr.data.taxonomy import (
    CLASS_WEIGHTS,
    CLASSES,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    iou_cxcywh,
    labels_from_doc,
    to_cxcywh,
)


def test_taxonomy_size_and_uniqueness():
    assert NUM_CLASSES == 37
    assert len(set(CLASSES)) == len(CLASSES), "a duplicate class name would alias two ids"


@pytest.mark.parametrize(
    ("index", "name"),
    [(0, "Question"), (1, "Paragraph"), (24, "Page-number"), (25, "Chapter-title"), (36, "Flag")],
)
def test_class_ids_are_frozen(index, name):
    """Ids live in every checkpoint. Only ever APPEND to CLASSES."""
    assert CLASSES[index] == name
    assert LABEL2ID[name] == index
    assert ID2LABEL[index] == name


def test_maps_agree_with_the_list():
    assert {c: i for i, c in enumerate(CLASSES)} == LABEL2ID
    assert dict(enumerate(CLASSES)) == ID2LABEL


def test_class_weights_name_real_classes():
    unknown = set(CLASS_WEIGHTS) - set(CLASSES)
    assert not unknown, f"CLASS_WEIGHTS names classes that do not exist: {sorted(unknown)}"


def test_to_cxcywh_converts_y_first_thousandths():
    # [y0, x0, y1, x1] = [0, 0, 500, 1000] is the top half of the page
    box = to_cxcywh([0, 0, 500, 1000])
    assert box is not None
    cx, cy, w, h = box
    assert cx == pytest.approx(0.5, abs=1e-3)
    assert cy == pytest.approx(0.25, abs=1e-3)
    assert w == pytest.approx(1.0, abs=1e-3)
    assert h == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("bad", [None, [], [1, 2, 3], [0, 0, 0, 0], [["nested"], 0, 1, 1], "x"])
def test_to_cxcywh_rejects_malformed_boxes(bad):
    """One bad box drops that box, not the page: converters differ and pages are precious."""
    assert to_cxcywh(bad) is None


def test_clamped_box_stays_inside_the_page():
    cx, cy, w, h = to_cxcywh([0, 0, 1000, 1000])
    assert cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6
    assert cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6


def test_iou_of_identical_and_disjoint_boxes():
    a = [0.5, 0.5, 0.2, 0.2]
    assert iou_cxcywh(a, a) == pytest.approx(1.0)
    assert iou_cxcywh(a, [0.9, 0.9, 0.05, 0.05]) == 0.0


def _doc(content, metadata=None):
    return {"content": content, "metadata": metadata or {}}


def test_parser_keeps_known_labels_and_drops_the_rest():
    doc = _doc(
        [
            {"bbox": [100, 50, 200, 900], "label": "Paragraph", "reading_order": 1},
            {"bbox": [300, 50, 400, 900], "label": "NotARealClass", "reading_order": 2},
            {"bbox": [0, 0, 0, 0], "label": "Paragraph", "reading_order": 3},
            "not even a dict",
        ]
    )
    boxes, classes, order = labels_from_doc(doc)
    assert classes == [LABEL2ID["Paragraph"]]
    assert len(boxes) == 1 and order == [1]


def test_header_reads_first_and_footer_last():
    doc = _doc(
        [{"bbox": [400, 50, 500, 900], "label": "Paragraph", "reading_order": 5}],
        {"header": {"bbox": [100, 100, 150, 900]}, "footer": {"bbox": [900, 100, 950, 900]}},
    )
    _, classes, order = labels_from_doc(doc)
    ranks = dict(zip(classes, order, strict=True))
    assert ranks[LABEL2ID["Header"]] < ranks[LABEL2ID["Paragraph"]]
    assert ranks[LABEL2ID["Footer"]] > ranks[LABEL2ID["Paragraph"]]


def test_tiny_header_stub_is_dropped():
    """Placeholder stubs like [0, 0, 5, 10] are not text regions."""
    doc = _doc(
        [{"bbox": [400, 50, 500, 900], "label": "Paragraph", "reading_order": 1}],
        {"header": {"bbox": [0, 0, 5, 10]}},
    )
    _, classes, _ = labels_from_doc(doc)
    assert LABEL2ID["Header"] not in classes


def test_full_width_edge_strip_dropped_only_when_asked():
    doc = _doc(
        [{"bbox": [400, 50, 500, 900], "label": "Paragraph", "reading_order": 1}],
        {"header": {"bbox": [0, 0, 100, 1000]}},
    )
    _, kept, _ = labels_from_doc(doc)
    _, dropped, _ = labels_from_doc(doc, drop_header_footer_strips=True)
    assert LABEL2ID["Header"] in kept, "default keeps the strip"
    assert LABEL2ID["Header"] not in dropped, "opt-in drops it"


def test_inset_header_survives_strip_dropping():
    """Only page-edge full-width strips are placeholders; a real header is narrower."""
    doc = _doc(
        [{"bbox": [400, 50, 500, 900], "label": "Paragraph", "reading_order": 1}],
        {"header": {"bbox": [100, 200, 160, 800]}},
    )
    _, classes, _ = labels_from_doc(doc, drop_header_footer_strips=True)
    assert LABEL2ID["Header"] in classes


def test_header_annotated_twice_is_not_duplicated():
    """A source that lists the header as content and as metadata must contribute one box."""
    shared = [100, 100, 150, 900]
    doc = _doc(
        [{"bbox": shared, "label": "Header", "reading_order": 1}],
        {"header": {"bbox": shared}},
    )
    _, classes, _ = labels_from_doc(doc)
    assert classes.count(LABEL2ID["Header"]) == 1


def test_empty_page_parses_to_nothing():
    boxes, classes, order = labels_from_doc(_doc([]))
    assert (boxes, classes, order) == ([], [], [])
