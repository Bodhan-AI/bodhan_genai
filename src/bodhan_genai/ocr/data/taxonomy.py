"""The 37-class layout taxonomy and the canonical page-annotation parser.

This is the *training* taxonomy for IndicDocLayout, and it is a different thing from
the inference contract in ``bodhan_genai.ocr.templates.contract``: this one is the raw
37-class label set the detector predicts, the contract maps those raw labels onto the
handful of coarse pipeline types that pick a prompt.

**Order is fixed. Only ever APPEND.** The ids are baked into every checkpoint, so
reordering silently relabels a trained model rather than raising:

*   ids 0-24  — the original 25 education classes
*   ids 25-26 — printed-only additions
*   ids 27-36 — magazine/newspaper furniture, populated only by the ``indicdlp-printed``
    source, so no other source's partial annotation conflicts with them

``labels_from_doc`` is deliberately the ONE parser used by the training blob cache and
by the val/test disk path both, because a split that parses its labels differently from
the split it is compared against produces a number that means nothing.
"""

from __future__ import annotations

CLASSES: list[str] = [
    # 0-24: the original education classes
    "Question", "Paragraph", "Answer", "List", "Title", "Section-title",
    "Equation", "Table", "Diagram", "Image", "MCQ", "Infobox",
    "Sub-section-title", "Expression", "Image-caption", "Placeholder-text",
    "Chart", "Solved-example", "Footnote", "Table-caption",
    "Sub-sub-section-title", "Footer", "Header", "Code", "Page-number",
    # 25-26: printed-only additions
    "Chapter-title", "Chapter-end-section",
    # 27-36: magazine / newspaper furniture (indicdlp-printed only)
    "Folio", "Reference", "Table-of-contents", "Index", "Advertisement",
    "Author", "Dateline", "Contact-info", "Website-link", "Flag",
]  # fmt: skip

LABEL2ID: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL: dict[int, str] = dict(enumerate(CLASSES))
NUM_CLASSES: int = len(CLASSES)

# sqrt-inverse block frequency, capped to [0.5, 5], measured over 155,922 blocks.
# Optional: the trainer only applies these when class weighting is switched on.
CLASS_WEIGHTS: dict[str, float] = {
    "Question": 0.50, "Paragraph": 0.50, "Equation": 0.50, "Answer": 0.50,
    "List": 0.50, "Section-title": 0.50, "Title": 0.50, "Image": 0.50,
    "Table": 0.52, "Diagram": 0.53, "MCQ": 0.74, "Expression": 1.00,
    "Image-caption": 1.11, "Infobox": 1.43, "Solved-example": 1.43,
    "Placeholder-text": 1.48, "Sub-section-title": 1.64, "Chart": 2.11,
    "Footnote": 4.63, "Table-caption": 5.0, "Sub-sub-section-title": 5.0,
    "Footer": 5.0, "Header": 5.0, "Code": 5.0, "Page-number": 5.0,
    "Chapter-title": 0.74, "Chapter-end-section": 4.5,
}  # fmt: skip

# Header/footer regions smaller than this are placeholder stubs (e.g. bbox [0, 0, 5, 10]),
# not real text, and teaching the model to emit them costs precision at the page edges.
MIN_HEADER_FOOTER_AREA = 1e-4


def to_cxcywh(bbox: list[float] | None) -> list[float] | None:
    """``[y0, x0, y1, x1]`` in 0-1000 to a clamped normalized ``[cx, cy, w, h]``.

    Returns ``None`` for a malformed or degenerate box rather than raising: page
    annotations come from several converters and a single bad box should drop that box,
    not the page. Note the source order is **y first**, which is easy to get backwards.
    """
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        y0, x0, y1, x1 = (float(c) / 1000 for c in bbox)
    except (TypeError, ValueError):
        return None
    if x1 - x0 <= 1e-3 or y1 - y0 <= 1e-3:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    cx, cy = min(max(cx, 1e-4), 1 - 1e-4), min(max(cy, 1e-4), 1 - 1e-4)
    # keep the box inside the page after clamping the centre
    w, h = min(w, 2 * min(cx, 1 - cx)), min(h, 2 * min(cy, 1 - cy))
    return [cx, cy, w, h]


def iou_cxcywh(a: list[float], b: list[float]) -> float:
    """IoU of two normalized ``[cx, cy, w, h]`` boxes."""
    iw = max(0.0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    ih = max(0.0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def labels_from_doc(
    doc: dict, *, drop_header_footer_strips: bool = False
) -> tuple[list[list[float]], list[int], list[int]]:
    """One page annotation to ``(boxes_cxcywh, class_ids, reading_order)``.

    Content blocks come first, then the optional ``metadata.header`` / ``metadata.footer``
    regions, which are given ranks either side of the content so the header reads first
    and the footer last.

    ``drop_header_footer_strips`` discards full-width header/footer boxes flush to a page
    edge (``[0, 0, 100, 1000]``-style placeholders that some handwritten sources emit).
    They are not real text regions, and training on them teaches the model to hallucinate
    a bar across the top or bottom of every page. Real, inset header/footer regions are
    kept either way.
    """
    boxes: list[list[float]] = []
    classes: list[int] = []
    order: list[int] = []

    for block in doc.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        label = block.get("label")
        if label not in LABEL2ID:
            continue
        box = to_cxcywh(block.get("bbox"))
        if box is None:
            continue
        boxes.append(box)
        classes.append(LABEL2ID[label])
        order.append(block.get("reading_order", len(order) + 1))

    metadata = doc.get("metadata") or {}
    ranks = order or [0]
    edges = (
        ("header", "Header", min(ranks) - 1),
        ("footer", "Footer", max(ranks) + 1),
    )
    for field, label, rank in edges:
        region = metadata.get(field) or {}
        box = to_cxcywh(region.get("bbox"))
        if box is None or box[2] * box[3] < MIN_HEADER_FOOTER_AREA:
            continue
        if drop_header_footer_strips and box[2] > 0.9:
            y0, y1 = box[1] - box[3] / 2, box[1] + box[3] / 2
            if (field == "header" and y0 < 0.02) or (field == "footer" and y1 > 0.98):
                continue
        label_id = LABEL2ID[label]
        # a source that annotates the header both as content and as metadata would
        # otherwise contribute the same region twice, with two different ranks
        if any(
            c == label_id and iou_cxcywh(b, box) > 0.5 for b, c in zip(boxes, classes, strict=True)
        ):
            continue
        boxes.append(box)
        classes.append(label_id)
        order.append(rank)

    return boxes, classes, order
