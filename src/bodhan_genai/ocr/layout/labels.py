"""IndicDocLayout's 37 education-domain layout classes = SUPERSET of the printed + handwritten
taxonomies.
Order is fixed; do not reorder and only APPEND (ids are baked into checkpoints). ids 0-24 are the
original 25; 25-26 are the printed-only additions (Chapter-title, Chapter-end-section); 27-36 are
the indicdlp-v2 magazine/newspaper additions. Class weights are sqrt-inverse block frequency
(capped [0.5, 5]); optional, used only if enabled.

These are the raw *labels*. The map onto the coarse pipeline *types* that select prompts lives in
``ocr.templates.contract`` -- see its module docstring for why the two vocabularies differ.
"""

CLASSES = [
    "Question",
    "Paragraph",
    "Answer",
    "List",
    "Title",
    "Section-title",
    "Equation",
    "Table",
    "Diagram",
    "Image",
    "MCQ",
    "Infobox",
    "Sub-section-title",
    "Expression",
    "Image-caption",
    "Placeholder-text",
    "Chart",
    "Solved-example",
    "Footnote",
    "Table-caption",
    "Sub-sub-section-title",
    "Footer",
    "Header",
    "Code",
    "Page-number",
    "Chapter-title",
    "Chapter-end-section",  # 25-26: printed-only additions (superset)
    # 27-36: indicdlp-v2 additions (magazine/newspaper furniture kept as their own classes;
    # only populated by the indicdlp-printed source -> no cross-dataset partial-annotation conflict).
    "Folio",
    "Reference",
    "Table-of-contents",
    "Index",
    "Advertisement",
    "Author",
    "Dateline",
    "Contact-info",
    "Website-link",
    "Flag",
]

LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# sqrt-inverse-frequency weights (measured on 155,922 blocks); rare classes upweighted.
CLASS_WEIGHTS = {
    "Question": 0.50,
    "Paragraph": 0.50,
    "Equation": 0.50,
    "Answer": 0.50,
    "List": 0.50,
    "Section-title": 0.50,
    "Title": 0.50,
    "Image": 0.50,
    "Table": 0.52,
    "Diagram": 0.53,
    "MCQ": 0.74,
    "Expression": 1.00,
    "Image-caption": 1.11,
    "Infobox": 1.43,
    "Solved-example": 1.43,
    "Placeholder-text": 1.48,
    "Sub-section-title": 1.64,
    "Chart": 2.11,
    "Footnote": 4.63,
    "Table-caption": 5.0,
    "Sub-sub-section-title": 5.0,
    "Footer": 5.0,
    "Header": 5.0,
    "Code": 5.0,
    "Page-number": 5.0,
    "Chapter-title": 0.74,
    "Chapter-end-section": 4.5,  # by frequency in label_frequency.csv
}

# --- Shared label parser (used by BOTH the training blob cache AND the val/test disk path,
#     so train/val/test are always consistent). Content blocks + metadata Header/Footer region
#     boxes (header reads first, footer last; placeholder stubs dropped; hw dedup vs content). ---
MIN_HF_AREA = 1e-4  # drop placeholder header/footer stubs (e.g. bbox [0,0,5,10])


def _to_cxcywh(bb):
    """[y0,x0,y1,x1] in 0-1000 -> clamped [cx,cy,w,h] normalized, or None if degenerate."""
    try:
        y0, x0, y1, x1 = [float(c) / 1000 for c in bb]
    except (TypeError, ValueError):
        return None  # malformed bbox (e.g. nested list) -> skip this box
    if x1 - x0 <= 1e-3 or y1 - y0 <= 1e-3:
        return None
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    cx, cy = min(max(cx, 1e-4), 1 - 1e-4), min(max(cy, 1e-4), 1 - 1e-4)
    w, h = min(w, 2 * min(cx, 1 - cx)), min(h, 2 * min(cy, 1 - cy))
    return [cx, cy, w, h]


def _iou(a, b):
    iw = max(0, min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2))
    ih = max(0, min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def labels_from_doc(d, drop_hf_strips=False):
    """Parse a page JSON dict -> (boxes[cxcywh], cls[ids], order). THE canonical parser.
    drop_hf_strips: skip metadata Header/Footer boxes that are full-width page-edge STRIPS (the
    hw-vs/hw-dps placeholders like [0,0,100,1000]) — they're not real text regions and teach the
    model to hallucinate top/bottom bars. Real (non-strip) header/footer regions are kept."""
    cls, boxes, order = [], [], []
    for b in d.get("content", []):
        bb, lab = b.get("bbox"), b.get("label")
        if not (isinstance(bb, list) and len(bb) == 4) or lab not in LABEL2ID:
            continue
        box = _to_cxcywh(bb)
        if box is None:
            continue
        cls.append(LABEL2ID[lab])
        boxes.append(box)
        order.append(b.get("reading_order", len(order) + 1))
    meta = d.get("metadata", {}) or {}
    ords = order or [0]
    for field, lab, ro in (
        ("header", "Header", min(ords) - 1),
        ("footer", "Footer", max(ords) + 1),
    ):
        bb = (meta.get(field) or {}).get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        box = _to_cxcywh(bb)
        if box is None or box[2] * box[3] < MIN_HF_AREA:
            continue
        if drop_hf_strips and box[2] > 0.9:  # full-width strip flush to a page edge = placeholder
            y0, y1 = box[1] - box[3] / 2, box[1] + box[3] / 2
            if (field == "header" and y0 < 0.02) or (field == "footer" and y1 > 0.98):
                continue
        lid = LABEL2ID[lab]
        if any(cls[i] == lid and _iou(boxes[i], box) > 0.5 for i in range(len(boxes))):
            continue
        cls.append(lid)
        boxes.append(box)
        order.append(ro)
    return boxes, cls, order
