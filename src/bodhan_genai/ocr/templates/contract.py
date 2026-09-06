"""IndicOCR's contract: prompts, block taxonomy, output schema.

Two vocabularies meet here and mixing them fails silently. **Labels** are what IndicDocLayout
emits (``ocr.layout.labels.CLASSES``); MARGINALIA and HEAD_FOOT match them case-SENSITIVELY.
**Types** are the coarse categories a label maps to, and select the prompt. OCR_SKIP_LABELS is
the exception: matched case-insensitively.

stdlib-only -- importable with no GPU stack and no PIL.
"""

from __future__ import annotations

from enum import StrEnum


class TableFormat(StrEnum):
    """HTML is the default: colspan/rowspan and in-cell breaks have no GFM spelling, so a
    merged-cell table rendered as markdown silently loses its structure."""

    HTML = "html"
    MARKDOWN = "markdown"


TEXT_PROMPT = (
    "Transcribe the text in this image. Write any mathematical expressions in LaTeX, "
    "using $...$ for inline math and $$...$$ for display equations."
)
EQUATION_PROMPT = "Output only the LaTeX for this equation image."
TABLE_PROMPTS = {
    TableFormat.HTML: (
        "Convert this table image to HTML. Preserve the structure exactly, using colspan and "
        "rowspan for merged cells and <br/> for line breaks within a cell. "
        "Output only the HTML table."
    ),
    TableFormat.MARKDOWN: (
        "Convert this table image to a GitHub-flavored markdown table. Output only the table."
    ),
}


def prompt_for(block_type: str, table_format: TableFormat = TableFormat.HTML) -> str:
    if block_type == "Table":
        return TABLE_PROMPTS[TableFormat(table_format)]
    if block_type == "Equation":
        return EQUATION_PROMPT
    return TEXT_PROMPT


# Labels absent from this map fall through to "Text" by design -- Question, Paragraph, Answer,
# List, MCQ, Code, Reference and the rest all carry prose.
LABEL_TO_TYPE = {
    "table": "Table",
    "table-caption": "Caption",
    "equation": "Equation",
    "expression": "Equation",
    "diagram": "Figure",
    "chart": "Figure",
    "image": "Picture",
    "image-caption": "Caption",
    "title": "Title",
    "chapter-title": "Title",
    "section-title": "SectionHeader",
    "sub-section-title": "SectionHeader",
    "sub-sub-section-title": "SectionHeader",
    "header": "PageHeader",
    "footer": "PageFooter",
    "page-number": "PageNumber",
    "folio": "PageNumber",
    "footnote": "Footnote",
}


def map_label(label) -> str:
    """Label -> pipeline type. Unknown labels are ``Text``."""
    return LABEL_TO_TYPE.get(str(label).strip().lower(), "Text")


# Exactly the types map_label can produce, less DROP_TYPES. test_contract asserts this, so a
# dead or undocumented type cannot creep in.
KEPT_BLOCK_TYPES = (
    "Text",
    "Title",
    "SectionHeader",
    "Table",
    "Equation",
    "Caption",
    "Footnote",
    "PageHeader",
    "PageFooter",
    "PageNumber",
)

# Never cropped, never reconstructed: pictorial regions have no text and invite hallucination.
# Deliberately narrow -- page numbers and margin text DO reach the recognizer.
DROP_TYPES = frozenset({"Figure", "Picture"})

# Cleaned as their own group so a page-spanning paragraph cannot swallow a page number.
MARGINALIA = frozenset({"Header", "Footer", "Page-number", "Folio"})
HEAD_FOOT = ("Header", "Footer")

# Never sent to the recognizer, but NOT deleted: these keep their place in the output with
# text "", so a consumer can still see what was detected and where.
OCR_SKIP_LABELS = frozenset({"header", "footer", "diagram", "image", "chart", "advertisement"})


def is_transcribed(label) -> bool:
    return str(label).strip().lower() not in OCR_SKIP_LABELS


OUTPUT_BLOCK_SCHEMA = {
    "order": "int    -- reading-order rank (0 = first)",
    "label": "str    -- raw IndicDocLayout label",
    "type": "str    -- pipeline type (see KEPT_BLOCK_TYPES)",
    "bbox_xyxy": "[float x4] -- pixel [x0, y0, x1, y1], clamped to the page",
    "conf": "float  -- detection confidence",
    "text": "str    -- transcription; '' when not sent to the recognizer",
}

OUTPUT_PAGE_SCHEMA = {
    "image": "str   -- source image filename",
    "width": "int   -- page width in pixels",
    "height": "int   -- page height in pixels",
    "blocks": "[BLOCK] -- in reading order",
}
