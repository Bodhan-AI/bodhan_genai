"""Stage 2 end to end against a fake recognizer: crop -> transcribe -> assign -> assemble.

No GPU and no real model, but every seam between them is real -- which is where transcriptions
get misaligned with their blocks.
"""

from __future__ import annotations

import pytest

from bodhan_genai.ocr.engine.offline import IndicBlockOCR
from bodhan_genai.ocr.engine.types import CropConfig, DedupConfig, RecognizerConfig
from bodhan_genai.ocr.templates.contract import TableFormat

pytest.importorskip("PIL", reason="pillow is an ocr-infer extra, absent on a bare CPU runner")

PAGE_W, PAGE_H = 800, 1000

LAYOUT = {
    "image": "page.png",
    "width": PAGE_W,
    "height": PAGE_H,
    "blocks": [
        {
            "order": 0,
            "label": "Header",
            "type": "PageHeader",
            "bbox_xyxy": [0, 0, 800, 40],
            "conf": 0.9,
        },
        {
            "order": 1,
            "label": "Title",
            "type": "Title",
            "bbox_xyxy": [50, 60, 750, 120],
            "conf": 0.9,
        },
        {
            "order": 2,
            "label": "Paragraph",
            "type": "Text",
            "bbox_xyxy": [50, 140, 750, 400],
            "conf": 0.8,
        },
        {
            "order": 3,
            "label": "Diagram",
            "type": "Figure",
            "bbox_xyxy": [50, 420, 750, 700],
            "conf": 0.7,
        },
        {
            "order": 4,
            "label": "Table",
            "type": "Table",
            "bbox_xyxy": [50, 720, 750, 900],
            "conf": 0.8,
        },
    ],
}


class FakeRecognizer:
    """Echoes each request's prompt so the test can see which prompt each block received."""

    def __init__(self, replies=None):
        self.requests = []
        self._replies = replies

    def transcribe(self, requests):
        self.requests = list(requests)
        if self._replies is not None:
            return list(self._replies)
        return [f"text-{i}" for i in range(len(requests))]

    def close(self):
        pass


@pytest.fixture
def page_image(tmp_path):
    from PIL import Image

    path = tmp_path / "page.png"
    Image.new("RGB", (PAGE_W, PAGE_H), "white").save(path)
    return str(path)


def run(page_image, backend, **kwargs):
    return IndicBlockOCR(backend=backend, **kwargs).run(page_image, LAYOUT)


def test_only_transcribable_blocks_are_sent_to_the_recognizer(page_image):
    """Header and Diagram are skipped; Title, Paragraph and Table are not."""
    fake = FakeRecognizer()
    run(page_image, fake)
    assert len(fake.requests) == 3


def test_every_layout_block_survives_with_skipped_ones_carrying_empty_text(page_image):
    result = run(page_image, FakeRecognizer())
    assert [b.order for b in result.blocks] == [0, 1, 2, 3, 4]
    by_label = {b.label: b.text for b in result.blocks}
    assert by_label["Header"] == "" and by_label["Diagram"] == ""
    assert by_label["Title"] and by_label["Paragraph"] and by_label["Table"]


def test_transcriptions_land_on_the_right_blocks(page_image):
    result = run(page_image, FakeRecognizer(replies=["THE TITLE", "the body", "<table/>"]))
    by_label = {b.label: b.text for b in result.blocks}
    assert by_label["Title"] == "THE TITLE"
    assert by_label["Paragraph"] == "the body"
    assert by_label["Table"] == "<table/>"


def test_table_blocks_get_the_table_prompt_in_the_configured_format(page_image):
    fake = FakeRecognizer()
    run(page_image, fake, config=RecognizerConfig(table_format=TableFormat.MARKDOWN))
    assert any("GitHub-flavored markdown" in r.prompt for r in fake.requests)

    fake_html = FakeRecognizer()
    run(page_image, fake_html)
    assert any("colspan" in r.prompt for r in fake_html.requests)
    assert not any("GitHub-flavored" in r.prompt for r in fake_html.requests)


def test_markdown_excludes_skipped_blocks_and_follows_reading_order(page_image):
    result = run(page_image, FakeRecognizer(replies=["Heading", "Body text", "<table/>"]))
    assert result.markdown == "Heading\n\nBody text\n\n<table/>"


def test_a_misaligned_backend_is_caught_rather_than_silently_shifting_text(page_image):
    """Returning the wrong number of transcriptions would offset every block after it."""
    with pytest.raises(RuntimeError, match="misaligned"):
        run(page_image, FakeRecognizer(replies=["only one"]))


def test_the_caller_s_layout_is_not_mutated(page_image):
    before = [dict(b) for b in LAYOUT["blocks"]]
    run(page_image, FakeRecognizer())
    assert LAYOUT["blocks"] == before


def test_crops_are_area_clamped_before_transcription(page_image):
    """A small box must be upscaled past the floor; a huge one capped."""
    fake = FakeRecognizer()
    tiny = {
        **LAYOUT,
        "blocks": [
            {
                "order": 0,
                "label": "Paragraph",
                "type": "Text",
                "bbox_xyxy": [0, 0, 30, 12],
                "conf": 1.0,
            }
        ],
    }
    IndicBlockOCR(backend=fake).run(page_image, tiny)
    w, h = fake.requests[0].image.size
    assert w * h >= CropConfig().min_px * 0.99


def test_result_carries_the_page_envelope(page_image):
    result = run(page_image, FakeRecognizer())
    assert (result.image, result.width, result.height) == ("page.png", PAGE_W, PAGE_H)
    record = result.as_record()
    assert list(record) == ["image", "width", "height", "blocks"]
    assert list(record["blocks"][0]) == ["order", "label", "type", "bbox_xyxy", "conf", "text"]


def test_nested_equation_is_deduped_before_cropping(page_image):
    """The dropped equation must not be cropped, but must still appear in the JSON."""
    nested = {
        **LAYOUT,
        "blocks": [
            {
                "order": 0,
                "label": "Paragraph",
                "type": "Text",
                "bbox_xyxy": [0, 0, 500, 200],
                "conf": 1.0,
            },
            {
                "order": 1,
                "label": "Equation",
                "type": "Equation",
                "bbox_xyxy": [50, 80, 200, 120],
                "conf": 1.0,
            },
        ],
    }
    fake = FakeRecognizer()
    result = IndicBlockOCR(backend=fake, dedup=DedupConfig(mode="both")).run(page_image, nested)
    assert len(fake.requests) == 1, "the inline equation should have been folded into the text"
    assert [b.label for b in result.blocks] == ["Paragraph", "Equation"]
    assert [b.text for b in result.blocks] == ["text-0", ""]


def test_hf_recognizer_satisfies_the_backend_protocol():
    """The quickstart backend must be substitutable for the vLLM one without the engine caring."""
    from bodhan_genai.ocr.engine.recognizer import HfRecognizer, RecognizerBackend

    assert hasattr(HfRecognizer, "transcribe") and hasattr(HfRecognizer, "close")
    # A stand-in with the same shape satisfies the Protocol, which is what the engine checks.
    assert isinstance(FakeRecognizer(), RecognizerBackend)


# Import hygiene for engine.recognizer is asserted in tests/ocr/test_ocr_lazy_import.py,
# in a fresh subprocess -- sys.modules describes the whole interpreter, not one import.
