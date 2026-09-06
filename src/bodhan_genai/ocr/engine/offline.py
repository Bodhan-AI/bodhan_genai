"""The two stages, and the pipeline that runs both.

    IndicDocLayout   page image   -> PageResult (blocks, no text)
    IndicBlockOCR    image+layout -> PageResult (blocks with text, plus markdown)
    IndicOCR   page image   -> both

Heavy imports live inside methods, so importing this module stays free.
"""

from __future__ import annotations

import json
import os

from bodhan_genai.ocr.engine.blocks import resolve_nested_equations
from bodhan_genai.ocr.engine.reconstruct import reconstruct
from bodhan_genai.ocr.engine.types import (
    CropConfig,
    DedupConfig,
    LayoutConfig,
    PageResult,
    RecognizerConfig,
)
from bodhan_genai.ocr.templates.contract import is_transcribed


def _open(image_path: str):
    from PIL import Image

    # Large scans and newspapers legitimately exceed PIL's decompression-bomb guard; layout
    # resizes to img_size and crops are area-clamped, so compute stays bounded regardless.
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(image_path).convert("RGB")


def _as_page(layout: PageResult | dict | str) -> PageResult:
    if isinstance(layout, PageResult):
        return layout
    if isinstance(layout, str):
        with open(layout, encoding="utf-8") as fh:
            layout = json.load(fh)
    return PageResult.from_record(layout)


class IndicDocLayout:
    """Stage 1 -- layout and reading order. Loads torch only, never vLLM."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: LayoutConfig | None = None,
        dedup: DedupConfig | None = None,
        backend=None,
    ) -> None:
        if backend is None:
            from bodhan_genai.ocr.engine.layout import IndicDocLayoutBackend

            backend = IndicDocLayoutBackend(ckpt, config, dedup)
        self.backend = backend

    def detect(self, image_path: str) -> PageResult:
        image = _open(image_path)
        return PageResult(
            image=os.path.basename(image_path),
            width=image.width,
            height=image.height,
            blocks=self.backend.detect(image),
        )

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> IndicDocLayout:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class IndicBlockOCR:
    """Stage 2 -- per-block transcription against a layout, which may be your own."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
        backend=None,
    ) -> None:
        self.config = config or RecognizerConfig()
        self.dedup = dedup or DedupConfig()
        self.crop = crop or CropConfig()
        if backend is None:
            from bodhan_genai.ocr.engine.recognizer_vllm import VllmRecognizer

            backend = VllmRecognizer(ckpt, self.config)
        self.backend = backend

    def run(self, image_path: str, layout: PageResult | dict | str) -> PageResult:
        """Every block of the layout comes back, in its original order. Blocks that were not
        transcribed carry ``text: ""`` rather than being dropped."""
        from bodhan_genai.ocr.engine.recognizer import build_requests

        image = _open(image_path)
        page = _as_page(layout)
        blocks = [b.copy() for b in page.blocks]

        eligible = resolve_nested_equations(
            [b for b in blocks if is_transcribed(b.label)], self.dedup
        )
        requests, orders = build_requests(eligible, image, self.crop, self.config.table_format)
        texts = self.backend.transcribe(requests)
        if len(texts) != len(orders):
            raise RuntimeError(
                f"recognizer returned {len(texts)} transcriptions for {len(orders)} crops; "
                "results would be misaligned"
            )

        by_order = dict(zip(orders, texts, strict=True))
        for block in blocks:
            block.text = (by_order.get(block.order) or "").strip()

        return PageResult(
            image=page.image,
            width=page.width,
            height=page.height,
            blocks=blocks,
            markdown=reconstruct(blocks),
        )

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> IndicBlockOCR:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class IndicOCR:
    """Both stages in one process."""

    def __init__(
        self,
        layout_ckpt: str | None = None,
        recognizer_ckpt: str | None = None,
        layout_config: LayoutConfig | None = None,
        recognizer_config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
    ) -> None:
        # vLLM FIRST. Its EngineCore forks/spawns at construction and must initialise CUDA
        # before the torch layout model touches the device; reversed, the child cannot re-init.
        self.ocr = IndicBlockOCR(recognizer_ckpt, recognizer_config, dedup, crop)
        self.layout = IndicDocLayout(layout_ckpt, layout_config, dedup)

    def detect(self, image_path: str) -> PageResult:
        return self.layout.detect(image_path)

    def parse(self, image_path: str) -> PageResult:
        return self.ocr.run(image_path, self.layout.detect(image_path))

    def close(self) -> None:
        self.ocr.close()
        self.layout.close()

    def __enter__(self) -> IndicOCR:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
