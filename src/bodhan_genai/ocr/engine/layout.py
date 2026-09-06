"""Layout backends: page image -> cleaned, reading-ordered blocks.

The two stages hand off a plain JSON layout, so stage 2 does not care where the layout came
from. :class:`LayoutBackend` makes that an interface rather than a claim.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bodhan_genai.ocr.engine.blocks import clamp_to_page, clean_layout
from bodhan_genai.ocr.engine.types import Block, DedupConfig, LayoutConfig, PageResult
from bodhan_genai.ocr.templates.contract import map_label

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


@runtime_checkable
class LayoutBackend(Protocol):
    """``detect`` must return blocks already cleaned and densely ordered: ``order`` a gap-free
    0-based rank. Stage 2 matches transcriptions back by ``order``, so gaps mis-assign text."""

    def detect(self, image: Image) -> list[Block]: ...

    def close(self) -> None: ...


def _densify(blocks: list[Block]) -> list[Block]:
    """Sort by the detector's reading order and renumber to a gap-free 0-based rank."""
    ordered = sorted(blocks, key=lambda b: b.order)
    for rank, block in enumerate(ordered):
        block.order = rank
    return ordered


class IndicDocLayoutBackend:
    """Our finetuned PP-DocLayoutV3 with an integrated reading-order head. Torch only --
    constructing this does not load vLLM, which is what lets stage 1 run alone."""

    def __init__(
        self,
        ckpt: str | None = None,
        config: LayoutConfig | None = None,
        dedup: DedupConfig | None = None,
    ) -> None:
        from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt
        from bodhan_genai.ocr.layout.infer import get_model

        self.config = config or LayoutConfig()
        self.dedup = dedup or DedupConfig()
        self.ckpt = resolve_ckpt("layout", ckpt)
        self.model = get_model(self.ckpt, device=self.config.device)

    def detect(self, image: Image) -> list[Block]:
        from bodhan_genai.ocr.layout.infer import infer

        width, height = image.size
        detections = infer(
            self.model,
            image,
            conf=self.config.conf,
            img_size=self.config.img_size,
            device=self.config.device,
        )

        # The model emits [y0, x0, y1, x1] normalised to 0-1000; the pipeline works in pixel
        # [x0, y0, x1, y1]. Axis swap and rescale happen here, once.
        blocks = []
        for det in detections:
            y0, x0, y1, x1 = det["bbox"]
            bbox = [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height]
            label = str(det["label"])
            blocks.append(
                Block(
                    order=det["reading_order"],
                    label=label,
                    type=map_label(label),
                    bbox_xyxy=[round(v, 1) for v in clamp_to_page(bbox, width, height)],
                    conf=round(float(det.get("score", 1.0)), 3),
                )
            )

        return _densify(clean_layout(blocks, self.dedup))

    def close(self) -> None:
        self.model = None


class JsonLayoutBackend:
    """Replay a layout produced elsewhere -- by stage 1, by hand, or by another detector.

    Assumed already clean, so no cleanup runs; blocks are only renumbered, which makes a
    hand-edited file usable without fixing ranks. Needs no torch.
    """

    def __init__(self, layout: str | dict | PageResult) -> None:
        if isinstance(layout, PageResult):
            self.page = layout
        else:
            if isinstance(layout, str):
                with open(layout, encoding="utf-8") as fh:
                    layout = json.load(fh)
            self.page = PageResult.from_record(layout)

    def detect(self, image: Image) -> list[Block]:
        """Copies, so renumbering cannot write back into the stored layout. ``image`` is
        accepted for interface parity and not read."""
        return _densify([b.copy() for b in self.page.blocks])

    def close(self) -> None:
        return None
