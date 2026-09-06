"""Turning boxes into the images the recognizer sees.

Split from ``engine.blocks`` because this is the only part that needs PIL -- keeping the
geometry PIL-free is what lets the cleanup rules be tested without an image library.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from bodhan_genai.ocr.engine.types import Block, CropConfig
from bodhan_genai.ocr.templates.contract import DROP_TYPES, is_transcribed

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image


def area_clamp(image: Image, cfg: CropConfig | None = None) -> Image:
    """Scale a crop so its area lands in ``[cfg.min_px, cfg.max_px]``, preserving aspect.

    Area rather than a side: pinning a side exploded elongated crops (a 122:1 rule line became
    ~32k image tokens and wedged the engine).
    """
    from PIL import Image as PILImage

    cfg = cfg or CropConfig()
    width, height = image.size
    pixels = width * height
    if pixels <= 0:
        return image

    if pixels < cfg.min_px:
        scale = math.sqrt(cfg.min_px / pixels)
    elif pixels > cfg.max_px:
        scale = math.sqrt(cfg.max_px / pixels)
    else:
        return image

    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), PILImage.LANCZOS
    )


def crop_for(block: Block, page: Image, cfg: CropConfig | None = None) -> Image | None:
    """The image for one block, or None if it should not be transcribed."""
    if block.type in DROP_TYPES or not is_transcribed(block.label):
        return None

    cfg = cfg or CropConfig()
    width, height = page.size
    x0, y0, x1, y1 = (round(v) for v in block.bbox_xyxy)
    if cfg.pad_px:  # recover glyph edges a tight box clips
        x0, y0 = x0 - cfg.pad_px, y0 - cfg.pad_px
        x1, y1 = x1 + cfg.pad_px, y1 + cfg.pad_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)

    if x1 <= x0 or y1 <= y0:  # rounding can collapse a thin rule to zero width
        return None
    return page.crop((x0, y0, x1, y1)).convert("RGB")
