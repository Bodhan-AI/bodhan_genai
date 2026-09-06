#!/usr/bin/env python3
"""Render a layout JSON over its page image, drawing each block's box and reading-order label.

Reads the <name>.layout.json written by the `layout` subcommand, so no model is loaded and no
GPU is needed. Pages smaller than TARGET on the long side are upscaled first, keeping the
annotations legible when the render is zoomed.

Usage:  python examples/ocr/viz_layout.py <*.layout.json> <image_dir> <out_dir>
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

TARGET = 2200  # minimum long side of the render

# One colour per label family: body text in blues, headings in reds, math in orange, tables in
# purple, visuals in gold, and page marginalia in green.
PALETTE = {
    "Paragraph": (33, 102, 172),
    "List": (33, 145, 140),
    "Infobox": (70, 130, 180),
    "Placeholder-text": (120, 144, 156),
    "Code": (84, 110, 122),
    "Title": (178, 24, 43),
    "Chapter-title": (178, 24, 43),
    "Section-title": (214, 96, 77),
    "Sub-section-title": (230, 140, 90),
    "Sub-sub-section-title": (240, 170, 120),
    "Question": (106, 61, 154),
    "Answer": (60, 140, 80),
    "MCQ": (140, 100, 180),
    "Solved-example": (90, 160, 110),
    "Equation": (230, 97, 1),
    "Expression": (240, 140, 40),
    "Table": (118, 42, 131),
    "Table-caption": (150, 90, 160),
    "Table-of-contents": (130, 60, 140),
    "Image": (200, 160, 20),
    "Diagram": (200, 140, 20),
    "Chart": (210, 170, 30),
    "Image-caption": (170, 140, 60),
    "Footnote": (130, 130, 130),
    "Reference": (120, 120, 120),
    "Header": (27, 158, 119),
    "Footer": (27, 158, 119),
    "Page-number": (27, 158, 119),
    "Folio": (27, 158, 119),
}
DEFAULT_COLOUR = (120, 120, 120)


def font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(layout_path, image_dir, out_dir):
    with open(layout_path, encoding="utf-8") as fh:
        layout = json.load(fh)
    img = Image.open(os.path.join(image_dir, layout["image"])).convert("RGB")

    scale = max(1.0, TARGET / max(img.size))
    if scale > 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(22, W // 46)
    f = font(size)
    pad = max(6, size // 4)
    width = max(4, W // 380)

    for b in sorted(layout["blocks"], key=lambda x: x["order"]):
        x0, y0, x1, y1 = (c * scale for c in b["bbox_xyxy"])
        colour = PALETTE.get(b["label"], DEFAULT_COLOUR)
        draw.rectangle([x0, y0, x1, y1], outline=(*colour, 255), width=width, fill=(*colour, 20))

        tag = f"{b['order']} · {b['label']}"
        tb = draw.textbbox((0, 0), tag, font=f, anchor="la")
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bx = min(max(0, x0), W - tw - 2 * pad - 2)  # keep the label inside the page
        by = min(max(0, y0), H - th - 2 * pad - 2)
        draw.rounded_rectangle(
            [bx, by, bx + tw + 2 * pad, by + th + 2 * pad],
            radius=pad,
            fill=(*colour, 255),
            outline=(255, 255, 255, 230),
            width=2,
        )
        draw.text((bx + pad, by + pad), tag, fill=(255, 255, 255, 255), font=f, anchor="la")

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    name = os.path.splitext(layout["image"])[0]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    out.save(path)
    print(f"{name}: {len(layout['blocks'])} blocks -> {path}")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3])
