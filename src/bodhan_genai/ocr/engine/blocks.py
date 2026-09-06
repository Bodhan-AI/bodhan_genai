"""Layout cleanup: geometry, duplicate suppression, nested-equation resolution.

Pure geometry on Blocks -- no PIL, no torch, no page image -- so the rules that decide what gets
transcribed stay testable with none of the GPU stack installed.
"""

from __future__ import annotations

from bodhan_genai.ocr.engine.types import Block, DedupConfig
from bodhan_genai.ocr.templates.contract import HEAD_FOOT, MARGINALIA, is_transcribed

TEXTLIKE = frozenset({"Text", "Title", "SectionHeader", "Caption", "Footnote"})


def area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def contained_frac(small, big) -> float:
    """Fraction of ``small``'s area lying inside ``big``."""
    ix0, iy0 = max(small[0], big[0]), max(small[1], big[1])
    ix1, iy1 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    a = area(small)
    return inter / a if a > 0 else 0.0


def clamp_to_page(bbox, width: int, height: int) -> list[float]:
    return [
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(float(width), bbox[2]),
        min(float(height), bbox[3]),
    ]


def clean_layout(blocks: list[Block], cfg: DedupConfig | None = None) -> list[Block]:
    """Drop duplicate and spurious boxes. Three rules; survivors keep input order.

    1. Nested duplicates -- a box ``cfg.contain`` inside a larger box of the same group goes.
    2. One header, one footer -- only the largest of each survives.
    3. Empty frames -- a header/footer wrapping nothing is dropped.
    """
    cfg = cfg or DedupConfig()
    n = len(blocks)
    box = lambda i: blocks[i].bbox_xyxy  # noqa: E731
    drop: set[int] = set()

    # Header/Footer are in neither group: rules 2 and 3 govern them entirely.
    groups = (
        lambda label: label not in MARGINALIA,
        lambda label: label in MARGINALIA and label not in HEAD_FOOT,
    )
    for in_group in groups:
        idxs = sorted(
            (i for i in range(n) if in_group(blocks[i].label)),
            key=lambda i: area(box(i)),
            reverse=True,
        )
        kept: list[int] = []
        for i in idxs:
            has_text = is_transcribed(blocks[i].label)
            # A block is never absorbed into a container that will not itself be transcribed:
            # otherwise a caption inside a Diagram is dropped for a container that is then
            # skipped, losing the text. One-directional -- a non-transcribed box may go anywhere.
            if any(
                contained_frac(box(i), box(j)) >= cfg.contain
                and (is_transcribed(blocks[j].label) or not has_text)
                for j in kept
            ):
                drop.add(i)
            else:
                kept.append(i)

    for label in HEAD_FOOT:
        group = [i for i in range(n) if blocks[i].label == label and i not in drop]
        if not group:
            continue
        biggest = max(group, key=lambda i: area(box(i)))
        drop.update(i for i in group if i != biggest)
        # Measured over range(n), NOT the survivors: a header whose only occupant was already
        # removed as a nested duplicate is still occupied, and deleting it loses a real header.
        wraps = any(
            blocks[k].label != label and contained_frac(box(k), box(biggest)) >= cfg.wrap
            for k in range(n)
        )
        if not wraps:
            drop.add(biggest)

    return [blocks[i] for i in range(n) if i not in drop]


def _nested_in(inner, outer, thresh: float) -> bool:
    # The strictness check stops two boxes over the same region each claiming to contain the
    # other, which would drop both.
    inner_area = area(inner)
    if inner_area <= 0:
        return False
    return contained_frac(inner, outer) >= thresh and inner_area < 0.95 * area(outer)


def _absorbs_equation(container: Block, mode: str) -> bool:
    if container.type == "Equation":
        # A display array is one outer box plus a box per row; collapsing keeps it one block.
        return mode != "text_only"
    if container.type in TEXTLIKE:
        # The text prompt already renders inline math as $...$, so a separate equation box
        # would emit it a second time as display math.
        return mode != "eq_only"
    return False


def resolve_nested_equations(blocks: list[Block], cfg: DedupConfig | None = None) -> list[Block]:
    """Drop equation boxes nested in a container that already covers them.

    Runs before cropping -- it needs only boxes and types, so resolving first avoids building
    crops that are immediately discarded.
    """
    cfg = cfg or DedupConfig()
    if not cfg.nest:
        return list(blocks)

    drop: set[int] = set()
    for i, inner in enumerate(blocks):
        if inner.type != "Equation":
            continue
        for j, container in enumerate(blocks):
            if (
                i != j
                and _nested_in(inner.bbox_xyxy, container.bbox_xyxy, cfg.nested)
                and _absorbs_equation(container, cfg.mode)
            ):
                drop.add(i)
                break
    return [b for k, b in enumerate(blocks) if k not in drop]
