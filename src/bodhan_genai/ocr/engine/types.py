"""Plain-data types and configuration.

Every tunable that used to be an environment variable read at import time lives here instead.
That fixes a real defect: callers corrected the old module-level defaults by setting os.environ
*before* importing the engine, two shipped callers set different values, and importing it
directly gave a third pipeline. These defaults ARE the canonical recipe.

stdlib-only -- importable with no GPU stack and no PIL.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from bodhan_genai.ocr.templates.contract import TableFormat


@dataclass
class Block:
    """One detected region. ``text`` is None before OCR, and "" for blocks deliberately not
    transcribed (kept in place rather than deleted)."""

    order: int
    label: str
    type: str
    bbox_xyxy: list[float]
    conf: float
    text: str | None = None

    def as_record(self) -> dict:
        # Key order is load-bearing: json.dump writes insertion order and the regression gate
        # compares byte for byte. Do not reorder.
        record: dict = {
            "order": self.order,
            "label": self.label,
            "type": self.type,
            "bbox_xyxy": [round(float(v), 1) for v in self.bbox_xyxy],
            "conf": round(float(self.conf), 3),
        }
        if self.text is not None:
            record["text"] = self.text
        return record

    @classmethod
    def from_record(cls, record: dict) -> Block:
        # `type` is derived when absent, so a third-party layout carrying only labels works.
        from bodhan_genai.ocr.templates.contract import map_label

        label = record.get("label", "")
        return cls(
            order=int(record["order"]),
            label=str(label),
            type=str(record.get("type") or map_label(label)),
            bbox_xyxy=[float(v) for v in record["bbox_xyxy"]],
            conf=float(record.get("conf", 1.0)),
            text=record.get("text"),
        )

    def copy(self) -> Block:
        return dataclasses.replace(self, bbox_xyxy=list(self.bbox_xyxy))


@dataclass
class PageResult:
    image: str
    width: int
    height: int
    blocks: list[Block] = field(default_factory=list)
    markdown: str | None = None

    def as_record(self) -> dict:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "blocks": [b.as_record() for b in self.blocks],
        }

    @classmethod
    def from_record(cls, record: dict) -> PageResult:
        return cls(
            image=str(record["image"]),
            width=int(record["width"]),
            height=int(record["height"]),
            blocks=[Block.from_record(b) for b in record.get("blocks", [])],
        )


@dataclass(frozen=True)
class LayoutConfig:
    conf: float = 0.5  # below ~0.4, stains and page borders start scoring as blocks
    img_size: int = 1024
    device: str = "cuda"


@dataclass(frozen=True)
class CropConfig:
    """The clamp is on area, not on a side: pinning a side exploded elongated crops (a 122:1
    rule line became ~32k image tokens and wedged the engine)."""

    min_px_side: int = 256  # largest single measured win; 0 disables upscaling
    max_px_side: int = 1536  # token ceiling
    pad_px: int = 0

    @property
    def min_px(self) -> int:
        return self.min_px_side**2

    @property
    def max_px(self) -> int:
        return self.max_px_side**2


#: both -- text-like OR a larger Equation | text_only -- text-like only | eq_only -- Equation only
DEDUP_MODES = ("both", "text_only", "eq_only")


@dataclass(frozen=True)
class DedupConfig:
    """IndicDocLayout over-produces equation boxes nested inside the paragraphs and display
    arrays that already contain them; transcribing both emits the same math twice."""

    nest: bool = True
    mode: str = "both"
    contain: float = 0.90  # duplicate threshold in clean_layout
    wrap: float = 0.5  # a header counts as occupied at this containment
    nested: float = 0.70  # nested-equation threshold

    def __post_init__(self) -> None:
        if self.mode not in DEDUP_MODES:
            raise ValueError(f"DedupConfig.mode must be one of {DEDUP_MODES}, got {self.mode!r}")


@dataclass(frozen=True)
class RecognizerConfig:
    max_model_len: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.0  # greedy: the only setting reproducible run to run
    gpu_memory_utilization: float = 0.80
    dtype: str = "bfloat16"
    # One giant batch over tens of thousands of multi-modal requests wedges the vLLM V1
    # scheduler at 100% util with no progress; ~2k chunks run clean.
    batch_size: int = 2048
    enforce_eager: bool = True  # skips ~4 min of torch.compile on a 0.8B model
    table_format: TableFormat = TableFormat.HTML

    def merged(self, **overrides) -> RecognizerConfig:
        """New config with non-None overrides applied; unknown keys raise."""
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"Unknown RecognizerConfig field(s): {', '.join(unknown)}")
        return dataclasses.replace(self, **{k: v for k, v in overrides.items() if v is not None})
