"""Corpus summary over a layout manifest: what is actually in the training mix.

The taxonomy has 37 classes and the mix has several sources with partial annotation —
magazine furniture exists only in one source, handwritten sets have no ``Code`` blocks,
and so on. A class that turns out to have forty examples across the whole corpus will
train to noise, and the only way to find that out before a multi-day run is to count.

Reads the page JSONs through the same :func:`labels_from_doc` the packer and the trainer
use, so the counts describe what training will actually see rather than what the raw
annotation files contain.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bodhan_genai.ocr.data.taxonomy import CLASSES, ID2LABEL, labels_from_doc

logger = logging.getLogger(__name__)


@dataclass
class CorpusSummary:
    pages: int = 0
    unreadable: int = 0
    empty_pages: int = 0
    blocks: int = 0
    by_source: Counter = field(default_factory=Counter)
    by_domain: Counter = field(default_factory=Counter)
    by_label: Counter = field(default_factory=Counter)
    blocks_per_page: list[int] = field(default_factory=list)

    @property
    def absent_labels(self) -> list[str]:
        """Classes with no examples at all — they cannot be learned from this mix."""
        return [c for c in CLASSES if self.by_label[c] == 0]

    def as_dict(self) -> dict:
        counts = sorted(self.blocks_per_page)
        median = counts[len(counts) // 2] if counts else 0
        return {
            "pages": self.pages,
            "unreadable": self.unreadable,
            "empty_pages": self.empty_pages,
            "blocks": self.blocks,
            "median_blocks_per_page": median,
            "by_source": dict(self.by_source),
            "by_domain": dict(self.by_domain),
            "by_label": {c: self.by_label[c] for c in CLASSES},
            "absent_labels": self.absent_labels,
        }


def summarize(
    manifest_path: str | Path, *, drop_header_footer_strips: bool = False, limit: int | None = None
) -> CorpusSummary:
    pages = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["pages"]
    if limit:
        pages = pages[:limit]

    out = CorpusSummary()
    for page in pages:
        path = Path(page["src"]) / "jsons" / f"{page['stem']}.json"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            out.unreadable += 1
            logger.debug("unreadable %s: %s", path, exc)
            continue
        _, classes, _ = labels_from_doc(doc, drop_header_footer_strips=drop_header_footer_strips)
        out.pages += 1
        out.by_source[page["source"]] += 1
        out.by_domain[page["domain"]] += 1
        out.blocks += len(classes)
        out.blocks_per_page.append(len(classes))
        if not classes:
            out.empty_pages += 1
        for class_id in classes:
            out.by_label[ID2LABEL[int(class_id)]] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.data.summarize",
        description="Count what a layout manifest actually contains.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None, help="write the summary as JSON here")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--drop-header-footer-strips", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = summarize(
        args.manifest,
        drop_header_footer_strips=args.drop_header_footer_strips,
        limit=args.limit,
    )
    payload = summary.as_dict()
    print(json.dumps(payload, indent=2))
    if summary.absent_labels:
        logger.warning(
            "%d class(es) have no examples in this manifest: %s",
            len(summary.absent_labels),
            summary.absent_labels,
        )
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
