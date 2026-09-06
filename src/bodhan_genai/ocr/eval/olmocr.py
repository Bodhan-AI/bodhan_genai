"""End-to-end page transcription against the olmOCR benchmark.

    python -m bodhan_genai.ocr.eval.olmocr --pages bench/pdfs --out runs/olmocr

**The published number was not measured with the shipped defaults.** IndicOCR's
recorded olmOCR score of 82.9 came from ``dedup_mode="text_only"`` and Markdown tables,
where the shipped defaults are ``dedup_mode="both"`` and HTML tables. Both defaults are
deliberate — HTML is the only way to express a merged cell, and ``both`` drops equations
nested inside paragraphs — but the benchmark's references are flat Markdown, so scoring
against them with the defaults measures a formatting mismatch rather than transcription
quality.

So the reproduction settings are the ones this module defaults to, and every run writes
them into ``settings.json`` beside the predictions. A number without the settings that
produced it is not reproducible, which is exactly the gap this closes.

Scoring itself is delegated to the official ``olmocr`` package when it is installed —
re-implementing someone else's benchmark metric is how you end up with a number that is
not comparable to theirs. Without it, predictions are still written and the command to
score them is printed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class BenchSettings:
    """The settings the published score was measured with."""

    dedup_mode: str = "text_only"
    table_format: str = "markdown"
    confidence: float = 0.5
    max_tokens: int = 2048
    gpu_memory_utilization: float = 0.85
    note: str = (
        "Non-default on purpose: the shipped defaults are dedup_mode='both' and "
        "table_format='html'. The olmOCR references are flat Markdown."
    )


def find_pages(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in PAGE_SUFFIXES)


def predict(
    pages_dir: str | Path,
    out_dir: str | Path,
    *,
    layout_ckpt: str | None = None,
    recognizer_ckpt: str | None = None,
    settings: BenchSettings | None = None,
    limit: int | None = None,
) -> dict:
    """Transcribe every page and write one ``.md`` per page plus ``settings.json``."""
    from bodhan_genai.ocr import DedupConfig, IndicOCR, RecognizerConfig, TableFormat

    settings = settings or BenchSettings()
    out_dir = Path(out_dir)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)

    pages = find_pages(pages_dir)
    if limit:
        pages = pages[:limit]
    if not pages:
        raise FileNotFoundError(f"no page images under {pages_dir}")
    logger.info("transcribing %d page(s) with %s", len(pages), asdict(settings))

    parser = IndicOCR(
        layout_ckpt=layout_ckpt,
        recognizer_ckpt=recognizer_ckpt,
        dedup=DedupConfig(mode=settings.dedup_mode),
        recognizer_config=RecognizerConfig(
            table_format=TableFormat(settings.table_format),
            max_tokens=settings.max_tokens,
            gpu_memory_utilization=settings.gpu_memory_utilization,
        ),
    )

    empty, blocks, started = 0, 0, time.perf_counter()
    try:
        for page in pages:
            result = parser.parse(str(page))
            markdown = result.markdown or ""
            blocks += len(result.blocks)
            if not markdown.strip():
                empty += 1
                logger.warning("empty transcription: %s", page.name)
            (out_dir / "pages" / f"{page.stem}.md").write_text(markdown, encoding="utf-8")
    finally:
        close = getattr(parser, "close", None)
        if callable(close):
            close()

    elapsed = time.perf_counter() - started
    report = {
        "pages": len(pages),
        "blocks": blocks,
        "empty_pages": empty,
        "seconds": round(elapsed, 1),
        "seconds_per_page": round(elapsed / len(pages), 2),
        "settings": asdict(settings),
    }
    (out_dir / "settings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "%d page(s) in %.1fs (%.2f s/page), %d empty",
        len(pages),
        elapsed,
        elapsed / len(pages),
        empty,
    )
    if empty:
        logger.warning(
            "%d page(s) transcribed to nothing. The benchmark scores an empty prediction "
            "as a miss, so this depresses the score as if it were a quality problem.",
            empty,
        )
    return report


def score(predictions_dir: str | Path, references_dir: str | Path) -> dict | None:
    """Score with the official olmOCR benchmark, if it is installed."""
    try:
        from olmocr.bench.runners import score_documents  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "the `olmocr` package is not installed, so predictions were written but not "
            "scored. Install it and run its bench over %s to get a comparable number.",
            predictions_dir,
        )
        return None
    return score_documents(str(predictions_dir), str(references_dir))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.eval.olmocr",
        description="Transcribe the olmOCR benchmark with the settings the published score used.",
    )
    parser.add_argument("--pages", required=True, help="directory of page images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--references", default=None, help="reference dir, to score as well")
    parser.add_argument("--layout-ckpt", default=None)
    parser.add_argument("--recognizer-ckpt", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shipped-defaults",
        action="store_true",
        help="use the shipped defaults (dedup both / HTML tables) instead of the "
        "reproduction settings — the score will NOT be comparable to the published one",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = (
        BenchSettings(dedup_mode="both", table_format="html", note="shipped defaults")
        if args.shipped_defaults
        else BenchSettings()
    )
    report = predict(
        args.pages,
        args.out,
        layout_ckpt=args.layout_ckpt,
        recognizer_ckpt=args.recognizer_ckpt,
        settings=settings,
        limit=args.limit,
    )
    if args.references:
        scored = score(Path(args.out) / "pages", args.references)
        if scored:
            report["score"] = scored
            (Path(args.out) / "settings.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
