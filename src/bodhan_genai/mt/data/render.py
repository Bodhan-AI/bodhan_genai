"""``python -m bodhan_genai.mt.data.render`` — bitext JSONL -> instruction JSONL.

Turns parallel text into the chat rows the trainer consumes. One input row can
yield two output rows (forward and reverse direction); each gets an instruction
phrasing drawn from :mod:`bodhan_genai.mt.templates.variants` with a **seeded**
RNG, so the model sees paraphrase diversity rather than one memorised string.

Output schema, one JSON object per line::

    {"messages": [{"role": "user",      "content": "Translate the following text into Hindi:\\n\\n..."},
                  {"role": "assistant", "content": "..."}],
     "corpus": "bpcc", "direction": "eng_Latn-hin_Deva",
     "src_lang": "eng_Latn", "tgt_lang": "hin_Deva",
     "src_name": "English", "tgt_name": "Hindi",
     "template_variant": "target_only", "template_id": 6}

Only ``messages`` is used for training; the rest is provenance, and it is what
makes a per-language or per-direction slice of the corpus possible afterwards.

**Ordering note:** do any upsampling or corpus mixing *before* this stage. The RNG
advances per row, so N copies of a row arriving here get N different phrasings
(useful augmentation); copies made after rendering would all share one phrasing.

Usage
-----
    python -m bodhan_genai.mt.data.render --config configs/mt/data/render.yaml
    python -m bodhan_genai.mt.data.render --config … --dry-run   # counts only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from bodhan_genai.mt.templates.variants import (
    choose_template,
    display_name,
    get_templates,
    render_instruction,
)

logger = logging.getLogger("mt.data.render")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class SourceConfig:
    """One parallel corpus and how to read a direction out of it."""

    path: str
    src_field: str
    tgt_field: str
    src_lang: str
    tgt_lang: str
    name: str = ""
    #: Fraction of rows ALSO emitted in the reverse direction (tgt -> src).
    #: Bitext is typically stored one-way; 1.0 makes the corpus fully bidirectional.
    reverse_fraction: float = 0.0
    #: Stop after this many input rows (None = all). Useful for smoke runs.
    limit: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.reverse_fraction <= 1.0:
            raise ValueError(
                f"sources[{self.name or self.path}].reverse_fraction must be in "
                f"[0, 1], got {self.reverse_fraction}"
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("sources[].limit must be > 0 or null")


@dataclass
class OutputConfig:
    train: str
    dev: str | None = None
    #: Fraction of rendered rows held out for the dev set the trainer evaluates on.
    dev_fraction: float = 0.01
    #: Hard cap on the dev set — eval runs every few hundred steps, so a dev set
    #: of 1 % of a 3 M-row corpus would dominate wall-clock for no extra signal.
    dev_max_rows: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 <= self.dev_fraction < 1.0:
            raise ValueError(f"output.dev_fraction must be in [0, 1), got {self.dev_fraction}")


@dataclass
class RenderConfig:
    sources: list[SourceConfig]
    output: OutputConfig
    seed: int = 42
    template_variant: str = "target_only"
    #: Language codes outside the served set, e.g. {"xyz_Deva": "Some Language"}.
    extra_languages: dict[str, str] = field(default_factory=dict)
    #: Drop exact duplicate (source, target) pairs across the whole render.
    dedup: bool = True


def load_config(config_path: str) -> RenderConfig:
    """Load a render YAML into :class:`RenderConfig`, failing loudly on typos."""
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    known = {"sources", "output", "seed", "template_variant", "extra_languages", "dedup"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{config_path}: unknown top-level key(s): {', '.join(unknown)}")

    if not raw.get("sources"):
        raise ValueError(f"{config_path}: at least one entry under `sources:` is required")
    if not raw.get("output", {}).get("train"):
        raise ValueError(f"{config_path}: `output.train:` is required")

    sources = [SourceConfig(**s) for s in raw["sources"]]
    variant = str(raw.get("template_variant", "target_only"))
    get_templates(variant)  # validate now, not per row

    return RenderConfig(
        sources=sources,
        output=OutputConfig(**raw["output"]),
        seed=int(raw.get("seed", 42)),
        template_variant=variant,
        extra_languages=dict(raw.get("extra_languages") or {}),
        dedup=bool(raw.get("dedup", True)),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _stable_seed(*parts: object) -> int:
    """Deterministic seed from arbitrary parts.

    ``hash()`` is salted per process for strings, so a render would not be
    reproducible across runs if we used it here.
    """
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _iter_jsonl(path: str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def render_row(
    src_text: str,
    tgt_text: str,
    *,
    src_lang: str,
    tgt_lang: str,
    corpus: str,
    variant: str,
    rng: random.Random,
    extra_languages: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render one directed pair into a chat row."""
    src_name = display_name(src_lang, extra_languages)
    tgt_name = display_name(tgt_lang, extra_languages)
    template_id = choose_template(rng, variant)
    instruction = render_instruction(
        get_templates(variant)[template_id], tgt_name, src_text, src_name
    )
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": tgt_text},
        ],
        "corpus": corpus,
        "direction": f"{src_lang}-{tgt_lang}",
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_name": src_name,
        "tgt_name": tgt_name,
        "template_variant": variant,
        "template_id": template_id,
    }


def _dedup_key(src_lang: str, tgt_lang: str, src_text: str, tgt_text: str) -> bytes:
    """Key identifying a directed pair.

    Keyed on the RAW texts, not the rendered instruction: each copy of a repeated
    row draws its own phrasing, so rendered instructions differ and would never
    compare equal. Direction is part of the key because en->hi and hi->en of the
    same pair are two legitimate training rows.
    """
    payload = "\x1f".join((src_lang, tgt_lang, src_text, tgt_text)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def render_source(
    source: SourceConfig, cfg: RenderConfig, stats: Counter
) -> Iterator[tuple[bytes, dict[str, Any]]]:
    """Render every usable row of one corpus, forward and (optionally) reverse.

    Yields ``(dedup_key, row)`` so :func:`render_all` can de-duplicate on the raw
    pair without the key having to live in the output schema.
    """
    corpus = source.name or Path(source.path).stem
    fwd_rng = random.Random(_stable_seed(cfg.seed, corpus, source.src_lang, source.tgt_lang))
    rev_rng = random.Random(_stable_seed(cfg.seed, corpus, source.tgt_lang, source.src_lang))
    # Independent of the template RNGs, so changing the phrasing bank does not
    # reshuffle which rows are reversed.
    pick_rng = random.Random(_stable_seed(cfg.seed, corpus, "reverse-pick"))

    for row in _iter_jsonl(source.path, source.limit):
        stats[f"{corpus}:read"] += 1
        src_text = (row.get(source.src_field) or "").strip()
        tgt_text = (row.get(source.tgt_field) or "").strip()
        if not src_text or not tgt_text:
            stats[f"{corpus}:empty"] += 1
            continue

        yield (
            _dedup_key(source.src_lang, source.tgt_lang, src_text, tgt_text),
            render_row(
                src_text,
                tgt_text,
                src_lang=source.src_lang,
                tgt_lang=source.tgt_lang,
                corpus=corpus,
                variant=cfg.template_variant,
                rng=fwd_rng,
                extra_languages=cfg.extra_languages,
            ),
        )
        stats[f"{corpus}:{source.src_lang}-{source.tgt_lang}"] += 1

        if source.reverse_fraction and pick_rng.random() < source.reverse_fraction:
            yield (
                _dedup_key(source.tgt_lang, source.src_lang, tgt_text, src_text),
                render_row(
                    tgt_text,
                    src_text,
                    src_lang=source.tgt_lang,
                    tgt_lang=source.src_lang,
                    corpus=corpus,
                    variant=cfg.template_variant,
                    rng=rev_rng,
                    extra_languages=cfg.extra_languages,
                ),
            )
            stats[f"{corpus}:{source.tgt_lang}-{source.src_lang}"] += 1


def render_all(cfg: RenderConfig, stats: Counter) -> Iterator[dict[str, Any]]:
    """Render every configured source, de-duplicating across corpora if asked.

    De-duplication is global, not per-source: the same pair appearing in two
    corpora is still one training example.
    """
    seen: set[bytes] = set()
    for source in cfg.sources:
        for key, row in render_source(source, cfg, stats):
            if cfg.dedup:
                if key in seen:
                    stats["dropped:duplicate"] += 1
                    continue
                seen.add(key)
            yield row


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.data.render",
        description="Render bitext JSONL into instruction chat rows for finetuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="render config YAML")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="count rows and report the mix without writing anything",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    stats: Counter = Counter()

    # Held-out selection is seeded and row-independent, so re-running the render
    # puts the same rows in dev.
    dev_rng = random.Random(_stable_seed(cfg.seed, "dev-split"))
    want_dev = cfg.output.dev and cfg.output.dev_fraction > 0

    train_rows = dev_rows = 0
    if args.dry_run:
        for _ in render_all(cfg, stats):
            train_rows += 1
        print(f"[dry-run] {train_rows} rows would be rendered")
    else:
        train_path = Path(cfg.output.train)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        dev_path = Path(cfg.output.dev) if want_dev else None
        if dev_path:
            dev_path.parent.mkdir(parents=True, exist_ok=True)

        # One streaming pass writing both splits, so a multi-million-row corpus
        # never has to be held in memory or read twice.
        with ExitStack() as stack:
            train_fh = stack.enter_context(open(train_path, "w", encoding="utf-8"))
            dev_fh = (
                stack.enter_context(open(dev_path, "w", encoding="utf-8")) if dev_path else None
            )
            for row in render_all(cfg, stats):
                line = json.dumps(row, ensure_ascii=False) + "\n"
                to_dev = (
                    dev_fh is not None
                    and dev_rows < cfg.output.dev_max_rows
                    and dev_rng.random() < cfg.output.dev_fraction
                )
                if to_dev:
                    dev_fh.write(line)
                    dev_rows += 1
                else:
                    train_fh.write(line)
                    train_rows += 1

        logger.info("wrote %d train rows -> %s", train_rows, train_path)
        if dev_path:
            logger.info("wrote %d dev rows -> %s", dev_rows, dev_path)

    for key in sorted(stats):
        logger.info("  %-48s %d", key, stats[key])

    if train_rows == 0:
        print("no rows rendered — check the field names in the config", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
