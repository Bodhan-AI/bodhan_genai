"""Romanized-translation and transliteration eval.

IN22-Gen carries no Roman references, so the existing harness cannot see any of
the model's Roman capability. This one scores four directions off a companion set
whose rows carry ``english`` / ``native`` / ``roman`` side by side, so one row
yields all four:

===============  ==========================  ===========================
direction        prompt                      metrics
===============  ==========================  ===========================
``en2roman``     translate -> Hindi (Roman)  BLEU + chrF++
``roman2en``     translate -> English        BLEU + chrF++
``xx2roman``     transliterate -> Hi (Roman) chrF++ + CER/WER + exact
``roman2xx``     transliterate -> Hindi      chrF++ + CER/WER + exact
===============  ==========================  ===========================

The split is deliberate. ``en2roman`` and ``roman2en`` are *translation* — the
content changes, so BLEU/chrF++ are the measures. ``xx2roman`` and ``roman2xx``
are *transliteration* — the content is fixed and only the script moves, so BLEU is
close to meaningless and character-level error rates carry the signal.

There is no code-mix direction here. The model has the capability, but no scored
harness exists for it, so any number would be unmeasured — see docs/mt/eval.md.

Usage::

    python -m bodhan_genai.mt.eval.roman --output-dir roman-results \\
        --url http://localhost:8000/v1 --langs hin_Deva tam_Taml --max-samples 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from bodhan_genai.mt.eval.metrics import (
    ErrorRate,
    PairScore,
    is_degenerate,
    score_error_rate,
    score_pair,
)
from bodhan_genai.mt.serving.client import DEFAULT_MODEL, MTClient

logger = logging.getLogger("mt.eval.roman")

#: The companion set: per-language configs whose rows carry english/native/roman.
DATASET = "bodhan-ai/IN22-GEN-Romanized-Transliterated"

#: One config per language. IN22-Gen carries a single code per language, so the
#: alternate scripts (kas_Deva, mni_Beng, snd_Arab) are absent by construction —
#: that is a property of the benchmark, not a gap to report as an error.
ROMAN_LANGS = (
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "gom_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
)

#: Translation directions, scored with BLEU/chrF++.
TRANSLATION_DIRECTIONS = ("en2roman", "roman2en")
#: Transliteration directions, scored with chrF++ and CER/WER.
TRANSLITERATION_DIRECTIONS = ("xx2roman", "roman2xx")
DIRECTIONS = TRANSLATION_DIRECTIONS + TRANSLITERATION_DIRECTIONS


def plan(direction: str, lang: str, row: dict[str, list[str]]) -> tuple[list[str], list[str], dict]:
    """Sources, references and the translate_batch kwargs for one direction."""
    english, native, roman = row["english"], row["native"], row["roman"]
    if direction == "en2roman":
        return english, roman, {"tgt_lang": lang, "script": "roman"}
    if direction == "roman2en":
        return roman, english, {"tgt_lang": "eng_Latn"}
    if direction == "xx2roman":
        return native, roman, {"tgt_lang": lang, "script": "roman", "task": "transliterate"}
    if direction == "roman2xx":
        return roman, native, {"tgt_lang": lang, "script": "native", "task": "transliterate"}
    raise ValueError(f"unknown direction {direction!r}. Supported: {', '.join(DIRECTIONS)}")


def load_config_split(dataset: str, lang: str, split: str, load_dataset):
    """Load one language config, from the Hub or from a local directory.

    A local directory is laid out one subdirectory per language, each holding the
    split's parquet files -- which is also how the Hub cache stores it. Supporting
    it means this eval runs on an air-gapped node, and against a private set
    without credentials, instead of failing on a 404.
    """
    root = Path(dataset)
    if root.is_dir():
        files = sorted(str(f) for f in (root / lang).glob(f"{split}-*.parquet"))
        if not files:
            files = sorted(str(f) for f in (root / lang).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split} parquet under {root / lang}")
        return load_dataset("parquet", data_files=files, split="train")
    return load_dataset(dataset, lang, split=split)


def run_direction(
    client: MTClient,
    direction: str,
    lang: str,
    row: dict[str, list[str]],
    *,
    num_workers: int,
    max_new_tokens: int,
) -> tuple[PairScore, ErrorRate | None, list[str], list[str], int, int]:
    """Generate and score one (language, direction) cell."""
    sources, references, kwargs = plan(direction, lang, row)
    results = client.translate_batch(
        sources, num_workers=num_workers, max_new_tokens=max_new_tokens, **kwargs
    )
    failed = [r for r in results if not r.ok]
    if failed:
        logger.warning("%s:%s — %d/%d errored", lang, direction, len(failed), len(results))
    truncated = sum(1 for r in results if r.truncated)
    if truncated:
        # A truncated transliteration is fluent and simply stops; it would
        # otherwise be scored as a quality problem rather than a budget one.
        logger.warning(
            "%s:%s — %d/%d TRUNCATED; raise --max-new-tokens",
            lang,
            direction,
            truncated,
            len(results),
        )

    predictions = [r.text for r in results]
    # A couple of runaway rows tank BLEU while barely moving chrF++, and they hit
    # whatever token budget they are given — so a stable truncated count across
    # two budgets means degeneration, not under-sizing. Counted, never dropped.
    degenerate = sum(
        is_degenerate(pred, src) for pred, src in zip(predictions, sources, strict=True)
    )
    if degenerate:
        logger.warning(
            "%s:%s — %d/%d DEGENERATE (looped or empty); raising --max-new-tokens will "
            "not fix these",
            lang,
            direction,
            degenerate,
            len(results),
        )
    score = score_pair(predictions, references)
    rate = None
    if direction in TRANSLITERATION_DIRECTIONS:
        rate = score_error_rate(predictions, references)
        logger.info(
            "%s:%s chrF++ %.2f  CER %.4f  WER %.4f",
            lang,
            direction,
            score.chrf,
            rate.cer,
            rate.wer,
        )
    else:
        logger.info("%s:%s BLEU %.2f  chrF++ %.2f", lang, direction, score.bleu, score.chrf)
    return score, rate, predictions, references, truncated, degenerate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.eval.roman",
        description="Score romanized translation and transliteration through an "
        "IndicTranslate server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", "-o", required=True)
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL)
    p.add_argument("--langs", nargs="*", help=f"default: all {len(ROMAN_LANGS)}")
    p.add_argument("--directions", nargs="*", default=list(DIRECTIONS), choices=DIRECTIONS)
    p.add_argument("--max-samples", type=int, default=0, help="0 = the whole split")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--split", default="test")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    client = MTClient(args.url, model=args.model, sampling=None)
    if not client.health():
        logger.error(
            "no server serving %r at %s — start one with scripts/mt/serve.sh",
            args.model,
            args.url,
        )
        return 1

    from datasets import load_dataset

    languages = args.langs or list(ROMAN_LANGS)
    unknown = [lang for lang in languages if lang not in ROMAN_LANGS]
    if unknown:
        p.error(f"no config for: {', '.join(unknown)}")

    # A Hub id names the folder after the repo; a local snapshot directory would
    # otherwise name it after a commit hash, which tells a reader nothing.
    label = Path(DATASET).name if Path(args.dataset).is_dir() else Path(args.dataset).name
    out_dir = Path(args.output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, PairScore] = {}
    rates: dict[str, ErrorRate] = {}
    predictions: dict[str, list[str]] = {}
    references: dict[str, list[str]] = {}
    truncations: dict[str, int] = {}
    degenerates: dict[str, int] = {}
    skipped: list[str] = []

    for lang in languages:
        try:
            ds = load_config_split(args.dataset, lang, args.split, load_dataset)
        except Exception as exc:  # a missing config is reported, not fatal
            logger.warning("skipping %s: %s", lang, exc)
            skipped.append(lang)
            continue
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        row = {col: list(ds[col]) for col in ("english", "native", "roman")}
        logger.info("%s: %d rows", lang, len(ds))

        for direction in args.directions:
            key = f"{lang}:{direction}"
            score, rate, preds, refs, trunc, degen = run_direction(
                client,
                direction,
                lang,
                row,
                num_workers=args.num_workers,
                max_new_tokens=args.max_new_tokens,
            )
            scores[key] = score
            predictions[key] = preds
            references[key] = refs
            truncations[key] = trunc
            degenerates[key] = degen
            if rate is not None:
                rates[key] = rate

    if not scores:
        logger.error("nothing was scored")
        return 1

    metrics: dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "languages": [lang for lang in languages if lang not in skipped],
        "configs_without_split": skipped,
        "pairwise": {
            key: {
                **scores[key].as_dict(),
                **({"error_rate": rates[key].as_dict()} if key in rates else {}),
                "n_truncated": truncations[key],
                "n_degenerate": degenerates[key],
            }
            for key in sorted(scores)
        },
    }
    # Per-direction aggregates, keyed ALL:<direction> so one direction can be
    # compared across runs without re-deriving it from the per-language cells.
    for direction in args.directions:
        keys = [k for k in scores if k.endswith(f":{direction}")]
        if not keys:
            continue
        preds = [p for k in keys for p in predictions[k]]
        refs = [r for k in keys for r in references[k]]
        entry: dict[str, Any] = {
            **score_pair(preds, refs).as_dict(),
            "n_langs": len(keys),
            "n_truncated": sum(truncations[k] for k in keys),
            "n_degenerate": sum(degenerates[k] for k in keys),
        }
        if direction in TRANSLITERATION_DIRECTIONS:
            entry["error_rate"] = score_error_rate(preds, refs).as_dict()
        metrics[f"ALL:{direction}"] = entry

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))

    print(f"\nwrote {out_dir}/metrics.json", file=sys.stderr)
    for direction in args.directions:
        entry = metrics.get(f"ALL:{direction}")
        if not entry:
            continue
        line = f"  {direction:10s} BLEU {entry['bleu']:6.2f}  chrF++ {entry['chrf']:6.2f}"
        if "error_rate" in entry:
            line += f"  CER {entry['error_rate']['cer']:.4f}  WER {entry['error_rate']['wer']:.4f}"
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
