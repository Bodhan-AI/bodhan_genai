"""``python -m bodhan_genai.mt.eval.in22`` — IN22 score replication.

Translates the IN22 benchmark through a running IndicTranslate server and scores it with
BLEU + chrF++. Run this whenever the inference path or a checkpoint changes: a
prompt-contract regression shows up here and almost nowhere else, because the output
stays fluent.

The full run is 22 languages x 2 directions x 1024 segments = 45,056 requests.
Start narrow while iterating::

    # smoke: one direction, one language, 32 segments
    python -m bodhan_genai.mt.eval.in22 --output-dir out/ --langs hin_Deva \\
        --directions en-xx --max-samples 32

    # the full run
    python -m bodhan_genai.mt.eval.in22 --output-dir out/

Requires a server: ``scripts/mt/serve.sh``.

Note the eval language list uses ``mni_Mtei`` and ``snd_Deva`` — one script per
language, 22 in total. The served set carries 25 language-script combinations; the
extra three are the alternate scripts, which IN22 does not cover.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bodhan_genai.mt.eval.metrics import (
    PairScore,
    macro_average,
    pooled_score,
    score_pair,
)
from bodhan_genai.mt.serving.client import DEFAULT_MODEL, MTClient
from bodhan_genai.mt.templates.prompt import resolve_language

logger = logging.getLogger("mt.eval.in22")

#: The 22 IN22 languages plus English. One script per language: a different script
#: choice measures a different thing, so this list is fixed.
IN22_LANGS: list[str] = [
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "eng_Latn",
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
]

PIVOT = "eng_Latn"
DATASET = "ai4bharat/IN22-Gen"


def build_tasks(
    dataset, languages: list[str], directions: str
) -> list[tuple[str, str, list[str], list[str]]]:
    """Build ``(src_lang, tgt_lang, sources, references)`` for each direction.

    IN22 is a multi-way parallel corpus: one row holds the same sentence in every
    language, so both directions of every pair come from the same rows.
    """
    tasks = []
    for lang in languages:
        if lang == PIVOT:
            continue
        if lang not in dataset.column_names:
            logger.warning("skipping %s: not a column in %s", lang, DATASET)
            continue
        eng = list(dataset[PIVOT])
        other = list(dataset[lang])
        if directions in ("both", "en-xx"):
            tasks.append((PIVOT, lang, eng, other))
        if directions in ("both", "xx-en"):
            tasks.append((lang, PIVOT, other, eng))
    return tasks


def run_task(
    client: MTClient,
    src_lang: str,
    tgt_lang: str,
    sources: list[str],
    references: list[str],
    *,
    num_workers: int,
    max_new_tokens: int,
) -> tuple[str, PairScore, list[str]]:
    """Translate and score one direction."""
    pair = f"{resolve_language(src_lang)}->{resolve_language(tgt_lang)}"
    logger.info("%s: %d segments", pair, len(sources))

    results = client.translate_batch(
        sources,
        tgt_lang=tgt_lang,
        src_lang=src_lang,
        num_workers=num_workers,
        max_new_tokens=max_new_tokens,
    )
    failed = [r for r in results if not r.ok]
    if failed:
        logger.warning("%s: %d/%d requests errored", pair, len(failed), len(results))

    predictions = [r.text for r in results]
    score = score_pair(predictions, references)
    logger.info("%s: BLEU %.2f  chrF++ %.2f", pair, score.bleu, score.chrf)
    return pair, score, predictions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.eval.in22",
        description="Translate and score the IN22 benchmark through a IndicTranslate server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", "-o", required=True, help="where results are written")
    p.add_argument("--url", default="http://localhost:8000/v1", help="OpenAI API root")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="served model name")
    p.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="subset of the IN22 languages (default: all 22)",
    )
    p.add_argument(
        "--directions",
        default="both",
        choices=("both", "en-xx", "xx-en"),
        help="which direction(s) to evaluate",
    )
    p.add_argument("--max-samples", type=int, default=None, help="cap segments per direction")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=64)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--split", default="test")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs one INFO line per request. A full run is 45,056 requests, which
    # buries the per-direction scores in transport noise.
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

    logger.info("loading %s [%s]", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info("%d segments per direction", len(dataset))

    languages = args.langs or IN22_LANGS
    unknown = [lang for lang in languages if lang not in IN22_LANGS]
    if unknown:
        p.error(f"not IN22 languages: {', '.join(unknown)}")

    tasks = build_tasks(dataset, languages, args.directions)
    if not tasks:
        logger.error("no directions to evaluate")
        return 1
    logger.info("%d direction(s), %d requests total", len(tasks), len(tasks) * len(dataset))

    out_dir = Path(args.output_dir) / Path(args.dataset).name
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, PairScore] = {}
    predictions: dict[str, list[str]] = {}
    references: dict[str, list[str]] = {}

    for src_lang, tgt_lang, sources, refs in tasks:
        pair, score, preds = run_task(
            client,
            src_lang,
            tgt_lang,
            sources,
            refs,
            num_workers=args.num_workers,
            max_new_tokens=args.max_new_tokens,
        )
        scores[pair] = score
        predictions[pair] = preds
        references[pair] = refs

    # Into and out of English are very different tasks, and one number over both
    # hides each.
    en_xx = {k: v for k, v in scores.items() if k.startswith("English->")}
    xx_en = {k: v for k, v in scores.items() if k.endswith("->English")}

    metrics: dict[str, object] = {
        "dataset": args.dataset,
        "model": args.model,
        "n_per_direction": len(dataset),
        "pairwise": {k: v.as_dict() for k, v in sorted(scores.items())},
    }
    # The headline aggregate is POOLED (all directions as one corpus). The macro
    # average is reported alongside it, clearly named, for spotting which
    # direction moved.
    for label, block in (("en_xx", en_xx), ("xx_en", xx_en)):
        if not block:
            continue
        keys = set(block)
        metrics[f"{label}_pooled"] = pooled_score(
            {k: predictions[k] for k in keys}, {k: references[k] for k in keys}
        ).as_dict()
        metrics[f"{label}_macro"] = macro_average(block).as_dict()

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (out_dir / "pairwise_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False)
    )
    (out_dir / "pairwise_references.json").write_text(
        json.dumps(references, indent=2, ensure_ascii=False)
    )

    print(f"\nwrote {out_dir}/metrics.json", file=sys.stderr)
    for label, key, block in (
        ("English->XX", "en_xx", en_xx),
        ("XX->English", "xx_en", xx_en),
    ):
        if not block:
            continue
        pooled = metrics[f"{key}_pooled"]
        print(
            f"  {label:<12} BLEU {pooled['bleu']:5.2f}  chrF++ {pooled['chrf']:5.2f}  "
            f"(pooled over {len(block)} directions)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
