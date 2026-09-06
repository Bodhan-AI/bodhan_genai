"""Translation metrics: corpus BLEU and chrF++ via sacrebleu.

Both are computed the way the published IndicTranslate scores were, so numbers from this
repo are comparable with the reference table:

*   ``corpus_bleu(preds, [refs])`` — sacrebleu defaults (``13a`` tokenizer).
*   ``corpus_chrf(preds, [refs], word_order=2)`` — ``word_order=2`` is what makes
    it chrF**++** rather than chrF. Dropping it silently reports a different metric
    that looks close enough to pass a casual review.

No normalization, transliteration or detokenization is applied anywhere: scoring
runs on the raw strings, as it did for the reference numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairScore:
    """Scores for one translation direction."""

    bleu: float
    chrf: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"bleu": self.bleu, "chrf": self.chrf, "n": self.n}


def score_pair(predictions: list[str], references: list[str]) -> PairScore:
    """Corpus BLEU + chrF++ for one direction.

    Empty predictions are kept, not filtered: a model that fails to translate a
    segment should be penalised for it, and dropping the row would quietly inflate
    the score. Use :func:`is_degenerate` to inspect suspect rows when a direction
    looks off.
    """
    import sacrebleu

    if len(predictions) != len(references):
        raise ValueError(
            f"prediction/reference count mismatch: {len(predictions)} vs {len(references)}"
        )
    if not predictions:
        raise ValueError("nothing to score")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    return PairScore(bleu=bleu.score, chrf=chrf.score, n=len(predictions))


def pooled_score(predictions: dict[str, list[str]], references: dict[str, list[str]]) -> PairScore:
    """Corpus score over every direction concatenated into one corpus.

    **This is the headline aggregate.** BLEU aggregates n-gram counts, so pooling
    every direction into one corpus and then scoring is not the same as averaging
    per-direction scores — the two differ by around a point. Quote which one you
    mean, and compare like with like.
    """
    if not predictions:
        raise ValueError("nothing to score")
    flat_preds = [p for key in sorted(predictions) for p in predictions[key]]
    flat_refs = [r for key in sorted(predictions) for r in references[key]]
    return score_pair(flat_preds, flat_refs)


def macro_average(scores: dict[str, PairScore]) -> PairScore:
    """Unweighted mean over directions — a secondary view, not the headline.

    Every direction gets an equal vote regardless of segment count, which makes a
    regression in one low-resource language visible instead of diluted. Useful for
    spotting *where* something moved; :func:`pooled_score` is the headline number.
    """
    if not scores:
        raise ValueError("no scores to average")
    n = len(scores)
    return PairScore(
        bleu=sum(s.bleu for s in scores.values()) / n,
        chrf=sum(s.chrf for s in scores.values()) / n,
        n=sum(s.n for s in scores.values()),
    )


def is_degenerate(
    prediction: str, source: str, *, min_chars: int = 200, ratio: float = 0.15
) -> bool:
    """True for an empty or runaway generation.

    A runaway is a prediction far longer than its source — the model looped
    instead of terminating. These are worth separating out because a couple of
    them tank BLEU while barely moving chrF++, which is the signature that tells
    "a few degenerate rows" apart from "a real regression".
    """
    if not prediction.strip():
        return True
    if len(prediction) < min_chars:
        return False
    return len(source) > 0 and (len(source) / len(prediction)) < ratio
