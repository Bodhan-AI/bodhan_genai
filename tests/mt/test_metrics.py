"""Metric definitions.

The definitions matter as much as the numbers: chrF++ without ``word_order=2`` is a
different metric that looks close enough to pass review, and a macro average is not
a pooled corpus score.
"""

from __future__ import annotations

import pytest

from bodhan_genai.mt.eval.metrics import (
    PairScore,
    is_degenerate,
    macro_average,
    pooled_score,
    score_pair,
)

sacrebleu = pytest.importorskip("sacrebleu", reason="sacrebleu backs both metrics")


# --------------------------------------------------------------------------- #
# Metric definitions
# --------------------------------------------------------------------------- #


def test_identical_prediction_scores_perfectly():
    refs = ["The committee approved the proposal.", "The meeting was postponed."]
    score = score_pair(list(refs), refs)
    assert score.bleu == pytest.approx(100.0, abs=1e-6)
    assert score.chrf == pytest.approx(100.0, abs=1e-6)
    assert score.n == 2


def test_chrf_is_chrf_plus_plus():
    """word_order=2 is what makes it chrF++; without it the number differs."""
    preds = ["the cat sat on the mat", "a dog barked loudly at noon"]
    refs = ["the cat sat on a mat", "the dog barked at noon"]
    got = score_pair(preds, refs).chrf
    plus_plus = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    plain = sacrebleu.corpus_chrf(preds, [refs]).score

    assert got == pytest.approx(plus_plus)
    assert got != pytest.approx(plain), "chrF and chrF++ coincided; test is not discriminating"


def test_worse_predictions_score_lower():
    refs = ["the committee approved the proposal after a long debate"] * 4
    good = score_pair(list(refs), refs)
    bad = score_pair(["totally unrelated words entirely"] * 4, refs)
    assert bad.bleu < good.bleu
    assert bad.chrf < good.chrf


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="count mismatch"):
        score_pair(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="nothing to score"):
        score_pair([], [])


def test_empty_predictions_are_penalised_not_dropped():
    """Filtering failed rows would quietly inflate the score."""
    refs = ["the cat sat on the mat"] * 4
    full = score_pair(list(refs), refs)
    with_holes = score_pair(["", "", refs[2], refs[3]], refs)
    assert with_holes.bleu < full.bleu
    assert with_holes.n == 4


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def test_pooled_is_not_the_same_as_macro():
    """BLEU aggregates n-gram counts, so pooling then scoring differs from averaging
    per-direction scores. Quoting the wrong one invents a regression."""
    preds = {
        "a": ["the cat sat on the mat"] * 8,
        "b": ["totally different words here entirely"] * 8,
    }
    refs = {
        "a": ["the cat sat on the mat"] * 8,
        "b": ["the cat sat on the mat"] * 8,
    }
    pooled = pooled_score(preds, refs)
    macro = macro_average({k: score_pair(preds[k], refs[k]) for k in preds})
    assert pooled.bleu != pytest.approx(macro.bleu)


def test_pooled_score_concatenates_every_direction():
    preds = {"a": ["x y z"], "b": ["p q r"]}
    refs = {"a": ["x y z"], "b": ["p q r"]}
    assert pooled_score(preds, refs).n == 2


def test_pooled_score_pairs_predictions_with_the_right_references():
    """Both dicts are walked in the same key order, so insertion order cannot
    misalign them — otherwise one direction's hypotheses would be scored against
    another's references and nothing would say so.

    Sentences are long enough for BLEU's 4-grams to exist; on a 3-token sentence
    corpus_bleu is 0 no matter how correct the output is.
    """
    a = "the committee approved the proposal after a long debate"
    b = "the meeting has been postponed until next Tuesday afternoon"
    # Prediction dict deliberately in the opposite insertion order to references.
    preds = {"second": [b], "first": [a]}
    refs = {"first": [a], "second": [b]}
    assert pooled_score(preds, refs).bleu == pytest.approx(100.0, abs=1e-6)

    # And a genuine misalignment must NOT score perfectly.
    swapped = {"first": [b], "second": [a]}
    assert pooled_score(swapped, refs).bleu < 50.0


def test_pooled_score_of_nothing_raises():
    with pytest.raises(ValueError, match="nothing to score"):
        pooled_score({}, {})


def test_macro_average_is_unweighted():
    scores = {
        "big": PairScore(bleu=10.0, chrf=20.0, n=1000),
        "small": PairScore(bleu=30.0, chrf=60.0, n=2),
    }
    avg = macro_average(scores)
    # Unweighted: a 2-segment direction counts as much as a 1000-segment one, so a
    # regression in one low-resource language stays visible.
    assert avg.bleu == pytest.approx(20.0)
    assert avg.chrf == pytest.approx(40.0)
    assert avg.n == 1002


def test_macro_average_of_nothing_raises():
    with pytest.raises(ValueError, match="no scores"):
        macro_average({})


def test_pair_score_as_dict_round_trips():
    assert PairScore(bleu=1.5, chrf=2.5, n=3).as_dict() == {"bleu": 1.5, "chrf": 2.5, "n": 3}


# --------------------------------------------------------------------------- #
# Degenerate-output detection
# --------------------------------------------------------------------------- #


def test_is_degenerate_flags_empty():
    assert is_degenerate("", "some source")
    assert is_degenerate("   ", "some source")


def test_is_degenerate_flags_runaway():
    """Output far longer than its source: the model looped instead of stopping."""
    assert is_degenerate("loop " * 200, "short source")


def test_is_degenerate_passes_normal_output():
    assert not is_degenerate("a fine translation", "some source")
    # Long but proportionate output is fine — Indic targets run 1.5-2x the source.
    assert not is_degenerate("word " * 60, "word " * 55)


def test_is_degenerate_thresholds_are_tunable():
    long_but_short_of_the_floor = "x" * 150
    assert not is_degenerate(long_but_short_of_the_floor, "tiny", min_chars=200)
    assert is_degenerate(long_but_short_of_the_floor, "tiny", min_chars=100)
