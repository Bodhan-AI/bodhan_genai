# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Tests for the offline LID policy comparison.

The corpus here is synthetic and constructed so the right answer is arithmetic,
not judgement: if this file passes, a number in the real report can be trusted to
mean what the header says it means.

Layout (290 labelled rows):
  hi  100 — 70 won by hi, 30 stolen by bgc
  ur  100 — 80 won by ur, 20 stolen by hi
  ta   50 — all won by ta
  xx   40 — a language outside the trained set; only 'vocab' can ever get it
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

try:  # installed package
    from bodhan_genai.asr.inference import lid_report
except ImportError:  # running against a checkout that is not on sys.path
    _here = pathlib.Path(__file__).resolve()
    for _cand in (
        _here.parents[1] / "src" / "lid_report.py",
        _here.parents[2] / "src" / "lid_report.py",
        _here.parents[1] / "src" / "bodhan_genai" / "asr" / "inference" / "lid_report.py",
    ):
        if _cand.exists():
            _spec = importlib.util.spec_from_file_location("lid_report", _cand)
            lid_report = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(lid_report)
            break
    else:  # pragma: no cover
        raise

TRAINED = set(lid_report.TRAINED)


def _row(truth, winner, p_win, runner=None, p_run=0.0):
    scores = dict.fromkeys(TRAINED | {"xx"}, 1e-6)
    scores[winner] = p_win
    if runner:
        scores[runner] = p_run
    return {"true_lang": truth, "scores": scores}


@pytest.fixture
def corpus():
    rows = []
    rows += [_row("hi", "hi", 0.93, "ur", 0.04) for _ in range(70)]
    rows += [_row("hi", "bgc", 0.55, "hi", 0.40) for _ in range(30)]
    rows += [_row("ur", "ur", 0.88, "hi", 0.09) for _ in range(80)]
    rows += [_row("ur", "hi", 0.60, "ur", 0.35) for _ in range(20)]
    rows += [_row("ta", "ta", 0.99) for _ in range(50)]
    rows += [_row("xx", "xx", 0.85, "hi", 0.05) for _ in range(40)]
    return rows


def test_unrestricted_accuracy_is_the_arithmetic_value(corpus):
    acc, n, _ = lid_report.evaluate(corpus, None)
    assert n == 290
    assert acc == pytest.approx((70 + 80 + 50 + 40) / 290)


def test_restricting_to_trained_loses_exactly_the_unreachable_rows(corpus):
    acc, n, per = lid_report.evaluate(corpus, TRAINED)
    assert n == 290
    assert acc == pytest.approx((70 + 80 + 50) / 290)
    # xx is still counted in the denominator — a policy does not get to hide the
    # traffic it cannot serve
    assert per["xx"] == [0, 40]


def test_excluding_a_confusable_neighbour_recovers_its_victim(corpus):
    """Dropping bgc hands the 30 stolen rows back to hi, and touches nothing else."""
    _, _, wide = lid_report.evaluate(corpus, TRAINED)
    _, _, narrow = lid_report.evaluate(corpus, TRAINED - {"bgc"})
    assert wide["hi"] == [70, 100]
    assert narrow["hi"] == [100, 100]
    assert wide["ur"] == narrow["ur"] == [80, 100]
    assert wide["ta"] == narrow["ta"] == [50, 50]


def test_predict_returns_argmax_over_the_candidate_subset():
    scores = {"hi": 0.4, "bgc": 0.55, "ta": 0.05}
    assert lid_report.predict(scores, None) == ("bgc", 0.55)
    assert lid_report.predict(scores, {"hi", "ta"}) == ("hi", 0.4)


def test_predict_on_an_empty_candidate_set_abstains_rather_than_crashing():
    assert lid_report.predict({"hi": 0.9}, set()) == (None, 0.0)


def test_rows_without_ground_truth_are_excluded_from_accuracy(corpus):
    _, n_before, _ = lid_report.evaluate(corpus, None)
    padded = corpus + [{"true_lang": None, "scores": {"hi": 0.9}}] * 17
    _, n_after, _ = lid_report.evaluate(padded, None)
    assert n_after == n_before


def test_loader_separates_error_rows_from_scored_rows(tmp_path):
    import json

    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"row": 0, "scores": {"hi": 0.9}, "true_lang": "hi"}),
                json.dumps({"row": 1, "error": "RuntimeError('bad audio')"}),
                "",
                json.dumps({"row": 2, "scores": {"ta": 0.8}, "true_lang": "ta"}),
            ]
        )
    )
    rows, n_err = lid_report.load([str(p)])
    assert len(rows) == 2
    assert n_err == 1
