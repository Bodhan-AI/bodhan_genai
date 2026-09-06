"""Tests for BestAndLastCheckpointKeeper.

The callback prunes ``output_dir/checkpoint-*`` to the union of:
  - top-N by ``eval_loss`` (lowest if greater_is_better=False)
  - last-N by step

Tests build a synthetic tree on a tmpdir and assert the right dirs survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from bodhan_genai.tts.training.callbacks import BestAndLastCheckpointKeeper


@dataclass
class _FakeArgs:
    output_dir: str


@dataclass
class _FakeState:
    log_history: list[dict] = field(default_factory=list)
    is_world_process_zero: bool = True


def _make_ckpt(root: Path, step: int) -> Path:
    p = root / f"checkpoint-{step}"
    p.mkdir(parents=True)
    (p / "trainer_state.json").write_text("{}")
    return p


def _existing_steps(root: Path) -> set[int]:
    return {int(p.name.rsplit("-", 1)[1]) for p in root.glob("checkpoint-*") if p.is_dir()}


def test_no_prune_when_under_threshold(tmp_path: Path):
    for step in (100, 200):
        _make_ckpt(tmp_path, step)
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(), None)
    assert _existing_steps(tmp_path) == {100, 200}


def test_keeps_last_n_when_no_eval_metrics(tmp_path: Path):
    """No eval_loss in log_history → falls back to last-N-only."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=[]), None)
    # Only last-3 by step survive; older two get pruned.
    assert _existing_steps(tmp_path) == {300, 400, 500}


def test_keeps_union_of_best_and_last(tmp_path: Path):
    """Lowest eval_loss is at step 100 (oldest). Last-3 are 300/400/500.
    Union → {100, 300, 400, 500} (4 dirs)."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    log_history = [
        {"step": 100, "eval_loss": 1.0},  # best
        {"step": 200, "eval_loss": 1.5},  # not in top 3, not in last 3 → pruned
        {"step": 300, "eval_loss": 1.4},
        {"step": 400, "eval_loss": 1.3},
        {"step": 500, "eval_loss": 1.2},
    ]
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    # best 3 by loss = {100, 500, 400}; last 3 by step = {300, 400, 500}
    # union = {100, 300, 400, 500}
    assert _existing_steps(tmp_path) == {100, 300, 400, 500}


def test_overlap_collapses_to_single_set(tmp_path: Path):
    """When best-N == last-N (all best are also most recent), keep just those."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    # Loss decreases monotonically — best are 500/400/300 = also last-3.
    log_history = [
        {"step": 100, "eval_loss": 5.0},
        {"step": 200, "eval_loss": 4.0},
        {"step": 300, "eval_loss": 3.0},
        {"step": 400, "eval_loss": 2.0},
        {"step": 500, "eval_loss": 1.0},
    ]
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    assert _existing_steps(tmp_path) == {300, 400, 500}


def test_greater_is_better_flips_ranking(tmp_path: Path):
    """For a metric where higher = better (e.g. accuracy)."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    log_history = [
        {"step": 100, "eval_acc": 0.90},  # best
        {"step": 200, "eval_acc": 0.50},
        {"step": 300, "eval_acc": 0.55},
        {"step": 400, "eval_acc": 0.60},
        {"step": 500, "eval_acc": 0.70},
    ]
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3, metric="eval_acc", greater_is_better=True)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    # best 3 by acc = {100, 500, 400}; last 3 = {300, 400, 500}
    # union = {100, 300, 400, 500}
    assert _existing_steps(tmp_path) == {100, 300, 400, 500}


def test_partial_metric_coverage(tmp_path: Path):
    """Some saves happen without an eval (save_steps != eval_steps).
    Those steps still get last-N protection but can't enter best-N."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    # Eval only at every other step. Step 200 has the lowest known loss.
    log_history = [
        {"step": 200, "eval_loss": 0.5},
        {"step": 400, "eval_loss": 0.7},
    ]
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    # best by loss (only steps with loss) = {200, 400}
    # last 3 by step                       = {300, 400, 500}
    # union = {200, 300, 400, 500}; step 100 pruned.
    assert _existing_steps(tmp_path) == {200, 300, 400, 500}


def test_non_rank_zero_is_a_noop(tmp_path: Path):
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    state = _FakeState(is_world_process_zero=False)
    cb.on_save(_FakeArgs(str(tmp_path)), state, None)
    assert _existing_steps(tmp_path) == {100, 200, 300, 400, 500}


def test_missing_output_dir_is_a_noop():
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    # Should silently return rather than crash.
    cb.on_save(_FakeArgs("/this/path/does/not/exist"), _FakeState(), None)


def test_uses_latest_log_entry_when_metric_logged_twice(tmp_path: Path):
    """If the same step appears twice in log_history (e.g. early eval + replayed),
    the most recent entry's value wins. Sanity test: making the last entry the
    best should keep that step."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    log_history = [
        {"step": 100, "eval_loss": 0.1},  # best initially
        {"step": 200, "eval_loss": 5.0},
        {"step": 300, "eval_loss": 5.0},
        {"step": 400, "eval_loss": 5.0},
        {"step": 500, "eval_loss": 5.0},
        {"step": 100, "eval_loss": 9.9},  # later overwrite — now step 100 is worst
    ]
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    # With step 100 worst, best-3 = {200, 300, 400} (or any subset since 4-way tie).
    # Either way step 100 is not in best, and not in last 3 → pruned.
    surviving = _existing_steps(tmp_path)
    assert 100 not in surviving
    assert {300, 400, 500} <= surviving


def test_negative_args_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        BestAndLastCheckpointKeeper(last_n=-1, best_k=3)


def test_both_zero_rejected():
    with pytest.raises(ValueError, match="at least one"):
        BestAndLastCheckpointKeeper(last_n=0, best_k=0)


def test_only_last_n_zero_falls_back_to_best_only(tmp_path: Path):
    """last_n=0, best_k=3 → keep best 3 by metric only; no last protection."""
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    log_history = [
        {"step": 100, "eval_loss": 1.0},
        {"step": 200, "eval_loss": 2.0},
        {"step": 300, "eval_loss": 3.0},
        {"step": 400, "eval_loss": 4.0},
        {"step": 500, "eval_loss": 5.0},
    ]
    cb = BestAndLastCheckpointKeeper(last_n=0, best_k=3)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    # Best 3 by lowest loss = {100, 200, 300}; nothing in last set.
    assert _existing_steps(tmp_path) == {100, 200, 300}


def test_only_best_k_zero_keeps_last_only(tmp_path: Path):
    for step in (100, 200, 300, 400, 500):
        _make_ckpt(tmp_path, step)
    log_history = [{"step": 100, "eval_loss": 0.001}]  # great loss but should be dropped
    cb = BestAndLastCheckpointKeeper(last_n=3, best_k=0)
    cb.on_save(_FakeArgs(str(tmp_path)), _FakeState(log_history=log_history), None)
    assert _existing_steps(tmp_path) == {300, 400, 500}
