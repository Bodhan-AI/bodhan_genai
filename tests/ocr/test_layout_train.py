"""Training pieces that are checkable without a GPU: the order loss, the sampler, EMA
and the config.

The model itself is not built here — that needs the PP-DocLayoutV3 weights and a GPU.
What is covered is everything that can silently be wrong while still running: a loss
that does not actually prefer the correct order, a sampler whose batches drift from the
configured mix, an EMA that never moves.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from bodhan_genai.ocr.training.config import LayoutTrainConfig  # noqa: E402
from bodhan_genai.ocr.training.dataset import MixedSourceSampler  # noqa: E402
from bodhan_genai.ocr.training.ema import ModelEma  # noqa: E402
from bodhan_genai.ocr.training.order_loss import decode_order, locality_gce  # noqa: E402


def _scores_for(order):
    """An antisymmetric matrix that encodes `order` exactly: S_ij > 0 iff i precedes j."""
    order = torch.as_tensor(order, dtype=torch.float32)
    return (order.unsqueeze(1) - order.unsqueeze(0)) * -5.0


# --------------------------------------------------------------------------- #
# Order loss
# --------------------------------------------------------------------------- #


def test_correct_order_scores_lower_than_reversed():
    order = torch.arange(6, dtype=torch.float32)
    assert locality_gce(_scores_for(order), order) < locality_gce(-_scores_for(order), order)


def test_loss_is_near_zero_for_a_confident_correct_matrix():
    order = torch.arange(5, dtype=torch.float32)
    assert locality_gce(_scores_for(order) * 4, order).item() < 0.01


def test_fewer_than_two_blocks_keeps_the_graph_alive():
    """A page with one block must not detach the order head from the graph."""
    scores = torch.zeros(1, 1, requires_grad=True)
    loss = locality_gce(scores, torch.zeros(1))
    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None


def test_loss_is_differentiable():
    order = torch.arange(4, dtype=torch.float32)
    scores = _scores_for(order).clone().requires_grad_(True)
    locality_gce(scores, order).backward()
    assert torch.isfinite(scores.grad).all()


def test_one_wrong_pair_costs_more_when_the_blocks_are_adjacent():
    """The point of the locality weight: pairs are weighted exp(-|delta rank| / tau).

    Isolate a *single* inverted pair. Swapping two distant elements of the permutation
    would not test this — it inverts a dozen pairs, several of them adjacent — so the
    score matrix is corrupted entry by entry instead.
    """
    order = torch.arange(8, dtype=torch.float32)
    correct = _scores_for(order)

    adjacent = correct.clone()
    adjacent[3, 4], adjacent[4, 3] = -correct[3, 4], -correct[4, 3]  # |delta rank| = 1

    distant = correct.clone()
    distant[0, 7], distant[7, 0] = -correct[0, 7], -correct[7, 0]  # |delta rank| = 7

    assert locality_gce(adjacent, order) > locality_gce(distant, order)


def test_a_bigger_perturbation_costs_more_overall():
    """Sanity check alongside the locality one: inverting many pairs must cost more than
    inverting one, however far apart they are."""
    order = torch.arange(8, dtype=torch.float32)
    one_pair = list(range(8))
    one_pair[3], one_pair[4] = one_pair[4], one_pair[3]
    assert locality_gce(_scores_for(one_pair), order) < locality_gce(-_scores_for(order), order)


def test_loss_ignores_gaps_in_the_rank_values():
    """Ranks arrive as raw reading_order values, which can skip numbers."""
    dense = torch.tensor([0.0, 1.0, 2.0])
    sparse = torch.tensor([5.0, 40.0, 900.0])
    scores = _scores_for([0, 1, 2])
    assert locality_gce(scores, dense) == pytest.approx(locality_gce(scores, sparse).item())


def test_decode_order_recovers_an_arbitrary_permutation():
    permutation = [3, 0, 4, 1, 2]
    decoded = decode_order(_scores_for(permutation)).tolist()
    expected = torch.as_tensor(permutation).argsort().tolist()
    assert decoded == expected


# --------------------------------------------------------------------------- #
# Mixed-source sampler
# --------------------------------------------------------------------------- #


def _source_ids(counts):
    import numpy as np

    return np.concatenate([np.full(n, s) for s, n in enumerate(counts)])


def test_every_batch_holds_the_configured_mix():
    from collections import Counter

    ids = _source_ids([600, 300, 100])
    sampler = MixedSourceSampler(ids, {0: 0.6, 1: 0.3, 2: 0.1}, batch_size=10, seed=1)
    assert sampler.counts == {0: 6, 1: 3, 2: 1}
    for batch in list(sampler)[:5]:
        assert Counter(int(ids[i]) for i in batch) == {0: 6, 1: 3, 2: 1}


def test_counts_always_sum_to_the_batch_size():
    """Largest-remainder apportionment: weights that do not divide evenly must not
    silently produce a short batch."""
    ids = _source_ids([100, 100, 100])
    sampler = MixedSourceSampler(ids, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, batch_size=10, seed=0)
    assert sum(sampler.counts.values()) == 10


def test_weights_need_not_be_normalized():
    ids = _source_ids([100, 100])
    sampler = MixedSourceSampler(ids, {0: 3.0, 1: 1.0}, batch_size=8, seed=0)
    assert sampler.counts == {0: 6, 1: 2}


def test_ranks_split_the_global_batch_and_keep_the_ratio():
    ids = _source_ids([600, 300, 100])
    kwargs = {"batch_size": 10, "seed": 7, "num_replicas": 2}
    rank0 = MixedSourceSampler(ids, {0: 0.6, 1: 0.3, 2: 0.1}, rank=0, **kwargs)
    rank1 = MixedSourceSampler(ids, {0: 0.6, 1: 0.3, 2: 0.1}, rank=1, **kwargs)
    first0, first1 = next(iter(rank0)), next(iter(rank1))
    assert len(first0) == len(first1) == 10
    assert not set(first0) & set(first1), "ranks must not see the same page in a step"


def test_epoch_changes_the_order():
    ids = _source_ids([100, 100])
    sampler = MixedSourceSampler(ids, {0: 0.5, 1: 0.5}, batch_size=10, seed=3)
    sampler.set_epoch(0)
    first = next(iter(sampler))
    sampler.set_epoch(1)
    assert first != next(iter(sampler))


def test_same_seed_and_epoch_reproduce_the_same_batches():
    ids = _source_ids([100, 100])
    a = MixedSourceSampler(ids, {0: 0.5, 1: 0.5}, batch_size=10, seed=5)
    b = MixedSourceSampler(ids, {0: 0.5, 1: 0.5}, batch_size=10, seed=5)
    assert next(iter(a)) == next(iter(b))


def test_a_source_that_rounds_to_zero_is_reported(caplog):
    ids = _source_ids([100, 100])
    with caplog.at_level("WARNING"):
        sampler = MixedSourceSampler(ids, {0: 0.99, 1: 0.01}, batch_size=4, seed=0)
    assert sampler.counts[1] == 0
    assert "never be sampled" in caplog.text, "a silently unused source must be flagged"


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


def test_ema_tracks_the_model_and_warms_up():
    model = torch.nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.9, warmup=10)
    assert ema.current_decay() == 0.0, "no updates yet"
    before = ema.shadow["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    assert 0.0 < ema.current_decay() < 0.9, "decay ramps rather than starting at the target"
    assert not torch.allclose(ema.shadow["weight"], before)


def test_ema_converges_towards_the_model():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEma(model, decay=0.5, warmup=1)
    with torch.no_grad():
        model.weight.fill_(5.0)
    for _ in range(40):
        ema.update(model)
    assert ema.shadow["weight"].mean().item() == pytest.approx(5.0, abs=1e-3)


def test_copy_to_writes_the_shadow_into_a_model():
    model = torch.nn.Linear(2, 2)
    ema = ModelEma(model, decay=0.5, warmup=1)
    with torch.no_grad():
        ema.shadow["weight"].fill_(3.0)
    ema.copy_to(model)
    assert torch.allclose(model.weight, torch.full_like(model.weight, 3.0))


def test_ema_state_round_trips():
    model = torch.nn.Linear(2, 2)
    ema = ModelEma(model, decay=0.99, warmup=5)
    ema.update(model)
    restored = ModelEma(model, decay=0.1, warmup=1)
    restored.load_state_dict(ema.state_dict())
    assert restored.updates == ema.updates
    assert restored.decay == ema.decay


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_unknown_config_key_raises(tmp_path):
    """A typo'd key that silently falls back to a default is found days into a run."""
    config = tmp_path / "c.yaml"
    config.write_text("cache_prefix: x\nlearing_rate: 3\n")
    with pytest.raises(ValueError, match="unknown config key"):
        LayoutTrainConfig.from_yaml(config)


def test_config_loads_and_overrides(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("cache_prefix: x\nepochs: 3\nbatch_size: 4\n")
    loaded = LayoutTrainConfig.from_yaml(config, epochs=9, batch_size=None)
    assert loaded.epochs == 9, "explicit overrides win"
    assert loaded.batch_size == 4, "None overrides are ignored, not applied"


def test_config_rejects_impossible_values():
    with pytest.raises(ValueError, match="batch_size"):
        LayoutTrainConfig(cache_prefix="x", batch_size=0)
    with pytest.raises(ValueError, match="grad_accum"):
        LayoutTrainConfig(cache_prefix="x", grad_accum=0)
    with pytest.raises(ValueError, match="non-negative"):
        LayoutTrainConfig(cache_prefix="x", source_weights={"a": -1.0})


def test_shipped_config_is_valid():
    """configs/ocr/train/layout.yaml must actually load."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "configs" / "ocr" / "train" / "layout.yaml"
    config = LayoutTrainConfig.from_yaml(path)
    assert config.lambda_order > 0
    assert config.max_grad_norm == pytest.approx(0.1), "Hungarian matching needs tight clipping"
    assert config.backbone_learning_rate < config.learning_rate
