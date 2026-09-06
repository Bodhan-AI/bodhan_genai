"""
Tests for bodhan_genai/tts/training/sampler.py.

Core invariants:
  1. _pack_ffd_sorted and _pack_ffd_linear produce identical bin assignments
  2. All sequences appear in exactly one bin
  3. No bin exceeds max_seq_len (except oversized sequences)
  4. _shard_bins pads instead of discarding
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bodhan_genai.tts.training.dataset import MixedDataset
from bodhan_genai.tts.training.sampler import (
    _HAS_NUMBA,
    _HAS_SORTED_CONTAINERS,
    SequencePackingSampler,
    require_fast_packing,
    require_numba_packing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_dataset(lengths: list[int]):
    """Create a minimal mock dataset with .lengths numpy array."""
    ds = MagicMock()
    ds.lengths = np.array(lengths, dtype=np.int32)
    ds.__len__ = lambda self: len(lengths)
    return ds


class _FakeParquetDataset:
    def __init__(self, lengths: list[int], source_path: str | None = None):
        self.lengths = np.array(lengths, dtype=np.int32)
        self.source_path = source_path or f"/tmp/fake_{id(self)}"

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": [int(idx)],
            "labels": [int(idx)],
        }


def _total_tokens_in_bins(bins: list[list[int]], lengths: np.ndarray) -> int:
    return sum(lengths[idx] for b in bins for idx in b)


def _all_indices_in_bins(bins: list[list[int]]) -> set[int]:
    return {idx for b in bins for idx in b}


# ---------------------------------------------------------------------------
# FFD packing — basic correctness
# ---------------------------------------------------------------------------


class TestPackFFD:
    def test_all_sequences_placed(self):
        """Every input sequence appears in exactly one bin."""
        lengths = [100, 200, 300, 400, 500, 150, 250, 350]
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)
        indices = np.arange(len(lengths), dtype=np.int64)
        bins = sampler._pack_ffd(indices)
        placed = _all_indices_in_bins(bins)
        assert placed == set(range(len(lengths)))

    def test_no_bin_exceeds_capacity(self):
        """No bin's total length exceeds max_seq_len (except oversized)."""
        lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        max_seq = 1000
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=max_seq, shuffle=False)
        indices = np.arange(len(lengths), dtype=np.int64)
        bins = sampler._pack_ffd(indices)
        for b in bins:
            total = sum(lengths[idx] for idx in b)
            if len(b) > 1:
                assert total <= max_seq

    def test_oversized_gets_own_bin(self):
        """A sequence longer than max_seq_len gets its own bin."""
        lengths = [100, 200, 2000]  # 2000 > 1000
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)
        indices = np.arange(len(lengths), dtype=np.int64)
        bins = sampler._pack_ffd(indices)
        # The oversized sequence (idx 2) should be alone
        oversized_bins = [b for b in bins if 2 in b]
        assert len(oversized_bins) == 1
        assert oversized_bins[0] == [2]

    def test_empty_indices(self):
        """Empty indices → empty bins."""
        ds = _make_mock_dataset([100, 200])
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)
        bins = sampler._pack_ffd(np.array([], dtype=np.int64))
        assert bins == []

    def test_single_sequence(self):
        """Single sequence → one bin."""
        ds = _make_mock_dataset([500])
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)
        bins = sampler._pack_ffd(np.array([0], dtype=np.int64))
        assert len(bins) == 1
        assert bins[0] == [0]


# ---------------------------------------------------------------------------
# FFD sorted vs linear parity
# ---------------------------------------------------------------------------


class TestFFDSortedLinearParity:
    @pytest.mark.parametrize("n_sequences", [10, 50, 200])
    def test_identical_bin_count(self, n_sequences):
        """Sorted and linear implementations must produce the same number of bins."""
        rng = np.random.default_rng(42)
        lengths = rng.integers(50, 500, size=n_sequences).tolist()
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)

        indices = np.arange(n_sequences, dtype=np.int64)
        sorted_order = np.argsort(-ds.lengths[indices])
        sorted_indices = indices[sorted_order]
        sorted_lengths = ds.lengths[indices][sorted_order]

        linear_bins, linear_oversized = sampler._pack_ffd_linear(sorted_indices, sorted_lengths)

        if _HAS_SORTED_CONTAINERS:
            sorted_bins, sorted_oversized = sampler._pack_ffd_sorted(sorted_indices, sorted_lengths)
            assert len(sorted_bins) == len(linear_bins)
            assert sorted_oversized == linear_oversized

    @pytest.mark.parametrize("n_sequences", [10, 50, 200])
    def test_identical_total_tokens(self, n_sequences):
        """Both implementations must pack exactly the same total tokens."""
        rng = np.random.default_rng(123)
        lengths = rng.integers(50, 500, size=n_sequences).tolist()
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)

        indices = np.arange(n_sequences, dtype=np.int64)
        sorted_order = np.argsort(-ds.lengths[indices])
        sorted_indices = indices[sorted_order]
        sorted_lengths = ds.lengths[indices][sorted_order]

        linear_bins, _ = sampler._pack_ffd_linear(sorted_indices, sorted_lengths)

        if _HAS_SORTED_CONTAINERS:
            sorted_bins, _ = sampler._pack_ffd_sorted(sorted_indices, sorted_lengths)
            linear_total = _total_tokens_in_bins(linear_bins, ds.lengths)
            sorted_total = _total_tokens_in_bins(sorted_bins, ds.lengths)
            assert linear_total == sorted_total

    @pytest.mark.parametrize("n_sequences", [10, 50, 200])
    def test_identical_sequences_placed(self, n_sequences):
        """Both must place exactly the same set of indices."""
        rng = np.random.default_rng(999)
        lengths = rng.integers(50, 500, size=n_sequences).tolist()
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)

        indices = np.arange(n_sequences, dtype=np.int64)
        sorted_order = np.argsort(-ds.lengths[indices])
        sorted_indices = indices[sorted_order]
        sorted_lengths = ds.lengths[indices][sorted_order]

        linear_bins, _ = sampler._pack_ffd_linear(sorted_indices, sorted_lengths)

        if _HAS_SORTED_CONTAINERS:
            sorted_bins, _ = sampler._pack_ffd_sorted(sorted_indices, sorted_lengths)
            assert _all_indices_in_bins(linear_bins) == _all_indices_in_bins(sorted_bins)


class TestBucketPackBackend:
    @pytest.mark.parametrize("n_sequences", [10, 50, 200])
    def test_bucket_places_every_sequence_once(self, n_sequences):
        rng = np.random.default_rng(2026)
        lengths = rng.integers(1, 500, size=n_sequences).tolist()
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(
            ds,
            max_seq_len=1000,
            shuffle=False,
            pack_backend="bucket",
        )

        bins = sampler._pack_ffd(np.arange(n_sequences, dtype=np.int64))
        placed = [idx for b in bins for idx in b]

        assert sorted(placed) == list(range(n_sequences))
        for b in bins:
            total = sum(lengths[idx] for idx in b)
            assert total <= 1000

    def test_bucket_matches_sortedlist_bin_count_when_available(self):
        if not _HAS_SORTED_CONTAINERS:
            pytest.skip("sortedcontainers not installed")
        rng = np.random.default_rng(99)
        lengths = rng.integers(1, 800, size=500).tolist()
        ds = _make_mock_dataset(lengths)
        indices = np.arange(len(lengths), dtype=np.int64)
        order = np.argsort(-ds.lengths[indices], kind="stable")
        sorted_indices = indices[order]
        sorted_lengths = ds.lengths[sorted_indices]

        sortedlist_sampler = SequencePackingSampler(
            ds,
            max_seq_len=1000,
            shuffle=False,
            pack_backend="sortedlist",
        )
        bucket_sampler = SequencePackingSampler(
            ds,
            max_seq_len=1000,
            shuffle=False,
            pack_backend="bucket",
        )

        sortedlist_bins, sortedlist_oversized = sortedlist_sampler._pack_ffd_sorted(
            sorted_indices,
            sorted_lengths,
        )
        bucket_bins, bucket_oversized = bucket_sampler._pack_bucket(
            sorted_indices,
            sorted_lengths,
        )

        assert len(bucket_bins) == len(sortedlist_bins)
        assert bucket_oversized == sortedlist_oversized


class TestNumbaBucketPackBackend:
    def test_numba_bucket_requires_numba_when_missing(self, monkeypatch):
        monkeypatch.setattr("bodhan_genai.tts.training.sampler._HAS_NUMBA", False)
        ds = _make_mock_dataset([100, 200, 300])
        sampler = SequencePackingSampler(
            ds,
            max_seq_len=500,
            shuffle=False,
            pack_backend="numba_bucket",
        )

        with pytest.raises(RuntimeError, match="numba is required"):
            sampler._pack_ffd(np.arange(3, dtype=np.int64))

    @pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
    def test_numba_bucket_matches_bucket_bin_count(self):
        rng = np.random.default_rng(314)
        lengths = rng.integers(1, 800, size=500).tolist()
        ds = _make_mock_dataset(lengths)
        indices = np.arange(len(lengths), dtype=np.int64)
        order = np.argsort(-ds.lengths[indices], kind="stable")
        sorted_indices = indices[order]
        sorted_lengths = ds.lengths[sorted_indices]

        bucket_sampler = SequencePackingSampler(
            ds,
            max_seq_len=1000,
            shuffle=False,
            pack_backend="bucket",
        )
        numba_sampler = SequencePackingSampler(
            ds,
            max_seq_len=1000,
            shuffle=False,
            pack_backend="numba_bucket",
        )

        bucket_bins, bucket_oversized = bucket_sampler._pack_bucket(
            sorted_indices,
            sorted_lengths,
        )
        numba_bins, numba_oversized = numba_sampler._pack_numba_bucket(
            sorted_indices,
            sorted_lengths,
        )

        assert len(numba_bins) == len(bucket_bins)
        assert numba_oversized == bucket_oversized
        assert _all_indices_in_bins(numba_bins) == set(range(len(lengths)))


# ---------------------------------------------------------------------------
# Bin sharding (pad vs discard)
# ---------------------------------------------------------------------------


class TestShardBins:
    def test_exact_multiple(self):
        """Bins count is exact multiple of world_size → no padding needed."""
        ds = _make_mock_dataset([100] * 10)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, rank=0, world_size=2)
        bins = [[i] for i in range(8)]  # 8 bins, world_size=2 → exact
        sharded = sampler._shard_bins(bins)
        assert len(sharded) == 4  # 8 / 2

    def test_pads_instead_of_discards(self):
        """Bins count NOT a multiple → padded, not truncated."""
        ds = _make_mock_dataset([100] * 10)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, rank=0, world_size=3)
        bins = [[i] for i in range(7)]  # 7 bins, world_size=3 → remainder=1

        # Old behavior would truncate to 6 bins (2 per rank)
        # New behavior pads to 9 bins (3 per rank)
        sharded = sampler._shard_bins(bins)
        assert len(sharded) == 3  # ceil(7/3) = 3

    def test_all_ranks_get_equal_bins(self):
        """After padding, all ranks get exactly the same number of bins."""
        ds = _make_mock_dataset([100] * 10)
        bins = [[i] for i in range(11)]  # 11 bins, world_size=4

        rank_counts = []
        for rank in range(4):
            sampler = SequencePackingSampler(ds, max_seq_len=1000, rank=rank, world_size=4)
            sharded = sampler._shard_bins(bins)
            rank_counts.append(len(sharded))

        assert len(set(rank_counts)) == 1  # all ranks have same count

    def test_no_data_lost(self):
        """All original bins appear in at least one rank's shard."""
        ds = _make_mock_dataset([100] * 10)
        bins = [[i] for i in range(7)]
        world_size = 3

        all_sharded = []
        for rank in range(world_size):
            sampler = SequencePackingSampler(ds, max_seq_len=1000, rank=rank, world_size=world_size)
            all_sharded.extend(sampler._shard_bins(bins))

        # All original bins must appear (plus padding duplicates)
        original_set = {tuple(b) for b in bins}
        sharded_set = {tuple(b) for b in all_sharded}
        assert original_set.issubset(sharded_set)

    def test_world_size_1(self):
        """Single rank → bins returned as-is."""
        ds = _make_mock_dataset([100] * 10)
        sampler = SequencePackingSampler(ds, max_seq_len=1000, rank=0, world_size=1)
        bins = [[i] for i in range(5)]
        sharded = sampler._shard_bins(bins)
        assert sharded == bins


# ---------------------------------------------------------------------------
# Full iteration
# ---------------------------------------------------------------------------


class TestSamplerIteration:
    def test_deterministic_across_calls(self):
        """Same epoch → same bins."""
        ds = _make_mock_dataset([100, 200, 300, 400, 500] * 20)
        s1 = SequencePackingSampler(ds, max_seq_len=1000, shuffle=True, seed=42)
        s2 = SequencePackingSampler(ds, max_seq_len=1000, shuffle=True, seed=42)
        bins1 = list(s1)
        bins2 = list(s2)
        assert bins1 == bins2

    def test_different_epochs_differ(self):
        """Different epochs → different bin orders (with shuffle=True)."""
        ds = _make_mock_dataset([100, 200, 300, 400, 500] * 20)
        s = SequencePackingSampler(ds, max_seq_len=1000, shuffle=True, seed=42)

        s.set_epoch(0)
        bins_0 = list(s)
        s.set_epoch(1)
        bins_1 = list(s)
        # Bin order should differ (content may differ for MixedDataset)
        assert bins_0 != bins_1

    def test_len_estimate(self):
        """__len__ returns a positive estimate."""
        ds = _make_mock_dataset([100, 200, 300])
        sampler = SequencePackingSampler(ds, max_seq_len=1000)
        assert len(sampler) >= 1

    def test_rank_local_packing_partitions_rows_across_ranks(self):
        lengths = [100, 200, 300, 400, 500, 600] * 10
        ds = _make_mock_dataset(lengths)
        world_size = 3

        per_rank_bins = []
        for rank in range(world_size):
            sampler = SequencePackingSampler(
                ds,
                max_seq_len=1000,
                shuffle=False,
                rank=rank,
                world_size=world_size,
                pack_backend="bucket",
                rank_local=True,
                equalize_rank_bins=False,
            )
            per_rank_bins.extend(list(sampler))

        placed = [idx for b in per_rank_bins for idx in b]
        assert sorted(placed) == list(range(len(lengths)))

    def test_rank_local_equalization_happens_after_local_packing(self, monkeypatch):
        lengths = [600, 600, 200, 200, 200, 200]
        ds = _make_mock_dataset(lengths)
        sampler = SequencePackingSampler(
            ds,
            max_seq_len=700,
            shuffle=False,
            rank=0,
            world_size=2,
            pack_backend="bucket",
            rank_local=True,
            equalize_rank_bins=True,
        )

        monkeypatch.setattr(sampler, "_distributed_max_int", lambda value: value + 1)

        bins = list(sampler)
        assert len(bins) == 3
        assert bins[-1] == bins[0]

        unique_local_indices = sorted({idx for b in bins[:-1] for idx in b})
        assert unique_local_indices == [0, 2, 4]

    def test_warns_once_for_expensive_global_packing(self, monkeypatch, caplog):
        monkeypatch.setattr("bodhan_genai.tts.training.sampler.GLOBAL_PACK_WARNING_THRESHOLD", 10)
        ds = _make_mock_dataset([1] * 10)
        sampler = SequencePackingSampler(
            ds,
            max_seq_len=2048,
            shuffle=False,
            rank=0,
            world_size=64,
            rank_local=False,
        )

        with caplog.at_level("WARNING"):
            list(sampler)

        messages = [
            rec.getMessage()
            for rec in caplog.records
            if "Sampler startup may take a few minutes" in rec.getMessage()
        ]
        assert len(messages) == 1
        assert "rank_local=false" in messages[0]
        assert "64" in messages[0]


class TestCachedOrdering:
    def test_mixed_dataset_sampling_is_deterministic(self):
        """Cached mixed-dataset ordering stays deterministic for a fixed seed/epoch."""
        mixed = MixedDataset(
            datasets=[
                _FakeParquetDataset([100, 400, 200, 300]),
                _FakeParquetDataset([150, 350, 250, 450]),
            ],
            ratios=[0.5, 0.5],
        )
        s1 = SequencePackingSampler(mixed, max_seq_len=1000, shuffle=False, seed=123)
        s2 = SequencePackingSampler(mixed, max_seq_len=1000, shuffle=False, seed=123)

        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        indices1, lengths1, _, _ = s1._ordered_indices_for_epoch(rng1)
        indices2, lengths2, _, _ = s2._ordered_indices_for_epoch(rng2)

        assert np.array_equal(indices1, indices2)
        assert np.array_equal(lengths1, lengths2)

    def test_mixed_dataset_all_selected_indices_appear_once_before_sharding(self):
        """When ratios select every member exactly once, coverage is preserved before sharding."""
        mixed = MixedDataset(
            datasets=[
                _FakeParquetDataset([100, 400]),
                _FakeParquetDataset([150, 350]),
            ],
            ratios=[0.5, 0.5],
        )
        sampler = SequencePackingSampler(mixed, max_seq_len=1000, shuffle=False, seed=7)

        ordered_indices, _, _, _ = sampler._ordered_indices_for_epoch(np.random.default_rng(7))
        assert np.array_equal(np.sort(ordered_indices), np.arange(len(mixed), dtype=np.int64))
        assert len(np.unique(ordered_indices)) == len(ordered_indices)

    def test_single_dataset_uses_cached_global_order(self):
        """Single-dataset ordering reuses the cached descending length order."""
        ds = _make_mock_dataset([150, 400, 100, 300])
        sampler = SequencePackingSampler(ds, max_seq_len=1000, shuffle=False)

        ordered_indices, ordered_lengths, sample_elapsed, order_elapsed = (
            sampler._ordered_indices_for_epoch(np.random.default_rng(0))
        )

        assert np.array_equal(ordered_indices, np.array([1, 3, 0, 2], dtype=np.int64))
        assert np.array_equal(ordered_lengths, np.array([400, 300, 150, 100], dtype=np.int32))
        assert sample_elapsed == 0.0
        assert order_elapsed == 0.0

    def test_sharded_bins_still_pad_in_cached_path(self):
        """Cached ordering still flows through the same shard padding behavior."""
        mixed = MixedDataset(
            datasets=[
                _FakeParquetDataset([600, 600]),
                _FakeParquetDataset([600, 600]),
            ],
            ratios=[0.5, 0.5],
        )
        sampler = SequencePackingSampler(
            mixed,
            max_seq_len=700,
            shuffle=False,
            seed=11,
            rank=0,
            world_size=3,
        )

        bins = list(sampler)
        assert len(bins) == 2


class TestFastPathRequirement:
    def test_require_fast_packing_fails_without_sortedcontainers(self, monkeypatch):
        monkeypatch.setattr("bodhan_genai.tts.training.sampler._HAS_SORTED_CONTAINERS", False)
        with pytest.raises(RuntimeError, match="sortedcontainers is required"):
            require_fast_packing("training")

    def test_require_numba_packing_fails_without_numba(self, monkeypatch):
        monkeypatch.setattr("bodhan_genai.tts.training.sampler._HAS_NUMBA", False)
        with pytest.raises(RuntimeError, match="numba is required"):
            require_numba_packing("training")
