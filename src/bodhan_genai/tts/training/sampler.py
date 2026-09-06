"""
SequencePackingSampler: First-Fit Decreasing bin packing with per-epoch reshuffling.

Algorithm per epoch:
  1. Sub-sample indices from each constituent dataset proportionally to ratios
     (only relevant for MixedDataset; for single dataset, use all indices)
  2. Sort selected indices by sequence length descending (FFD)
  3. Pack into bins of max_seq_len tokens using first-fit greedy assignment
  4. Shard bins across DDP ranks: rank_bins = all_bins[rank::world_size]
  5. Shuffle bin order within each rank

Each __iter__ call produces different bins due to the epoch counter.
Call set_epoch(epoch) before each epoch to change the sub-sampling seed.

Why FFD over greedy online packing:
  FFD achieves <= 11/9 x OPT + 6/9 bins (provably near-optimal).
  Online greedy has no such guarantee and performs worse on skewed distributions
  (e.g., TTS audio where utterance lengths vary widely).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

from bodhan_genai.tts.training.dataset import (
    MixedDataset,
    ParquetTokenDataset,
)

logger = logging.getLogger(__name__)

try:
    from sortedcontainers import SortedList

    _HAS_SORTED_CONTAINERS = True
except ImportError:
    SortedList = None
    _HAS_SORTED_CONTAINERS = False

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    njit = None
    _HAS_NUMBA = False

VALID_PACK_BACKENDS = ("auto", "sortedlist", "bucket", "numba_bucket", "linear")
GLOBAL_PACK_WARNING_THRESHOLD = 1_000_000


def _identity_decorator(fn):
    return fn


# Keep Numba's disk cache off for distributed training. With many ranks
# importing from a shared filesystem, cache=True writes .nbi/.nbc files under
# training/__pycache__ and can fail with NFS "stale file handle" errors.
_optional_njit = njit(cache=False) if _HAS_NUMBA else _identity_decorator


@_optional_njit
def _fenwick_add_numba(tree, size, idx, delta):
    while idx <= size:
        tree[idx] += delta
        idx += idx & -idx


@_optional_njit
def _fenwick_prefix_sum_numba(tree, idx):
    total = 0
    while idx > 0:
        total += tree[idx]
        idx -= idx & -idx
    return total


@_optional_njit
def _fenwick_lower_bound_numba(tree, size, target):
    idx = 0
    bit = 1
    while bit << 1 <= size:
        bit <<= 1
    while bit:
        next_idx = idx + bit
        if next_idx <= size and tree[next_idx] < target:
            idx = next_idx
            target -= tree[next_idx]
        bit >>= 1
    return idx + 1


@_optional_njit
def _pack_numba_bucket_kernel(sorted_indices, sorted_lengths, max_seq_len):
    n = len(sorted_indices)
    item_values = np.empty(n, dtype=np.int64)
    item_next = np.empty(n, dtype=np.int64)
    bin_heads = np.empty(n, dtype=np.int64)
    bin_counts = np.zeros(n, dtype=np.int64)
    bucket_heads = np.empty(max_seq_len + 1, dtype=np.int64)
    bucket_next = np.empty(n, dtype=np.int64)
    nonempty = np.zeros(max_seq_len + 1, dtype=np.int64)

    for i in range(n):
        bin_heads[i] = -1
        bucket_next[i] = -1
    for i in range(max_seq_len + 1):
        bucket_heads[i] = -1

    item_count = 0
    bin_count = 0
    oversized_count = 0

    for pos in range(n):
        idx = int(sorted_indices[pos])
        seq_len = int(sorted_lengths[pos])

        if seq_len > max_seq_len:
            bin_id = bin_count
            bin_count += 1
            oversized_count += 1
        else:
            need = seq_len
            if need < 1:
                need = 1
            before = _fenwick_prefix_sum_numba(nonempty, need - 1)
            total = _fenwick_prefix_sum_numba(nonempty, max_seq_len)
            if total > before:
                remaining = _fenwick_lower_bound_numba(nonempty, max_seq_len, before + 1)
                bin_id = bucket_heads[remaining]
                bucket_heads[remaining] = bucket_next[bin_id]
                bucket_next[bin_id] = -1
                if bucket_heads[remaining] == -1:
                    _fenwick_add_numba(nonempty, max_seq_len, remaining, -1)

                new_remaining = remaining - seq_len
                if new_remaining > 0:
                    if bucket_heads[new_remaining] == -1:
                        _fenwick_add_numba(nonempty, max_seq_len, new_remaining, 1)
                    bucket_next[bin_id] = bucket_heads[new_remaining]
                    bucket_heads[new_remaining] = bin_id
            else:
                bin_id = bin_count
                bin_count += 1
                new_remaining = max_seq_len - seq_len
                if new_remaining > 0:
                    if bucket_heads[new_remaining] == -1:
                        _fenwick_add_numba(nonempty, max_seq_len, new_remaining, 1)
                    bucket_next[bin_id] = bucket_heads[new_remaining]
                    bucket_heads[new_remaining] = bin_id

        item_values[item_count] = idx
        item_next[item_count] = bin_heads[bin_id]
        bin_heads[bin_id] = item_count
        bin_counts[bin_id] += 1
        item_count += 1

    offsets = np.empty(bin_count + 1, dtype=np.int64)
    offsets[0] = 0
    for bin_id in range(bin_count):
        offsets[bin_id + 1] = offsets[bin_id] + bin_counts[bin_id]

    flat_indices = np.empty(item_count, dtype=np.int64)
    for bin_id in range(bin_count):
        write_pos = offsets[bin_id + 1]
        item_pos = bin_heads[bin_id]
        while item_pos != -1:
            write_pos -= 1
            flat_indices[write_pos] = item_values[item_pos]
            item_pos = item_next[item_pos]

    return flat_indices, offsets, oversized_count


class _FenwickTree:
    """Tracks which remaining-capacity buckets are non-empty."""

    def __init__(self, size: int):
        self.size = int(size)
        self.tree = [0] * (self.size + 1)

    def add(self, idx: int, delta: int) -> None:
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def prefix_sum(self, idx: int) -> int:
        if idx <= 0:
            return 0
        idx = min(idx, self.size)
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

    def lower_bound(self, target: int) -> int:
        """Return smallest index whose prefix sum is >= target."""
        idx = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            next_idx = idx + bit
            if next_idx <= self.size and self.tree[next_idx] < target:
                idx = next_idx
                target -= self.tree[next_idx]
            bit >>= 1
        return idx + 1


def require_fast_packing(context: str) -> None:
    """Fail fast when the fast SortedList packing dependency is unavailable."""
    if _HAS_SORTED_CONTAINERS:
        return
    raise RuntimeError(
        "sortedcontainers is required for fast sequence packing during "
        f"{context}. Install it in the active environment with "
        "`pip install sortedcontainers` or ensure the project environment is active."
    )


def require_numba_packing(context: str) -> None:
    """Fail fast when the Numba packing dependency is unavailable."""
    if _HAS_NUMBA:
        return
    raise RuntimeError(
        "numba is required for numba_bucket sequence packing during "
        f"{context}. Install it in the active environment with "
        "`pip install numba>=0.64.0` or set data.train.packing.backend='bucket'."
    )


class SequencePackingSampler(Sampler):
    """
    Packs variable-length sequences into fixed-size bins of max_seq_len tokens.

    Each yielded item is a list[int] of dataset indices that together fit
    within max_seq_len tokens. The PackingCollator concatenates them.

    Args:
        dataset: ParquetTokenDataset or MixedDataset.
        max_seq_len: Maximum tokens per packed bin.
        shuffle: Whether to shuffle bin order (True for train, False for eval).
        seed: Base random seed. Each epoch uses seed + epoch.
        rank: Current DDP rank (0-indexed).
        world_size: Total number of DDP processes.
    """

    def __init__(
        self,
        dataset: ParquetTokenDataset | MixedDataset,
        max_seq_len: int,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        pack_backend: str = "auto",
        rank_local: bool = False,
        equalize_rank_bins: bool = True,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        if pack_backend not in VALID_PACK_BACKENDS:
            raise ValueError(
                f"pack_backend must be one of {VALID_PACK_BACKENDS}, got {pack_backend!r}"
            )
        self.pack_backend = pack_backend
        self.rank_local = bool(rank_local)
        self.equalize_rank_bins = bool(equalize_rank_bins)
        self.epoch = 0

        self.lengths = dataset.lengths
        self._is_mixed = isinstance(dataset, MixedDataset)
        self._global_indices = np.arange(len(self.lengths), dtype=np.int64)
        self._global_sort_order = np.argsort(-self.lengths, kind="stable")
        self._global_sorted_indices = self._global_indices[self._global_sort_order]
        self._global_sorted_lengths = self.lengths[self._global_sorted_indices]
        self._selection_mask = np.zeros(len(self.lengths), dtype=bool)
        self._selection_counts = np.zeros(len(self.lengths), dtype=np.int32)
        self._logged_epoch0_timing = False
        self._logged_global_pack_warning = False
        # Cached actual per-rank bin count from the most recent __iter__,
        # used by __len__ once we've actually packed at least once. Before
        # the first pack we fall back to an approximation.
        self._cached_bins_per_rank: int | None = None

        self._dataset_index_ranges: tuple[np.ndarray, ...] = ()
        self._samples_per_dataset: tuple[int, ...] = ()
        if self._is_mixed:
            ds: MixedDataset = self.dataset  # type: ignore[assignment]
            total_target = len(self.lengths)
            self._dataset_index_ranges = tuple(
                np.arange(start, start + size, dtype=np.int64)
                for start, size in zip(ds.offsets, ds._sizes, strict=False)
            )
            self._samples_per_dataset = tuple(
                max(1, int(total_target * ratio)) for ratio in ds.ratios
            )

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to change sub-sampling and shuffling."""
        self.epoch = epoch

    # ------------------------------------------------------------------
    # Sub-sampling for MixedDataset
    # ------------------------------------------------------------------

    def _ordered_indices_for_epoch(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Build a descending-by-length index order for this epoch.

        For a single dataset this is the precomputed global descending order.
        For MixedDataset we sub-sample according to ratios, mark the sampled
        members in a reusable mask, and then filter the cached global order.
        """
        if not self._is_mixed:
            return self._global_sorted_indices, self._global_sorted_lengths, 0.0, 0.0

        touched_parts: list[np.ndarray] = []
        sample_start = time.perf_counter()

        for ds_indices, n_samples in zip(
            self._dataset_index_ranges,
            self._samples_per_dataset,
            strict=False,
        ):
            size = len(ds_indices)
            if n_samples <= size:
                if n_samples == size:
                    chosen = ds_indices
                else:
                    chosen = ds_indices[rng.choice(size, size=n_samples, replace=False)]
                self._selection_mask[chosen] = True
                self._selection_counts[chosen] = 1
                touched_parts.append(chosen)
            else:
                chosen = rng.choice(ds_indices, size=n_samples, replace=True)
                unique_chosen, counts = np.unique(chosen, return_counts=True)
                self._selection_mask[unique_chosen] = True
                self._selection_counts[unique_chosen] = counts
                touched_parts.append(unique_chosen)
        sample_elapsed = time.perf_counter() - sample_start

        if not touched_parts:
            empty_indices = np.empty(0, dtype=np.int64)
            empty_lengths = np.empty(0, dtype=self.lengths.dtype)
            return empty_indices, empty_lengths, sample_elapsed, 0.0

        if len(touched_parts) == 1:
            touched = touched_parts[0]
        else:
            touched = np.concatenate(touched_parts)

        order_start = time.perf_counter()
        selected_mask = self._selection_mask[self._global_sorted_indices]
        selected_unique = self._global_sorted_indices[selected_mask]
        selected_unique_lengths = self._global_sorted_lengths[selected_mask]
        selected_counts = self._selection_counts[selected_unique]

        if np.any(selected_counts > 1):
            ordered_indices = np.repeat(selected_unique, selected_counts)
            ordered_lengths = np.repeat(selected_unique_lengths, selected_counts)
        else:
            ordered_indices = selected_unique
            ordered_lengths = selected_unique_lengths

        self._selection_mask[touched] = False
        self._selection_counts[touched] = 0
        order_elapsed = time.perf_counter() - order_start
        return ordered_indices, ordered_lengths, sample_elapsed, order_elapsed

    # ------------------------------------------------------------------
    # First-Fit Decreasing bin packing
    # ------------------------------------------------------------------

    def _pack_ffd(self, indices: np.ndarray) -> list[list[int]]:
        """
        First-Fit Decreasing packing.

        Sorts by length descending, then assigns each sequence to the first
        bin with enough remaining capacity. Opens a new bin if none fits.

        Uses a SortedList (O(n log m)) when sortedcontainers is installed,
        falling back to O(n * m) linear scan otherwise.

        Sequences longer than max_seq_len get their own bin; the collator
        will truncate them.
        """
        lengths = self.lengths[indices]
        sorted_order = np.argsort(-lengths, kind="stable")
        sorted_indices = indices[sorted_order]
        sorted_lengths = lengths[sorted_order]
        return self._pack_sorted_indices(sorted_indices, sorted_lengths)

    def _pack_sorted_indices(
        self,
        sorted_indices: np.ndarray,
        sorted_lengths: np.ndarray,
    ) -> list[list[int]]:
        """Pack a pre-sorted descending-by-length index list."""

        oversized_count = 0
        backend = self._resolved_pack_backend()

        if backend == "bucket":
            bins, oversized_count = self._pack_bucket(
                sorted_indices,
                sorted_lengths,
            )
        elif backend == "numba_bucket":
            bins, oversized_count = self._pack_numba_bucket(
                sorted_indices,
                sorted_lengths,
            )
        elif backend == "sortedlist":
            bins, oversized_count = self._pack_ffd_sorted(sorted_indices, sorted_lengths)
        elif backend == "linear":
            bins, oversized_count = self._pack_ffd_linear(sorted_indices, sorted_lengths)
        else:
            raise AssertionError(f"unhandled pack backend {backend!r}")

        self._log_oversized_warning(oversized_count, len(sorted_indices))
        return bins

    def _resolved_pack_backend(self) -> str:
        if self.pack_backend == "auto":
            return "sortedlist" if _HAS_SORTED_CONTAINERS else "linear"
        if self.pack_backend == "sortedlist" and not _HAS_SORTED_CONTAINERS:
            require_fast_packing("sortedlist pack backend")
        if self.pack_backend == "numba_bucket" and not _HAS_NUMBA:
            require_numba_packing("numba_bucket pack backend")
        return self.pack_backend

    def _pack_ffd_sorted(
        self, sorted_indices: np.ndarray, sorted_lengths: np.ndarray
    ) -> tuple[list[list[int]], int]:
        """
        O(n log m) FFD using SortedList keyed by remaining capacity.

        Each entry in the SortedList is (remaining_capacity, bin_index).
        bisect_left((seq_len,)) finds the first bin with enough room.
        """
        bins: list[list[int]] = []
        # SortedList of (remaining_capacity, bin_index) — sorted ascending by remaining
        sl: SortedList = SortedList()
        oversized_count = 0

        for idx, seq_len in zip(sorted_indices, sorted_lengths, strict=False):
            idx = int(idx)
            seq_len = int(seq_len)

            if seq_len > self.max_seq_len:
                bins.append([idx])
                oversized_count += 1
                continue

            # Find first bin with remaining >= seq_len (O(log m))
            pos = sl.bisect_left((seq_len,))
            if pos < len(sl):
                remaining, bin_id = sl[pos]
                sl.pop(pos)
                bins[bin_id].append(idx)
                new_remaining = remaining - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))
            else:
                # No existing bin fits — open a new one
                bin_id = len(bins)
                bins.append([idx])
                new_remaining = self.max_seq_len - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))

        return bins, oversized_count

    def _pack_bucket(
        self,
        ordered_indices: np.ndarray,
        ordered_lengths: np.ndarray,
    ) -> tuple[list[list[int]], int]:
        """
        Best-fit packing using fixed remaining-capacity buckets.

        The tree tracks non-empty capacity buckets in O(log max_seq_len). This
        avoids SortedList's per-bin tuple churn while preserving the same
        "smallest remaining capacity that fits" placement rule.
        """
        bins: list[list[int]] = []
        buckets: list[list[int]] = [[] for _ in range(self.max_seq_len + 1)]
        nonempty = _FenwickTree(self.max_seq_len)
        oversized_count = 0

        def add_bucket(remaining: int, bin_id: int) -> None:
            if remaining <= 0:
                return
            bucket = buckets[remaining]
            was_empty = len(bucket) == 0
            bucket.append(bin_id)
            if was_empty:
                nonempty.add(remaining, 1)

        def pop_bucket(remaining: int) -> int:
            bucket = buckets[remaining]
            bin_id = bucket.pop()
            if not bucket:
                nonempty.add(remaining, -1)
            return bin_id

        for idx, seq_len in zip(ordered_indices, ordered_lengths, strict=False):
            idx = int(idx)
            seq_len = int(seq_len)

            if seq_len > self.max_seq_len:
                bins.append([idx])
                oversized_count += 1
                continue

            need = max(1, seq_len)
            before = nonempty.prefix_sum(need - 1)
            if nonempty.prefix_sum(self.max_seq_len) > before:
                remaining = nonempty.lower_bound(before + 1)
                bin_id = pop_bucket(remaining)
                bins[bin_id].append(idx)
                add_bucket(remaining - seq_len, bin_id)
            else:
                bin_id = len(bins)
                bins.append([idx])
                add_bucket(self.max_seq_len - seq_len, bin_id)

        return bins, oversized_count

    def _pack_numba_bucket(
        self,
        ordered_indices: np.ndarray,
        ordered_lengths: np.ndarray,
    ) -> tuple[list[list[int]], int]:
        """
        Best-fit bucket packing through a Numba-compiled kernel.

        The kernel keeps the same placement rule as `_pack_bucket` but avoids
        Python list/object churn across millions of samples.
        """
        require_numba_packing("numba_bucket pack backend")
        flat_indices, offsets, oversized_count = _pack_numba_bucket_kernel(
            np.asarray(ordered_indices, dtype=np.int64),
            np.asarray(ordered_lengths, dtype=np.int64),
            int(self.max_seq_len),
        )
        bins = [
            flat_indices[int(offsets[i]) : int(offsets[i + 1])].tolist()
            for i in range(len(offsets) - 1)
        ]
        return bins, int(oversized_count)

    def _pack_ffd_linear(
        self, sorted_indices: np.ndarray, sorted_lengths: np.ndarray
    ) -> tuple[list[list[int]], int]:
        """O(n * m) fallback when sortedcontainers is not installed."""
        bins: list[list[int]] = []
        bin_remaining: list[int] = []
        oversized_count = 0

        for idx, seq_len in zip(sorted_indices, sorted_lengths, strict=False):
            idx = int(idx)
            seq_len = int(seq_len)

            if seq_len > self.max_seq_len:
                bins.append([idx])
                bin_remaining.append(0)
                oversized_count += 1
                continue

            placed = False
            for b in range(len(bins)):
                if bin_remaining[b] >= seq_len:
                    bins[b].append(idx)
                    bin_remaining[b] -= seq_len
                    placed = True
                    break

            if not placed:
                bins.append([idx])
                bin_remaining.append(self.max_seq_len - seq_len)

        return bins, oversized_count

    def _log_oversized_warning(
        self,
        oversized_count: int,
        total_count: int,
    ) -> None:
        if oversized_count <= 0:
            return
        frac = oversized_count / max(1, total_count)
        # Quiet zone (<= 1%): single line at INFO so we know it happened.
        # Loud zone (> 1%): ERROR with actionable advice. The collator silently
        # truncates these to max_seq_len, which means the tail tokens of the
        # offending samples are never seen in training; > 1% is a real data
        # quality problem worth raising every epoch instead of swallowing.
        if frac > 0.01:
            logger.error(
                f"Epoch {self.epoch}: {oversized_count}/{total_count} sequences "
                f"({frac:.1%}) exceed max_seq_len={self.max_seq_len} and will be "
                "TRUNCATED by the collator. Increase max_seq_len, filter long "
                "samples upstream, or accept the truncation by silencing this."
            )
        else:
            logger.info(
                f"Epoch {self.epoch}: {oversized_count}/{total_count} sequences "
                f"({frac:.2%}) exceed max_seq_len={self.max_seq_len} and will be "
                "truncated (within the 1% tolerance)."
            )

    def _pack_best_fit(
        self,
        ordered_indices: np.ndarray,
        ordered_lengths: np.ndarray,
    ) -> tuple[list[list[int]], int]:
        """
        Online best-fit packing for a pre-ordered sequence stream.

        Uses a SortedList keyed by remaining capacity and always places a
        sequence into the smallest bin that can fit it.
        """
        backend = self._resolved_pack_backend()
        if backend == "bucket":
            return self._pack_bucket(ordered_indices, ordered_lengths)
        if backend == "numba_bucket":
            return self._pack_numba_bucket(ordered_indices, ordered_lengths)

        require_fast_packing("best-fit packing")
        bins: list[list[int]] = []
        sl: SortedList = SortedList()
        oversized_count = 0

        for idx, seq_len in zip(ordered_indices, ordered_lengths, strict=False):
            idx = int(idx)
            seq_len = int(seq_len)

            if seq_len > self.max_seq_len:
                bins.append([idx])
                oversized_count += 1
                continue

            pos = sl.bisect_left((seq_len, -1))
            if pos < len(sl):
                remaining, bin_id = sl.pop(pos)
                bins[bin_id].append(idx)
                new_remaining = remaining - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))
            else:
                bin_id = len(bins)
                bins.append([idx])
                new_remaining = self.max_seq_len - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))

        return bins, oversized_count

    def _pack_best_fit_with_affinity(
        self,
        ordered_indices: np.ndarray,
        ordered_lengths: np.ndarray,
        ordered_dataset_ids: np.ndarray,
        expected_counts: np.ndarray,
    ) -> tuple[list[list[int]], int]:
        """
        Best-fit packing with a weight-aware bin-affinity tie-break.

        Among bins with the same smallest remaining capacity that fit the next
        sequence, prefer the bin whose dominant dataset has the largest current
        representation deficit relative to the expected stage-2 target counts.
        """
        require_fast_packing("best-fit affinity packing")
        bins: list[list[int]] = []
        sl: SortedList = SortedList()
        oversized_count = 0
        observed_counts = np.zeros_like(expected_counts, dtype=np.int64)
        bin_dataset_counts: list[dict[int, int]] = []
        bin_dominant_dataset: list[int] = []

        for idx, seq_len, dataset_id in zip(
            ordered_indices,
            ordered_lengths,
            ordered_dataset_ids,
            strict=False,
        ):
            idx = int(idx)
            seq_len = int(seq_len)
            dataset_id = int(dataset_id)

            if seq_len > self.max_seq_len:
                bins.append([idx])
                bin_dataset_counts.append({dataset_id: 1})
                bin_dominant_dataset.append(dataset_id)
                observed_counts[dataset_id] += 1
                oversized_count += 1
                continue

            pos = sl.bisect_left((seq_len, -1))
            chosen_pos: int | None = None
            chosen_bin_id: int | None = None
            if pos < len(sl):
                best_remaining = sl[pos][0]
                chosen_deficit = float("-inf")
                scan_pos = pos
                while scan_pos < len(sl) and sl[scan_pos][0] == best_remaining:
                    _, candidate_bin_id = sl[scan_pos]
                    dominant_dataset = bin_dominant_dataset[candidate_bin_id]
                    deficit = float(
                        expected_counts[dominant_dataset] - observed_counts[dominant_dataset]
                    )
                    if deficit > chosen_deficit or (
                        deficit == chosen_deficit
                        and (chosen_bin_id is None or candidate_bin_id < chosen_bin_id)
                    ):
                        chosen_deficit = deficit
                        chosen_pos = scan_pos
                        chosen_bin_id = candidate_bin_id
                    scan_pos += 1

            if chosen_pos is not None and chosen_bin_id is not None:
                remaining, bin_id = sl.pop(chosen_pos)
                bins[bin_id].append(idx)
                counts = bin_dataset_counts[bin_id]
                counts[dataset_id] = counts.get(dataset_id, 0) + 1
                dominant_dataset = bin_dominant_dataset[bin_id]
                dominant_count = counts[dominant_dataset]
                current_count = counts[dataset_id]
                if current_count > dominant_count or (
                    current_count == dominant_count and dataset_id < dominant_dataset
                ):
                    bin_dominant_dataset[bin_id] = dataset_id
                new_remaining = remaining - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))
            else:
                bin_id = len(bins)
                bins.append([idx])
                bin_dataset_counts.append({dataset_id: 1})
                bin_dominant_dataset.append(dataset_id)
                new_remaining = self.max_seq_len - seq_len
                if new_remaining > 0:
                    sl.add((new_remaining, bin_id))

            observed_counts[dataset_id] += 1

        return bins, oversized_count

    # ------------------------------------------------------------------
    # DDP sharding
    # ------------------------------------------------------------------

    def _rank_local_ordered_subset(
        self,
        ordered_indices: np.ndarray,
        ordered_lengths: np.ndarray,
        ordered_dataset_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Pre-shard an ordered row stream so each rank packs only its slice."""
        if not self.rank_local or self.world_size <= 1:
            return ordered_indices, ordered_lengths, ordered_dataset_ids
        rank_slice = slice(self.rank, None, self.world_size)
        if ordered_dataset_ids is None:
            return ordered_indices[rank_slice], ordered_lengths[rank_slice], None
        return (
            ordered_indices[rank_slice],
            ordered_lengths[rank_slice],
            ordered_dataset_ids[rank_slice],
        )

    def _shard_bins(self, bins: list[list[int]]) -> list[list[int]]:
        """
        Shard bins across DDP ranks by interleaving.

        Pads up to a multiple of world_size by duplicating the last few bins
        so that all data is seen every epoch (no bins are discarded).
        """
        remainder = len(bins) % self.world_size
        if remainder != 0:
            pad_count = self.world_size - remainder
            # Duplicate bins from the beginning to fill the gap. When
            # world_size >> bin_count (small eval sets, tiny final epochs),
            # this can repeat a meaningful fraction of bins — log it once per
            # epoch so silent data-leakage doesn't go unnoticed.
            pad_ratio = pad_count / max(1, len(bins))
            if pad_ratio > 0.10:
                logger.warning(
                    f"Epoch {self.epoch}: padding {pad_count}/{len(bins)} bins "
                    f"({pad_ratio:.1%}) by duplication to align with "
                    f"world_size={self.world_size}. Some samples will be seen "
                    "multiple times this epoch."
                )
            bins = bins + bins[:pad_count]
        return bins[self.rank :: self.world_size]

    def _distributed_max_int(self, value: int) -> int:
        """All-reduce a small integer when distributed training is initialized."""
        if self.world_size <= 1:
            return int(value)
        try:
            import torch
            import torch.distributed as dist
        except Exception:
            return int(value)
        if not dist.is_available() or not dist.is_initialized():
            return int(value)

        if torch.cuda.is_available():
            tensor = torch.tensor([int(value)], dtype=torch.int64, device="cuda")
        else:
            tensor = torch.tensor([int(value)], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def _equalize_rank_local_bins(self, bins: list[list[int]]) -> list[list[int]]:
        """
        Pad rank-local bin lists up to the distributed max count.

        Global sharding naturally gives every rank equal length. Rank-local
        packing needs this small all-reduce so DDP sees the same number of
        batches on every rank.
        """
        if not self.rank_local or not self.equalize_rank_bins:
            return bins
        target_count = self._distributed_max_int(len(bins))
        if target_count <= len(bins):
            return bins
        if not bins:
            raise RuntimeError(
                "rank-local packing produced zero bins on this rank while at "
                "least one other rank has data; disable rank_local packing for "
                "very small datasets or reduce world_size."
            )
        pad_count = target_count - len(bins)
        return bins + [bins[i % len(bins)] for i in range(pad_count)]

    def _maybe_log_global_packing_warning(
        self,
        *,
        global_selected_count: int,
        stage_name: str | None = None,
    ) -> None:
        """Warn once when every rank is about to pack the full global stream."""
        if self._logged_global_pack_warning:
            return
        if self.rank != 0 or self.rank_local or self.world_size <= 1:
            return
        if global_selected_count < GLOBAL_PACK_WARNING_THRESHOLD:
            return
        stage_note = f" stage={stage_name}" if stage_name is not None else ""
        logger.warning(
            "Sampler startup may take a few minutes: rank_local=false means each "
            "of %d ranks will pack the full global ordered stream (%d samples)%s "
            "before DDP sharding. Enable rank-local packing to reduce startup cost.",
            self.world_size,
            global_selected_count,
            stage_note,
        )
        self._logged_global_pack_warning = True

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        ordered_indices, ordered_lengths, sample_elapsed, order_elapsed = (
            self._ordered_indices_for_epoch(rng)
        )
        global_selected_count = len(ordered_indices)
        self._maybe_log_global_packing_warning(global_selected_count=global_selected_count)
        ordered_indices, ordered_lengths, _ = self._rank_local_ordered_subset(
            ordered_indices,
            ordered_lengths,
        )

        pack_start = time.perf_counter()
        bins = self._pack_sorted_indices(ordered_indices, ordered_lengths)
        pack_elapsed = time.perf_counter() - pack_start

        shard_start = time.perf_counter()
        if self.rank_local:
            bins = self._equalize_rank_local_bins(bins)
        else:
            bins = self._shard_bins(bins)
        shard_elapsed = time.perf_counter() - shard_start
        self._cached_bins_per_rank = len(bins)

        # 4. Shuffle bin order
        if self.shuffle:
            order = rng.permutation(len(bins))
            bins = [bins[i] for i in order]

        if self.epoch == 0 and not self._logged_epoch0_timing:
            logger.info(
                "Sampler epoch 0 timings: sub_sample=%.3fs order=%.3fs pack=%.3fs shard=%.3fs "
                "(selected=%d, bins=%d, rank=%d/%d)",
                sample_elapsed,
                order_elapsed,
                pack_elapsed,
                shard_elapsed,
                len(ordered_indices),
                len(bins),
                self.rank,
                self.world_size,
            )
            if self.rank_local:
                logger.info(
                    "Rank-local packing active: backend=%s selected_local=%d selected_global=%d "
                    "(rank=%d/%d)",
                    self._resolved_pack_backend(),
                    len(ordered_indices),
                    global_selected_count,
                    self.rank,
                    self.world_size,
                )
            self._logged_epoch0_timing = True

        yield from bins

    def __len__(self) -> int:
        """Actual per-rank bin count if we've packed at least once this run;
        otherwise a token-volume estimate (HF Trainer calls __len__ early to
        size its progress bar, before __iter__ has run)."""
        if self._cached_bins_per_rank is not None:
            return self._cached_bins_per_rank
        total_tokens = int(self.lengths.sum())
        return max(1, total_tokens // self.max_seq_len // self.world_size)


def build_train_sampler(
    dataset: ParquetTokenDataset | MixedDataset,
    max_seq_len: int,
    shuffle: bool,
    seed: int,
    rank: int,
    world_size: int,
    pack_backend: str = "auto",
    rank_local: bool = False,
    equalize_rank_bins: bool = True,
) -> SequencePackingSampler:
    return SequencePackingSampler(
        dataset=dataset,
        max_seq_len=max_seq_len,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
        pack_backend=pack_backend,
        rank_local=rank_local,
        equalize_rank_bins=equalize_rank_bins,
    )
