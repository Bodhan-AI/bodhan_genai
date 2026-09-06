"""
Tests for training/collator.py.

Core invariants:
  - Output shape is always [1, max_seq_len]
  - position_ids reset to 0 at each sequence boundary
  - Padding regions: input_ids=pad_token_id, labels=-100, attn_mask=0, pos_ids=0
  - packing_efficiency = packed_tokens / max_seq_len
"""

from __future__ import annotations

import pytest
import torch

from bodhan_genai.tts.training.collator import PackingCollator

# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_basic_shape(self):
        collator = PackingCollator(max_seq_len=100, pad_token_id=0)
        samples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]
        batch = collator(samples)
        assert batch["input_ids"].shape == (1, 100)
        assert batch["labels"].shape == (1, 100)
        assert batch["position_ids"].shape == (1, 100)
        assert batch["attention_mask"].shape == (1, 100)

    def test_empty_samples(self):
        collator = PackingCollator(max_seq_len=50, pad_token_id=0)
        batch = collator([])
        assert batch["input_ids"].shape == (1, 50)
        assert batch["packing_efficiency"] == 0.0

    def test_dtype(self):
        collator = PackingCollator(max_seq_len=50, pad_token_id=0)
        samples = [{"input_ids": [1, 2], "labels": [3, 4]}]
        batch = collator(samples)
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["position_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# Padding behavior
# ---------------------------------------------------------------------------


class TestPadding:
    def test_short_sample_padded(self):
        """Sample shorter than max_seq_len → rest is padded."""
        collator = PackingCollator(max_seq_len=10, pad_token_id=99)
        samples = [{"input_ids": [1, 2, 3], "labels": [4, 5, 6]}]
        batch = collator(samples)
        ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        mask = batch["attention_mask"][0]

        # Real tokens
        assert ids[:3].tolist() == [1, 2, 3]
        assert labels[:3].tolist() == [4, 5, 6]
        assert mask[:3].tolist() == [1, 1, 1]

        # Padding
        assert ids[3:].tolist() == [99] * 7
        assert labels[3:].tolist() == [-100] * 7
        assert mask[3:].tolist() == [0] * 7

    def test_exact_fit_no_padding(self):
        """Sample exactly max_seq_len → no padding."""
        collator = PackingCollator(max_seq_len=5, pad_token_id=0)
        samples = [{"input_ids": [1, 2, 3, 4, 5], "labels": [6, 7, 8, 9, 10]}]
        batch = collator(samples)
        assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]
        assert batch["packing_efficiency"] == 1.0


# ---------------------------------------------------------------------------
# Position IDs — sequence boundary resets
# ---------------------------------------------------------------------------


class TestPositionIds:
    def test_single_sequence(self):
        """Single sequence: position_ids = [0, 1, 2, ...]."""
        collator = PackingCollator(max_seq_len=10, pad_token_id=0)
        samples = [{"input_ids": [10, 20, 30], "labels": [10, 20, 30]}]
        batch = collator(samples)
        pos = batch["position_ids"][0]
        assert pos[:3].tolist() == [0, 1, 2]
        assert pos[3:].tolist() == [0] * 7  # padding

    def test_two_sequences_reset(self):
        """Two packed sequences: position_ids reset at boundary."""
        collator = PackingCollator(max_seq_len=10, pad_token_id=0)
        samples = [
            {"input_ids": [10, 20, 30], "labels": [10, 20, 30]},
            {"input_ids": [40, 50], "labels": [40, 50]},
        ]
        batch = collator(samples)
        pos = batch["position_ids"][0]
        # First sequence: positions 0, 1, 2
        assert pos[:3].tolist() == [0, 1, 2]
        # Second sequence: positions reset to 0, 1
        assert pos[3:5].tolist() == [0, 1]
        # Padding
        assert pos[5:].tolist() == [0] * 5

    def test_three_sequences(self):
        """Three packed sequences: each starts at position 0."""
        collator = PackingCollator(max_seq_len=20, pad_token_id=0)
        samples = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [5, 6], "labels": [5, 6]},
            {"input_ids": [7, 8, 9], "labels": [7, 8, 9]},
        ]
        batch = collator(samples)
        pos = batch["position_ids"][0]
        assert pos[:4].tolist() == [0, 1, 2, 3]
        assert pos[4:6].tolist() == [0, 1]
        assert pos[6:9].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Packing efficiency
# ---------------------------------------------------------------------------


class TestPackingEfficiency:
    def test_efficiency_calculation(self):
        collator = PackingCollator(max_seq_len=100, pad_token_id=0)
        samples = [
            {"input_ids": list(range(40)), "labels": list(range(40))},
            {"input_ids": list(range(35)), "labels": list(range(35))},
        ]
        batch = collator(samples)
        assert batch["packing_efficiency"] == pytest.approx(0.75)

    def test_full_pack_efficiency(self):
        collator = PackingCollator(max_seq_len=10, pad_token_id=0)
        samples = [
            {"input_ids": list(range(10)), "labels": list(range(10))},
        ]
        batch = collator(samples)
        assert batch["packing_efficiency"] == 1.0

    def test_empty_efficiency(self):
        collator = PackingCollator(max_seq_len=10, pad_token_id=0)
        batch = collator([])
        assert batch["packing_efficiency"] == 0.0


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_sample_exceeding_max_seq_len(self):
        """Single sample longer than max_seq_len → truncated."""
        collator = PackingCollator(max_seq_len=5, pad_token_id=0)
        samples = [{"input_ids": [1, 2, 3, 4, 5, 6, 7], "labels": [1, 2, 3, 4, 5, 6, 7]}]
        batch = collator(samples)
        assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]

    def test_second_sample_truncated_when_overflow(self):
        """First sample fills most of bin, second is truncated to fit."""
        collator = PackingCollator(max_seq_len=6, pad_token_id=0)
        samples = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},
        ]
        batch = collator(samples)
        ids = batch["input_ids"][0].tolist()
        assert ids == [1, 2, 3, 4, 5, 6]  # second truncated to 2 tokens

    def test_third_sample_dropped_when_full(self):
        """Bin already full → third sample entirely dropped."""
        collator = PackingCollator(max_seq_len=4, pad_token_id=0)
        samples = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [5, 6], "labels": [5, 6]},
        ]
        batch = collator(samples)
        ids = batch["input_ids"][0].tolist()
        assert ids == [1, 2, 3, 4]  # only first sample fits


# ---------------------------------------------------------------------------
# Independent batches (no shared state)
# ---------------------------------------------------------------------------


class TestIndependence:
    def test_consecutive_calls_independent(self):
        """Each collator call produces an independent batch (no state leakage)."""
        collator = PackingCollator(max_seq_len=10, pad_token_id=0)

        batch1 = collator([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}])
        batch2 = collator([{"input_ids": [4, 5], "labels": [4, 5]}])

        assert batch1["input_ids"][0][:3].tolist() == [1, 2, 3]
        assert batch2["input_ids"][0][:2].tolist() == [4, 5]
        # Verify batch1 wasn't mutated by batch2
        assert batch1["input_ids"][0][:3].tolist() == [1, 2, 3]
