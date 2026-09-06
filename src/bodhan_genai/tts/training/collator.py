"""
PackingCollator: concatenates packed sequences into fixed-shape tensors.

The sampler yields lists of dataset indices (one bin = one batch).
The DataLoader fetches those samples and passes them here.

Output tensors are always [1, max_seq_len] — static shapes for torch.compile.

Cross-sequence attention prevention:
  position_ids are reset to 0 at each sequence boundary within a pack.
  With Flash Attention 2 (Transformers >= 4.44), FA2 detects these resets
  and internally builds cu_seqlens for flash_attn_varlen_func, enforcing
  block-diagonal attention. No 2D attention mask is needed.

Padding:
  - input_ids: padded with pad_token_id
  - labels: padded with -100 (ignored by CrossEntropyLoss)
  - position_ids: padded with 0
  - attention_mask: 1 for real tokens, 0 for padding
"""

from __future__ import annotations

import torch


class PackingCollator:
    """
    Concatenates a list of samples into a single [1, max_seq_len] batch.

    Args:
        max_seq_len: Fixed output sequence length.
        pad_token_id: Token ID used to pad input_ids.
    """

    def __init__(self, max_seq_len: int, pad_token_id: int) -> None:
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Args:
            samples: List of dicts with keys "input_ids" and "labels" (lists of ints).

        Returns:
            Dict with tensors of shape [1, max_seq_len]:
              input_ids, labels, position_ids, attention_mask
        """
        # Allocate fresh tensors each call — each DataLoader worker needs its
        # own memory, so pre-allocated + .copy() saved nothing.
        input_ids = torch.full((self.max_seq_len,), self.pad_token_id, dtype=torch.long)
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        position_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        attn_mask = torch.zeros(self.max_seq_len, dtype=torch.long)

        offset = 0
        for sample in samples:
            ids = sample["input_ids"]
            lbls = sample["labels"]

            remaining = self.max_seq_len - offset
            if remaining <= 0:
                break

            seq_len = min(len(ids), remaining)
            end = offset + seq_len

            input_ids[offset:end] = torch.tensor(ids[:seq_len], dtype=torch.long)
            labels[offset:end] = torch.tensor(lbls[:seq_len], dtype=torch.long)
            position_ids[offset:end] = torch.arange(seq_len, dtype=torch.long)
            attn_mask[offset:end] = 1

            offset = end

        packed_len = offset

        return {
            "input_ids": input_ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "attention_mask": attn_mask.unsqueeze(0),
            "packing_efficiency": packed_len / self.max_seq_len,
        }
