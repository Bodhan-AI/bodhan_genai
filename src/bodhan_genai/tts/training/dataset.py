"""
Dataset classes for token-in, token-out training.

ParquetTokenDataset: wraps a HuggingFace `datasets.Dataset` loaded from parquet
  shards. Arrow caching lives under HF_HOME (typically /tmp/$USER/hf_cache).
MixedDataset: combines multiple ParquetTokenDatasets with sampling ratios.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from bodhan_genai.tts.training.config import DataSplitConfig


def _resolve_parquet_shards(path: Path) -> list[Path]:
    """Find parquet shards under `path`.

    Search order:
      1. ``<path>/*.parquet``    — shards directly under the path.
      2. ``<path>/**/*.parquet`` — recursive fallback for arbitrary nesting
         (e.g. config points at a parent dir that contains per-language
         subdirs each holding their own shards).
    """
    if not path.is_dir():
        raise FileNotFoundError(f"No .parquet files found at {path} (not a directory).")
    shards = sorted(path.glob("*.parquet"))
    if shards:
        return shards
    shards = sorted(path.rglob("*.parquet"))
    if shards:
        return shards
    raise FileNotFoundError(f"No .parquet files found under {path}.")


class ParquetTokenDataset(Dataset):
    """
    Wraps a HuggingFace `datasets.Dataset` loaded from parquet shards.

    Expected Parquet columns:
      input_ids — list[int32]: full token sequence
      labels    — list[int32]: CPT = input_ids; SFT = -100 for user turn
      length    — int32: len(input_ids)

    `datasets.load_dataset` builds an Arrow cache under HF_HOME; the sampler
    reads `self.lengths` (materialized once at __init__) and the collator
    consumes the dict returned by __getitem__.
    """

    def __init__(self, parquet_path: str):
        """
        Args:
            parquet_path: Path to a directory of .parquet files (or a single file).
        """
        path = Path(parquet_path)
        if path.is_file():
            data_files = [str(path)]
        else:
            data_files = [str(p) for p in _resolve_parquet_shards(path)]

        self.source_path = str(path)
        self.shards: list[Path] = sorted(Path(p) for p in data_files)

        self._hf_dataset = load_dataset(
            "parquet",
            data_files={"train": data_files},
            split="train",
        )
        # Materialize lengths as a numpy array for fast sampler access — the
        # sampler reads this many times per epoch.
        self.lengths = np.array(self._hf_dataset["length"], dtype=np.int32)

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self._hf_dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
        }


class MixedDataset(Dataset):
    """
    Combines multiple ParquetTokenDataset instances with sampling ratios.

    Presents a single flat index space across all constituent datasets.
    The SequencePackingSampler reads .lengths and .ratios to perform
    ratio-weighted sub-sampling per epoch.

    Global index → (dataset_idx, local_idx) mapping is maintained as
    a numpy array for O(1) lookup.
    """

    def __init__(
        self,
        datasets: list[ParquetTokenDataset],
        ratios: list[float],
    ):
        assert len(datasets) == len(ratios), "datasets and ratios must have same length"
        assert all(r > 0 for r in ratios), "all ratios must be positive"

        self.datasets = datasets
        self.source_paths = [d.source_path for d in datasets]
        # Normalize ratios to sum to 1.0
        total = sum(ratios)
        self.ratios = [r / total for r in ratios]

        # Build global index mapping
        self.offsets: list[int] = []
        all_lengths: list[np.ndarray] = []
        self._dataset_ids = np.empty(0, dtype=np.int32)
        self._sizes = [len(d) for d in datasets]

        offset = 0
        id_arrays = []
        for i, ds in enumerate(datasets):
            self.offsets.append(offset)
            all_lengths.append(ds.lengths)
            id_arrays.append(np.full(len(ds), i, dtype=np.int32))
            offset += len(ds)

        self.lengths = np.concatenate(all_lengths)
        self._dataset_ids = np.concatenate(id_arrays)
        self._total = offset

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, global_idx: int) -> dict:
        ds_idx = int(self._dataset_ids[global_idx])
        local_idx = global_idx - self.offsets[ds_idx]
        return self.datasets[ds_idx][local_idx]


def build_mixed_dataset(
    split_cfg: DataSplitConfig | None,
) -> MixedDataset | ParquetTokenDataset | None:
    """
    Build a dataset from a DataSplitConfig.
    Returns MixedDataset for multiple entries, ParquetTokenDataset for one,
    or None if config is None.
    """
    if split_cfg is None:
        return None

    entries = split_cfg.datasets
    if not entries:
        return None

    if len(entries) == 1:
        return ParquetTokenDataset(entries[0].path)

    datasets = [ParquetTokenDataset(e.path) for e in entries]
    ratios = [e.ratio for e in entries]
    return MixedDataset(datasets, ratios)
