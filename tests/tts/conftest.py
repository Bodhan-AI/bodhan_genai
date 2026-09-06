"""
Shared fixtures for the bodhan_genai test suite.

Provides mock SNAC models, temporary directories, sample data generators,
and Parquet file helpers used across multiple test modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Mock SNAC model
# ---------------------------------------------------------------------------


class MockSNACModel:
    """
    Fake SNAC model that returns deterministic codes for any input waveform.

    Mimics the real SNAC interface:
      - encode(audio_tensor) → [codes_0, codes_1, codes_2]
        where audio_tensor is (B, 1, T)
        codes_0 shape: (B, 1, N)
        codes_1 shape: (B, 1, 2N)
        codes_2 shape: (B, 1, 4N)
      - hop_length, vq_strides, attn_window_size attributes

    The mock produces N = padded_len // hop_length // vq_strides[0] frames,
    matching real SNAC padding logic.
    """

    def __init__(self, device: str = "cpu"):
        self.hop_length = 441  # real SNAC 24kHz default: prod([3,3,7,7])
        self.vq_strides = [8, 4, 2, 1]
        self.attn_window_size = 32
        self._device = device

    def encode(self, audio_data: torch.Tensor) -> list[torch.Tensor]:
        """
        Deterministic encoding: codes are derived from input length for reproducibility.
        Each code value is (frame_index % 4096) to stay in valid range.

        Real SNAC returns codes as list of 3 tensors with shapes:
          codes[0]: (B, N)     — coarsest
          codes[1]: (B, 2N)
          codes[2]: (B, 4N)    — finest
        """
        import math

        B = audio_data.shape[0]
        T = audio_data.shape[-1]

        # Mirror SNAC's preprocess padding
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size)
        pad_to = self.hop_length * lcm
        padded_T = math.ceil(T / pad_to) * pad_to

        N = padded_T // self.hop_length // self.vq_strides[0]

        device = audio_data.device

        # Deterministic codes: use arange modulo 4096, shape (B, *)
        codes_0 = (torch.arange(N, device=device) % 4096).unsqueeze(0).expand(B, -1)
        codes_1 = (torch.arange(2 * N, device=device) % 4096).unsqueeze(0).expand(B, -1)
        codes_2 = (torch.arange(4 * N, device=device) % 4096).unsqueeze(0).expand(B, -1)

        return [codes_0.clone(), codes_1.clone(), codes_2.clone()]

    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Fake decode: return zeros of appropriate length (batch-aware)."""
        n = codes[0].shape[-1]
        B = codes[0].shape[0] if codes[0].dim() > 1 else 1
        T = n * self.hop_length * self.vq_strides[0]
        return torch.zeros(B, 1, T)

    def eval(self):
        return self

    def to(self, device):
        self._device = device
        return self


@pytest.fixture
def mock_snac_model():
    """Provide a MockSNACModel on CPU."""
    return MockSNACModel(device="cpu")


# ---------------------------------------------------------------------------
# Frozen-layout mock tokenizer
# ---------------------------------------------------------------------------

# Frozen token contract (see docs): bos 128000, eot 128009, speech 128257/128258,
# human 128259/128260, ai 128261/128262, snac base 128266, wrappers 156938-156941.
FROZEN_SPECIALS = {
    "<|start_of_speech|>": 128257,
    "<|end_of_speech|>": 128258,
    "<|start_of_human|>": 128259,
    "<|end_of_human|>": 128260,
    "<|start_of_ai|>": 128261,
    "<|end_of_ai|>": 128262,
    "<|pad|>": 128263,
    "<|snac_0|>": 128266,
    "<|speaker>": 156938,
    "<speaker|>": 156939,
    "<|style>": 156940,
    "<style|>": 156941,
    "<|eot_id|>": 128009,
}


class FrozenLayoutTokenizer:
    """Mock tokenizer implementing the frozen llama3-TTS special-token layout.

    ``encode`` maps text deterministically into a low id range (word ids
    1000-3999, newline 198) so text ids can never collide with structural ids.
    ``convert_calls`` records every ``convert_tokens_to_ids`` lookup so tests
    can assert ids are resolved from the tokenizer, never hardcoded.
    """

    name_or_path = "frozen-layout-mock"
    unk_token_id = 0
    bos_token_id = 128000
    eos_token_id = 128009

    def __init__(self):
        self.convert_calls: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "builders must never add special tokens"
        if text == "\n":
            return [198]
        return [1000 + (sum(map(ord, w)) % 3000) for w in text.split()]

    def convert_tokens_to_ids(self, token: str):
        self.convert_calls.append(token)
        return FROZEN_SPECIALS.get(token)


@pytest.fixture
def frozen_tokenizer():
    """Provide a fresh FrozenLayoutTokenizer."""
    return FrozenLayoutTokenizer()


# ---------------------------------------------------------------------------
# Temporary directories and files
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory (pytest built-in)."""
    return tmp_path


@pytest.fixture
def sample_jsonl(tmp_path) -> Path:
    """Create a small JSONL file with 10 rows."""
    path = tmp_path / "manifest.jsonl"
    rows = [
        {
            "audio_filepath": f"/data/audio_{i:04d}.wav",
            "text": f"Sample text number {i}",
            "speaker": f"spk_{i % 3}",
            "audio_caption": "",
        }
        for i in range(10)
    ]
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def sample_parquet_dir(tmp_path) -> Path:
    """Create a directory with 2 small Parquet shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp_path / "parquet_data"
    shard_dir.mkdir()

    schema = pa.schema(
        [
            pa.field("audio_filepath", pa.string()),
            pa.field("text", pa.string()),
            pa.field("speaker", pa.string()),
        ]
    )

    for shard_id in range(2):
        rows = {
            "audio_filepath": [f"/data/audio_{shard_id}_{i}.wav" for i in range(5)],
            "text": [f"Text {shard_id}_{i}" for i in range(5)],
            "speaker": [f"spk_{i % 2}" for i in range(5)],
        }
        table = pa.table(rows, schema=schema)
        pq.write_table(table, shard_dir / f"shard_{shard_id:05d}.parquet")

    return shard_dir


@pytest.fixture
def sample_training_parquet(tmp_path) -> Path:
    """Create a directory with training-format Parquet (input_ids, labels, length)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp_path / "training_data"
    shard_dir.mkdir()

    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("labels", pa.list_(pa.int32())),
            pa.field("length", pa.int32()),
        ]
    )

    rng = np.random.default_rng(42)
    for shard_id in range(2):
        n_rows = 50
        lengths = rng.integers(50, 500, size=n_rows).tolist()
        rows = {
            "input_ids": [rng.integers(0, 1000, size=n).tolist() for n in lengths],
            "labels": [rng.integers(0, 1000, size=n).tolist() for n in lengths],
            "length": lengths,
        }
        table = pa.table(rows, schema=schema)
        pq.write_table(table, shard_dir / f"shard_{shard_id:05d}.parquet")

    return shard_dir


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def random_waveforms():
    """Generate a list of random audio waveforms of varying lengths."""
    rng = np.random.default_rng(42)

    def _make(n: int = 5, min_len: int = 4800, max_len: int = 48000) -> list[np.ndarray]:
        lengths = rng.integers(min_len, max_len, size=n)
        return [rng.normal(0, 0.1, size=m).astype(np.float32) for m in lengths]

    return _make


@pytest.fixture
def audio_token_map() -> dict[str, list[int]]:
    """Fake audio_token_map for 10 audio files with deterministic token IDs."""
    rng = np.random.default_rng(42)
    return {
        f"/data/audio_{i:04d}.wav": rng.integers(
            256000, 284672, size=rng.integers(50, 200)
        ).tolist()
        for i in range(10)
    }
