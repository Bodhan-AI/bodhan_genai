"""
Tests for bodhan_genai.tts.data.compile (Parquet-native version).

Covers: ParquetShardWriter, tokenized-row contract, and the sft-only
training_mode guard.
"""

from __future__ import annotations

import inspect

import pyarrow.parquet as pq
import pytest

from bodhan_genai.tts.data.compile import (
    ParquetShardWriter,
    _init_worker,
)

# ---------------------------------------------------------------------------
# ParquetShardWriter
# ---------------------------------------------------------------------------


class TestParquetShardWriter:
    def test_writes_shards(self, tmp_path):
        writer = ParquetShardWriter(tmp_path / "output", shard_size=3)
        for i in range(7):
            writer.write({"input_ids": [i, i + 1], "labels": [i, i + 1], "length": 2})
        writer.close()
        shards = sorted((tmp_path / "output").glob("shard_*.parquet"))
        assert len(shards) == 3  # 3 + 3 + 1

    def test_total_written(self, tmp_path):
        writer = ParquetShardWriter(tmp_path / "output", shard_size=5)
        for i in range(12):
            writer.write({"input_ids": [i], "labels": [i], "length": 1})
        writer.close()
        assert writer.total_written == 12

    def test_shard_id_start(self, tmp_path):
        writer = ParquetShardWriter(tmp_path / "output", shard_size=5, shard_id_start=10)
        for i in range(6):
            writer.write({"input_ids": [i], "labels": [i], "length": 1})
        writer.close()
        shards = sorted((tmp_path / "output").glob("shard_*.parquet"))
        assert shards[0].name == "shard_00010.parquet"

    def test_empty_close(self, tmp_path):
        writer = ParquetShardWriter(tmp_path / "output", shard_size=5)
        writer.close()
        shards = list((tmp_path / "output").glob("shard_*.parquet"))
        assert len(shards) == 0
        assert writer.total_written == 0

    def test_parquet_schema(self, tmp_path):
        writer = ParquetShardWriter(tmp_path / "output", shard_size=10)
        writer.write({"input_ids": [1, 2, 3], "labels": [4, 5, 6], "length": 3})
        writer.close()
        table = pq.read_table(str(tmp_path / "output" / "shard_00000.parquet"))
        assert set(table.column_names) == {"input_ids", "labels", "length"}


# ---------------------------------------------------------------------------
# Tokenized row contract
# ---------------------------------------------------------------------------


class TestTokenizedRowContract:
    """Tokenized rows should be consumable directly without synthetic keys."""

    def test_audio_row_shape(self):
        row = {
            "token_ids": [256000, 260096, 264192],
            "text": "hello",
            "speaker": "spk_0",
            "style": "happy",
            "accent": "indian english",
            "audio_caption": "",
        }
        entry = {
            "token_ids": row["token_ids"],
            "text": row["text"],
            "speaker": row["speaker"],
            "style": row["style"],
            "accent": row["accent"],
            "audio_caption": row["audio_caption"],
        }
        assert entry["token_ids"] == row["token_ids"]
        assert entry["style"] == "happy"
        assert entry["accent"] == "indian english"

    def test_text_only_row_shape(self):
        row = {
            "token_ids": [],
            "text": "hello",
            "speaker": "",
            "style": "",
            "accent": "",
            "audio_caption": "",
        }
        entry = {
            "token_ids": row["token_ids"],
            "text": row["text"],
            "speaker": row["speaker"],
            "style": row["style"],
            "accent": row["accent"],
            "audio_caption": row["audio_caption"],
        }
        assert entry["token_ids"] == []


# ---------------------------------------------------------------------------
# Simplified knobs — llama route is always full-sequence loss
# ---------------------------------------------------------------------------


class TestSimplifiedKnobs:
    def test_init_worker_signature(self):
        """Worker init takes only (tokenizer_path, drop_style) — no model_type /
        mask_prompt: the llama route always trains with full-sequence loss."""
        params = list(inspect.signature(_init_worker).parameters)
        assert params == ["tokenizer_path", "drop_style"]

    def test_main_rejects_non_sft_training_mode(self, tmp_path, monkeypatch):
        import sys

        import yaml

        from bodhan_genai.tts.data.compile import main

        cfg = {
            "models": {"tokenizer_path": "/nonexistent/tokenizer"},
            "processing": {"training_mode": "cpt"},
            "input": {"datasets": []},
        }
        cfg_path = tmp_path / "compile.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setattr(sys, "argv", ["compile", "--config", str(cfg_path)])
        with pytest.raises(ValueError, match="sft template with full loss"):
            main()
