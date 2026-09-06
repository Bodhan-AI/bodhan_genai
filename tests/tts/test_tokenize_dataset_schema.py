"""Focused tests for tokenized-row metadata schema plumbing."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

# No ray stub needed: bodhan_genai.tts.data.tokenize keeps ray imports inside main().
from bodhan_genai.tts.data.tokenize import _resolve_columns, _write_shard_atomic


def test_resolve_columns_includes_optional_style_and_accent():
    cols = _resolve_columns({"audio": "wav", "style": "style_col", "accent": "accent_col"})
    assert cols["audio"] == "wav"
    assert cols["style"] == "style_col"
    assert cols["accent"] == "accent_col"
    assert cols["speaker"] == ""


def test_write_shard_persists_style_and_accent(tmp_path):
    out_dir = tmp_path / "tokenized"
    out_dir.mkdir()

    _write_shard_atomic(
        out_dir,
        0,
        [
            {
                "text": "hello",
                "token_ids": [1, 2, 3],
                "language": "en",
                "speaker": "spk",
                "style": "happy",
                "accent": "indian english",
                "audio_caption": "",
                "audio_filepath": "/tmp/a.wav",
            },
            {
                "text": "world",
                "token_ids": [4, 5],
                "language": "en",
                "speaker": "",
                "style": "",
                "accent": "",
                "audio_caption": "",
                "audio_filepath": "/tmp/b.wav",
            },
        ],
    )

    table = pq.read_table(str(out_dir / "shard_00000.parquet"))
    assert "style" in table.column_names
    assert "accent" in table.column_names
    assert table.column("style").to_pylist() == ["happy", ""]
    assert table.column("accent").to_pylist() == ["indian english", ""]


# ---------------------------------------------------------------------------
# Sample data configs load and carry the expected top-level sections
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_tokenize_yaml_loads():
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "tts" / "data" / "tokenize.yaml").read_text())
    assert cfg["ray"]["address"] is None
    assert cfg["models"]["snac_model_path"]
    assert cfg["models"]["tokenizer_path"]
    assert cfg["processing"]["workers_per_gpu"] == 2
    assert cfg["processing"]["encode_batch_size"] == 32
    ds = cfg["datasets"][0]
    assert "jsonl" in ds["source"]
    cols = _resolve_columns(ds["columns"])
    assert cols["audio"] == "audio_filepath"
    assert ds["output"]["rows_per_shard"] == 10000


def test_compile_yaml_loads():
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "tts" / "data" / "compile.yaml").read_text())
    assert cfg["models"]["tokenizer_path"]
    proc = cfg["processing"]
    assert proc["training_mode"] == "sft"
    assert proc["drop_style"] is False
    assert proc["split_by_language"] is True
    assert proc["split_by_source"] is False
    ds = cfg["input"]["datasets"][0]
    assert {"input_dir", "output_dir", "source_name"} <= set(ds)
