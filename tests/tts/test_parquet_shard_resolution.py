"""Tests for ParquetTokenDataset's shard resolver.

The resolver tries direct shards under the literal path first, then falls
back to a recursive glob. We do not silently rewrite the user's path: if
they point at a non-existent dir, they get a FileNotFoundError. (Earlier
versions had heuristic fallbacks for a ``split_by_source=true`` doubled-
segment quirk in ``scripts/compile_dataset.py``; that bug is fixed at the
compile side now, so those fallbacks are gone.)

Tests stay cheap by hitting ``_resolve_parquet_shards`` directly (no HF
dataset construction, no real parquet content).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bodhan_genai.tts.training.dataset import _resolve_parquet_shards


def _touch_shards(directory: Path, n: int = 2) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    shards = [directory / f"shard_{i:05d}.parquet" for i in range(n)]
    for s in shards:
        s.touch()
    return shards


def test_resolve_direct_shards(tmp_path: Path):
    """Layout: split_by_source=False, split_by_language=True.
    Config points at <output_dir>/<lang>/, shards are directly inside."""
    lang_dir = tmp_path / "rasa" / "hi"
    expected = _touch_shards(lang_dir)
    found = _resolve_parquet_shards(lang_dir)
    assert sorted(found) == sorted(expected)


def test_resolve_recursive_fallback(tmp_path: Path):
    """Shards nested deeper than 1 level under literal path. Recursive glob
    should still find them."""
    deep = tmp_path / "rasa" / "extra" / "hi"
    expected = _touch_shards(deep)
    found = _resolve_parquet_shards(tmp_path / "rasa")
    assert sorted(found) == sorted(expected)


def test_resolve_doubled_via_basename(tmp_path: Path):
    """Layout: <output_dir>/<source>/<source>/<lang> where config points at
    <output_dir>/<source>. The resolver should find shards under
    <path>/<basename(path)>/."""
    src = tmp_path / "rasa"
    expected = _touch_shards(src / "rasa")
    found = _resolve_parquet_shards(src)
    assert sorted(found) == sorted(expected)


def test_resolve_raises_when_nothing_found(tmp_path: Path):
    """If neither the literal path nor any sibling layout has shards, the
    resolver should raise — the user needs feedback, not silent success."""
    empty = tmp_path / "missing" / "hi"
    with pytest.raises(FileNotFoundError, match=r"No \.parquet files found"):
        _resolve_parquet_shards(empty)


def test_resolve_prefers_direct_over_walk(tmp_path: Path):
    """If shards exist BOTH directly under the path and in a sibling layout,
    the direct hit must win — that is the path the user explicitly wrote."""
    lang_dir = tmp_path / "rasa" / "hi"
    direct = _touch_shards(lang_dir)
    # Also create a doubled-segment competitor that would otherwise match
    _touch_shards(tmp_path / "rasa" / "rasa" / "hi")
    found = _resolve_parquet_shards(lang_dir)
    assert sorted(found) == sorted(direct)
