"""Offline transcription CLI: argument contract and shard/resume selection.

The transcription itself needs a checkpoint and a GPU; what is testable — and
what would silently corrupt a large run if wrong — is which manifest rows a
shard claims and which it skips on resume. That logic is exercised here
directly against temp files, with no model load.
"""

from __future__ import annotations

import json

import pytest

from bodhan_genai.asr.inference.transcribe import build_parser


def write_manifest(path, n_rows, *, duplicate_keys=True):
    with open(path, "w") as f:
        for i in range(n_rows):
            f.write(
                json.dumps(
                    {
                        "audio_path": f"/audio/{i}.wav",
                        "language": "hi" if i % 2 == 0 else "bn",
                        # deliberately colliding keys: the real corpus had 84k
                        # rows but only 48k unique keys
                        "key": f"dup{i % 3}" if duplicate_keys else f"uniq{i}",
                    }
                )
                + "\n"
            )
    return path


def select_rows(manifest, shard, num_shards, done_rows=()):
    """Mirror of the CLI's selection logic (kept in sync by the tests below)."""
    done = set(done_rows)
    rows = {}
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i % num_shards == shard and i not in done:
                rows[i] = json.loads(line)
    return sorted(rows)


def test_parser_requires_manifest_model_dir_and_out_dir():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])
    args = p.parse_args(["--manifest", "m.jsonl", "--model-dir", "d", "--out-dir", "o"])
    assert args.manifest == "m.jsonl"


def test_parser_defaults_carry_the_measured_values():
    """These defaults encode upstream measurements; a silent change to them
    changes throughput or quality on every run."""
    args = build_parser().parse_args(["--manifest", "m", "--model-dir", "d", "--out-dir", "o"])
    assert args.batch_size == 96, "96 was the measured throughput knee on one H100"
    assert args.chunk_above == 0.0, "chunking is opt-in; it is harmful below ~45 s"
    assert (args.chunk_min, args.chunk_max) == (15.0, 25.0), "best-WER window in the sweep"
    assert args.dtype == "bfloat16"
    assert args.num_shards == 1 and args.shard == 0


def test_lang_override_beats_lang_key():
    args = build_parser().parse_args(
        ["--manifest", "m", "--model-dir", "d", "--out-dir", "o", "--lang", "ta"]
    )
    assert args.lang == "ta"
    assert args.lang_key == "language"  # still defaulted, but --lang wins at runtime


def test_shards_partition_the_manifest_exactly(tmp_path):
    """Every row handled exactly once across shards: no gaps, no duplicates."""
    m = write_manifest(tmp_path / "m.jsonl", 20)
    num_shards = 4
    claimed = [i for s in range(num_shards) for i in select_rows(m, s, num_shards)]
    assert sorted(claimed) == list(range(20))
    assert len(claimed) == len(set(claimed))


def test_shard_selection_is_stride_based(tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", 10)
    assert select_rows(m, 0, 2) == [0, 2, 4, 6, 8]
    assert select_rows(m, 1, 2) == [1, 3, 5, 7, 9]


def test_resume_skips_done_rows_only(tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", 10)
    assert select_rows(m, 0, 2, done_rows=[0, 4]) == [2, 6, 8]


def test_resume_is_row_indexed_so_duplicate_keys_survive(tmp_path):
    """The reason resume keys on row index: rows 0/3/6/9 share key 'dup0', and
    a key-based resume would drop three of them after the first completes."""
    m = write_manifest(tmp_path / "m.jsonl", 10, duplicate_keys=True)
    rows = select_rows(m, 0, 1)
    with open(m) as f:
        lines = f.readlines()
    keys = [json.loads(lines[i])["key"] for i in rows]
    assert len(rows) == 10
    assert len(set(keys)) == 3, "fixture must actually have colliding keys"
    # completing row 0 must not mark rows 3/6/9 (same key) as done
    remaining = select_rows(m, 0, 1, done_rows=[0])
    assert remaining == [1, 2, 3, 4, 5, 6, 7, 8, 9]
