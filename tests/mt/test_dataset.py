"""The training-dataset loader: length filtering, caching, and the build lock.

Driven with a fake tokenizer so this needs no checkpoint and runs on CPU. The fake
mimics the one behaviour that matters here: ``apply_chat_template(tokenize=True,
return_dict=True)`` hands back a mapping whose ``input_ids`` is the token list.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("datasets", reason="the loader is built on HF datasets")
pytest.importorskip("filelock", reason="the cache build is serialised with a file lock")

from bodhan_genai.mt.data.dataset import (
    KEEP_COLUMN,
    load_training_dataset,
    rendered_length,
)


class FakeTokenizer:
    """One token per whitespace word across the whole conversation, plus 4 for the
    template's structural tokens (bos + turn markers)."""

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(
        self, messages, add_generation_prompt=False, tokenize=True, return_dict=True, **kwargs
    ):
        self.calls += 1
        words = sum(len(m["content"].split()) for m in messages)
        return {"input_ids": list(range(words + 4))}


class BatchedFakeTokenizer(FakeTokenizer):
    """Some tokenizer/template combinations return a batch of one; the loader has
    to cope rather than measuring len([[...]]) == 1."""

    def apply_chat_template(self, messages, **kwargs):
        out = super().apply_chat_template(messages, **kwargs)
        return {"input_ids": [out["input_ids"]]}


def _rows(n: int, words: int = 5) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": " ".join(["w"] * words)},
                {"role": "assistant", "content": " ".join(["t"] * words)},
            ],
            "corpus": "probe",
            "direction": "eng_Latn-hin_Deva",
        }
        for _ in range(n)
    ]


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return str(path)


# --------------------------------------------------------------------------- #
# Length measurement
# --------------------------------------------------------------------------- #


def test_rendered_length_measures_the_full_chat_not_the_raw_text():
    example = {
        KEEP_COLUMN: [
            {"role": "user", "content": "one two three"},
            {"role": "assistant", "content": "four five"},
        ]
    }
    # 5 words + 4 structural tokens.
    assert rendered_length(example, FakeTokenizer(), 100)["rendered_len"] == 9


def test_rendered_length_unwraps_a_batch_of_one():
    """The trap: apply_chat_template(tokenize=True) returns a BatchEncoding whose
    len() is the key count, not the token count — the length has to come off
    ["input_ids"], and that may itself be nested."""
    example = {KEEP_COLUMN: [{"role": "user", "content": "one two"}]}
    assert rendered_length(example, BatchedFakeTokenizer(), 100)["rendered_len"] == 6


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_keeps_only_the_messages_column(tmp_path):
    path = _write(tmp_path / "train.jsonl", _rows(6))
    ds = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "cache"), "train", 100, num_proc=1
    )
    assert ds.column_names == [KEEP_COLUMN]
    assert len(ds) == 6


def test_filters_rows_over_max_seq_length(tmp_path):
    rows = _rows(4, words=3) + _rows(3, words=50)  # 10 tokens vs 104 tokens
    path = _write(tmp_path / "train.jsonl", rows)
    ds = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "cache"), "train", 20, num_proc=1
    )
    assert len(ds) == 4


def test_boundary_length_is_kept(tmp_path):
    """<= max_seq_length, not <: a row exactly at the limit trains fine."""
    path = _write(tmp_path / "train.jsonl", _rows(1, words=3))  # 10 tokens
    ds = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "cache"), "train", 10, num_proc=1
    )
    assert len(ds) == 1


def test_everything_filtered_raises_instead_of_returning_empty(tmp_path):
    path = _write(tmp_path / "train.jsonl", _rows(3, words=50))
    with pytest.raises(ValueError, match="nothing left to train on"):
        load_training_dataset(
            FakeTokenizer(), path, str(tmp_path / "cache"), "train", 5, num_proc=1
        )


def test_missing_messages_column_raises_with_the_fix(tmp_path):
    path = _write(tmp_path / "train.jsonl", [{"instruction": "a", "target": "b"}])
    with pytest.raises(ValueError, match=r"bodhan_genai\.mt\.data\.render"):
        load_training_dataset(
            FakeTokenizer(), path, str(tmp_path / "cache"), "train", 100, num_proc=1
        )


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_second_call_hits_the_cache_without_rereading_the_corpus(tmp_path, monkeypatch):
    """The second call must not touch the raw JSONL at all.

    Asserted by making ``load_dataset`` explode rather than by counting tokenizer
    calls: HF datasets memoises ``.map`` on a content fingerprint of its own, so a
    call count can read as zero even on a genuine rebuild.
    """
    import datasets

    path = _write(tmp_path / "train.jsonl", _rows(5))
    cache = str(tmp_path / "cache")

    ds = load_training_dataset(FakeTokenizer(), path, cache, "train", 100, num_proc=1)
    assert len(ds) == 5
    assert (tmp_path / "cache" / "processed_dataset" / "train").exists()

    def _explode(*args, **kwargs):
        raise AssertionError("cache miss: the raw corpus was read again")

    monkeypatch.setattr(datasets, "load_dataset", _explode)
    cached = load_training_dataset(FakeTokenizer(), path, cache, "train", 100, num_proc=1)
    assert len(cached) == 5
    assert cached.column_names == [KEEP_COLUMN]


def test_splits_are_cached_separately(tmp_path):
    train = _write(tmp_path / "train.jsonl", _rows(6))
    dev = _write(tmp_path / "dev.jsonl", _rows(2))
    cache = str(tmp_path / "cache")

    train_ds = load_training_dataset(FakeTokenizer(), train, cache, "train", 100, num_proc=1)
    dev_ds = load_training_dataset(FakeTokenizer(), dev, cache, "dev", 100, num_proc=1)
    assert len(train_ds) == 6
    assert len(dev_ds) == 2


def test_the_build_is_serialised_with_a_per_split_file_lock(tmp_path, monkeypatch):
    """The lock is what lets every rank call this concurrently without an NCCL
    barrier; losing it is a silent regression to a race on the cache directory.

    Checked by recording the lock construction, not by looking for the file
    afterwards — filelock unlinks it on release.
    """
    import filelock

    acquired: list[str] = []
    real_lock = filelock.FileLock

    class RecordingLock(real_lock):  # type: ignore[misc, valid-type]
        def __init__(self, lock_file, *args, **kwargs):
            acquired.append(str(lock_file))
            super().__init__(lock_file, *args, **kwargs)

    monkeypatch.setattr(filelock, "FileLock", RecordingLock)

    cache = tmp_path / "cache"
    load_training_dataset(
        FakeTokenizer(),
        _write(tmp_path / "train.jsonl", _rows(3)),
        str(cache),
        "train",
        100,
        num_proc=1,
    )
    load_training_dataset(
        FakeTokenizer(),
        _write(tmp_path / "dev.jsonl", _rows(2, words=4)),
        str(cache),
        "dev",
        100,
        num_proc=1,
    )

    assert acquired == [
        str(cache / ".dataset_build.train.lock"),
        str(cache / ".dataset_build.dev.lock"),
    ], "each split must take its own lock, so train and dev do not serialise on each other"


def test_shuffle_seed_is_deterministic(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"row {i}"},
                {"role": "assistant", "content": f"t {i}"},
            ]
        }
        for i in range(30)
    ]
    path = _write(tmp_path / "train.jsonl", rows)

    a = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "c1"), "train", 100, num_proc=1, shuffle_seed=7
    )
    b = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "c2"), "train", 100, num_proc=1, shuffle_seed=7
    )
    c = load_training_dataset(
        FakeTokenizer(), path, str(tmp_path / "c3"), "train", 100, num_proc=1, shuffle_seed=99
    )

    order_a = [r["messages"][0]["content"] for r in a]
    order_b = [r["messages"][0]["content"] for r in b]
    order_c = [r["messages"][0]["content"] for r in c]
    assert order_a == order_b
    assert order_a != order_c
