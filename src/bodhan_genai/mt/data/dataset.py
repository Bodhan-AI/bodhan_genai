"""Training-dataset loading: rendered JSONL -> a cached, length-filtered HF dataset.

The trainer wants a ``datasets.Dataset`` whose only column is ``messages``. This
module gets it there and caches the result, because tokenizing a multi-million-row
corpus takes long enough that doing it once per rank per run is not acceptable.

Two details that look incidental and are not:

*   **A file lock, not a distributed barrier.** Under ``accelerate launch`` every
    rank runs this code. Coordinating with ``dist.barrier()`` means initialising
    the process group early and risking an NCCL timeout on a big corpus. A
    ``filelock.FileLock`` has neither problem: the first rank in builds the cache,
    the rest block on the lock and then load it from disk in seconds.
*   **The length filter measures the rendered chat prompt**, not the raw text.
    Filtering on ``len(text)`` lets rows through that blow past ``max_seq_length``
    once the template and special tokens are added, and TRL then truncates them —
    silently teaching the model to emit unterminated translations.

Heavy imports (``datasets``, ``filelock``) live inside the functions so importing
this module stays cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mt.data.dataset")

#: Everything except this is dropped before caching — provenance columns are
#: useful in the rendered JSONL but would just bloat the tokenizer cache.
KEEP_COLUMN = "messages"


def rendered_length(example: dict[str, Any], tokenizer, max_seq_length: int) -> dict[str, Any]:
    """Measure one example's fully rendered chat length.

    Returns the example plus ``rendered_len``. The trap this avoids:
    ``apply_chat_template(..., tokenize=True)`` returns a ``BatchEncoding``, and
    ``len()`` on that is 2 (the number of keys), not the token count — so the
    length must be read off ``["input_ids"]``.
    """
    encoded = tokenizer.apply_chat_template(
        example[KEEP_COLUMN],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
    )
    ids = encoded["input_ids"]
    # Some tokenizer/template combinations return a batch of one.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return {"rendered_len": len(ids)}


def load_training_dataset(
    tokenizer,
    filename: str,
    cache_dir: str,
    split: str,
    max_seq_length: int,
    num_proc: int = 16,
    shuffle_seed: int = 42,
):
    """Load rendered JSONL into a cached ``datasets.Dataset`` of ``messages``.

    ``split`` names the cache subdirectory ("train" / "dev"), so the two splits of
    one run never collide. Re-running with an existing cache returns it directly;
    delete ``<cache_dir>/processed_dataset/<split>`` to force a rebuild.

    The tokenizer must already carry the training chat template
    (``GEMMA4_TRL_TEMPLATE``) — the length filter has to measure what the trainer
    will actually feed the model.
    """
    from datasets import load_dataset, load_from_disk
    from filelock import FileLock

    cache_root = Path(cache_dir)
    cache_path = cache_root / "processed_dataset" / split
    lock_path = cache_root / f".dataset_build.{split}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # One rank builds; the others queue here and then hit the fast path above.
    with FileLock(str(lock_path)):
        if cache_path.exists():
            logger.info("loading cached %s dataset from %s", split, cache_path)
            return load_from_disk(str(cache_path))

        logger.info("building %s dataset from %s", split, filename)
        ds = load_dataset("json", data_files=filename)["train"]
        before = len(ds)

        drop = [c for c in ds.column_names if c != KEEP_COLUMN]
        if KEEP_COLUMN not in ds.column_names:
            raise ValueError(
                f"{filename}: expected a {KEEP_COLUMN!r} column — render the corpus "
                f"with `python -m bodhan_genai.mt.data.render` first "
                f"(found: {', '.join(ds.column_names)})"
            )

        ds = ds.map(
            lambda x: rendered_length(x, tokenizer, max_seq_length),
            remove_columns=drop,
            num_proc=num_proc,
            desc=f"measuring {split} lengths",
        )
        ds = ds.filter(
            lambda x: x["rendered_len"] <= max_seq_length,
            num_proc=num_proc,
            desc=f"filtering {split} to <= {max_seq_length} tokens",
        )
        after = len(ds)
        if after < before:
            logger.info(
                "%s: dropped %d/%d rows over %d tokens (%.2f%%)",
                split,
                before - after,
                before,
                max_seq_length,
                100.0 * (before - after) / max(before, 1),
            )
        if after == 0:
            raise ValueError(
                f"{filename}: every row exceeds max_seq_length={max_seq_length}; "
                f"nothing left to train on"
            )

        ds = ds.remove_columns(["rendered_len"])
        ds = ds.shuffle(seed=shuffle_seed)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(cache_path))
        logger.info("cached %d %s rows -> %s", after, split, cache_path)
        return ds
