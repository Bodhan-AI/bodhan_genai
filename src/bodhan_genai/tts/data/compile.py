"""
Stage 2: Compile tokenized Parquet into training-ready sequences.

Reads tokenized Parquet (from bodhan_genai.tts.data.tokenize) where each row already
contains token_ids + metadata, builds training sequences via chat templates,
and writes training Parquet (input_ids, labels, length).

Input schema (from Stage 1):
  text, token_ids (list<int32>), language, speaker, audio_caption

Output schema:
  input_ids  — list<int32>: full token sequence
  labels     — list<int32>: full copy of input_ids (full-sequence loss)
  length     — int32: token sequence length

Fault tolerance:
  compile_manifest.json tracks per-shard status. Completed shards are skipped.

Usage:
  python -m bodhan_genai.tts.data.compile --config configs/tts/data/compile.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from bodhan_genai.tts.templates.chat import build_sequence, get_template_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ROWS_PER_SHARD = 100_000
# Parquet row-group granularity inside each shard. ParquetTokenDataset
# (training/dataset.py) decodes a whole row group on every cache miss, so
# writing one giant row group per shard (the pyarrow default) makes random
# access prohibitively expensive. 1024 keeps per-miss decode in the low-MB
# range without inflating metadata.
ROW_GROUP_SIZE = 1024
TMP_COMPILE_DIRNAME = ".compile_tmp"

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("input_ids", pa.large_list(pa.int32())),
        pa.field("labels", pa.large_list(pa.int32())),
        pa.field("length", pa.int32()),
    ]
)

_WORKER_TOKENIZER = None
_WORKER_TMPL = None
_WORKER_DROP_STYLE = False


def _available_cpu_count() -> int:
    """Best-effort CPU count respecting scheduler affinity when available."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


class ParquetShardWriter:
    """Accumulates rows and flushes to Parquet shards periodically."""

    def __init__(
        self,
        output_dir: Path,
        shard_size: int = ROWS_PER_SHARD,
        shard_id_start: int = 0,
    ):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._shard_idx = shard_id_start
        self.total_written: int = 0
        self._schema = _OUTPUT_SCHEMA

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_path = self.output_dir / f"shard_{self._shard_idx:05d}.parquet"
        table = pa.table(
            {
                "input_ids": [row["input_ids"] for row in self._buffer],
                "labels": [row["labels"] for row in self._buffer],
                "length": [row["length"] for row in self._buffer],
            },
            schema=self._schema,
        )
        pq.write_table(
            table,
            shard_path,
            compression="snappy",
            row_group_size=ROW_GROUP_SIZE,
        )
        logger.info(f"Wrote {len(self._buffer)} rows → {shard_path}")
        self.total_written += len(self._buffer)
        self._shard_idx += 1
        self._buffer.clear()

    def close(self) -> None:
        self._flush()


def _sanitize_split_value(value: str | None, fallback: str = "unknown") -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return sanitized.strip("._-") or fallback


def _clear_output_shards(output_dir: Path) -> None:
    for shard_path in output_dir.glob("shard_*.parquet"):
        shard_path.unlink()


def _iter_temp_dataset_dirs(temp_base_dir: Path) -> list[Path]:
    if not temp_base_dir.exists():
        return []
    return sorted(
        path for path in temp_base_dir.rglob("*") if path.is_dir() and any(path.glob("*.parquet"))
    )


def _relative_split_dir(
    split_by_source: bool,
    split_by_language: bool,
    source_value: str,
    language_value: str,
    output_dir_name: str | None = None,
) -> Path:
    """Build the per-row sub-path under output_dir.

    When ``split_by_source`` is true but ``output_dir`` already ends with
    ``source_value`` (the common convention — configs set the dataset's name
    as the final segment of ``output_dir``), drop the redundant segment so we
    don't produce ``<output_dir>/<source>/<source>/<lang>``.
    """
    sanitized_source = _sanitize_split_value(source_value)
    parts: list[str] = []
    if split_by_source and sanitized_source != output_dir_name:
        parts.append(sanitized_source)
    if split_by_language:
        parts.append(_sanitize_split_value(language_value))
    return Path(*parts) if parts else Path("__all__")


def _rewrite_sorted_dataset(
    temp_dir: Path,
    final_dir: Path,
    shard_size: int,
) -> int:
    """Load all temp shards for one dataset split, sort by length desc, and rewrite."""
    temp_shards = sorted(temp_dir.glob("*.parquet"))
    if not temp_shards:
        logger.warning(f"No temp shards found in {temp_dir}; skipping final write.")
        return 0

    tables = []
    for shard_path in temp_shards:
        table = pq.read_table(str(shard_path))
        if table.schema != _OUTPUT_SCHEMA:
            table = table.cast(_OUTPUT_SCHEMA, safe=False)
        tables.append(table)
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    sorted_table = table.sort_by([("length", "descending")])

    final_dir.mkdir(parents=True, exist_ok=True)
    _clear_output_shards(final_dir)

    total_rows = sorted_table.num_rows
    for shard_idx, start in enumerate(range(0, total_rows, shard_size)):
        shard = sorted_table.slice(start, shard_size)
        shard_path = final_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(shard, shard_path, compression="snappy")
        logger.info(f"Wrote sorted shard {shard_idx} ({shard.num_rows} rows) → {shard_path}")

    return total_rows


class SplitWriterManager:
    """Routes compiled rows into split-specific temp shard writers."""

    def __init__(
        self,
        temp_base_dir: Path,
        final_base_dir: Path,
        shard_size: int,
        split_by_source: bool,
        split_by_language: bool,
    ) -> None:
        self.temp_base_dir = temp_base_dir
        self.final_base_dir = final_base_dir
        self.shard_size = shard_size
        self.split_by_source = split_by_source
        self.split_by_language = split_by_language
        self._writers: dict[Path, ParquetShardWriter] = {}

    def write(self, row: dict, source_value: str, language_value: str) -> None:
        rel_dir = _relative_split_dir(
            split_by_source=self.split_by_source,
            split_by_language=self.split_by_language,
            source_value=source_value,
            language_value=language_value,
            output_dir_name=self.final_base_dir.name,
        )
        writer = self._writers.get(rel_dir)
        if writer is None:
            temp_dir = self.temp_base_dir / rel_dir
            existing_shards = _count_existing_output_shards(temp_dir)
            writer = ParquetShardWriter(
                temp_dir,
                shard_size=self.shard_size,
                shard_id_start=existing_shards,
            )
            self._writers[rel_dir] = writer
        writer.write(row)

    def close_all(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def finalize(self) -> dict[str, int]:
        results: dict[str, int] = {}
        for temp_dir in _iter_temp_dataset_dirs(self.temp_base_dir):
            rel_dir = temp_dir.relative_to(self.temp_base_dir)
            final_dir = (
                self.final_base_dir if rel_dir == Path("__all__") else self.final_base_dir / rel_dir
            )
            results[str(final_dir)] = _rewrite_sorted_dataset(
                temp_dir=temp_dir,
                final_dir=final_dir,
                shard_size=self.shard_size,
            )
        return results


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "compile_manifest.json"


def _load_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save_manifest_atomic(manifest: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(path)


def _count_existing_output_shards(output_dir: Path) -> int:
    return len(sorted(output_dir.glob("shard_*.parquet")))


def _write_rows_atomic(output_path: Path, rows: list[dict]) -> None:
    """Write one parquet file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "input_ids": [row["input_ids"] for row in rows],
            "labels": [row["labels"] for row in rows],
            "length": [row["length"] for row in rows],
        },
        schema=_OUTPUT_SCHEMA,
    )

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        suffix=".parquet.tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pq.write_table(table, str(tmp_path), compression="snappy")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------------


def _init_worker(tokenizer_path: str, drop_style: bool = False) -> None:
    """Initializer for process workers so each process loads tokenizer once."""
    global _WORKER_TOKENIZER, _WORKER_TMPL, _WORKER_DROP_STYLE

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    _WORKER_DROP_STYLE = bool(drop_style)
    _WORKER_TMPL = get_template_ids(_WORKER_TOKENIZER)


# ---------------------------------------------------------------------------
# Single-shard processing
# ---------------------------------------------------------------------------


def _process_shard(
    shard_path: Path,
    training_mode: str,
    source_value: str,
    split_by_source: bool,
    split_by_language: bool,
    output_dir_name: str | None = None,
) -> tuple[int, dict[str, list[dict]], list[dict]]:
    """
    Process one tokenized Parquet shard into split-bucketed training sequences.

    Each row has token_ids + metadata directly — no join needed.

    Returns (rows_in, rows grouped by relative split directory, dropped rows).
    """
    if _WORKER_TOKENIZER is None or _WORKER_TMPL is None:
        raise RuntimeError("Worker tokenizer/template not initialized")

    table = pq.read_table(str(shard_path))
    rows_dict = table.to_pydict()
    n = table.num_rows
    del table

    rows_in = 0
    split_rows: dict[str, list[dict]] = {}
    failed: list[dict] = []

    for i in range(n):
        rows_in += 1

        token_ids = rows_dict["token_ids"][i] or []
        text = rows_dict.get("text", [""] * n)[i] or ""
        language = rows_dict.get("language", [""] * n)[i] or ""
        speaker = rows_dict.get("speaker", [""] * n)[i] or ""
        style = "" if _WORKER_DROP_STYLE else (rows_dict.get("style", [""] * n)[i] or "")
        accent = rows_dict.get("accent", [""] * n)[i] or ""
        audio_caption = rows_dict.get("audio_caption", [""] * n)[i] or ""

        # Optional Stage-1 column: conversation rows carry pre-formatted
        # <|speaker>...<speaker|> text plus an is_conversation flag so
        # build_sequence dispatches to the conversation template.
        is_conversation = bool(rows_dict.get("is_conversation", [False] * n)[i])

        entry = {
            "token_ids": token_ids,
            "text": text,
            "is_conversation": is_conversation,
            "speaker_id": speaker,
            "style": style,
            "accent": accent,
            "audio_caption": audio_caption,
        }

        result = build_sequence(
            entry=entry,
            tokenizer=_WORKER_TOKENIZER,
            tmpl=_WORKER_TMPL,
        )
        if result is None:
            # build_sequence rejected this row (missing/short/unhandled). Record it
            # rather than drop silently — driven up to <output_dir>/compile_failed/.
            failed.append(
                {
                    "text": text,
                    "audio_filepath": rows_dict.get("audio_filepath", [""] * n)[i] or "",
                    "language": language,
                }
            )
            continue

        rel_dir = _relative_split_dir(
            split_by_source=split_by_source,
            split_by_language=split_by_language,
            source_value=source_value,
            language_value=language,
            output_dir_name=output_dir_name,
        )
        split_rows.setdefault(str(rel_dir), []).append(result)

    return rows_in, split_rows, failed


def _process_shard_to_temp(
    shard_id: int,
    shard_path: str,
    temp_base_dir: str,
    training_mode: str,
    source_value: str,
    split_by_source: bool,
    split_by_language: bool,
) -> tuple[int, int, int]:
    """Worker entry point: process one shard and write deterministic temp parts."""
    # temp_base_dir is <output_dir>/.compile_tmp, so its parent's name is the
    # dataset output dir name — pass it so the temp rel-dir dedups the source
    # segment (matching SplitWriterManager.write) instead of doubling it.
    rows_in, split_rows, failed = _process_shard(
        shard_path=Path(shard_path),
        training_mode=training_mode,
        source_value=source_value,
        split_by_source=split_by_source,
        split_by_language=split_by_language,
        output_dir_name=Path(temp_base_dir).parent.name,
    )

    rows_out = 0
    temp_root = Path(temp_base_dir)
    for rel_dir_str, rows in split_rows.items():
        if not rows:
            continue
        rel_dir = Path(rel_dir_str)
        output_path = temp_root / rel_dir / f"part_{shard_id:05d}.parquet"
        _write_rows_atomic(output_path, rows)
        rows_out += len(rows)

    # Record dropped rows to a unique per-shard file OUTSIDE .compile_tmp (so it
    # survives finalize). One file per shard => no cross-worker write contention.
    if failed:
        failed_path = temp_root.parent / "compile_failed" / f"shard_{shard_id:05d}.jsonl"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "w", encoding="utf-8") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    return shard_id, rows_in, rows_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Compile tokenized datasets to training Parquet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split-by-language", action="store_true")
    parser.add_argument("--split-by-source", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer_path: str = cfg["models"]["tokenizer_path"]
    processing_cfg = cfg.get("processing", {})
    training_mode: str = processing_cfg.get("training_mode", "sft")
    # When true, drop the per-row `style` tag from the sequence (no style
    # conditioning) — e.g. Rasa's CONV/NEWS/HAPPY domain+emotion labels.
    drop_style: bool = bool(processing_cfg.get("drop_style", False))
    split_by_language: bool = args.split_by_language or processing_cfg.get(
        "split_by_language", False
    )
    split_by_source: bool = args.split_by_source or processing_cfg.get("split_by_source", False)
    configured_num_workers = int(processing_cfg.get("num_workers", 0) or 0)

    if training_mode != "sft":
        raise ValueError(
            f"training_mode must be 'sft', got {training_mode!r}. The Llama route always "
            "trains with full-sequence loss, so the Orpheus-style pretraining route IS the "
            "sft template with full loss — use training_mode: sft."
        )

    datasets_cfg: list[dict] = cfg["input"]["datasets"]

    for ds_cfg in datasets_cfg:
        input_dir = Path(ds_cfg["input_dir"])
        output_dir = Path(ds_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Input:         {input_dir}")
        logger.info(f"Output:        {output_dir}")
        logger.info(f"Training mode: {training_mode} (full-sequence loss)")
        logger.info(f"Split by source:   {split_by_source}")
        logger.info(f"Split by language: {split_by_language}")

        source_value = ds_cfg.get("source_name") or input_dir.name
        temp_output_dir = output_dir / TMP_COMPILE_DIRNAME

        input_shards = sorted(input_dir.glob("*.parquet"))
        if not input_shards:
            logger.warning(f"No .parquet files in {input_dir}. Skipping.")
            continue
        logger.info(f"Input shards:  {len(input_shards)}")

        mf_path = _manifest_path(output_dir)
        if mf_path.exists():
            # Key by source path, not positional id. Previous positional-id
            # scheme silently corrupted resume state if shard count changed
            # between runs: existing_manifest[idx=N] would map to a DIFFERENT
            # input_shards[N] file. Path-keyed dedup tolerates added/removed
            # shards and only resumes "done" entries whose path is still present.
            existing_by_path = {
                str(item.get("path", "")): item
                for item in _load_manifest(mf_path)
                if item.get("path")
            }
            manifest = []
            for idx, sp in enumerate(input_shards):
                item = existing_by_path.get(str(sp), {})
                manifest.append(
                    {
                        "id": idx,
                        "path": str(sp),
                        "status": item.get("status", "pending"),
                        "rows_in": int(item.get("rows_in", 0) or 0),
                        "rows_out": int(item.get("rows_out", 0) or 0),
                        "error": item.get("error"),
                    }
                )
            done = sum(1 for s in manifest if s["status"] == "done")
            stale = len(existing_by_path) - sum(
                1 for sp in input_shards if str(sp) in existing_by_path
            )
            logger.info(
                f"Resuming: {done}/{len(manifest)} shards done"
                + (f" (dropped {stale} stale manifest entries)" if stale else "")
            )
            _save_manifest_atomic(manifest, mf_path)
        else:
            manifest = [
                {
                    "id": idx,
                    "path": str(sp),
                    "status": "pending",
                    "rows_in": 0,
                    "rows_out": 0,
                    "error": None,
                }
                for idx, sp in enumerate(input_shards)
            ]
            _save_manifest_atomic(manifest, mf_path)

        all_done = all(s["status"] == "done" for s in manifest)
        if all_done and not temp_output_dir.exists():
            total_out = sum(s["rows_out"] for s in manifest)
            logger.info(f"All shards done ({total_out} rows). Nothing to do.")
            continue

        pending = [shard_meta for shard_meta in manifest if shard_meta["status"] != "done"]
        if pending:
            num_workers = (
                configured_num_workers if configured_num_workers > 0 else _available_cpu_count()
            )
            num_workers = max(1, min(num_workers, len(pending)))
            logger.info(f"Compiling pending shards with {num_workers} worker process(es)")

            future_to_meta = {}
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(tokenizer_path, drop_style),
            ) as executor:
                for shard_meta in pending:
                    shard_idx = shard_meta["id"]
                    logger.info(
                        f"Submitting shard {shard_idx}/{len(manifest) - 1}: "
                        f"{Path(shard_meta['path']).name}"
                    )
                    future = executor.submit(
                        _process_shard_to_temp,
                        shard_idx,
                        shard_meta["path"],
                        str(temp_output_dir),
                        training_mode,
                        source_value,
                        split_by_source,
                        split_by_language,
                    )
                    future_to_meta[future] = shard_meta

                for future in as_completed(future_to_meta):
                    shard_meta = future_to_meta[future]
                    shard_idx = shard_meta["id"]
                    try:
                        _finished_id, rows_in, rows_out = future.result()
                        shard_meta["status"] = "done"
                        shard_meta["rows_in"] = rows_in
                        shard_meta["rows_out"] = rows_out
                        shard_meta["error"] = None
                        logger.info(f"Shard {shard_idx} done: {rows_in} in, {rows_out} out")
                    except Exception as e:
                        shard_meta["status"] = "failed"
                        shard_meta["error"] = str(e)
                        logger.error(f"Shard {shard_idx} failed: {e}", exc_info=True)
                    finally:
                        _save_manifest_atomic(manifest, mf_path)

        done = [s for s in manifest if s["status"] == "done"]
        failed = [s for s in manifest if s["status"] == "failed"]
        total_out = sum(s["rows_out"] for s in done)
        logger.info(
            f"Complete: {len(done)}/{len(manifest)} shards, {len(failed)} failed, {total_out} rows."
        )
        if failed:
            logger.warning(f"Failed: {[s['id'] for s in failed]}")
            continue

        if temp_output_dir.exists():
            logger.info("Finalizing sorted output shards ...")
            # Always wipe stale top-level shards. Previously this was gated on
            # split_by_source/split_by_language, but without splits old shards
            # from a prior run silently get mixed with new (unsorted) output.
            # Split-aware sub-dirs are cleared inside _rewrite_sorted_dataset.
            _clear_output_shards(output_dir)
            finalizer = SplitWriterManager(
                temp_base_dir=temp_output_dir,
                final_base_dir=output_dir,
                shard_size=ROWS_PER_SHARD,
                split_by_source=split_by_source,
                split_by_language=split_by_language,
            )
            written = finalizer.finalize()
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            for target_dir, rows in written.items():
                logger.info(f"Finalized {rows} rows → {target_dir}")

    logger.info("\nAll datasets compiled.")


if __name__ == "__main__":
    main()
