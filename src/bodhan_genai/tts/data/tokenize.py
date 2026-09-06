"""
SNAC-encode audio datasets to sharded Parquet.

Two source types are supported in the config:

  - jsonl      : path to a JSONL file. Each line is a JSON object whose
                 `columns.audio` field holds an audio filepath. Audio is
                 loaded via torchaudio and resampled to 24 kHz.

  - hf_dataset : name of a HuggingFace Hub dataset (optional `subset` /
                 `split` / `trust_remote_code`). Audio is read from an
                 HF Audio feature — already decoded by the `datasets`
                 library — and resampled to 24 kHz if needed.

Output schema (identical for both sources):
  text             — string
  token_ids        — list<int32>: SNAC tokens for main audio
  language         — string
  speaker          — string
  style            — string
  accent           — string
  audio_caption    — string

Fault tolerance:
  tokenize_manifest.json tracks per-shard status. Completed shards are skipped
  on restart.

Config format:

  datasets:
    # JSONL source
    - source:
        jsonl: /path/to/manifest.jsonl
      columns:
        audio: audio_filepath
        text:  text
        # optional: language, speaker, style, accent, audio_caption
      output:
        dir: /path/to/output
        rows_per_shard: 10000

    # HF Hub source
    - source:
        hf_dataset: mozilla-foundation/common_voice_17_0
        subset: en
        split: train
        trust_remote_code: false
      columns:
        audio: audio          # HF Audio feature column
        text:  sentence
        speaker: client_id
      output:
        dir: /path/to/output
        rows_per_shard: 10000

Usage:
  python -m bodhan_genai.tts.data.tokenize --config configs/tts/data/tokenize.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from bodhan_genai.tts.codec.snac import batch_encode_audio, encode_audio, load_snac_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 24_000


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string()),
        pa.field("token_ids", pa.list_(pa.int32())),
        pa.field("language", pa.string()),
        pa.field("speaker", pa.string()),
        pa.field("style", pa.string()),
        pa.field("accent", pa.string()),
        pa.field("audio_caption", pa.string()),
        # The row's own audio_filepath, kept as a string for provenance /
        # debugging (JSONL sources only; empty for HF Audio features).
        pa.field("audio_filepath", pa.string()),
    ]
)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "tokenize_manifest.json"


def _load_manifest(path: Path) -> tuple[list[dict], int | None]:
    """Return (shard_list, rows_per_shard_from_manifest). The persisted format
    is either the legacy bare-list form (rows_per_shard unknown) or the new
    wrapped form ``{"rows_per_shard": N, "shards": [...]}``."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    return data.get("shards", []), data.get("rows_per_shard")


def _save_manifest_atomic(
    manifest: list[dict],
    path: Path,
    rows_per_shard: int | None = None,
) -> None:
    tmp = path.with_suffix(".json.tmp")
    payload: dict | list = (
        {"rows_per_shard": int(rows_per_shard), "shards": manifest}
        if rows_per_shard is not None
        else manifest
    )
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(path)


def _check_rows_per_shard(
    mf_path: Path,
    persisted_rows_per_shard: int | None,
    current_rows_per_shard: int,
) -> None:
    """Refuse to resume if the user changed rows_per_shard between runs —
    shard boundaries shift, so old `done` entries would map to NEW shards
    holding different rows. Legacy manifests (persisted_rows_per_shard=None)
    just emit a one-line warning so existing pipelines aren't broken."""
    if persisted_rows_per_shard is None:
        logger.warning(
            f"{mf_path}: legacy manifest without rows_per_shard metadata; "
            f"resume assumes current rows_per_shard={current_rows_per_shard} "
            "matches the original run."
        )
        return
    if int(persisted_rows_per_shard) != int(current_rows_per_shard):
        raise ValueError(
            f"{mf_path}: rows_per_shard changed between runs "
            f"(was {persisted_rows_per_shard}, now {current_rows_per_shard}). "
            "Shard boundaries shift, so existing `done` shards would map to "
            "different rows. Delete the manifest + output dir to start fresh, "
            "or revert rows_per_shard in the config."
        )


def _sync_manifest(manifest: list[dict], num_shards: int) -> list[dict]:
    synced: list[dict] = []
    for shard_id in range(num_shards):
        existing = manifest[shard_id] if shard_id < len(manifest) else {}
        synced.append(
            {
                "id": shard_id,
                "status": existing.get("status", "pending"),
                "rows": existing.get("rows", 0),
                "error": existing.get("error"),
            }
        )
    return synced


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {lineno} in {path}: {e}")
    return rows


def _shard_rows(rows: list[dict], rows_per_shard: int) -> list[list[dict]]:
    if rows_per_shard <= 0:
        raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard}")
    n = len(rows)
    num_shards = math.ceil(n / rows_per_shard)
    return [rows[i * rows_per_shard : (i + 1) * rows_per_shard] for i in range(num_shards)]


def _resolve_audio_path(filepath: str, source_root: str) -> str:
    if not filepath:
        return ""
    path = Path(filepath).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(source_root) / path).resolve())


# ---------------------------------------------------------------------------
# Audio loading helpers (used inside actors)
# ---------------------------------------------------------------------------


def _load_audio_from_path(filepath: str) -> np.ndarray:
    """
    Load audio from a filepath using torchaudio, mix down to mono, resample
    to 24 kHz, return as float32 numpy array. Empty array on failure.
    """
    import torchaudio

    if not filepath:
        return np.zeros(0, dtype=np.float32)
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception as e:
        logger.warning(f"Failed to load audio {filepath}: {e}")
        return np.zeros(0, dtype=np.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
        )
    return waveform.squeeze(0).numpy().astype(np.float32)


def _audio_from_hf_feature(audio: Any) -> np.ndarray:
    """
    Extract a float32 mono 24 kHz waveform from an HF Audio feature value.

    `datasets >= 3.0` returns a torchcodec `AudioDecoder` (not a dict) which
    supports `__getitem__` for "array" / "sampling_rate" / "path" but does NOT
    implement `.get()`. Older `datasets` versions return a plain dict. We use
    bracket indexing (works for both) and fall back to `get_all_samples()` if
    the feature only exposes the decoder API.
    """
    import torchaudio

    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(audio, str):
        return _load_audio_from_path(audio)

    array: Any = None
    sr: int = 0

    try:
        array = audio["array"]
        sr = int(audio["sampling_rate"])
    except (KeyError, TypeError, AttributeError):
        try:
            samples = audio.get_all_samples()
            data = samples.data
            array = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
            sr = int(samples.sample_rate)
        except Exception:
            array = None
            sr = 0

    if array is None or sr <= 0:
        path: Any = None
        try:
            path = audio["path"]
        except (KeyError, TypeError, AttributeError):
            path = None
        if isinstance(path, str):
            return _load_audio_from_path(path)
        return np.zeros(0, dtype=np.float32)

    waveform = torch.as_tensor(np.asarray(array, dtype=np.float32))
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if sr != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
        )
    return waveform.squeeze(0).numpy().astype(np.float32)


def _row_waveform(row: dict, col: str, source_type: str, source_root: str) -> np.ndarray:
    """Extract a waveform from a row given the source type (jsonl | hf)."""
    if not col:
        return np.zeros(0, dtype=np.float32)
    value = row.get(col)
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if source_type == "hf":
        return _audio_from_hf_feature(value)
    # jsonl
    path = _resolve_audio_path(value if isinstance(value, str) else "", source_root)
    return _load_audio_from_path(path)


# ---------------------------------------------------------------------------
# Ray Actor for GPU SNAC encoding
# ---------------------------------------------------------------------------


class SNACTokenizeActor:
    """Ray actor that owns a SNAC model on a single GPU.

    Defined as a plain class so this module imports without ray; main()
    wraps it with ``ray.remote(SNACTokenizeActor)`` before spawning actors.
    """

    _LONG_AUDIO_THRESHOLD = 60 * TARGET_SAMPLE_RATE  # 60 s at 24 kHz

    def __init__(
        self,
        snac_model_path: str,
        audio_token_base_id: int,
        encode_batch_size: int = 8,
    ) -> None:
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._audio_token_base_id = audio_token_base_id
        self._encode_batch_size = encode_batch_size
        self._snac_model = load_snac_model(snac_model_path, device=self._device)
        logger.info(f"SNACTokenizeActor ready on {self._device}")

    def _encode_waveforms(self, waveforms: list[np.ndarray]) -> list[list[int]]:
        """Encode a list of waveforms, batching short ones and falling back for long."""
        short: list[np.ndarray] = []
        short_indices: list[int] = []
        results: list[list[int]] = [[] for _ in waveforms]

        for i, w in enumerate(waveforms):
            if len(w) == 0:
                continue
            if len(w) > self._LONG_AUDIO_THRESHOLD:
                try:
                    results[i] = encode_audio(
                        self._snac_model, w, self._audio_token_base_id, device=self._device
                    )
                except Exception as e:
                    logger.warning(f"Long audio encode failed: {e}")
            else:
                short.append(w)
                short_indices.append(i)

        for batch_start in range(0, len(short), self._encode_batch_size):
            batch_wavs = short[batch_start : batch_start + self._encode_batch_size]
            batch_idxs = short_indices[batch_start : batch_start + self._encode_batch_size]
            try:
                encoded = batch_encode_audio(
                    self._snac_model, batch_wavs, self._audio_token_base_id, device=self._device
                )
                for j, idx in enumerate(batch_idxs):
                    results[idx] = encoded[j] if j < len(encoded) else []
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM on batch of {len(batch_wavs)}, falling back to single")
                torch.cuda.empty_cache()
                for j, idx in enumerate(batch_idxs):
                    with contextlib.suppress(Exception):
                        results[idx] = encode_audio(
                            self._snac_model,
                            batch_wavs[j],
                            self._audio_token_base_id,
                            device=self._device,
                        )

        return results

    def process_rows(
        self,
        shard_id: int,
        rows: list[dict],
        source_type: str,
        source_root: str,
        col_audio: str,
        col_text: str,
        col_language: str,
        col_speaker: str,
        col_style: str,
        col_accent: str,
        col_audio_caption: str,
    ) -> tuple[int, list[dict], list[str], list[dict]]:
        """
        Process a shard delivered as plain dicts. Rows may come from:
          - JSONL (source_type="jsonl"): audio fields are filepath strings
          - HF Hub (source_type="hf"):   audio fields are HF Audio dicts
                                         with pre-decoded arrays.
        """
        errors: list[str] = []
        output_rows: list[dict] = []
        failed_rows: list[dict] = []

        audio_waveforms: list[np.ndarray] = []

        for row in rows:
            audio_waveforms.append(_row_waveform(row, col_audio, source_type, source_root))

        token_ids_list = self._encode_waveforms(audio_waveforms)

        n_dropped = 0
        for i, row in enumerate(rows):
            # For JSONL sources col_audio is a string filepath; for HF sources
            # it's an Audio dict (no top-level path), so audio_filepath ends up
            # empty there. Path-based downstream pairings only apply to JSONL.
            audio_fp = row.get(col_audio, "") if source_type == "jsonl" else ""
            if not isinstance(audio_fp, str):
                audio_fp = ""
            # Drop rows whose audio failed to encode. _encode_waveforms swallows
            # per-row exceptions and leaves token_ids_list[i] = []; without this
            # guard we'd silently write zero-length sequences into parquet and
            # corrupt downstream packing. Distinguish "no audio input" (len(w)==0)
            # — for which empty token_ids are expected — from "had audio, failed
            # to encode": the latter is a real drop and gets surfaced.
            if not token_ids_list[i]:
                # No token_ids: either audio failed to encode, or the row had no
                # audio input. Either way record the ORIGINAL row to failed.jsonl
                # (driver side) so nothing is silently dropped.
                failed_rows.append(row)
                if len(audio_waveforms[i]) > 0:
                    errors.append(f"encode_failed: shard={shard_id} idx={i} path={audio_fp!r}")
                    logger.warning(
                        f"Dropping row with failed audio encode: shard={shard_id} "
                        f"idx={i} path={audio_fp!r}"
                    )
                n_dropped += 1
                continue
            output_rows.append(
                {
                    "text": row.get(col_text, "") or "",
                    "token_ids": token_ids_list[i],
                    "language": row.get(col_language, "") or "",
                    "speaker": row.get(col_speaker, "") or "",
                    "style": row.get(col_style, "") or "",
                    "accent": row.get(col_accent, "") or "",
                    "audio_caption": row.get(col_audio_caption, "") or "",
                    "audio_filepath": audio_fp,
                }
            )

        if n_dropped:
            logger.info(
                f"shard {shard_id}: dropped {n_dropped}/{len(rows)} rows "
                f"(encode failed or empty audio input)"
            )
        return shard_id, output_rows, errors, failed_rows


def _append_failed_rows(output_dir: Path, rows: list[dict]) -> None:
    """Record rows that could not be tokenized (audio encode failed or no audio
    input) verbatim to <output_dir>/failed.jsonl so nothing is silently dropped."""
    if not rows:
        return
    fp = Path(output_dir) / "failed.jsonl"
    with open(fp, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


def _write_shard_atomic(output_dir: Path, shard_id: int, rows: list[dict]) -> Path:
    """Write tokenized rows to Parquet atomically."""
    final_path = output_dir / f"shard_{shard_id:05d}.parquet"
    if final_path.exists():
        return final_path

    table = pa.table(
        {
            "text": [r["text"] for r in rows],
            "token_ids": [pa.array(r["token_ids"], type=pa.int32()) for r in rows],
            "language": [r["language"] for r in rows],
            "speaker": [r["speaker"] for r in rows],
            "style": [r.get("style", "") for r in rows],
            "accent": [r.get("accent", "") for r in rows],
            "audio_caption": [r["audio_caption"] for r in rows],
            "audio_filepath": [r.get("audio_filepath", "") for r in rows],
        },
        schema=_OUTPUT_SCHEMA,
    )

    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".parquet.tmp", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.rename(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return final_path


# ---------------------------------------------------------------------------
# Shard iteration over sources
# ---------------------------------------------------------------------------


def _iter_hf_shards(ds, rows_per_shard: int) -> Iterator[tuple[int, list[dict]]]:
    """
    Stream an HF Dataset one shard at a time to keep driver memory bounded.
    Each shard is a list of plain dicts (audio feature already decoded).
    """
    n = len(ds)
    num_shards = math.ceil(n / rows_per_shard) if n > 0 else 0
    for shard_id in range(num_shards):
        start = shard_id * rows_per_shard
        end = min(start + rows_per_shard, n)
        # ds.select + to_list materializes this shard's rows into Python dicts
        # with HF Audio features decoded (via the `datasets` Audio() feature).
        shard_ds = ds.select(range(start, end))
        rows = shard_ds.to_list()
        yield shard_id, rows


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------


def _resolve_columns(columns: dict) -> dict:
    return {
        "audio": columns.get("audio", "audio_filepath"),
        "text": columns.get("text", "text"),
        "language": columns.get("language", "language"),
        "speaker": columns.get("speaker") or "",
        "style": columns.get("style") or "",
        "accent": columns.get("accent") or "",
        "audio_caption": columns.get("audio_caption") or "",
    }


def _run_shards(
    pool,
    output_dir: Path,
    source_type: str,
    source_root: str,
    cols: dict,
    shards: Iterator[tuple[int, list[dict]]],
    total_shards: int | None,
    manifest: list[dict],
    mf_path: Path,
    rows_per_shard: int | None = None,
) -> None:
    """Submit pending shards to the actor pool and write results as they return."""
    from tqdm import tqdm

    pending: list[tuple[int, list[dict]]] = []
    for shard_id, rows in shards:
        if shard_id < len(manifest) and manifest[shard_id]["status"] == "done":
            continue
        pending.append((shard_id, rows))

    if not pending:
        logger.info("All shards already done. Nothing to do.")
        return

    def _submit(actor, item):
        shard_idx, shard_rows = item
        return actor.process_rows.remote(
            shard_idx,
            shard_rows,
            source_type,
            source_root,
            cols["audio"],
            cols["text"],
            cols["language"],
            cols["speaker"],
            cols["style"],
            cols["accent"],
            cols["audio_caption"],
        )

    results_iter = pool.map_unordered(_submit, pending)
    total_rows = 0
    total_errors = 0
    total_failed = 0
    total = total_shards if total_shards is not None else len(pending)

    with tqdm(total=len(pending), desc="Tokenizing shards") as pbar:
        for shard_id, output_rows, errors, failed_rows in results_iter:
            total_errors += len(errors)
            if output_rows:
                _write_shard_atomic(output_dir, shard_id, output_rows)
                total_rows += len(output_rows)
            if failed_rows:
                _append_failed_rows(output_dir, failed_rows)
                total_failed += len(failed_rows)
            for err in errors:
                logger.warning(f"Error: {err}")

            # The shard IS complete: its untokenizable rows are recorded to
            # failed.jsonl (not lost), and re-running would deterministically
            # re-fail them. So mark "done" even when rows were sidelined — this
            # avoids the old behaviour where any encode error re-ran the shard
            # forever. `error` is kept for visibility.
            manifest[shard_id]["status"] = "done"
            manifest[shard_id]["rows"] = len(output_rows)
            manifest[shard_id]["error"] = "; ".join(errors) if errors else None
            _save_manifest_atomic(manifest, mf_path, rows_per_shard=rows_per_shard)

            pbar.update(1)
            pbar.set_postfix(rows=total_rows, errors=total_errors, sidelined=total_failed)

    done = sum(1 for s in manifest if s["status"] == "done")
    failed = sum(1 for s in manifest if s["status"] == "failed")
    logger.info(
        f"Complete: {done}/{total} shards, {failed} failed, {total_rows} rows written, "
        f"{total_failed} rows sidelined to failed.jsonl"
    )


def _process_jsonl_dataset(ds_cfg: dict, pool) -> None:
    """Process a single JSONL-sourced dataset end-to-end."""
    jsonl_path: str = ds_cfg["source"]["jsonl"]
    cols = _resolve_columns(ds_cfg.get("columns", {}))
    output_cfg: dict = ds_cfg.get("output", {})
    output_dir = Path(output_cfg["dir"])
    rows_per_shard: int = int(output_cfg.get("rows_per_shard", 10_000))
    source_root = str(Path(jsonl_path).resolve().parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"JSONL input: {jsonl_path}")
    logger.info(f"Output:      {output_dir}  (rows_per_shard={rows_per_shard})")

    logger.info("Loading JSONL...")
    all_rows = _load_jsonl(jsonl_path)
    logger.info(f"Total rows: {len(all_rows)}")
    if not all_rows:
        logger.info("JSONL is empty. Nothing to tokenize.")
        return

    shards = _shard_rows(all_rows, rows_per_shard)
    logger.info(f"Shards: {len(shards)}")

    mf_path = _manifest_path(output_dir)
    if mf_path.exists():
        loaded, persisted_rps = _load_manifest(mf_path)
        _check_rows_per_shard(mf_path, persisted_rps, rows_per_shard)
        manifest = _sync_manifest(loaded, len(shards))
        done = sum(1 for s in manifest if s.get("status") == "done")
        logger.info(f"Resuming: {done}/{len(manifest)} shards done")
    else:
        manifest = [
            {"id": i, "status": "pending", "rows": 0, "error": None} for i in range(len(shards))
        ]
    _save_manifest_atomic(manifest, mf_path, rows_per_shard=rows_per_shard)

    if all(s["status"] == "done" for s in manifest):
        logger.info("All shards done. Nothing to do.")
        return

    shard_iter = ((shard_id, rows) for shard_id, rows in enumerate(shards))
    _run_shards(
        pool=pool,
        output_dir=output_dir,
        source_type="jsonl",
        source_root=source_root,
        cols=cols,
        shards=shard_iter,
        total_shards=len(shards),
        manifest=manifest,
        mf_path=mf_path,
        rows_per_shard=rows_per_shard,
    )


def _process_hf_dataset(ds_cfg: dict, pool, cache_dir: str | None) -> None:
    """Process a HuggingFace Hub dataset end-to-end (audio decoded in driver)."""
    from datasets import load_dataset

    source_cfg = ds_cfg["source"]
    hf_name: str = source_cfg["hf_dataset"]
    subset = source_cfg.get("subset")
    split = source_cfg.get("split", "train")
    trust_remote_code = bool(source_cfg.get("trust_remote_code", False))

    cols = _resolve_columns(ds_cfg.get("columns", {}))
    output_cfg: dict = ds_cfg.get("output", {})
    output_dir = Path(output_cfg["dir"])
    rows_per_shard: int = int(output_cfg.get("rows_per_shard", 10_000))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"HF dataset: {hf_name!r} subset={subset!r} split={split!r}")
    logger.info(f"Output:     {output_dir}  (rows_per_shard={rows_per_shard})")

    ds = load_dataset(
        hf_name,
        name=subset,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"Total rows: {len(ds)}")

    max_rows = ds_cfg.get("processing", {}).get("max_rows")
    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))
        logger.info(f"After max_rows cap: {len(ds)}")

    if len(ds) == 0:
        logger.info("Dataset is empty. Nothing to tokenize.")
        return

    num_shards = math.ceil(len(ds) / rows_per_shard)
    logger.info(f"Shards: {num_shards}")

    mf_path = _manifest_path(output_dir)
    if mf_path.exists():
        loaded, persisted_rps = _load_manifest(mf_path)
        _check_rows_per_shard(mf_path, persisted_rps, rows_per_shard)
        manifest = _sync_manifest(loaded, num_shards)
        done = sum(1 for s in manifest if s.get("status") == "done")
        logger.info(f"Resuming: {done}/{len(manifest)} shards done")
    else:
        manifest = [
            {"id": i, "status": "pending", "rows": 0, "error": None} for i in range(num_shards)
        ]
    _save_manifest_atomic(manifest, mf_path, rows_per_shard=rows_per_shard)

    if all(s["status"] == "done" for s in manifest):
        logger.info("All shards done. Nothing to do.")
        return

    _run_shards(
        pool=pool,
        output_dir=output_dir,
        source_type="hf",
        source_root="",  # not used for HF sources
        cols=cols,
        shards=_iter_hf_shards(ds, rows_per_shard),
        total_shards=num_shards,
        manifest=manifest,
        mf_path=mf_path,
        rows_per_shard=rows_per_shard,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SNAC-encode audio from JSONL manifests or HuggingFace Hub datasets"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # ray is heavy (and absent on CPU test nodes) — import it only when a
    # tokenization run actually starts, never at module import time.
    import ray

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ray_address: str | None = os.environ.get("RAY_ADDRESS") or cfg.get("ray", {}).get("address")
    snac_model_path: str = cfg["models"]["snac_model_path"]
    tokenizer_path: str = cfg["models"]["tokenizer_path"]
    datasets_cfg: list[dict] = cfg["datasets"]
    cache_dir: str | None = cfg.get("cache_dir")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    audio_token_base_id = tokenizer.convert_tokens_to_ids("<|snac_0|>")
    if audio_token_base_id is None or audio_token_base_id == tokenizer.unk_token_id:
        raise ValueError(
            f"<|snac_0|> not found in tokenizer at {tokenizer_path!r}. This repo does not "
            "build tokenizers — point models.tokenizer_path at an extended frozen-layout "
            "tokenizer that already contains the SNAC audio tokens."
        )
    logger.info(f"Audio token base ID: {audio_token_base_id}")
    del tokenizer

    if ray_address:
        ray.init(address=ray_address)
    else:
        ray.init()

    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus == 0:
        logger.warning("No GPUs available — encoding will be slow")
        num_actors = 1
        actor_num_gpus = 0
    else:
        workers_per_gpu = max(1, int(cfg.get("processing", {}).get("workers_per_gpu", 2)))
        num_actors = max(1, available_gpus * workers_per_gpu)
        actor_num_gpus = 1.0 / workers_per_gpu
    logger.info(
        f"Ray: {available_gpus} GPUs available → {num_actors} actors "
        f"({actor_num_gpus:.2f} GPU each)"
    )

    for ds_cfg in datasets_cfg:
        source = ds_cfg.get("source", {})
        encode_batch_size = int(
            ds_cfg.get("processing", {}).get(
                "encode_batch_size",
                cfg.get("processing", {}).get("encode_batch_size", 8),
            )
        )
        actor_cls = ray.remote(SNACTokenizeActor).options(num_gpus=actor_num_gpus)
        actors = [
            actor_cls.remote(snac_model_path, audio_token_base_id, encode_batch_size)
            for _ in range(num_actors)
        ]
        pool = ray.util.ActorPool(actors)

        try:
            if "jsonl" in source:
                _process_jsonl_dataset(ds_cfg, pool=pool)
            elif "hf_dataset" in source:
                _process_hf_dataset(ds_cfg, pool=pool, cache_dir=cache_dir)
            else:
                logger.warning(
                    f"Dataset config missing 'source.jsonl' or 'source.hf_dataset'. "
                    f"Skipping: {ds_cfg}"
                )
        finally:
            for actor in actors:
                ray.kill(actor)

    ray.shutdown()
    logger.info("\nAll datasets tokenized.")


if __name__ == "__main__":
    main()
