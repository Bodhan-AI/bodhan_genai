"""Shared helpers for the two-phase disagg-eval pipeline.

Phase A (eval/generate_audios.py) writes 24 kHz WAVs + manifest.jsonl.
Phase B (eval/run_eval.py) reads the manifest, scores each row, rewrites
manifest.jsonl in place with the score columns merged in.

Both phases use ``AudioRow`` as the on-disk schema, ``audio_filename``
for the deterministic per-row WAV name (so resume + diff are trivial),
and ``write_wav_24k`` / ``read_wav_any`` for codebase-consistent I/O.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SNAC_SAMPLE_RATE = 24_000
METRIC_SAMPLE_RATE = 16_000


@dataclass
class AudioRow:
    """Single sample in ``manifest.jsonl``.

    Base columns are written by Phase A; the score columns are filled in
    by the metric stages in Phase B. Unset score columns simply aren't
    serialized — JSON consumers should treat absence as "metric disabled
    for this row".
    """

    # ---- Phase A: identity + generation ------------------------------------
    row_idx: int
    audio_filepath: str  # reference audio path (input)
    text: str
    language: str
    speaker_id: str = ""
    gen_audio_path: str = ""  # relative to the checkpoint dir, e.g. "audio/0000_3f7a.wav"
    ok: bool = False
    error: str | None = None
    gen_duration_sec: float = 0.0
    gen_wall_sec: float = 0.0

    # ---- Phase B: scoring (all optional; absent until the metric runs) ------
    pred_text: str | None = None
    asr_backend: str | None = None
    wer: float | None = None
    cer: float | None = None
    mos: float | None = None
    speaker_similarity: float | None = None
    mcd: float | None = None
    # Per-criterion judge scores live under arbitrary "judge_<name>" keys
    # — keep them in a free-form dict so the rubric stays pluggable.
    judge: dict[str, float] = field(default_factory=dict)
    judge_parse_ok: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize as a flat JSON-ready dict; drop None entries and the
        empty ``judge`` dict, and inline ``judge_<name>`` keys."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "judge":
                continue
            val = getattr(self, f.name)
            if val is None:
                continue
            out[f.name] = val
        for name, score in (self.judge or {}).items():
            out[f"judge_{name}"] = score
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> AudioRow:
        """Parse a manifest row, lifting ``judge_<name>`` keys back into ``judge``."""
        judge_dict: dict[str, float] = {}
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k.startswith("judge_") and k != "judge_parse_ok":
                judge_dict[k[len("judge_") :]] = float(v)
            else:
                kwargs[k] = v
        if judge_dict:
            kwargs["judge"] = judge_dict
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        return cls(**kwargs)


def audio_filename(row_idx: int, audio_filepath: str, text: str) -> str:
    """Deterministic per-row WAV name: ``NNNN_<sha8>.wav``.

    Stable across checkpoints so the same sample lands at the same path —
    makes Phase A's resume check and cross-checkpoint diff trivial.
    """
    digest = hashlib.sha256((audio_filepath + "\x00" + text).encode("utf-8")).hexdigest()[:8]
    return f"{int(row_idx):04d}_{digest}.wav"


def write_wav_24k(path: str | Path, audio: np.ndarray) -> None:
    """Write a float32 mono 24 kHz waveform to disk atomically.

    Uses ``soundfile`` (torchaudio's WAV I/O now requires torchcodec+FFmpeg,
    which isn't guaranteed on compute nodes). Atomic via a tmp file in the
    same directory + ``os.replace`` so a crashed Phase A never leaves a
    half-written WAV that Phase A's resume check would mistake for a
    completed row.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".wav", dir=str(path.parent))
    os.close(fd)
    try:
        sf.write(tmp, arr, samplerate=SNAC_SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = SNAC_SAMPLE_RATE) -> bytes:
    """Wrap raw little-endian int16 PCM in an in-memory WAV container.

    Used by the serving offline endpoint (response body). Same soundfile
    backend as ``write_wav_24k``, imported lazily."""
    import io

    import soundfile as sf

    arr = np.frombuffer(pcm, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, arr, samplerate=int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def read_wav_any(path: str | Path, target_sr: int | None = None) -> np.ndarray:
    """Load a WAV as mono float32. Optionally resample to ``target_sr``."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    wf = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if target_sr is not None and sr != target_sr:
        import librosa

        wf = librosa.resample(wf, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wf, dtype=np.float32)


def read_manifest(path: str | Path) -> list[AudioRow]:
    """Read a manifest.jsonl as a list of ``AudioRow``."""
    rows: list[AudioRow] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(AudioRow.from_json(json.loads(line)))
            except Exception as e:
                raise ValueError(f"Bad manifest line {lineno} in {path}: {e}") from e
    return rows


def write_manifest_atomic(path: str | Path, rows: list[AudioRow]) -> None:
    """Atomically rewrite manifest.jsonl via tmp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_manifest_", suffix=".jsonl", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r.to_json(), ensure_ascii=False))
                f.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
