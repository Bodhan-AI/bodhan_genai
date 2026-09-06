"""Volume-consistency utilities for chunked long-form synthesis.

The offline recipe follows the proven gemma-tts chunk combiner: per chunk
trim leading/trailing silence, loudness-normalize to a target LUFS
(ITU-R BS.1770 via pyloudnorm when available, RMS fallback otherwise),
concatenate with silence gaps, then peak-normalize the whole utterance to
leave headroom. ``StreamingLoudnessNormalizer`` is the causal (frame-by-frame)
counterpart used for chunks that must be emitted before their audio is
complete.

Module imports are numpy-only; ``pyloudnorm`` and ``librosa`` are optional
accelerators imported lazily with pure-numpy fallbacks.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LUFS = -23.0
DEFAULT_PEAK_DBFS = -1.0
DEFAULT_TRIM_DB = 30.0


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(y: np.ndarray) -> float:
    """Root-mean-square level of a float waveform (0.0 for empty input)."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def peak_normalize(y: np.ndarray, peak_dbfs: float = DEFAULT_PEAK_DBFS) -> np.ndarray:
    """Scale so the loudest sample sits at ``peak_dbfs`` (headroom, no clipping)."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y * (_db_to_lin(peak_dbfs) / peak)
    return y.astype(np.float32)


# Gain safety rails shared by every normalization path: never boost more than
# this (a breath/noise-only chunk must not become full-loudness hiss), and
# treat gated levels below the floor as non-program material (pass through).
MAX_NORM_GAIN_DB = 12.0
ACTIVE_FLOOR_DBFS = -45.0

_warned_missing: set[str] = set()


def _warn_once(dep: str, msg: str) -> None:
    if dep not in _warned_missing:
        _warned_missing.add(dep)
        logger.warning(msg)


def _apply_gain_db(y: np.ndarray, gain_db: float, context: str) -> np.ndarray:
    clamped = float(np.clip(gain_db, -MAX_NORM_GAIN_DB, MAX_NORM_GAIN_DB))
    if clamped != gain_db:
        logger.warning(
            "%s: normalization gain %.1f dB clamped to %.1f dB", context, gain_db, clamped
        )
    return (y * _db_to_lin(clamped)).astype(np.float32)


def active_rms(y: np.ndarray, sample_rate: int = 24_000) -> float:
    """Silence-gated RMS: level over 20 ms frames above the activity floor.

    This is the level-measurement domain the causal
    :class:`StreamingLoudnessNormalizer` effectively tracks (it holds gain
    through silent frames), so batch paths that must level-match a causally
    normalized stream should normalize against THIS, not full-segment RMS."""
    if y.size == 0:
        return 0.0
    frame = max(1, int(sample_rate * 0.02))
    n_frames = int(np.ceil(y.size / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[: y.size] = y.astype(np.float64)
    frame_rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
    voiced = frame_rms[frame_rms > _db_to_lin(ACTIVE_FLOOR_DBFS)]
    if voiced.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(voiced**2)))


def normalize_active_rms(
    y: np.ndarray,
    target_db: float = DEFAULT_TARGET_LUFS,
    sample_rate: int = 24_000,
) -> np.ndarray:
    """Normalize the silence-gated RMS to ``target_db`` (gain-clamped).

    Used by the streaming exact path so prefetched chunks land in the same
    level domain as the causal normalizer — one target across the hybrid seam.
    Segments with no active frames pass through unchanged."""
    level = active_rms(y, sample_rate)
    if level <= _db_to_lin(ACTIVE_FLOOR_DBFS):
        return y.astype(np.float32)
    return _apply_gain_db(y, target_db - 20.0 * np.log10(level), "normalize_active_rms")


def normalize_loudness(
    y: np.ndarray,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    sample_rate: int = 24_000,
) -> np.ndarray:
    """Loudness-normalize to ``target_lufs`` (ITU-R BS.1770 via pyloudnorm,
    RMS fallback for segments too short to meter). The applied gain is clamped
    to ±12 dB and near-silent segments pass through, so breath/noise-only
    chunks are never boosted to program loudness. Output may exceed ±1.0 —
    callers apply a peak cap afterwards."""
    if y.size == 0:
        return y.astype(np.float32)
    if active_rms(y, sample_rate) <= _db_to_lin(ACTIVE_FLOOR_DBFS):
        logger.warning("normalize_loudness: segment below activity floor — passing through")
        return y.astype(np.float32)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loud = meter.integrated_loudness(y.astype(np.float64))
        if np.isfinite(loud):
            return _apply_gain_db(y, target_lufs - loud, "normalize_loudness[lufs]")
    except ImportError:
        _warn_once(
            "pyloudnorm",
            "pyloudnorm not installed — loudness normalization degrades to RMS "
            "(install the [infer]/[serve] extra for the documented LUFS recipe)",
        )
    except Exception:  # segment shorter than the meter block
        pass
    level = rms(y)
    if level > 1e-9:
        return _apply_gain_db(y, target_lufs - 20.0 * np.log10(level), "normalize_loudness[rms]")
    return y.astype(np.float32)


def trim_silence(
    y: np.ndarray,
    top_db: float = DEFAULT_TRIM_DB,
    sample_rate: int = 24_000,
) -> np.ndarray:
    """Trim leading/trailing silence only (internal pauses preserved).

    Uses ``librosa.effects.trim`` when available; otherwise a 20 ms frame-RMS
    threshold relative to the loudest frame. Never returns an empty array."""
    if y.size == 0:
        return y.astype(np.float32)
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return trimmed.astype(np.float32) if trimmed.size else y.astype(np.float32)
    except ImportError:
        _warn_once(
            "librosa",
            "librosa not installed — silence trimming uses the numpy frame-RMS fallback",
        )
    except Exception:
        pass

    frame = max(1, int(sample_rate * 0.02))
    n_frames = int(np.ceil(y.size / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[: y.size] = y.astype(np.float64)
    frame_rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
    ref = float(frame_rms.max())
    if ref <= 1e-9:
        return y.astype(np.float32)
    keep = np.nonzero(frame_rms > ref * _db_to_lin(-abs(top_db)))[0]
    if keep.size == 0:
        return y.astype(np.float32)
    start = int(keep[0]) * frame
    end = min(y.size, (int(keep[-1]) + 1) * frame)
    trimmed = y[start:end]
    return trimmed.astype(np.float32) if trimmed.size else y.astype(np.float32)


def warm_dsp(sample_rate: int = 24_000) -> None:
    """Pre-warm the lazy DSP dependencies off the request path.

    librosa's first ``effects.trim`` call pays a ~20 s lazy-submodule/numba
    initialization; pyloudnorm's first meter is ~10 ms. Call this at service
    startup (the serving replica does, before it reports ready) so the first
    chunked request never stalls the event loop."""
    dummy = np.zeros(sample_rate // 2, dtype=np.float32)
    dummy[::100] = 0.1
    trim_silence(dummy, sample_rate=sample_rate)
    normalize_loudness(dummy, sample_rate=sample_rate)


class StreamingLoudnessNormalizer:
    """Causal per-frame gain smoother: keeps live-streamed PCM near a target
    level without buffering the chunk.

    A frame-RMS EMA tracks the program level (fast attack when louder, slow
    release when quieter — asymmetry avoids pumping on pauses); the applied
    gain slews toward ``target_rms / ema`` and is clamped to ±``max_gain_db``.
    Output samples are hard-limited to ``limiter_peak`` full-scale. State is
    intentionally long-lived: feed successive chunks through ONE instance so
    the level stays consistent across chunk boundaries.
    """

    def __init__(
        self,
        target_lufs: float = DEFAULT_TARGET_LUFS,
        sample_rate: int = 24_000,
        attack: float = 0.35,
        release: float = 0.08,
        max_gain_db: float = 12.0,
        limiter_peak: float = 0.985,
    ) -> None:
        self._target_rms = _db_to_lin(target_lufs)
        self._attack = float(attack)
        self._release = float(release)
        self._max_gain = _db_to_lin(abs(max_gain_db))
        self._min_gain = 1.0 / self._max_gain
        self._limit = float(limiter_peak)
        self._ema: float | None = None  # program-level estimate (float RMS)
        self._gain = 1.0
        # ~ -60 dBFS: below this the frame is treated as silence (gain held).
        self._silence_floor = 1e-3

    def process(self, frame: np.ndarray | bytes) -> bytes:
        """Normalize one int16 PCM frame (array or raw bytes) -> int16 bytes."""
        i16 = np.frombuffer(frame, dtype=np.int16) if isinstance(frame, bytes) else frame
        if i16.size == 0:
            return b""
        y = i16.astype(np.float32) / 32767.0
        level = rms(y)
        if level > self._silence_floor:
            if self._ema is None:
                # Cold start: there is no preceding audio to step against, so
                # snap straight to the desired gain — slewing from 1.0 would
                # play the first ~1 s at the wrong level with an audible fade.
                self._ema = level
                self._gain = float(
                    np.clip(self._target_rms / level, self._min_gain, self._max_gain)
                )
            else:
                coeff = self._attack if level > self._ema else self._release
                self._ema += coeff * (level - self._ema)
                desired = float(
                    np.clip(self._target_rms / self._ema, self._min_gain, self._max_gain)
                )
                # Slew the applied gain so consecutive frames never step audibly —
                # asymmetric: reduce gain fast (protects against sudden loudness),
                # raise it slowly (no pumping after pauses).
                coeff = self._attack if desired < self._gain else self._release
                self._gain += coeff * (desired - self._gain)
        out = np.clip(y * self._gain, -self._limit, self._limit)
        return (out * 32767.0).astype(np.int16).tobytes()
