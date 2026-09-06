"""Loudness utilities: level math, trimming, and the causal streaming normalizer."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from bodhan_genai.tts.engine.loudness import (
    StreamingLoudnessNormalizer,
    _db_to_lin,
    normalize_loudness,
    peak_normalize,
    rms,
    trim_silence,
)

SR = 24_000


def tone(seconds: float, amp: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestLevelMath:
    def test_rms(self):
        assert rms(np.zeros(100, dtype=np.float32)) == 0.0
        assert rms(np.ones(100, dtype=np.float32)) == pytest.approx(1.0)
        assert rms(np.array([], dtype=np.float32)) == 0.0

    def test_peak_normalize_hits_target(self):
        y = tone(0.1, amp=0.25)
        out = peak_normalize(y, peak_dbfs=-1.0)
        assert np.max(np.abs(out)) == pytest.approx(_db_to_lin(-1.0), rel=1e-4)

    def test_peak_normalize_silence_passthrough(self):
        y = np.zeros(100, dtype=np.float32)
        assert np.array_equal(peak_normalize(y), y)

    def test_normalize_loudness_rms_fallback(self, monkeypatch):
        real_import = builtins.__import__

        def no_pyln(name, *a, **k):
            if name == "pyloudnorm":
                raise ImportError("blocked for test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_pyln)
        # amp 0.03 / 0.3: both within the +/-12 dB normalization gain clamp
        quiet = tone(0.2, amp=0.03)
        loud = tone(0.2, amp=0.3)
        target = -23.0
        for y in (quiet, loud):
            out = normalize_loudness(y, target_lufs=target, sample_rate=SR)
            assert rms(out) == pytest.approx(_db_to_lin(target), rel=1e-3)

    def test_normalize_loudness_gain_clamped(self, monkeypatch):
        """A breath/noise-level chunk must NOT be boosted to program loudness:
        gain is clamped to +12 dB (would need ~+20 dB here)."""
        y = tone(0.2, amp=0.007)  # ~ -43 dBFS RMS, above the -45 activity floor
        out = normalize_loudness(y, target_lufs=-23.0, sample_rate=SR)
        assert rms(out) <= rms(y) * _db_to_lin(12.0) * 1.05
        assert rms(out) < _db_to_lin(-23.0)  # did not reach program level

    def test_normalize_loudness_floor_passthrough(self):
        """Below the activity floor the segment passes through unchanged."""
        y = tone(0.2, amp=0.0002)  # ~ -74 dBFS: noise floor
        out = normalize_loudness(y, target_lufs=-23.0, sample_rate=SR)
        assert np.array_equal(out, y.astype(np.float32))

    def test_normalize_active_rms_hits_target(self):
        from bodhan_genai.tts.engine.loudness import active_rms, normalize_active_rms

        # voiced burst padded with silence: gated RMS ignores the padding
        pad = np.zeros(SR // 4, dtype=np.float32)
        y = np.concatenate([pad, tone(0.3, amp=0.05), pad])
        out = normalize_active_rms(y, target_db=-23.0, sample_rate=SR)
        assert active_rms(out, SR) == pytest.approx(_db_to_lin(-23.0), rel=0.05)

    def test_normalize_loudness_silence_passthrough(self):
        y = np.zeros(1000, dtype=np.float32)
        assert np.array_equal(normalize_loudness(y), y)


class TestTrimSilence:
    def test_numpy_fallback_trims_edges(self, monkeypatch):
        real_import = builtins.__import__

        def no_librosa(name, *a, **k):
            if name == "librosa":
                raise ImportError("blocked for test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_librosa)
        pad = np.zeros(SR // 2, dtype=np.float32)  # 500ms silence
        speech = tone(0.3, amp=0.5)
        y = np.concatenate([pad, speech, pad])
        out = trim_silence(y, top_db=30.0, sample_rate=SR)
        assert out.size < y.size
        assert out.size >= speech.size * 0.9  # speech body kept

    def test_never_returns_empty(self, monkeypatch):
        real_import = builtins.__import__
        monkeypatch.setattr(
            builtins,
            "__import__",
            lambda n, *a, **k: (
                (_ for _ in ()).throw(ImportError()) if n == "librosa" else real_import(n, *a, **k)
            ),
        )
        silence = np.zeros(1000, dtype=np.float32)
        assert trim_silence(silence).size == 1000


class TestStreamingNormalizer:
    FRAME = 2048  # samples, matching the serving frame size

    def frames(self, y: np.ndarray) -> list[np.ndarray]:
        i16 = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        n = (i16.size // self.FRAME) * self.FRAME
        return [i16[i : i + self.FRAME] for i in range(0, n, self.FRAME)]

    def out_rms(self, chunks: list[bytes]) -> float:
        y = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32767.0
        return rms(y)

    def test_converges_toward_target(self):
        norm = StreamingLoudnessNormalizer(target_lufs=-23.0, sample_rate=SR)
        target = _db_to_lin(-23.0)
        # amp 0.03 -> RMS ~0.021: needs ~+10.5 dB, inside the 12 dB gain clamp
        outs = [norm.process(f) for f in self.frames(tone(3.0, amp=0.03))]
        tail = self.out_rms(outs[-10:])
        assert abs(tail - target) / target < 0.35  # converged near target

    def test_limiter_and_gain_clamp(self):
        norm = StreamingLoudnessNormalizer(max_gain_db=12.0, limiter_peak=0.985)
        outs = [norm.process(f) for f in self.frames(tone(1.0, amp=0.9))]
        y = np.frombuffer(b"".join(outs), dtype=np.int16).astype(np.float32) / 32767.0
        assert np.max(np.abs(y)) <= 0.985 + 1e-3
        # quiet input can gain at most +12 dB
        norm2 = StreamingLoudnessNormalizer(max_gain_db=12.0)
        outs2 = [norm2.process(f) for f in self.frames(tone(1.0, amp=0.001))]
        assert self.out_rms(outs2) <= 0.001 * _db_to_lin(12.0) * 1.2

    def test_state_persists_across_chunks(self):
        """Two bursts at different input levels end up near-equal loudness when
        fed through ONE normalizer (the cross-chunk consistency property)."""
        norm = StreamingLoudnessNormalizer(target_lufs=-23.0, sample_rate=SR)
        quiet_out = [norm.process(f) for f in self.frames(tone(2.5, amp=0.03))]
        loud_out = [norm.process(f) for f in self.frames(tone(2.5, amp=0.30))]
        r1 = self.out_rms(quiet_out[-8:])
        r2 = self.out_rms(loud_out[-8:])
        assert abs(r1 - r2) / max(r1, r2) < 0.4

    def test_silence_holds_gain(self):
        norm = StreamingLoudnessNormalizer()
        [norm.process(f) for f in self.frames(tone(1.0, amp=0.05))]
        gain_before = norm._gain
        zero = np.zeros(self.FRAME, dtype=np.int16)
        out = norm.process(zero)
        assert norm._gain == gain_before
        assert np.frombuffer(out, dtype=np.int16).sum() == 0

    def test_bytes_in_bytes_out(self):
        norm = StreamingLoudnessNormalizer()
        raw = (tone(0.1, amp=0.1)[: self.FRAME] * 32767).astype(np.int16).tobytes()
        out = norm.process(raw)
        assert isinstance(out, bytes) and len(out) == len(raw)
        assert norm.process(b"") == b""


class TestColdStartSnap:
    FRAME = 2048

    def test_first_voiced_frame_already_near_target(self):
        """Cold start snaps gain instead of slewing from 1.0 — no audible
        fade-in over the first seconds (review finding)."""
        norm = StreamingLoudnessNormalizer(target_lufs=-23.0, sample_rate=SR)
        target = _db_to_lin(-23.0)
        y = tone(0.5, amp=0.03)  # needs ~+10.5 dB, inside the clamp
        first = (np.clip(y[: self.FRAME], -1, 1) * 32767).astype(np.int16)
        out = np.frombuffer(norm.process(first), dtype=np.int16).astype(np.float32) / 32767.0
        assert rms(out) == pytest.approx(target, rel=0.15)  # snapped, not ramping
