"""Tests for bodhan_genai.tts.engine.types — SamplingConfig merge semantics and
TTSResult duration/RTF accounting + WAV save roundtrip. numpy/soundfile only."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bodhan_genai.tts.engine import SamplingConfig, TTSResult

# ---------------------------------------------------------------------------
# SamplingConfig
# ---------------------------------------------------------------------------


def test_defaults():
    cfg = SamplingConfig()
    assert cfg.temperature == 0.6
    assert cfg.top_p == 0.95
    assert cfg.top_k == -1
    assert cfg.repetition_penalty == 1.1
    assert cfg.max_new_tokens == 2048


def test_frozen():
    cfg = SamplingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.temperature = 0.9


def test_merged_ignores_none_and_applies_overrides():
    cfg = SamplingConfig()
    merged = cfg.merged(temperature=0.9, top_p=None, max_new_tokens=128)
    assert merged.temperature == 0.9
    assert merged.max_new_tokens == 128
    assert merged.top_p == cfg.top_p  # None ignored
    assert merged.repetition_penalty == cfg.repetition_penalty
    # all-None merge is a no-op copy
    assert cfg.merged(temperature=None) == cfg


def test_merged_returns_new_instance_and_leaves_original_untouched():
    cfg = SamplingConfig()
    merged = cfg.merged(temperature=0.2)
    assert merged is not cfg
    assert isinstance(merged, SamplingConfig)
    assert cfg.temperature == 0.6


def test_merged_rejects_unknown_keys():
    with pytest.raises(TypeError, match="top_kk"):
        SamplingConfig().merged(top_kk=5)


# ---------------------------------------------------------------------------
# TTSResult
# ---------------------------------------------------------------------------


def test_duration_and_rtf_math():
    audio = np.zeros(36_000, dtype=np.float32)  # 1.5 s at 24 kHz
    r = TTSResult(audio=audio, gen_time_s=0.6, decode_time_s=0.15)
    assert r.duration_s == pytest.approx(1.5)
    assert r.rtf == pytest.approx((0.6 + 0.15) / 1.5)


def test_empty_result_defaults():
    r = TTSResult()
    assert r.sample_rate == 24_000
    assert r.prompt_tokens == 0 and r.generated_tokens == 0 and r.audio_tokens == 0
    assert r.error is None
    assert r.duration_s == 0.0
    assert r.rtf == 0.0  # no audio -> no div-by-zero


def test_save_roundtrip(tmp_path):
    import soundfile as sf

    t = np.arange(24_000, dtype=np.float32) / 24_000.0
    audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    r = TTSResult(audio=audio)

    out = r.save(tmp_path / "tone.wav")
    assert out == tmp_path / "tone.wav"
    assert out.exists()

    data, sr = sf.read(str(out), dtype="float32")
    assert sr == 24_000
    assert data.shape == audio.shape
    np.testing.assert_allclose(data, audio, atol=2.0 / 32767.0)  # PCM_16 quantization
