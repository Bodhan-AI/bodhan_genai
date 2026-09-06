"""
Tests for bodhan_genai.tts.codec.snac.

Core invariant: batch_encode_audio must produce identical output to
calling encode_audio on each waveform individually.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest
import torch

from bodhan_genai.tts.codec.snac import (
    SNAC_CODEBOOK_SIZE,
    SNAC_NUM_CODEBOOKS,
    SNAC_TOTAL_AUDIO_TOKENS,
    _autocast_context,
    _compute_valid_code_frames,
    _maybe_compile_model,
    _remove_duplicate_frames,
    batch_encode_audio,
    encode_audio,
    load_snac_model,
    tokens_to_audio_token_ids,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_ID = 256000  # arbitrary base token id for tests


# ---------------------------------------------------------------------------
# _remove_duplicate_frames
# ---------------------------------------------------------------------------


class TestRemoveDuplicateFrames:
    def test_no_duplicates(self):
        """All frames distinct → no removal."""
        # 3 frames, each with unique c0
        frames = torch.tensor(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
            ]
        )
        result = _remove_duplicate_frames(frames)
        assert result.shape[0] == 21  # 3 * 7

    def test_all_duplicates(self):
        """All frames have same c0 → collapse to one frame."""
        frame = torch.tensor([100, 2, 3, 4, 5, 6, 7])
        frames = frame.repeat(5)
        result = _remove_duplicate_frames(frames)
        assert result.shape[0] == 7  # only first frame kept

    def test_consecutive_duplicates_only(self):
        """Non-consecutive duplicates are NOT removed (only consecutive)."""
        frames = torch.tensor(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,  # c0=1
                2,
                0,
                0,
                0,
                0,
                0,
                0,  # c0=2
                1,
                0,
                0,
                0,
                0,
                0,
                0,  # c0=1 again (non-consecutive)
            ]
        )
        result = _remove_duplicate_frames(frames)
        assert result.shape[0] == 21  # all 3 frames kept

    def test_single_frame(self):
        """Single frame → returned as-is."""
        frames = torch.tensor([10, 20, 30, 40, 50, 60, 70])
        result = _remove_duplicate_frames(frames)
        assert torch.equal(result, frames)

    def test_invalid_length_raises(self):
        """Length not divisible by 7 → ValueError."""
        with pytest.raises(ValueError, match="not divisible"):
            _remove_duplicate_frames(torch.tensor([1, 2, 3]))

    def test_preserves_non_c0_differences(self):
        """Frames with same c0 but different other codes are still deduplicated."""
        frames = torch.tensor(
            [
                5,
                10,
                20,
                30,
                40,
                50,
                60,
                5,
                99,
                88,
                77,
                66,
                55,
                44,  # same c0=5
            ]
        )
        result = _remove_duplicate_frames(frames)
        assert result.shape[0] == 7  # second frame removed (same c0)

    def test_empty_input(self):
        """Empty tensor with correct shape → empty result."""
        frames = torch.tensor([], dtype=torch.long)
        # 0 is divisible by 7
        result = _remove_duplicate_frames(frames)
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# _compute_valid_code_frames
# ---------------------------------------------------------------------------


class TestComputeValidCodeFrames:
    def test_known_alignment(self, mock_snac_model):
        """Verify frame count for known audio length."""
        # With hop_length=441, vq_strides[0]=8, attn_window_size=32
        # lcm(8, 32) = 32, pad_to = 441 * 32 = 14112
        # For 24000 samples: padded = ceil(24000/14112) * 14112 = 2 * 14112 = 28224
        # n_frames = 28224 // 441 // 8 = 8
        n = _compute_valid_code_frames(24000, mock_snac_model)
        expected_padded = math.ceil(24000 / (441 * 32)) * (441 * 32)
        expected_frames = expected_padded // 441 // 8
        assert n == expected_frames

    def test_exact_alignment(self, mock_snac_model):
        """Audio length that's an exact multiple of pad_to."""
        pad_to = 441 * math.lcm(8, 32)
        n = _compute_valid_code_frames(pad_to, mock_snac_model)
        assert n == pad_to // 441 // 8

    def test_zero_samples(self, mock_snac_model):
        """Zero audio samples → 0 frames."""
        assert _compute_valid_code_frames(0, mock_snac_model) == 0

    def test_one_sample(self, mock_snac_model):
        """Single sample still produces at least one padded block."""
        n = _compute_valid_code_frames(1, mock_snac_model)
        assert n >= 1

    def test_monotonic(self, mock_snac_model):
        """Longer audio → same or more frames (monotonic non-decreasing)."""
        prev = 0
        for length in [100, 1000, 10000, 50000, 100000]:
            n = _compute_valid_code_frames(length, mock_snac_model)
            assert n >= prev
            prev = n


# ---------------------------------------------------------------------------
# encode_audio (single waveform)
# ---------------------------------------------------------------------------


class TestEncodeAudio:
    def test_output_range(self, mock_snac_model):
        """All token IDs must be in [BASE_ID, BASE_ID + 28672)."""
        audio = np.random.default_rng(42).normal(0, 0.1, size=24000).astype(np.float32)
        ids = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        assert len(ids) > 0
        assert all(BASE_ID <= t < BASE_ID + SNAC_TOTAL_AUDIO_TOKENS for t in ids)

    def test_output_length_divisible_by_7(self, mock_snac_model):
        """After deduplication, length is still divisible by 7."""
        audio = np.random.default_rng(42).normal(0, 0.1, size=48000).astype(np.float32)
        ids = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        assert len(ids) % SNAC_NUM_CODEBOOKS == 0

    def test_deterministic(self, mock_snac_model):
        """Same input → same output."""
        audio = np.random.default_rng(42).normal(0, 0.1, size=24000).astype(np.float32)
        ids_1 = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        ids_2 = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        assert ids_1 == ids_2

    def test_different_base_id_shifts_output(self, mock_snac_model):
        """Changing base_id shifts all token IDs by the difference."""
        audio = np.random.default_rng(42).normal(0, 0.1, size=24000).astype(np.float32)
        ids_a = encode_audio(mock_snac_model, audio, 100000, device="cpu")
        ids_b = encode_audio(mock_snac_model, audio, 200000, device="cpu")
        # Frame structure should be identical, just offset
        assert len(ids_a) == len(ids_b)
        for a, b in zip(ids_a, ids_b, strict=False):
            ids_a.index(a) % SNAC_NUM_CODEBOOKS
            expected_diff = 200000 - 100000
            assert b - a == expected_diff


# ---------------------------------------------------------------------------
# batch_encode_audio — parity with encode_audio
# ---------------------------------------------------------------------------


class TestBatchEncodeAudio:
    def test_single_waveform_matches_encode_audio(self, mock_snac_model):
        """Batch of 1 must produce identical length and content to encode_audio."""
        audio = np.random.default_rng(42).normal(0, 0.1, size=24000).astype(np.float32)
        single = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        batched = batch_encode_audio(mock_snac_model, [audio], BASE_ID, device="cpu")
        assert len(batched) == 1
        assert len(batched[0]) == len(single), (
            f"Length: batch={len(batched[0])}, single={len(single)}"
        )
        assert batched[0] == single, "Content mismatch between batch-of-1 and single encode"

    def test_multi_waveform_matches_encode_audio(self, mock_snac_model, random_waveforms):
        """Each waveform in a batch must match its individual encode_audio output — both length and content."""
        waveforms = random_waveforms(n=4, min_len=4800, max_len=48000)

        # Individual
        singles = [encode_audio(mock_snac_model, w, BASE_ID, device="cpu") for w in waveforms]

        # Batched
        batched = batch_encode_audio(mock_snac_model, waveforms, BASE_ID, device="cpu")

        assert len(batched) == len(singles)
        for i, (s, b) in enumerate(zip(singles, batched, strict=False)):
            assert len(s) == len(b), (
                f"Waveform {i} length mismatch: single={len(s)}, batch={len(b)}"
            )
            assert s == b, f"Waveform {i} content mismatch at first diff: " + str(
                next((j, sv, bv) for j, (sv, bv) in enumerate(zip(s, b, strict=False)) if sv != bv)
            )

    def test_empty_waveforms_list(self, mock_snac_model):
        """Empty input → empty output."""
        result = batch_encode_audio(mock_snac_model, [], BASE_ID, device="cpu")
        assert result == []

    def test_mixed_lengths(self, mock_snac_model):
        """Batch with very different lengths (4:1 ratio) — length and content must match."""
        rng = np.random.default_rng(42)
        short = rng.normal(0, 0.1, size=2400).astype(np.float32)  # 0.1s
        long = rng.normal(0, 0.1, size=96000).astype(np.float32)  # 4.0s

        singles = [
            encode_audio(mock_snac_model, short, BASE_ID, device="cpu"),
            encode_audio(mock_snac_model, long, BASE_ID, device="cpu"),
        ]
        batched = batch_encode_audio(mock_snac_model, [short, long], BASE_ID, device="cpu")

        for i, (s, b) in enumerate(zip(singles, batched, strict=False)):
            assert len(s) == len(b), f"Waveform {i} length: single={len(s)}, batch={len(b)}"
            assert s == b, f"Waveform {i} content mismatch"

    def test_all_outputs_in_valid_range(self, mock_snac_model, random_waveforms):
        """All token IDs across all batch elements must be in valid range."""
        waveforms = random_waveforms(n=6)
        batched = batch_encode_audio(mock_snac_model, waveforms, BASE_ID, device="cpu")
        for i, ids in enumerate(batched):
            for t in ids:
                assert BASE_ID <= t < BASE_ID + SNAC_TOTAL_AUDIO_TOKENS, (
                    f"Waveform {i}: token {t} out of range [{BASE_ID}, {BASE_ID + SNAC_TOTAL_AUDIO_TOKENS})"
                )

    def test_all_outputs_divisible_by_7(self, mock_snac_model, random_waveforms):
        """Each batch element's token count is divisible by 7."""
        waveforms = random_waveforms(n=6)
        batched = batch_encode_audio(mock_snac_model, waveforms, BASE_ID, device="cpu")
        for i, ids in enumerate(batched):
            assert len(ids) % SNAC_NUM_CODEBOOKS == 0, (
                f"Waveform {i}: {len(ids)} tokens not divisible by {SNAC_NUM_CODEBOOKS}"
            )

    def test_batch_deterministic(self, mock_snac_model, random_waveforms):
        """Same waveforms → same batch output."""
        waveforms = random_waveforms(n=3)
        r1 = batch_encode_audio(mock_snac_model, waveforms, BASE_ID, device="cpu")
        r2 = batch_encode_audio(mock_snac_model, waveforms, BASE_ID, device="cpu")
        assert r1 == r2


class TestCompilePlumbing:
    def test_maybe_compile_model_preserves_default_behavior(self, monkeypatch):
        calls = []

        def fake_compile(model, **kwargs):
            calls.append((model, kwargs))
            return {"compiled": model, "kwargs": kwargs}

        monkeypatch.setattr(torch, "compile", fake_compile)
        result = _maybe_compile_model("model")

        assert result["compiled"] == "model"
        assert calls == [("model", {})]

    def test_maybe_compile_model_passes_requested_mode(self, monkeypatch):
        calls = []

        def fake_compile(model, **kwargs):
            calls.append(kwargs)
            return model

        monkeypatch.setattr(torch, "compile", fake_compile)
        _maybe_compile_model(
            "model",
            compile_mode="max-autotune",
            compile_dynamic=False,
            compile_fullgraph=True,
        )

        assert calls == [
            {
                "mode": "max-autotune",
                "fullgraph": True,
                "dynamic": False,
            }
        ]

    def test_load_snac_model_uses_compile_controls(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "snac"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "pytorch_model.bin").write_text("stub", encoding="utf-8")

        class FakeModel:
            def __init__(self):
                self.loaded_state = None
                self.device = None
                self.dtype = None

            def load_state_dict(self, state_dict):
                self.loaded_state = state_dict

            def eval(self):
                return self

            def to(self, device=None, dtype=None):
                self.device = device
                self.dtype = dtype
                return self

        class FakeSNAC:
            @staticmethod
            def from_config(_path):
                return FakeModel()

        fake_compile_calls = []

        def fake_compile(model, **kwargs):
            fake_compile_calls.append(kwargs)
            return model

        monkeypatch.setitem(sys.modules, "snac", types.SimpleNamespace(SNAC=FakeSNAC))
        monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"weights": 1})
        monkeypatch.setattr(torch, "compile", fake_compile)

        model = load_snac_model(
            str(model_dir),
            device="cpu",
            model_dtype=torch.bfloat16,
            compile_mode="max-autotune-no-cudagraphs",
        )

        assert isinstance(model, FakeModel)
        assert model.loaded_state == {"weights": 1}
        assert model.device == "cpu"
        assert model.dtype == torch.bfloat16
        assert fake_compile_calls == [{"mode": "max-autotune-no-cudagraphs"}]

    def test_autocast_context_disables_fp32_and_cpu(self):
        with _autocast_context(device="cpu", autocast_dtype=torch.bfloat16):
            pass
        with _autocast_context(device="cuda", autocast_dtype=torch.float32):
            pass


# ---------------------------------------------------------------------------
# tokens_to_audio_token_ids
# ---------------------------------------------------------------------------


class TestTokensToAudioTokenIds:
    def test_basic_offsets(self):
        """Verify position-based offset mapping."""
        N = 2
        c0 = list(range(N))
        c1 = list(range(2 * N))
        c2 = list(range(4 * N))
        ids = tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=False, device="cpu")
        assert len(ids) == 7 * N
        # First frame: [c0[0]+off0, c1[0]+off1, c2[0]+off2, c2[1]+off3, c1[1]+off4, c2[2]+off5, c2[3]+off6]
        assert ids[0] == 0 + BASE_ID + 0 * SNAC_CODEBOOK_SIZE
        assert ids[1] == 0 + BASE_ID + 1 * SNAC_CODEBOOK_SIZE
        assert ids[2] == 0 + BASE_ID + 2 * SNAC_CODEBOOK_SIZE

    def test_deduplication(self):
        """Duplicate c0 values get removed."""
        N = 3
        c0 = [5, 5, 10]  # first two frames have same c0
        c1 = list(range(2 * N))
        c2 = list(range(4 * N))
        ids = tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=True, device="cpu")
        # Two unique c0 values → 2 * 7 = 14 tokens
        assert len(ids) == 14

    def test_no_deduplication(self):
        """With deduplicate=False, duplicates preserved."""
        N = 3
        c0 = [5, 5, 10]
        c1 = list(range(2 * N))
        c2 = list(range(4 * N))
        ids = tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=False, device="cpu")
        assert len(ids) == 21

    def test_empty_codes(self):
        """Empty code lists → empty output."""
        ids = tokens_to_audio_token_ids([[], [], []], BASE_ID, device="cpu")
        assert ids == []
