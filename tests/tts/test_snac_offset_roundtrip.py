"""Round-trip tests for the SNAC offset scheme at the frozen llama3 base id.

Pins the frozen vocab contract: <|snac_0|>=128266, 28,672 SNAC tokens laid out
as id = base + (position % 7) * 4096, so the audio vocab occupies
[128266, 156938). Runtime code always resolves the base from the tokenizer —
the literal here only guards the layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodhan_genai.tts.codec.snac import (
    SNAC_CODEBOOK_SIZE,
    SNAC_NUM_CODEBOOKS,
    SNAC_TOTAL_AUDIO_TOKENS,
    _parse_token_ids_to_frames,
    batch_decode_audio,
    decode_audio,
    encode_audio,
    tokens_to_audio_token_ids,
)

BASE_ID = 128266  # frozen <|snac_0|> id in the llama3-TTS vocab
AUDIO_HI = BASE_ID + SNAC_TOTAL_AUDIO_TOKENS  # 156938 (first speaker-wrapper id)


def test_frozen_vocab_arithmetic():
    assert SNAC_NUM_CODEBOOKS == 7
    assert SNAC_CODEBOOK_SIZE == 4096
    assert SNAC_TOTAL_AUDIO_TOKENS == 28672
    assert AUDIO_HI == 156938


# ---------------------------------------------------------------------------
# encode_audio at the frozen base
# ---------------------------------------------------------------------------


class TestEncodeAtFrozenBase:
    def test_ids_within_frozen_range(self, mock_snac_model):
        audio = np.random.default_rng(0).normal(0, 0.1, size=24000).astype(np.float32)
        ids = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        assert len(ids) > 0
        assert all(BASE_ID <= t < AUDIO_HI for t in ids)

    def test_position_offset_per_flattened_index(self, mock_snac_model):
        """(id - base) // 4096 must equal position % 7 for every flattened token."""
        audio = np.random.default_rng(0).normal(0, 0.1, size=48000).astype(np.float32)
        ids = encode_audio(mock_snac_model, audio, BASE_ID, device="cpu")
        assert len(ids) % SNAC_NUM_CODEBOOKS == 0
        for i, t in enumerate(ids):
            assert (t - BASE_ID) // SNAC_CODEBOOK_SIZE == i % SNAC_NUM_CODEBOOKS


# ---------------------------------------------------------------------------
# tokens_to_audio_token_ids → decode-path parsing round trip
# ---------------------------------------------------------------------------


class TestOffsetRoundTrip:
    def _make_codes(self, n_frames: int):
        rng = np.random.default_rng(7)
        # Distinct consecutive c0 values so deduplication never fires.
        c0 = (np.arange(n_frames) * 13 + 5) % SNAC_CODEBOOK_SIZE
        c1 = rng.integers(0, SNAC_CODEBOOK_SIZE, size=2 * n_frames)
        c2 = rng.integers(0, SNAC_CODEBOOK_SIZE, size=4 * n_frames)
        return c0.tolist(), c1.tolist(), c2.tolist()

    def test_parse_recovers_raw_codes(self):
        c0, c1, c2 = self._make_codes(5)
        ids = tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=False, device="cpu")
        assert all(BASE_ID <= t < AUDIO_HI for t in ids)

        frames = _parse_token_ids_to_frames(ids, BASE_ID)
        assert frames is not None and frames.shape == (5, SNAC_NUM_CODEBOOKS)
        for i in range(5):
            # Frame i → [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]
            expected = [
                c0[i],
                c1[2 * i],
                c2[4 * i],
                c2[4 * i + 1],
                c1[2 * i + 1],
                c2[4 * i + 2],
                c2[4 * i + 3],
            ]
            assert frames[i].tolist() == expected

    def test_decode_audio_produces_pcm_of_expected_length(self, mock_snac_model):
        c0, c1, c2 = self._make_codes(4)
        ids = tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=False, device="cpu")
        pcm = decode_audio(mock_snac_model, ids, BASE_ID, device="cpu")
        # Mock decode emits n_frames * hop * vq_stride samples of int16 PCM.
        expected_samples = 4 * mock_snac_model.hop_length * mock_snac_model.vq_strides[0]
        assert isinstance(pcm, bytes)
        assert len(pcm) == expected_samples * 2


# ---------------------------------------------------------------------------
# Range guard
# ---------------------------------------------------------------------------


class TestRangeGuard:
    def test_decode_audio_rejects_id_below_base(self, mock_snac_model):
        ids = [BASE_ID + p * SNAC_CODEBOOK_SIZE for p in range(SNAC_NUM_CODEBOOKS)]
        ids[0] = BASE_ID - 1  # e.g. a token from a different backbone's vocab
        with pytest.raises(ValueError, match="outside the audio-vocab range"):
            decode_audio(mock_snac_model, ids, BASE_ID, device="cpu")

    def test_decode_audio_rejects_id_at_or_above_hi(self, mock_snac_model):
        ids = [BASE_ID + p * SNAC_CODEBOOK_SIZE for p in range(SNAC_NUM_CODEBOOKS)]
        ids[-1] = AUDIO_HI  # first speaker-wrapper id, one past the audio vocab
        with pytest.raises(ValueError, match="outside the audio-vocab range"):
            decode_audio(mock_snac_model, ids, BASE_ID, device="cpu")


# ---------------------------------------------------------------------------
# batch_decode_audio row isolation
# ---------------------------------------------------------------------------


class TestBatchDecodeIsolation:
    def test_bad_row_is_none_without_poisoning_batch(self, mock_snac_model):
        rng = np.random.default_rng(3)

        def make_row(n_frames: int) -> list[int]:
            c0 = ((np.arange(n_frames) * 17 + 1) % SNAC_CODEBOOK_SIZE).tolist()
            c1 = rng.integers(0, SNAC_CODEBOOK_SIZE, size=2 * n_frames).tolist()
            c2 = rng.integers(0, SNAC_CODEBOOK_SIZE, size=4 * n_frames).tolist()
            return tokens_to_audio_token_ids([c0, c1, c2], BASE_ID, deduplicate=False, device="cpu")

        good_a = make_row(2)
        good_b = make_row(3)
        bad = list(good_a)
        bad[3] = 5000  # a text-vocab id — outside [BASE_ID, AUDIO_HI)

        results = batch_decode_audio(
            mock_snac_model,
            [good_a, bad, good_b],
            BASE_ID,
            device="cpu",
        )
        assert results[1] is None
        spf = mock_snac_model.hop_length * mock_snac_model.vq_strides[0]
        assert isinstance(results[0], bytes) and len(results[0]) == 2 * spf * 2
        assert isinstance(results[2], bytes) and len(results[2]) == 3 * spf * 2
