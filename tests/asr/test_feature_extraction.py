"""Mel front-end: shape/length contract and the parity-critical details.

No checkpoint needed — the extractor constructs with default buffers, which is
enough to exercise the framing math, the normalization, and the resampler's
sample-count contract. Absolute mel VALUES depend on the checkpoint's `fb`
buffer and are covered by the upstream GPU parity gates, not here.
"""

from __future__ import annotations

import pytest
import torch

from bodhan_genai.asr.model import IndicTranscribeFeatureExtractor


@pytest.fixture(scope="module")
def fe():
    """Extractor with a REALISTIC mel filterbank installed.

    The default `fb` buffer is all zeros (real values come from the
    checkpoint), which makes every mel bin the same constant — log of the
    zero-guard — so the per-feature normalization divides by its 1e-5 std
    floor and amplifies pure floating-point residue. Tests of the
    normalization math need a non-degenerate filterbank.

    A dense random positive bank is used rather than a true mel bank because
    at this model's n_mels=128 / n_freqs=257 a real mel bank leaves several
    bins all-zero (torchaudio warns about exactly this), which would leave
    those bins degenerate and defeat the point. Absolute mel values against
    the real checkpoint buffer are covered by the upstream GPU parity gates,
    not here.
    """
    fx = IndicTranscribeFeatureExtractor()
    g = torch.Generator().manual_seed(0)
    fx.fb.copy_(torch.rand(1, fx.n_mels, fx.n_fft // 2 + 1, generator=g) + 0.1)
    return fx


def test_seq_len_is_floor_div_hop_plus_one(fe):
    """NeMo's exact framing count; off-by-one here shifts every downstream mask."""
    lens = torch.tensor([16000, 15999, 160, 161])
    expected = torch.tensor([101, 100, 2, 2])
    assert torch.equal(fe.get_seq_len(lens), expected)


def test_forward_shapes_and_lengths(fe):
    audio = torch.randn(3, 16000) * 0.1
    lens = torch.tensor([16000, 8000, 4000])
    feats, feat_lens = fe(audio, lens)
    assert feats.shape == (3, fe.n_mels, 101)
    assert torch.equal(feat_lens, fe.get_seq_len(lens))
    assert feats.dtype == torch.float32, "NeMo forces fp32 here regardless of model dtype"


def test_padded_frames_are_zeroed(fe):
    """Rows shorter than the batch max must not leak energy into padding."""
    audio = torch.randn(2, 16000) * 0.1
    lens = torch.tensor([16000, 4000])
    feats, feat_lens = fe(audio, lens)
    tail = feats[1, :, int(feat_lens[1]) :]
    assert torch.count_nonzero(tail) == 0


def test_normalization_is_per_feature_over_true_length(fe):
    """Mean ~0 / std ~1 per mel bin over VALID frames only (unbiased std)."""
    audio = torch.randn(1, 32000) * 0.1
    lens = torch.tensor([32000])
    feats, feat_lens = fe(audio, lens)
    valid = feats[0, :, : int(feat_lens[0])]
    assert torch.allclose(valid.mean(dim=1), torch.zeros(fe.n_mels), atol=1e-4)
    # std += 1e-5 happens after the sqrt, so this is ~1 but not exactly 1
    assert torch.allclose(valid.std(dim=1, unbiased=True), torch.ones(fe.n_mels), atol=1e-2)


def test_single_frame_audio_raises(fe):
    """One frame makes the unbiased (n-1) std divide by zero; NeMo raises too."""
    with pytest.raises(ValueError, match="shorter than one hop"):
        fe(torch.randn(1, 100), torch.tensor([100]))


def test_default_buffers_are_placeholders_not_usable_features():
    """Guard against a real trap: an extractor built WITHOUT a checkpoint has an
    all-zero filterbank, so every mel bin collapses to log(zero-guard) and the
    output carries no signal. Anyone who sees plausible-looking (B, 128, T)
    output from a bare IndicTranscribeFeatureExtractor() is looking at noise.
    """
    bare = IndicTranscribeFeatureExtractor()
    assert torch.count_nonzero(bare.fb) == 0
    feats, _ = bare(torch.randn(1, 16000) * 0.1, torch.tensor([16000]))
    # constant pre-normalization -> std floor dominates -> tiny fp residue only
    assert feats.abs().max() < 1.0


def test_resample_is_identity_at_native_rate(fe):
    wav = torch.randn(16000)
    assert torch.equal(fe.resample(wav, 16000), wav)


@pytest.mark.parametrize("orig_sr,seconds", [(24000, 1.0), (44100, 0.5), (8000, 2.0)])
def test_resample_sample_count_matches_lhotse_rounding(fe, orig_sr, seconds):
    """torchaudio returns ceil; production (lhotse) uses ROUND_HALF_UP, and the
    extra tail sample would shift every frame boundary."""
    wav = torch.randn(int(orig_sr * seconds))
    out = fe.resample(wav, orig_sr)
    assert out.shape[-1] == round(seconds * fe.sample_rate)


def test_resamplers_are_cached_per_rate(fe):
    local = IndicTranscribeFeatureExtractor()
    local.resample(torch.randn(2400), 24000)
    local.resample(torch.randn(2400), 24000)
    local.resample(torch.randn(4410), 44100)
    assert sorted(local._resamplers) == [24000, 44100]
