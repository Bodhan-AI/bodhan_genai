"""Waveform collation and file-slice reads.

These guard the production front-end contract: sub-1 s audio is padded to 1 s
*centred*, mono is a channel mean, and — the subtle one — a covering-span read
must produce byte-identical samples to per-chunk reads.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from bodhan_genai.asr.engine.audio_input import (
    _as_mono_tensor,
    collate_waveforms,
    read_slice,
    read_span_and_slice,
)
from bodhan_genai.asr.model import IndicTranscribeFeatureExtractor

SR = 16000


@pytest.fixture(scope="module")
def fe():
    return IndicTranscribeFeatureExtractor()


def test_mono_from_channels_first_and_last():
    stereo_cf = torch.stack([torch.ones(100), torch.full((100,), 3.0)])  # (2, 100)
    assert torch.allclose(_as_mono_tensor(stereo_cf), torch.full((100,), 2.0))
    stereo_cl = stereo_cf.T.contiguous()  # (100, 2)
    assert torch.allclose(_as_mono_tensor(stereo_cl), torch.full((100,), 2.0))


def test_numpy_input_is_accepted():
    out = _as_mono_tensor(np.ones(50, dtype=np.float32))
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32


def test_rejects_3d_input():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _as_mono_tensor(torch.zeros(2, 3, 4))


def test_sub_one_second_audio_is_centre_padded(fe):
    """Production pads to 1 s with pad_direction='both'; an off-centre pad
    shifts every frame and moves the transcript."""
    wav = torch.ones(4000)  # 0.25 s
    batch, lens = collate_waveforms([wav], fe)
    assert batch.shape == (1, SR)
    assert int(lens[0]) == SR, "the padded length is what the encoder sees"
    off = round((SR - 4000) / 2)
    assert torch.count_nonzero(batch[0, :off]) == 0
    assert torch.allclose(batch[0, off : off + 4000], torch.ones(4000))
    assert torch.count_nonzero(batch[0, off + 4000 :]) == 0


def test_longer_audio_is_left_aligned_with_true_lengths(fe):
    a, b = torch.ones(SR * 2), torch.ones(SR)
    batch, lens = collate_waveforms([a, b], fe)
    assert batch.shape == (2, SR * 2)
    assert lens.tolist() == [SR * 2, SR]
    assert torch.count_nonzero(batch[1, SR:]) == 0


def test_pre_collated_tensor_passes_through(fe):
    pre = torch.ones(3, 1234)
    batch, lens = collate_waveforms(pre, fe)
    assert torch.equal(batch, pre)
    assert lens.tolist() == [1234] * 3


def test_empty_input_raises(fe):
    with pytest.raises(ValueError, match="no audio given"):
        collate_waveforms([], fe)


def test_read_slice_returns_requested_span(tmp_path, fe):
    path = tmp_path / "a.wav"
    sig = np.sin(np.arange(SR * 3) * 0.01).astype(np.float32)
    # subtype="FLOAT": .wav defaults to 16-bit PCM, whose ~3e-5 quantization
    # step would swamp an exactness check that is about slicing, not codecs.
    sf.write(path, sig, SR, subtype="FLOAT")
    out = read_slice(str(path), 1.0, 2.0, fe)
    assert out.shape[0] == SR
    assert np.allclose(out.numpy(), sig[SR : 2 * SR], atol=1e-6)


def test_span_and_slice_is_sample_identical_to_per_chunk_reads(tmp_path, fe):
    """The documented reason this function cuts at the NATIVE rate: resampling
    once and then slicing is cheaper but produces DIFFERENT samples, which would
    silently change output relative to transcripts already on disk.
    """
    path = tmp_path / "b.wav"
    sig = np.sin(np.arange(24000 * 5) * 0.01).astype(np.float32)
    sf.write(path, sig, 24000, subtype="FLOAT")  # 24 kHz -> resampling happens
    spans = [(0.5, 1.5), (2.0, 3.0), (3.5, 4.5)]

    grouped = read_span_and_slice(str(path), spans, fe)
    individual = [read_slice(str(path), a, b, fe) for a, b in spans]

    assert len(grouped) == len(individual)
    for g, i in zip(grouped, individual, strict=True):
        assert torch.equal(g, i), "grouped read diverged from per-chunk reads"


def test_span_and_slice_empty_spans(tmp_path, fe):
    assert read_span_and_slice(str(tmp_path / "nope.wav"), [], fe) == []
