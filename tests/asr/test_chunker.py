"""Silence-aware chunker: segmentation invariants on synthetic audio.

Pure torch, no model and no checkpoint — these run in CI. The invariants here
are the ones a caller depends on: chunks tile the input exactly (no dropped or
duplicated audio), never exceed the cap, and prefer real pauses over hard cuts.
"""

from __future__ import annotations

import itertools

import torch

from bodhan_genai.asr.engine.chunker import (
    ChunkConfig,
    chunk_audio,
    frame_db,
    silence_runs,
    split_points,
)

SR = 16000


def speech(seconds: float, amp: float = 0.1) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(int(seconds * SR), generator=g) * amp


def silence(seconds: float) -> torch.Tensor:
    return torch.zeros(int(seconds * SR))


def alternating(n_blocks: int, speech_s: float = 8.0, gap_s: float = 1.0) -> torch.Tensor:
    parts = []
    for _ in range(n_blocks):
        parts.extend([speech(speech_s), silence(gap_s)])
    return torch.cat(parts)


def test_short_audio_is_one_chunk():
    wav = speech(10.0)
    assert split_points(wav, SR, ChunkConfig(min_chunk=15, max_chunk=25)) == [(0, wav.numel())]


def test_chunks_tile_the_input_exactly():
    """Contiguous and complete: no sample dropped, none transcribed twice."""
    wav = alternating(6)
    segs = split_points(wav, SR, ChunkConfig(min_chunk=15, max_chunk=25))
    assert segs[0][0] == 0
    assert segs[-1][1] == wav.numel()
    for (_, end), (start, _) in itertools.pairwise(segs):
        assert end == start


def test_chunks_respect_max_chunk():
    wav = alternating(8)
    cfg = ChunkConfig(min_chunk=15, max_chunk=25)
    segs = split_points(wav, SR, cfg)
    assert len(segs) > 1, "this fixture must actually split"
    for a, b in segs:
        assert (b - a) / SR <= cfg.max_chunk + 1e-6


def test_cuts_land_in_silence_not_mid_speech():
    """The point of the module: split at pauses, not on a fixed grid."""
    wav = alternating(6)  # speech 8s / silence 1s, so pauses are at 8-9, 17-18, ...
    segs = split_points(wav, SR, ChunkConfig(min_chunk=15, max_chunk=25))
    for _, end in segs[:-1]:  # last boundary is end-of-audio, not a cut
        t = end / SR
        # every gap starts at k*9 + 8 and lasts 1 s
        offset = t % 9.0
        assert 8.0 <= offset <= 9.0, f"cut at {t:.2f}s (offset {offset:.2f}) is not in a pause"


def test_hard_cut_when_no_silence_exists():
    """Continuous speech has no pauses; the packer must still bound chunk length."""
    wav = speech(60.0)
    cfg = ChunkConfig(min_chunk=15, max_chunk=25)
    segs = split_points(wav, SR, cfg)
    assert len(segs) >= 3
    for a, b in segs:
        assert (b - a) / SR <= cfg.max_chunk + 1e-6
    assert segs[-1][1] == wav.numel()


def test_short_tail_is_merged():
    """A 0.5 s trailing chunk would be its own encoder batch row for nothing."""
    cfg = ChunkConfig(min_chunk=10, max_chunk=15, min_tail=2.0)
    wav = torch.cat([alternating(3, speech_s=8.0, gap_s=1.0), speech(0.5)])
    segs = split_points(wav, SR, cfg)
    assert (segs[-1][1] - segs[-1][0]) / SR >= cfg.min_tail


def test_chunk_audio_yields_aligned_samples_and_times():
    wav = alternating(6)
    for _i, start_s, end_s, samples in chunk_audio(
        wav, SR, ChunkConfig(min_chunk=15, max_chunk=25)
    ):
        assert samples.numel() == round((end_s - start_s) * SR)
        assert torch.equal(samples, wav[round(start_s * SR) : round(end_s * SR)])


def test_frame_db_and_silence_runs_separate_speech_from_silence():
    wav = torch.cat([speech(2.0), silence(1.0), speech(2.0)])
    cfg = ChunkConfig()
    db = frame_db(wav, SR, cfg)
    runs = silence_runs(db, cfg)
    assert runs, "the 1 s gap must be detected"
    # at least one detected run should overlap the true 2.0-3.0 s gap
    hop_s = cfg.hop_ms / 1000
    assert any(a * hop_s < 3.0 and b * hop_s > 2.0 for a, b in runs)
