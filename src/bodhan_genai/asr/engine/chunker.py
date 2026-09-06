# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Silence-aware segmentation for long-form audio.

The checkpoint was trained with max_duration: 30 s, and decoding is capped at
min(1024, enc_frames + 50) tokens, so long recordings must be split before
transcription. Splitting on a fixed grid cuts words in half; this splits inside
silences, choosing the pause closest to the far end of a [min_chunk, max_chunk]
window so chunks stay long (fewer chunks = less boundary loss and less padding
waste) without exceeding the cap.

Energy-based VAD, pure torch — no new dependencies, deterministic, and cheap
(one STFT-free framing pass per file). Good enough for finding pauses in speech;
for noisy field recordings a neural VAD would segment better, but this needs no
model and no download.

Returns sample ranges; the caller transcribes them and joins the texts.

Gate chunking on a DURATION THRESHOLD (~45 s), not on everything: quality is
flat to ~45 s and collapses past 60 s, so chunking short audio is
neutral-to-harmful. See docs/asr/caveats.md for the measured curve.
"""

from dataclasses import dataclass

import torch


@dataclass
class ChunkConfig:
    min_chunk: float = 10.0  # prefer chunks at least this long
    max_chunk: float = 15.0  # never exceed (hard cut if no silence found)
    min_sil: float = 0.20  # a pause must last this long to be a split point
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    floor_pct: float = 10.0  # noise floor percentile of frame energies
    floor_margin_db: float = 8.0  # silence = below (floor + margin)
    min_tail: float = 2.0  # merge a trailing chunk shorter than this


def frame_db(wav: torch.Tensor, sr: int, cfg: ChunkConfig) -> torch.Tensor:
    """Per-frame energy in dB (hop-aligned, one value per hop)."""
    win = max(1, int(sr * cfg.frame_ms / 1000))
    hop = max(1, int(sr * cfg.hop_ms / 1000))
    if wav.numel() < win:
        wav = torch.nn.functional.pad(wav, (0, win - wav.numel()))
    frames = wav.unfold(0, win, hop)  # (n_frames, win)
    rms = frames.pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
    return 20.0 * torch.log10(rms)


def silence_runs(db: torch.Tensor, cfg: ChunkConfig) -> list[tuple[int, int]]:
    """Contiguous [start, end) frame ranges that are silent and long enough."""
    floor = torch.quantile(db, cfg.floor_pct / 100.0)
    peak = db.max()
    thr = torch.minimum(floor + cfg.floor_margin_db, peak - 20.0)
    sil = (db < thr).tolist()
    min_frames = max(1, int(cfg.min_sil * 1000 / cfg.hop_ms))
    runs, start = [], None
    for i, s in enumerate(sil):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    if start is not None and len(sil) - start >= min_frames:
        runs.append((start, len(sil)))
    return runs


def split_points(
    wav: torch.Tensor, sr: int, cfg: ChunkConfig | None = None
) -> list[tuple[int, int]]:
    """Segment `wav` into (start_sample, end_sample) ranges.

    Greedy: from each chunk start, consider silences whose midpoint lands in
    [min_chunk, max_chunk]; take the LAST such silence (longest chunk that still
    fits). If none exists, look for any silence before max_chunk; failing that,
    hard-cut at max_chunk.
    """
    cfg = ChunkConfig() if cfg is None else cfg
    n = wav.numel()
    total = n / sr
    if total <= cfg.max_chunk:
        return [(0, n)]

    hop = max(1, int(sr * cfg.hop_ms / 1000))
    db = frame_db(wav, sr, cfg)
    runs = silence_runs(db, cfg)
    # candidate cut samples: the middle of each silence run
    cuts = [((a + b) // 2) * hop for a, b in runs]

    out, pos = [], 0
    while pos < n:
        if (n - pos) / sr <= cfg.max_chunk:
            out.append((pos, n))
            break
        lo = pos + int(cfg.min_chunk * sr)
        hi = pos + int(cfg.max_chunk * sr)
        window = [c for c in cuts if lo <= c <= hi]
        if not window:  # relax: any pause before hi
            window = [c for c in cuts if pos + int(0.5 * sr) < c <= hi]
        cut = window[-1] if window else hi  # else hard cut
        out.append((pos, cut))
        pos = cut

    # merge a too-short tail into its predecessor
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) / sr < cfg.min_tail:
        a, _ = out[-2]
        _, b = out[-1]
        out = [*out[:-2], (a, b)]
    return out


def chunk_audio(wav: torch.Tensor, sr: int, cfg: ChunkConfig | None = None):
    """Yield (index, start_s, end_s, samples) for each chunk."""
    cfg = ChunkConfig() if cfg is None else cfg
    for i, (a, b) in enumerate(split_points(wav, sr, cfg)):
        yield i, a / sr, b / sr, wav[a:b]
