# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Waveform input: raw arrays/tensors, and slices of a file.

The existing paths take **file paths** and read whole files
(``IndicASREngine.load_audio``). Two callers need more:

* a pipeline that already holds decoded audio (or reads one long span once and slices it
  in memory) wants to hand over **arrays** without a round-trip through a temp wav;
* a pipeline that transcribes **segments** of a multi-hour recording wants
  ``(path, start_s, end_s)`` rather than one file per chunk.

Everything here reproduces the production front-end exactly, so output is unchanged:

* resample via ``feature_extractor.resample`` — torchaudio, matching what Lhotse used
  to produce every transcript on disk (a ``soxr``-based resampler such as
  ``librosa.load``'s default is *not* equivalent: it moved text on 5 of 48 chunks);
* pad sub-1 s audio to 1 s **centred**, which is production's
  ``pad_min_duration=1.0`` / ``pad_direction='both'``;
* mono by channel mean.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import soundfile as sf
import torch

Waveform = np.ndarray | torch.Tensor


def _as_mono_tensor(wav: Waveform) -> torch.Tensor:
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(wav))
    if not isinstance(wav, torch.Tensor):
        raise TypeError(f"expected numpy array or torch tensor, got {type(wav)}")
    wav = wav.detach().to(torch.float32)
    if wav.ndim == 2:  # (C, S) or (S, C) -> mono
        wav = wav.mean(dim=0 if wav.shape[0] < wav.shape[1] else 1)
    elif wav.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D waveform, got shape {tuple(wav.shape)}")
    return wav.contiguous()


def read_slice(path: str, start_s: float, end_s: float, feature_extractor) -> torch.Tensor:
    """One ``[start_s, end_s)`` span of a file, at the model's sample rate.

    A seek + partial read, never a whole-file decode -- the point is to transcribe a
    15 s chunk of a multi-hour recording without reading the hours.
    """
    with sf.SoundFile(path) as fh:
        native = fh.samplerate
        fh.seek(int(float(start_s) * native))
        frames = int((float(end_s) - float(start_s)) * native)
        wav = fh.read(frames, dtype="float32", always_2d=False)
    if wav.size == 0:
        raise ValueError(f"empty read: {path} [{start_s}, {end_s})")
    return feature_extractor.resample(_as_mono_tensor(wav), native)


def read_span_and_slice(path: str, spans: Sequence[tuple[float, float]], feature_extractor) -> list:
    """Read ONE covering span, return a slice per ``(start_s, end_s)``.

    Per-chunk reads pay an open + seek + resample each; one sequential read of the
    covering span pays them once and the slices are free views. Measured on a 5.8 h
    episode with 3,177 chunks: **529.7 s of per-chunk reads vs 12.8 s** for one read
    plus slicing.

    Only worth it when the spans are close together -- the caller should keep the
    covering span from ballooning (e.g. group chunks that are contiguous in time),
    because this reads ``[min(start), max(end))`` whether or not the middle is wanted.

    **The cut happens at the native rate, and each slice is resampled on its own.**
    Resampling the whole span and then cutting is *not* equivalent: the resampler's
    filter sees different edges and the cut points land on different filter phases, so
    the samples differ from what a per-chunk read produces. That was measured -- an
    earlier version of this function did exactly that and failed an equality check
    against per-chunk reads. Since every transcript on disk came from per-chunk reads,
    the cheap-but-different version would have silently changed output. This ordering
    keeps the saving that actually matters (one open + one sequential read instead of N
    seeks) while staying sample-identical.
    """
    if not spans:
        return []
    lo = min(float(a) for a, _ in spans)
    hi = max(float(b) for _, b in spans)
    with sf.SoundFile(path) as fh:
        native = fh.samplerate
        fh.seek(int(lo * native))
        raw = fh.read(int((hi - lo) * native), dtype="float32", always_2d=False)
    if raw.size == 0:
        raise ValueError(f"empty read: {path} [{lo}, {hi})")
    raw = _as_mono_tensor(raw)
    out = []
    for a, b in spans:
        # native-domain cut, mirroring read_slice's own int() truncation exactly
        i = int(float(a) * native) - int(lo * native)
        n = int((float(b) - float(a)) * native)
        piece = raw[i : i + n]
        out.append(feature_extractor.resample(piece, native))
    return out


def collate_waveforms(audio, feature_extractor, *, sample_rate: int | None = None):
    """``(B, S) float32`` batch + lengths, from arrays/tensors or a collated tensor.

    Reproduces ``IndicASREngine.collate``: sub-1 s rows are padded to 1 s **centred**
    (production's ``pad_direction='both'``) and their length is reported as the padded
    length, because the encoder is fed the padded row.

    ``sample_rate`` resamples first; omit it when the audio is already at the model rate.
    """
    if isinstance(audio, torch.Tensor) and audio.ndim == 2:
        batch = audio.to(torch.float32)
        lens = torch.full((batch.size(0),), batch.size(1), dtype=torch.int64)
        return batch, lens

    wavs = [_as_mono_tensor(w) for w in audio]
    if sample_rate is not None and sample_rate != feature_extractor.sample_rate:
        wavs = [feature_extractor.resample(w, sample_rate) for w in wavs]
    if not wavs:
        raise ValueError("no audio given")

    min_len = feature_extractor.sample_rate
    lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.int64)
    pad_max = max(int(lens.max()), min_len)
    batch = torch.zeros(len(wavs), pad_max, dtype=torch.float32)
    for i, w in enumerate(wavs):
        n = w.shape[0]
        if n < min_len:
            off = round((min_len - n) / 2)
            batch[i, off : off + n] = w
            lens[i] = min_len
        else:
            batch[i, :n] = w
    return batch, lens
