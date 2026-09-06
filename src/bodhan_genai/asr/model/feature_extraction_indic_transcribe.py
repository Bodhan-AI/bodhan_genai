# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0 — see NOTICE.
#
# Derived from NeMo's FilterbankFeatures for inference parity.
"""GPU-capable mel front-end for IndicTranscribe, numerically matching NeMo.

Parity-critical details:
  - resampling (24k eval audio -> 16k) MUST be torchaudio's sinc_interp_hann
    Resample (lhotse vendors exactly this); librosa/soxr do NOT match.
  - window: hann(400, periodic=False) — loaded from the checkpoint buffer.
  - stft(n_fft=512, hop=160, win=400, center=True, pad_mode='reflect').
  - magnitude -> power(2) -> mel matmul with the checkpoint `fb` buffer
    -> log(x + 2**-24).
  - per-feature normalization over TRUE lengths: mean, then UNBIASED (n-1)
    std, then `std += 1e-5` AFTER the sqrt.
  - seq_len = floor(samples / 160) + 1; padded frames zero-filled at the end.
  - dither/pad_to are inert at inference (training-only / 0). Runs in fp32
    regardless of model dtype (NeMo hard-forces fp32 here too).
"""

import os

import torch
import torch.nn as nn
import torchaudio

FEATURE_FILE = "feature_extractor.safetensors"

CONSTANT = 1e-5
LOG_ZERO_GUARD = 2.0**-24


class IndicTranscribeFeatureExtractor(nn.Module):
    sample_rate = 16000
    n_fft = 512
    win_length = 400
    hop_length = 160
    n_mels = 128

    def __init__(self):
        super().__init__()
        self.preemph = 0.97
        # Placeholders; real values come from the converted checkpoint buffers.
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=False))
        self.register_buffer("fb", torch.zeros(1, self.n_mels, self.n_fft // 2 + 1))
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    # ---- audio loading ------------------------------------------------------

    def resample(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Match lhotse's production path: torchaudio sinc_interp_hann defaults,
        then trim/pad to lhotse's ROUND_HALF_UP(duration * new_sr) sample count
        (torchaudio returns ceil, one extra tail sample for e.g. 24k->16k)."""
        if orig_sr == self.sample_rate:
            return audio
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
        out = self._resamplers[orig_sr](audio)
        from decimal import ROUND_HALF_UP, Decimal

        duration = audio.shape[-1] / orig_sr
        n = int(Decimal(round(duration * self.sample_rate, 8)).quantize(0, rounding=ROUND_HALF_UP))
        if out.shape[-1] > n:
            out = out[..., :n]
        elif out.shape[-1] < n:
            out = torch.nn.functional.pad(out, (0, n - out.shape[-1]))
        return out

    # ---- features -----------------------------------------------------------

    def get_seq_len(self, sample_lens: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(sample_lens, self.hop_length) + 1

    @torch.no_grad()
    def forward(
        self, audio: torch.Tensor, sample_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """audio: (B, S) float32 16 kHz, padded; sample_lens: (B,) true lengths.
        Returns (features (B, 128, T) fp32, feat_lens (B,))."""
        x = audio.to(torch.float32)
        seq_len = self.get_seq_len(sample_lens).to(torch.long)
        if bool((seq_len <= 1).any()):
            # NeMo raises here too: a single frame makes the unbiased std NaN
            raise ValueError(
                "audio shorter than one hop (160 samples); pad to >= 1s "
                "as the production pipeline does"
            )

        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(dtype=torch.float),
                center=True,
                return_complex=True,
            )
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))
        x = x.pow(2.0)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + LOG_ZERO_GUARD)

        # per-feature normalization over valid frames (NeMo normalize_batch)
        b, _, t = x.shape
        time_steps = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        denom = valid_mask.sum(dim=1)
        mean = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(dim=2) / denom.unsqueeze(1)
        std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - mean.unsqueeze(2), 0.0) ** 2, dim=2)
            / (denom.unsqueeze(1) - 1.0)
        )
        std += CONSTANT
        x = (x - mean.unsqueeze(2)) / std.unsqueeze(2)

        x = x.masked_fill(~valid_mask.unsqueeze(1), 0.0)
        return x, seq_len

    # ---- persistence --------------------------------------------------------

    def save_pretrained(self, save_dir: str):
        from safetensors.torch import save_file

        os.makedirs(save_dir, exist_ok=True)
        save_file(
            {"fb": self.fb.contiguous(), "window": self.window.contiguous()},
            os.path.join(save_dir, FEATURE_FILE),
        )

    @classmethod
    def from_pretrained(
        cls, load_dir: str, device: str | None = None
    ) -> "IndicTranscribeFeatureExtractor":
        from safetensors.torch import load_file

        from bodhan_genai.asr.checkpoints import resolve_file

        fe = cls()
        # safetensors reads the filesystem only; resolve_file accepts a directory or a repo id.
        tensors = load_file(resolve_file(load_dir, FEATURE_FILE))
        fe.fb.copy_(tensors["fb"])
        fe.window.copy_(tensors["window"])
        if device is not None:
            fe.to(device)
        fe.eval()
        return fe
