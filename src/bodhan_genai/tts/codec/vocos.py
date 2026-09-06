"""Vocos decoder for SNAC codes — a drop-in replacement for SNAC's own decoder.

SNAC has three parts: encoder, quantizer, decoder. Only the *decoder* is replaced
here. The quantizer is still needed to turn discrete codes into the ``z_q`` latent
that Vocos consumes, so :class:`VocosSnac` holds a real SNAC model and delegates
everything except :meth:`decode` to it.

That keeps the swap invisible to callers: ``decode_audio``, ``batch_decode_audio``
and ``decode_window_batch`` all call ``snac_model.decode(codes)`` and neither know
nor care which decoder is behind it.

Architecture follows Vocos (arXiv:2306.00814): a ConvNeXt-1D backbone at frame rate
plus a single exp-magnitude + phase iSTFT head. Weights are the fine-tuned decoder
published alongside the SFT checkpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_VOCOS_REPO = "bodhan-ai/indic-speak"
DEFAULT_VOCOS_FILE = "vocos/best.pt"

# SNAC's latent width at 24 kHz; the checkpoint's own config supplies the rest.
_LATENT_DIM = 768


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init: float):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class ISTFTHead(nn.Module):
    """Predict magnitude + phase, then one centered inverse STFT.

    Pads a single feature frame (replicate) before the iSTFT so that
    ``hop * (T_feat + 1 - 1) == 512 * L`` exactly — no off-by-hop trims.
    """

    def __init__(self, dim: int, n_fft: int, hop: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] -> [B, 1, T*hop]
        x = torch.nn.functional.pad(x, (0, 1), mode="replicate")
        p = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, phase = p.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        spec = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        wav = torch.istft(
            spec, self.n_fft, self.hop, self.n_fft, self.window.to(spec.real.dtype), center=True
        )
        return wav.unsqueeze(1)


class VocosDecoder(nn.Module):
    """SNAC ``z_q`` ``[B, 768, L]`` -> waveform ``[B, 1, 512*L]`` at 24 kHz."""

    def __init__(
        self,
        latent_dim: int = _LATENT_DIM,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_blocks_pre: int = 2,
        num_blocks_post: int = 8,
        upsample: int = 2,
        n_fft: int = 1024,
        hop: int = 256,
    ):
        super().__init__()
        if upsample * hop != 512:
            raise ValueError(
                f"upsample*hop must be 512 to preserve SNAC's samples-per-step, "
                f"got {upsample}*{hop}={upsample * hop}"
            )
        self.upsample = upsample
        ls = 1.0 / (num_blocks_pre + num_blocks_post)
        self.stem = nn.Conv1d(latent_dim, dim, kernel_size=7, padding=3)
        self.pre = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_pre)]
        )
        self.up_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.post = nn.ModuleList(
            [ConvNeXtBlock(dim, intermediate_dim, ls) for _ in range(num_blocks_post)]
        )
        self.final_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = ISTFTHead(dim, n_fft, hop)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.stem(z_q)
        for b in self.pre:
            x = b(x)
        x = self.up_conv(torch.repeat_interleave(x, self.upsample, dim=-1))
        for b in self.post:
            x = b(x)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)


# Config keys that describe *which* decoder variant was trained, rather than
# arguments to this one. This module implements the istft head with no noise
# injection; any other value needs a different module, so it must fail loudly.
_VARIANT_KEYS = {"head_type": "istft", "noise_inject": False}


def _build_from_config(model_cfg: dict) -> VocosDecoder:
    cfg = dict(model_cfg)
    for key, supported in _VARIANT_KEYS.items():
        value = cfg.pop(key, supported)
        if value != supported:
            raise ValueError(
                f"this decoder implements {key}={supported!r}; checkpoint asks for "
                f"{key}={value!r}, which needs a different module"
            )
    return VocosDecoder(latent_dim=_LATENT_DIM, **cfg)


def load_vocos_decoder(
    vocos_path: str | None = None,
    device: str = "cuda",
    *,
    repo_id: str = DEFAULT_VOCOS_REPO,
    weights: str = "ema",
) -> VocosDecoder:
    """Load the fine-tuned decoder from a local ``.pt`` or the Hub.

    The checkpoint carries its own hyperparameters under ``config.model``, so the
    module is always built to match its weights rather than to module defaults.

    ``weights="ema"`` (the default) prefers the EMA shadow, which beats the live
    weights on every validation split. Checkpoint layouts differ:

    * training checkpoints hold ``vocos`` (live) **and** ``ema``
    * older ``best.pt`` files were written as ``ema.full_state_dict(vocos)`` — their
      ``vocos`` key already holds EMA weights and there is no separate ``ema`` key

    Preferring ``ema`` and falling back to ``vocos`` therefore yields the EMA weights
    for both layouts. Pass ``weights="vocos"`` to force the live weights.
    """
    if vocos_path is None or not os.path.isfile(str(vocos_path)):
        if vocos_path is None:
            from huggingface_hub import hf_hub_download

            vocos_path = hf_hub_download(repo_id=repo_id, filename=DEFAULT_VOCOS_FILE)
        else:
            raise FileNotFoundError(f"vocos checkpoint not found: {vocos_path}")

    ckpt = torch.load(vocos_path, map_location="cpu", weights_only=False)
    key = "ema" if (weights == "ema" and isinstance(ckpt.get("ema"), dict)) else "vocos"
    if key not in ckpt:
        raise KeyError(f"{vocos_path} has no {key!r} state dict (keys: {sorted(ckpt)[:6]})")
    decoder = _build_from_config((ckpt.get("config") or {}).get("model") or {})
    decoder.load_state_dict(ckpt[key])
    logger.info(
        "Loaded Vocos decoder from %s (step %s, %s weights)",
        vocos_path,
        ckpt.get("step", "?"),
        key,
    )
    return decoder.eval().to(device)


class VocosSnac:
    """A SNAC model whose decoder is the fine-tuned Vocos decoder.

    Duck-types the SNAC model: every attribute other than :meth:`decode` — including
    ``encode``, ``vq_strides`` and ``hop_length`` — is delegated to the wrapped model,
    so existing decode call sites work unchanged.
    """

    def __init__(self, snac_model: Any, vocos: VocosDecoder):
        self._snac = snac_model
        self._vocos = vocos

    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Hierarchical SNAC codes -> waveform ``[B, 1, T]``."""
        z_q = self._snac.quantizer.from_codes(codes)
        return self._vocos(z_q.float())

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the wrapper itself.
        return getattr(self._snac, name)

    def __repr__(self) -> str:
        return f"VocosSnac({self._snac.__class__.__name__} + VocosDecoder)"


def wrap_with_vocos(
    snac_model: Any, vocos_path: str | None = None, device: str = "cuda"
) -> VocosSnac:
    """Convenience: load the decoder and wrap an existing SNAC model with it."""
    return VocosSnac(snac_model, load_vocos_decoder(vocos_path, device=device))


def resolve_decoder(snac_model: Any, spec: Any, device: str = "cuda") -> Any:
    """Apply a ``vocos`` setting to a freshly loaded SNAC model.

    ``spec`` is ``False``/``"false"`` (keep SNAC's decoder), ``True``/``"true"``
    (fetch the published decoder from the Hub), or a path to a local checkpoint.

    Loading failures propagate rather than falling back to SNAC's decoder: quietly
    decoding with a different vocoder than the one asked for would be worse than
    failing.
    """
    if isinstance(spec, str):
        text = spec.strip()
        if text.lower() in ("false", "0", "none", ""):
            return snac_model
        path = None if text.lower() in ("true", "1") else text
    elif spec:
        path = None
    else:
        return snac_model

    return wrap_with_vocos(snac_model, path, device=device)
