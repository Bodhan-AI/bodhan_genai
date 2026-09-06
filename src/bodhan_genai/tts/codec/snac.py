"""
bodhan_genai.tts.codec.snac — SNAC audio codec encoder/decoder, decoupled from vocabulary.

The SNAC 24kHz model outputs 3 codebooks at different temporal rates (1:2:4):
  c0: [seq_len]      (12 Hz, coarsest level)
  c1: [2*seq_len]    (23 Hz)
  c2: [4*seq_len]    (47 Hz, finest level)

These are flattened to 7 tokens per frame:
  Frame i → [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]]

Each raw code (0-4095) is offset by: audio_token_base_id + position_index * 4096
  Position 0 (c0[i]):    base + 0*4096
  Position 1 (c1[2i]):   base + 1*4096
  Position 2 (c2[4i]):   base + 2*4096
  Position 3 (c2[4i+1]): base + 3*4096
  Position 4 (c1[2i+1]): base + 4*4096
  Position 5 (c2[4i+2]): base + 5*4096
  Position 6 (c2[4i+3]): base + 6*4096

Consecutive duplicate frames (same c0 value) are removed.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

SNAC_NUM_CODEBOOKS = 7
SNAC_CODEBOOK_SIZE = 4096
SNAC_TOTAL_AUDIO_TOKENS = SNAC_NUM_CODEBOOKS * SNAC_CODEBOOK_SIZE  # 28,672

# Orpheus-style streaming decode: a fixed 4-frame (28-token) window decoded per
# emitted frame; only the middle frame's audio is kept, so the convolutional
# context on both sides hides the window seams.
SNAC_WINDOW_FRAMES = 4
SNAC_WINDOW_TOKENS = SNAC_WINDOW_FRAMES * SNAC_NUM_CODEBOOKS  # 28


def _maybe_compile_model(
    model: object,
    *,
    compile_model: bool = True,
    compile_mode: str | None = None,
    compile_backend: str = "inductor",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = None,
    compile_options: dict[str, Any] | None = None,
) -> object:
    """Optionally wrap a model with ``torch.compile``.

    Defaults preserve the historical behavior here: compile once with torch's
    defaults. Extra knobs are opt-in for benchmark scripts that want to try a
    more aggressive fixed-shape mode.
    """
    if not compile_model:
        return model

    compile_kwargs: dict[str, Any] = {}
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    if compile_backend != "inductor":
        compile_kwargs["backend"] = compile_backend
    if compile_fullgraph:
        compile_kwargs["fullgraph"] = True
    if compile_dynamic is not None:
        compile_kwargs["dynamic"] = compile_dynamic
    if compile_options is not None:
        compile_kwargs["options"] = dict(compile_options)
    return torch.compile(model, **compile_kwargs)


def load_snac_model(
    snac_model_path: str,
    device: str = "cuda",
    *,
    model_dtype: torch.dtype | None = None,
    compile_model: bool = True,
    compile_mode: str | None = None,
    compile_backend: str = "inductor",
    compile_fullgraph: bool = False,
    compile_dynamic: bool | None = None,
    compile_options: dict[str, Any] | None = None,
) -> object:
    """Load and return a SNAC model, optionally compiled with ``torch.compile``."""
    from snac import SNAC

    if os.path.isdir(snac_model_path):
        config_path = os.path.join(snac_model_path, "config.json")
        model_path_bin = os.path.join(snac_model_path, "pytorch_model.bin")
        model = SNAC.from_config(config_path)
        state_dict = torch.load(model_path_bin, map_location=device, weights_only=True)
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=snac_model_path, filename="config.json")
        model_path_bin = hf_hub_download(repo_id=snac_model_path, filename="pytorch_model.bin")
        model = SNAC.from_config(config_path)
        state_dict = torch.load(model_path_bin, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model = model.eval()
    if model_dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=model_dtype)
    model = _maybe_compile_model(
        model,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_options=compile_options,
    )
    if compile_model:
        logger.info(
            "SNAC model loaded and compiled on %s (mode=%s, backend=%s, fullgraph=%s, dynamic=%s)",
            device,
            compile_mode or "<torch-default>",
            compile_backend,
            compile_fullgraph,
            compile_dynamic,
        )
    else:
        logger.info("SNAC model loaded on %s without torch.compile", device)
    return model


def _autocast_context(
    *,
    device: str,
    autocast_dtype: torch.dtype | None,
):
    use_cuda = device.startswith("cuda") if isinstance(device, str) else False
    if not use_cuda:
        return nullcontext()
    if autocast_dtype is None:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    if autocast_dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)


def _remove_duplicate_frames(codes_tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove consecutive duplicate frames from a flattened SNAC token tensor.

    Two frames are considered duplicates if they share the same c0 (first) token.
    Input shape: [7 * num_frames]
    Output shape: [7 * num_unique_frames]
    """
    if codes_tensor.shape[0] % SNAC_NUM_CODEBOOKS != 0:
        raise ValueError(
            f"Input length {codes_tensor.shape[0]} not divisible by {SNAC_NUM_CODEBOOKS}"
        )

    frames = codes_tensor.view(-1, SNAC_NUM_CODEBOOKS)  # [num_frames, 7]

    if frames.shape[0] == 1:
        return codes_tensor

    first_tokens = frames[:, 0]
    diff_mask = torch.ones(frames.shape[0], dtype=torch.bool, device=frames.device)
    diff_mask[1:] = first_tokens[1:] != first_tokens[:-1]

    return frames[diff_mask].reshape(-1)


def encode_audio(
    snac_model: object,
    audio: np.ndarray,
    audio_token_base_id: int,
    device: str = "cuda",
    autocast_dtype: torch.dtype | None = None,
) -> list[int]:
    """
    Encode a raw 24kHz mono audio waveform to flattened SNAC token IDs.

    Args:
        snac_model: Loaded SNAC model (from load_snac_model).
        audio: numpy array of shape [samples], float32, 24kHz mono.
        audio_token_base_id: Token ID of <|snac_0|> in the extended tokenizer.
        device: CUDA device string.

    Returns:
        List of token IDs in range [audio_token_base_id, audio_token_base_id + 28672).
        Consecutive duplicate frames are removed.
    """
    with torch.inference_mode(), _autocast_context(device=device, autocast_dtype=autocast_dtype):
        audio_tensor = torch.tensor(audio, device=device, dtype=torch.float32)
        # SNAC expects [batch, channels, samples]
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        codes = snac_model.encode(audio_tensor)

        c0 = codes[0][0]  # [seq_len]
        c1 = codes[1][0]  # [2*seq_len]
        c2 = codes[2][0]  # [4*seq_len]

        base = audio_token_base_id
        offset_tensor = torch.tensor(
            [
                base + 0 * SNAC_CODEBOOK_SIZE,
                base + 1 * SNAC_CODEBOOK_SIZE,
                base + 2 * SNAC_CODEBOOK_SIZE,
                base + 3 * SNAC_CODEBOOK_SIZE,
                base + 4 * SNAC_CODEBOOK_SIZE,
                base + 5 * SNAC_CODEBOOK_SIZE,
                base + 6 * SNAC_CODEBOOK_SIZE,
            ],
            device=device,
            dtype=c0.dtype,
        )

        # Vectorized interleaving: 7 tokens per frame
        all_codes = torch.stack(
            [
                c0 + offset_tensor[0],  # c0[i]
                c1[::2] + offset_tensor[1],  # c1[2i]
                c2[::4] + offset_tensor[2],  # c2[4i]
                c2[1::4] + offset_tensor[3],  # c2[4i+1]
                c1[1::2] + offset_tensor[4],  # c1[2i+1]
                c2[2::4] + offset_tensor[5],  # c2[4i+2]
                c2[3::4] + offset_tensor[6],  # c2[4i+3]
            ],
            dim=1,
        ).reshape(-1)  # [7 * seq_len]

    all_codes = _remove_duplicate_frames(all_codes)
    return all_codes.cpu().numpy().astype(np.int64).tolist()


def _compute_valid_code_frames(num_audio_samples: int, snac_model: object) -> int:
    """
    Compute how many SNAC code frames correspond to num_audio_samples of audio.

    Mirrors SNAC's internal right-pad + encode logic so we can trim codes
    produced from zero-padded (batched) input back to the correct length.

    SNAC pads audio to: ceil(T / (hop_length * lcm)) * (hop_length * lcm)
    Then codes[0] length = padded_T / hop_length / vq_strides[0]
    That coarsest-stream length == number of 7-token frames to keep.
    """
    hop_length = snac_model.hop_length
    vq_stride_0 = snac_model.vq_strides[0]
    attn_window = getattr(snac_model, "attn_window_size", None) or 1
    lcm = math.lcm(vq_stride_0, attn_window)
    pad_to = hop_length * lcm

    padded_len = math.ceil(num_audio_samples / pad_to) * pad_to
    n_frames = padded_len // hop_length // vq_stride_0
    return n_frames


def batch_encode_audio(
    snac_model: object,
    waveforms: list[np.ndarray],
    audio_token_base_id: int,
    device: str = "cuda",
    autocast_dtype: torch.dtype | None = None,
) -> list[list[int]]:
    """
    Encode multiple 24kHz mono waveforms in a single batched SNAC forward pass.

    Pads all waveforms to the length of the longest, runs one batched encode(),
    then trims each sample's codes to the correct frame count based on its
    original length.

    Args:
        snac_model: Loaded SNAC model (from load_snac_model).
        waveforms: List of numpy arrays, each shape [samples], float32, 24kHz mono.
        audio_token_base_id: Token ID of <|snac_0|>.
        device: CUDA device string.

    Returns:
        List of token ID lists (one per input waveform). Empty list for samples
        that failed to encode.
    """
    if not waveforms:
        return []

    original_lengths = [w.shape[0] for w in waveforms]
    max_len = max(original_lengths)

    # Pad all waveforms to uniform length and stack into (B, 1, max_len)
    padded = torch.zeros(len(waveforms), 1, max_len, device=device, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        t = torch.from_numpy(w).to(device=device, dtype=torch.float32)
        padded[i, 0, : t.shape[0]] = t

    with torch.inference_mode(), _autocast_context(device=device, autocast_dtype=autocast_dtype):
        # codes: list of 3 tensors with shapes (B, N), (B, 2N), (B, 4N)
        codes = snac_model.encode(padded)

    base = audio_token_base_id
    offsets = [base + p * SNAC_CODEBOOK_SIZE for p in range(SNAC_NUM_CODEBOOKS)]

    results: list[list[int]] = []
    for b, orig_len in enumerate(original_lengths):
        try:
            n_frames = _compute_valid_code_frames(orig_len, snac_model)
            # Guard: model may produce fewer frames than expected
            n_frames = min(n_frames, codes[0].shape[-1])

            if n_frames == 0:
                results.append([])
                continue

            # Extract this sample's codes and trim to valid frames
            c0 = codes[0][b, :n_frames]  # [n_frames]
            c1 = codes[1][b, : 2 * n_frames]  # [2 * n_frames]
            c2 = codes[2][b, : 4 * n_frames]  # [4 * n_frames]

            # Vectorized interleaving: 7 tokens per frame
            all_codes = torch.stack(
                [
                    c0 + offsets[0],
                    c1[::2] + offsets[1],
                    c2[::4] + offsets[2],
                    c2[1::4] + offsets[3],
                    c1[1::2] + offsets[4],
                    c2[2::4] + offsets[5],
                    c2[3::4] + offsets[6],
                ],
                dim=1,
            ).reshape(-1)

            all_codes = _remove_duplicate_frames(all_codes)
            results.append(all_codes.cpu().numpy().astype(np.int64).tolist())

        except Exception as e:
            logger.warning(f"Batch encode: sample {b} failed: {e}")
            results.append([])

    return results


def decode_audio(
    snac_model: object,
    token_ids: list[int],
    audio_token_base_id: int,
    device: str = "cuda",
    autocast_dtype: torch.dtype | None = None,
) -> bytes | None:
    """
    Decode flattened SNAC token IDs back to audio waveform bytes (int16 PCM).

    Args:
        snac_model: Loaded SNAC model.
        token_ids: List of token IDs produced by encode_audio.
        audio_token_base_id: Token ID of <|snac_0|> (same value used during encoding).
        device: CUDA device string.

    Returns:
        Audio as bytes (int16 PCM, 24kHz mono), or None if fewer than 7 tokens.
    """
    if len(token_ids) < SNAC_NUM_CODEBOOKS:
        return None

    # Cross-backbone safety: if these token IDs were produced with a different
    # tokenizer's audio_token_base_id (e.g. caller passes llama3-vocab IDs with
    # another backbone's base, or vice versa), the per-frame range checks below may
    # coincidentally pass while the decoded audio is garbage. Refuse upfront.
    audio_lo = audio_token_base_id
    audio_hi = audio_token_base_id + SNAC_NUM_CODEBOOKS * SNAC_CODEBOOK_SIZE
    id_min = min(token_ids)
    id_max = max(token_ids)
    if id_min < audio_lo or id_max >= audio_hi:
        raise ValueError(
            f"decode_audio received token_ids outside the audio-vocab range for "
            f"audio_token_base_id={audio_token_base_id}: expected [{audio_lo}, "
            f"{audio_hi}), got [{id_min}, {id_max}]. This usually means the "
            f"token_ids were tokenized by a different backbone's tokenizer."
        )

    # Strip offsets: raw_code = token_id - audio_token_base_id - position_index * 4096
    raw_codes = [
        x - audio_token_base_id - (i % SNAC_NUM_CODEBOOKS) * SNAC_CODEBOOK_SIZE
        for i, x in enumerate(token_ids)
    ]

    num_frames = len(raw_codes) // SNAC_NUM_CODEBOOKS
    raw_codes = raw_codes[: num_frames * SNAC_NUM_CODEBOOKS]

    # Validate frames — vectorized numpy, no threadpool overhead
    arr = np.array(raw_codes, dtype=np.int32).reshape(num_frames, SNAC_NUM_CODEBOOKS)
    valid_mask = np.all((arr >= 0) & (arr < SNAC_CODEBOOK_SIZE), axis=1)
    invalid_count = int((~valid_mask).sum())

    if invalid_count > 0:
        logger.warning(f"Skipping {invalid_count}/{num_frames} out-of-range frames")

    valid_arr = arr[valid_mask]  # [n, 7]
    n = valid_arr.shape[0]

    if n == 0:
        raise ValueError("No valid frames found — cannot decode audio.")

    # Build interleaved codebook tensors from the validated numpy array (zero-copy via from_numpy)
    codes_0 = torch.from_numpy(valid_arr[:, 0].copy()).to(device, dtype=torch.int32)

    codes_1_raw = np.empty(n * 2, dtype=np.int32)
    codes_1_raw[0::2] = valid_arr[:, 1]
    codes_1_raw[1::2] = valid_arr[:, 4]
    codes_1 = torch.from_numpy(codes_1_raw).to(device)

    codes_2_raw = np.empty(n * 4, dtype=np.int32)
    codes_2_raw[0::4] = valid_arr[:, 2]
    codes_2_raw[1::4] = valid_arr[:, 3]
    codes_2_raw[2::4] = valid_arr[:, 5]
    codes_2_raw[3::4] = valid_arr[:, 6]
    codes_2 = torch.from_numpy(codes_2_raw).to(device)

    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    with torch.inference_mode(), _autocast_context(device=device, autocast_dtype=autocast_dtype):
        audio_hat = snac_model.decode(codes)

    audio_np = audio_hat.detach().cpu().numpy()
    # Clip before int16 conversion to prevent overflow
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


def _parse_token_ids_to_frames(
    token_ids: list[int],
    audio_token_base_id: int,
) -> np.ndarray | None:
    """Parse one flattened SNAC token sequence into a validated ``[n_frames, 7]``
    int32 array of raw codes (offsets stripped), or ``None`` if the sequence is
    unusable (too short, all tokens outside the audio vocab, or no valid frame).

    Shares the offset/range logic with ``decode_audio`` but, unlike it, never
    raises on out-of-range IDs: in batch mode a single bad sequence must not
    poison the rest, so the caller treats ``None`` as a per-row failure. (If the
    whole batch comes back ``None``, that surfaces a wrong ``audio_token_base_id``
    just as loudly.)
    """
    if len(token_ids) < SNAC_NUM_CODEBOOKS:
        return None
    ids = np.asarray(token_ids, dtype=np.int64)
    audio_lo = audio_token_base_id
    audio_hi = audio_token_base_id + SNAC_NUM_CODEBOOKS * SNAC_CODEBOOK_SIZE
    if int(ids.min()) < audio_lo or int(ids.max()) >= audio_hi:
        return None

    num_frames = ids.shape[0] // SNAC_NUM_CODEBOOKS
    ids = ids[: num_frames * SNAC_NUM_CODEBOOKS]
    pos = np.arange(num_frames * SNAC_NUM_CODEBOOKS) % SNAC_NUM_CODEBOOKS
    raw = (ids - audio_token_base_id - pos * SNAC_CODEBOOK_SIZE).reshape(
        num_frames, SNAC_NUM_CODEBOOKS
    )
    valid_mask = np.all((raw >= 0) & (raw < SNAC_CODEBOOK_SIZE), axis=1)
    valid = raw[valid_mask]
    if valid.shape[0] == 0:
        return None
    return valid.astype(np.int32)


def batch_decode_audio(
    snac_model: object,
    token_ids_list: list[list[int]],
    audio_token_base_id: int,
    device: str = "cuda",
    autocast_dtype: torch.dtype | None = None,
    max_batch_size: int = 32,
) -> list[bytes | None]:
    """Decode many flattened SNAC token sequences with batched SNAC forwards.

    Counterpart to ``batch_encode_audio``. Sequences are sorted by frame count
    and chunked into groups of ``<= max_batch_size``; each chunk is zero-padded
    to its own longest member, decoded in a single ``snac_model.decode`` call,
    and every waveform is trimmed back to its real length
    (``n_frames * samples_per_frame``).

    Sorting keeps similar-length sequences together so the zero padding — whose
    only cost is a slight convolutional bleed into the final frames of the
    shorter members and some wasted compute — stays minimal.

    Returns a list aligned with ``token_ids_list``; an entry is ``None`` when
    that sequence had no decodable frames (see ``_parse_token_ids_to_frames``).
    """
    n = len(token_ids_list)
    results: list[bytes | None] = [None] * n
    if n == 0:
        return results

    parsed: list[tuple[int, np.ndarray]] = []
    for i, tids in enumerate(token_ids_list):
        frames = _parse_token_ids_to_frames(list(tids), audio_token_base_id)
        if frames is not None:
            parsed.append((i, frames))
    if not parsed:
        return results

    parsed.sort(key=lambda t: t[1].shape[0])
    bsz = max(1, int(max_batch_size))

    for start in range(0, len(parsed), bsz):
        chunk = parsed[start : start + bsz]
        max_frames = max(f.shape[0] for _, f in chunk)
        B = len(chunk)

        # (B, max_frames, 7) zero-padded; zeros are valid codes, trimmed off later.
        padded = np.zeros((B, max_frames, SNAC_NUM_CODEBOOKS), dtype=np.int32)
        n_frames = np.empty(B, dtype=np.int64)
        for b, (_, f) in enumerate(chunk):
            padded[b, : f.shape[0], :] = f
            n_frames[b] = f.shape[0]

        # Interleave the 7 per-frame positions back into the 3 codebook streams,
        # mirroring decode_audio's single-sequence layout (positions 1/4 -> c1,
        # positions 2/3/5/6 -> c2).
        c1 = np.empty((B, 2 * max_frames), dtype=np.int32)
        c1[:, 0::2] = padded[:, :, 1]
        c1[:, 1::2] = padded[:, :, 4]
        c2 = np.empty((B, 4 * max_frames), dtype=np.int32)
        c2[:, 0::4] = padded[:, :, 2]
        c2[:, 1::4] = padded[:, :, 3]
        c2[:, 2::4] = padded[:, :, 5]
        c2[:, 3::4] = padded[:, :, 6]

        codes = [
            torch.from_numpy(padded[:, :, 0].copy()).to(device, dtype=torch.int32),
            torch.from_numpy(c1).to(device, dtype=torch.int32),
            torch.from_numpy(c2).to(device, dtype=torch.int32),
        ]

        with (
            torch.inference_mode(),
            _autocast_context(device=device, autocast_dtype=autocast_dtype),
        ):
            audio_hat = snac_model.decode(codes)  # (B, 1, T)

        audio_np = audio_hat.detach().to(torch.float32).cpu().numpy()
        total_t = audio_np.shape[-1]
        samples_per_frame = total_t // max_frames if max_frames > 0 else total_t

        for b, (orig_idx, _) in enumerate(chunk):
            valid_len = int(n_frames[b]) * samples_per_frame if samples_per_frame > 0 else total_t
            wave = np.clip(audio_np[b, 0, :valid_len], -1.0, 1.0)
            results[orig_idx] = (wave * 32767).astype(np.int16).tobytes()

    return results


def decode_window_batch(
    decode_fn,
    window_codes: np.ndarray,
    device: str = "cuda",
    autocast_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Decode a batch of W-frame sliding windows; return frame index 1 per window
    (the emitted frame ``k`` — the window starts at ``k-1``, so it's always at
    index 1). W is inferred from the input width (W*7 codes), so this supports
    both the 4-frame (28-code) and 3-frame (21-code) windows.

    ``window_codes``: int array ``(B, W*7)`` of RAW codes in ``[0, 4096)``
    (offsets stripped, clamped, zero-padded at sequence edges). ``decode_fn`` is
    ``snac_model.decode`` (optionally CUDA-graph compiled). **B and W constant
    across calls** so one captured graph is reused.

    Returns int16 numpy ``(B, samples_per_frame)`` (2048 at 24 kHz).
    """
    wc = np.asarray(window_codes, dtype=np.int32)
    if wc.ndim != 2 or wc.shape[1] % SNAC_NUM_CODEBOOKS != 0:
        raise ValueError(f"window_codes must be (B, W*{SNAC_NUM_CODEBOOKS}); got {wc.shape}")
    B = wc.shape[0]
    W = wc.shape[1] // SNAC_NUM_CODEBOOKS
    if W < 2:
        raise ValueError(
            f"window must be >= 2 frames (got {W}); the kept frame needs a left neighbour"
        )
    frames = wc.reshape(B, W, SNAC_NUM_CODEBOOKS)

    codes_0 = frames[:, :, 0]
    c1 = np.empty((B, 2 * W), dtype=np.int32)
    c1[:, 0::2] = frames[:, :, 1]
    c1[:, 1::2] = frames[:, :, 4]
    c2 = np.empty((B, 4 * W), dtype=np.int32)
    c2[:, 0::4] = frames[:, :, 2]
    c2[:, 1::4] = frames[:, :, 3]
    c2[:, 2::4] = frames[:, :, 5]
    c2[:, 3::4] = frames[:, :, 6]

    codes = [
        torch.from_numpy(codes_0.copy()).to(device, dtype=torch.int32),
        torch.from_numpy(c1).to(device, dtype=torch.int32),
        torch.from_numpy(c2).to(device, dtype=torch.int32),
    ]
    with torch.inference_mode(), _autocast_context(device=device, autocast_dtype=autocast_dtype):
        audio_hat = decode_fn(codes)  # (B, 1, W*samples_per_frame)

    total_t = audio_hat.shape[-1]
    spf = total_t // W
    mid = audio_hat[:, :, spf : 2 * spf]  # keep frame k (index 1; window starts at k-1)
    a = mid.detach().to(torch.float32).cpu().numpy()[:, 0, :]
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767).astype(np.int16)


def tokens_to_audio_token_ids(
    raw_snac_codes: list[list[int]],
    audio_token_base_id: int,
    deduplicate: bool = True,
    device: str = "cpu",
) -> list[int]:
    """
    Convert already-computed SNAC codes (3 lists at different rates) to token IDs.

    This is an alternative to encode_audio when you already have SNAC codes
    (e.g., from a pre-encoded dataset) and just need to apply offsets.

    Args:
        raw_snac_codes: [c0, c1, c2] where c0 has length N, c1 has 2N, c2 has 4N.
        audio_token_base_id: Token ID of <|snac_0|>.
        deduplicate: Whether to remove consecutive duplicate frames.
        device: Device for tensor operations.

    Returns:
        List of offset token IDs.
    """
    c0 = torch.tensor(raw_snac_codes[0], dtype=torch.long, device=device)
    c1 = torch.tensor(raw_snac_codes[1], dtype=torch.long, device=device)
    c2 = torch.tensor(raw_snac_codes[2], dtype=torch.long, device=device)

    base = audio_token_base_id
    all_codes = torch.stack(
        [
            c0 + base + 0 * SNAC_CODEBOOK_SIZE,
            c1[::2] + base + 1 * SNAC_CODEBOOK_SIZE,
            c2[::4] + base + 2 * SNAC_CODEBOOK_SIZE,
            c2[1::4] + base + 3 * SNAC_CODEBOOK_SIZE,
            c1[1::2] + base + 4 * SNAC_CODEBOOK_SIZE,
            c2[2::4] + base + 5 * SNAC_CODEBOOK_SIZE,
            c2[3::4] + base + 6 * SNAC_CODEBOOK_SIZE,
        ],
        dim=1,
    ).reshape(-1)

    if deduplicate:
        all_codes = _remove_duplicate_frames(all_codes)

    return all_codes.numpy().astype(np.int64).tolist()
