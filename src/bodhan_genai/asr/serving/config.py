# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""ASR server configuration: a dataclass + argparse builder.

Knob names mirror the offline engine's where they mean the same thing, so an
operator moving between `asr.inference.transcribe` and the server has one
mental model — the same convention the TTS serving config follows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class ServeConfig:
    # --- model ---
    # Local directory or Hub repo id; resolved by asr.engine.checkpoints.resolve_ckpt.
    model_dir: str = ""  # "" -> the published default
    dtype: str = "bfloat16"  # WER-neutral vs fp32 (docs/asr/caveats.md)

    # --- replica topology ---
    num_replicas: int = -1  # -1 = auto: one per visible GPU
    gpus_per_replica: float = 1.0
    max_ongoing_requests: int = 32  # per-replica admission cap = concurrent
    # STREAMING sessions one GPU sustains in real time. Measured with the slot
    # pool AND encoder/decoder overlap: 32 hold real time (rtf 1.02, no drift),
    # 36 drops to 14% and 40 to 0%. It was 24 with the encoder running inline.
    # Past the ceiling sessions do NOT degrade one at a time -- they share the
    # GPU and all fall behind together, so admission control is the only
    # protection. See docs/asr/serving.md.
    max_queued_requests: int = 256

    # --- offline / long-form ---
    batch_size: int = 96  # measured throughput knee on one H100
    chunk_above: float = 45.0  # long-form threshold; below it, decode whole
    chunk_min: float = 15.0
    chunk_max: float = 25.0

    # --- streaming (WS /asr/stream) ---
    # Spans are cut at pauses and each is decoded ONCE (serving/streaming.py).
    # The previous LocalAgreement scheme re-decoded a growing buffer and cost
    # 4.43x the encoder work and ~4.3x the decoder tokens actually needed.
    #
    # Trailing pause that closes a span. 0.5 s is the production convention;
    # below ~0.35 s ordinary breath pauses close spans mid-phrase. NOTE: this
    # cannot be tuned on the current bench corpus, which is synthetic TTS output
    # with pause p99 of only 0.56 s -- it needs real conversational audio.
    stream_endpoint_silence_s: float = 0.5
    # Force-cut bound, and therefore the hard ceiling on time-to-first-text:
    # a pause-free stretch yields nothing until this elapses. This is the
    # latency SLO knob, so it is set to the SLO (5 s) rather than to whatever
    # maximises throughput.
    stream_max_segment_s: float = 5.0
    # Interim text: re-decode the open span this often. NOT free -- each partial
    # is a full re-decode, and measured against no partials at all it costs
    # roughly half the capacity gain (4.43x -> 2.17x at 2.0 s, -> 1.56x at
    # 1.0 s). 0 disables partials.
    stream_partial_interval_s: float = 2.0
    # Never emit a span shorter than this; an AED given a fraction of a second
    # of audio invents a word.
    stream_min_segment_s: float = 0.6
    # Continuous-batching slot pool (serving/slot_engine.py). Sessions are
    # admitted into free slots and stepped together under a CUDA graph; a slot
    # is evicted the moment its decode hits EOS. This replaced a fixed batch,
    # which could not be graphed without waiting to fill and paid max-length
    # for every row. Measured on one H100: 5.8-6.0 ms/step eager vs 2.5-2.7 ms
    # graphed (2.3x), nearly flat from 8 to 32 slots, plus ~1.28x from not
    # paying the longest row's length for everyone.
    #
    # Slots are cheap because graphed step cost barely moves with pool size;
    # what they cost is KV memory, which is bounded by the fixed window above.
    stream_slots: int = 32
    stream_admit_batch: int = 16  # pending requests encoded per admission pass
    stream_no_cuda_graphs: bool = False  # escape hatch if capture ever misbehaves
    # Encoder on a producer thread + side CUDA stream instead of inline in the
    # scheduler. The encoder is 42% of GPU time, and run inline it stalls every
    # queued decode behind it whenever sessions come due together — worth 1.33x
    # (24 -> 32 realtime sessions/GPU). Off only for debugging.
    stream_overlap_encode: bool = True
    # Sample rate the client is expected to send. Audio is accepted as raw
    # little-endian int16 PCM, mono — the same wire format the TTS server emits,
    # so a caller can pipe one into the other.
    stream_sample_rate: int = 16000

    # --- ingress ---
    host: str = "0.0.0.0"
    port: int = 8000


def add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model_dir",
        default=ServeConfig.model_dir,
        help="Converted IndicTranscribe checkpoint: local directory or Hub repo id. "
        "Empty uses the published default.",
    )
    p.add_argument("--dtype", default=ServeConfig.dtype, choices=["bfloat16", "float32"])
    p.add_argument(
        "--num_replicas",
        type=int,
        default=ServeConfig.num_replicas,
        help="-1 = auto-detect (one replica per visible GPU).",
    )
    p.add_argument("--gpus_per_replica", type=float, default=ServeConfig.gpus_per_replica)
    p.add_argument(
        "--max_ongoing_requests",
        type=int,
        default=ServeConfig.max_ongoing_requests,
        help="Per-replica concurrency cap = admission control.",
    )
    p.add_argument("--max_queued_requests", type=int, default=ServeConfig.max_queued_requests)
    p.add_argument("--batch_size", type=int, default=ServeConfig.batch_size)
    p.add_argument(
        "--chunk_above",
        type=float,
        default=ServeConfig.chunk_above,
        help="Long-form threshold in seconds; audio at or below it is decoded whole "
        "(chunking shorter audio measurably hurts).",
    )
    p.add_argument("--chunk_min", type=float, default=ServeConfig.chunk_min)
    p.add_argument("--chunk_max", type=float, default=ServeConfig.chunk_max)
    p.add_argument(
        "--stream_endpoint_silence_s",
        type=float,
        default=ServeConfig.stream_endpoint_silence_s,
        help="Streaming: trailing pause that closes a span and emits final text.",
    )
    p.add_argument(
        "--stream_max_segment_s",
        type=float,
        default=ServeConfig.stream_max_segment_s,
        help="Streaming: force-cut bound; the hard ceiling on time-to-first-text.",
    )
    p.add_argument(
        "--stream_partial_interval_s",
        type=float,
        default=ServeConfig.stream_partial_interval_s,
        help="Streaming: how often to re-decode the open span for interim text. "
        "Each partial is a full re-decode; 0 disables them (cheapest).",
    )
    p.add_argument("--stream_min_segment_s", type=float, default=ServeConfig.stream_min_segment_s)
    p.add_argument(
        "--stream_slots",
        type=int,
        default=ServeConfig.stream_slots,
        help="Continuous-batching pool size. Graphed step cost is nearly flat in "
        "slots, so this is close to free capacity until the GPU saturates.",
    )
    p.add_argument("--stream_admit_batch", type=int, default=ServeConfig.stream_admit_batch)
    p.add_argument(
        "--stream_no_cuda_graphs",
        action="store_true",
        help="Disable CUDA-graph capture of the decode step (2.3x slower; escape hatch).",
    )
    p.add_argument(
        "--stream_no_overlap_encode",
        action="store_true",
        help="Run the encoder inline in the scheduler instead of on a side stream "
        "(1.33x fewer realtime sessions; debugging only).",
    )
    p.add_argument("--stream_sample_rate", type=int, default=ServeConfig.stream_sample_rate)
    p.add_argument("--host", default=ServeConfig.host)
    p.add_argument("--port", type=int, default=ServeConfig.port)


def config_from_args(args: argparse.Namespace) -> ServeConfig:
    fields = ServeConfig.__dataclass_fields__
    cfg = ServeConfig(**{k: getattr(args, k) for k in fields if hasattr(args, k)})
    # the CLI exposes the negative (store_true) form of an on-by-default knob
    if getattr(args, "stream_no_overlap_encode", False):
        cfg.stream_overlap_encode = False
    return cfg
