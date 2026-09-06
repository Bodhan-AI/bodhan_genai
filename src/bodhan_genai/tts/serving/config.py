"""Server configuration: a dataclass + argparse builder (mirrors the engine-knob
names used in the offline vLLM eval path so operators have one mental model)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass
class ServeConfig:
    # --- model / tokenizer / codec ---
    checkpoint_path: str = "bodhan-ai/indic-speak"  # local path or HF id
    tokenizer_path: str = ""  # "" = use checkpoint_path
    snac_model_path: str = "hubertsiuzdak/snac_24khz"
    # --- replica / admission topology ---
    num_replicas: int = -1  # -1 = auto: torch.cuda.device_count(), resolved in
    # app.py. 1 LLM engine + 1 SNAC decoder per GPU.
    gpus_per_replica: float = 1.0  # Ray num_gpus per replica. 0.5 packs 2 replicas/GPU
    # (multi-tenancy under MPS): each replica's event loop
    # carries half the streams — the per-replica serving
    # ceiling (~2k tok/s through AsyncLLM+WS) binds before
    # the GPU does. Scale gpu_memory_utilization down to fit.
    max_ongoing_requests: int = 16  # per-replica admission cap (concurrency)
    max_queued_requests: int = 256  # router queue beyond which we 503/overload
    # --- vLLM engine knobs ---
    gpu_memory_utilization: float = 0.85  # leave headroom for in-process SNAC
    max_model_len: int = 8192
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096  # chunked-prefill chunk size (0 = vLLM default 2048).
    # Tuned @ 24/replica on Rasa: 4096 gave the best RTF p99
    # (1.27 vs 1.56@2048, 1.90@8192; 512 starves throughput).
    enable_chunked_prefill: bool = True
    quantization: str = ""  # "" = none (bf16 weights); "fp8" = online dynamic FP8
    # weights (compute stays bf16). Faster decode (HBM-bound),
    # but can shift audio-token logits -> check quality.
    kv_cache_dtype: str = "auto"  # "auto" = model dtype; "fp8" halves KV memory + KV-read
    # bandwidth (less preemption, faster decode); quality risk.
    dtype: str = "bfloat16"
    enforce_eager: bool = False
    async_scheduling: bool = False  # vLLM v1 async scheduling overlaps CPU scheduling with GPU
    # exec via a batch queue + async output copy. Default OFF:
    # it caused a sporadic CUDA illegal-memory-access that killed
    # an EngineCore ~6 min into a sustained 58k run (fault in
    # step_with_batch_queue/async_copy_ready_event.synchronize).
    # The offline path uses the sync LLMEngine and never hit it.
    enable_prefix_caching: bool = False
    seed: int = 0
    # --- SNAC decode knobs ---
    snac_cudagraph_batch: int = 32
    snac_flush_interval_ms: float = 4.0
    snac_compile_mode: str = "reduce-overhead"
    # EXPERIMENTAL topology: "colocated" (default; 1 LLM + 1 SNAC per GPU) or
    # "pooled" (LLM-only replicas on N-1 GPUs; num_snac_actors SNAC actors packed
    # on the remaining GPU — run the node under CUDA MPS so they overlap). With
    # "pooled" set --num_replicas to GPUs-1 (e.g. 7 on an 8-GPU node).
    snac_topology: str = "colocated"
    num_snac_actors: int = 3
    snac_pool_gpu_fraction: float = 0.33
    snac_window_frames: int = 3  # Orpheus sliding-window size; emit the middle frame.
    # 3 (tuned default): ~16% better RTF p99 + ~20% better
    # TTFP p99 vs 4, perceptually identical (seam delta below
    # SNAC's noise floor). 4 = max right-context if ever needed.
    # --- default sampling (per-request can override) ---
    # NOTE: these are production-tuned for the streaming server; the library
    # default (bodhan_genai.tts.engine.types.SamplingConfig) is 0.6 / 0.95 /
    # repetition_penalty 1.1.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = -1
    repetition_penalty: float = 1.0  # 1.0 = disabled — matches the server's historical
    # behavior (the engine now always passes one).
    max_new_tokens: int = 2048
    # --- streaming / backpressure ---
    per_request_queue_max: int = 64  # buffered frames before backpressure
    frames_per_message: int = 2  # group N frames per streamed message (1 frame=85ms);
    # first frame sent alone for low TTFB, then grouped.
    record_dir: str = ""  # if set, full-utterance WAVs are written here by a
    # BACKGROUND process pool — off the streaming hot path,
    # never counted in latency.
    # --- long-form chunked synthesis (ChunkedIndicStreamingTTS) ---
    # Per-request {"chunked": true} opts in; chunked_default flips the server
    # default. Knob semantics mirror engine.chunked / engine.loudness.
    chunked_default: bool = False
    chunk_min_chars: int = 16
    chunk_max_chars: int = 300
    chunk_first_chars: int = 120  # ramp: small chunk 0 = low TTFA; 0 disables
    chunk_gap_ms: float = 250.0
    chunk_target_lufs: float = -23.0
    chunk_peak_dbfs: float = -1.0
    chunk_trim_db: float = 30.0
    # --- ingress ---
    host: str = "0.0.0.0"
    port: int = 8000

    def engine_kwargs(self) -> dict[str, Any]:
        """Kwargs for ``vllm.AsyncEngineArgs`` (one engine per replica, gmu<1 to
        co-locate SNAC). tensor_parallel_size stays 1 (data-parallel across GPUs
        via Serve replicas)."""
        kw = dict(
            model=self.checkpoint_path,
            tokenizer=self.tokenizer_path or self.checkpoint_path,
            dtype=self.dtype,
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            max_model_len=int(self.max_model_len),
            max_num_seqs=int(self.max_num_seqs),
            enable_chunked_prefill=bool(self.enable_chunked_prefill),
            enforce_eager=bool(self.enforce_eager),
            async_scheduling=bool(self.async_scheduling),
            enable_prefix_caching=bool(self.enable_prefix_caching),
            disable_log_stats=True,
            trust_remote_code=True,
            seed=int(self.seed),
        )
        if int(self.max_num_batched_tokens) > 0:
            kw["max_num_batched_tokens"] = int(self.max_num_batched_tokens)
        if self.quantization:
            kw["quantization"] = self.quantization
        if self.kv_cache_dtype and self.kv_cache_dtype != "auto":
            kw["kv_cache_dtype"] = self.kv_cache_dtype
        return kw


def add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--checkpoint_path",
        default=ServeConfig.checkpoint_path,
        help="Model checkpoint: local path or HF id.",
    )
    p.add_argument(
        "--tokenizer_path",
        default=ServeConfig.tokenizer_path,
        help="Tokenizer path; '' = use --checkpoint_path.",
    )
    p.add_argument("--snac_model_path", default=ServeConfig.snac_model_path)
    p.add_argument(
        "--num_replicas",
        type=int,
        default=ServeConfig.num_replicas,
        help="-1 = auto-detect (one replica per visible GPU).",
    )
    p.add_argument(
        "--gpus_per_replica",
        type=float,
        default=ServeConfig.gpus_per_replica,
        help="Ray num_gpus per replica; 0.5 = 2 replicas/GPU multi-tenancy (use with MPS).",
    )
    p.add_argument(
        "--max_ongoing_requests",
        type=int,
        default=ServeConfig.max_ongoing_requests,
        help="Per-replica concurrency cap = admission control.",
    )
    p.add_argument("--max_queued_requests", type=int, default=ServeConfig.max_queued_requests)
    p.add_argument(
        "--gpu_memory_utilization", type=float, default=ServeConfig.gpu_memory_utilization
    )
    p.add_argument("--max_model_len", type=int, default=ServeConfig.max_model_len)
    p.add_argument("--max_num_seqs", type=int, default=ServeConfig.max_num_seqs)
    p.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=ServeConfig.max_num_batched_tokens,
        help="Chunked-prefill chunk size (0=vLLM default). Smaller reduces decode stalls.",
    )
    p.add_argument("--dtype", default=ServeConfig.dtype, choices=["bfloat16", "float16", "auto"])
    p.add_argument(
        "--quantization",
        default=ServeConfig.quantization,
        help="'' = bf16 weights; 'fp8' = online dynamic FP8 (faster decode, check audio quality).",
    )
    p.add_argument(
        "--kv_cache_dtype",
        default=ServeConfig.kv_cache_dtype,
        help="'auto' = model dtype KV; 'fp8' halves KV mem+bandwidth (check audio quality).",
    )
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument(
        "--async_scheduling",
        action="store_true",
        help="Enable vLLM async scheduling (OFF by default; it caused a CUDA IMA that "
        "killed an EngineCore mid-run). Leave off for stability.",
    )
    p.add_argument("--enable_prefix_caching", action="store_true")
    p.add_argument("--seed", type=int, default=ServeConfig.seed)
    p.add_argument("--snac_cudagraph_batch", type=int, default=ServeConfig.snac_cudagraph_batch)
    p.add_argument(
        "--snac_window_frames",
        type=int,
        default=ServeConfig.snac_window_frames,
        help="Sliding-window frames (4=seamless; 3=lower latency, less right-context).",
    )
    p.add_argument(
        "--snac_flush_interval_ms", type=float, default=ServeConfig.snac_flush_interval_ms
    )
    p.add_argument(
        "--snac_topology",
        default=ServeConfig.snac_topology,
        choices=["colocated", "pooled"],
        help="EXPERIMENTAL: 'pooled' = LLM-only replicas + SNAC actor pool on a dedicated GPU.",
    )
    p.add_argument("--num_snac_actors", type=int, default=ServeConfig.num_snac_actors)
    p.add_argument(
        "--snac_pool_gpu_fraction", type=float, default=ServeConfig.snac_pool_gpu_fraction
    )
    p.add_argument(
        "--snac_compile_mode",
        default=ServeConfig.snac_compile_mode,
        help="torch.compile mode for snac.decode ('reduce-overhead'=CUDA graphs; '' disables).",
    )
    p.add_argument("--temperature", type=float, default=ServeConfig.temperature)
    p.add_argument("--top_p", type=float, default=ServeConfig.top_p)
    p.add_argument("--top_k", type=int, default=ServeConfig.top_k)
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=ServeConfig.repetition_penalty,
        help="Default repetition penalty (1.0 = disabled, the historical server behavior).",
    )
    p.add_argument("--max_new_tokens", type=int, default=ServeConfig.max_new_tokens)
    p.add_argument("--per_request_queue_max", type=int, default=ServeConfig.per_request_queue_max)
    p.add_argument("--frames_per_message", type=int, default=ServeConfig.frames_per_message)
    p.add_argument(
        "--record_dir",
        default=ServeConfig.record_dir,
        help="If set, write full-utterance WAVs here via a background process pool "
        "(off the streaming hot path). Empty = no recording.",
    )
    p.add_argument(
        "--chunked_default",
        action="store_true",
        default=ServeConfig.chunked_default,
        help="Serve long-form chunked synthesis by default (requests can still "
        'opt out with {"chunked": false}).',
    )
    p.add_argument("--chunk_min_chars", type=int, default=ServeConfig.chunk_min_chars)
    p.add_argument("--chunk_max_chars", type=int, default=ServeConfig.chunk_max_chars)
    p.add_argument("--chunk_first_chars", type=int, default=ServeConfig.chunk_first_chars)
    p.add_argument("--chunk_gap_ms", type=float, default=ServeConfig.chunk_gap_ms)
    p.add_argument("--chunk_target_lufs", type=float, default=ServeConfig.chunk_target_lufs)
    p.add_argument("--chunk_peak_dbfs", type=float, default=ServeConfig.chunk_peak_dbfs)
    p.add_argument("--chunk_trim_db", type=float, default=ServeConfig.chunk_trim_db)
    p.add_argument("--host", default=ServeConfig.host)
    p.add_argument("--port", type=int, default=ServeConfig.port)


def config_from_args(args: argparse.Namespace) -> ServeConfig:
    fields = ServeConfig.__dataclass_fields__
    return ServeConfig(**{k: getattr(args, k) for k in fields if hasattr(args, k)})
