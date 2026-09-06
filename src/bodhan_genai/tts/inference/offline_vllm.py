"""Offline batch TTS generation using vLLM offline inference + batched SNAC decode.

Two-phase variant ("generate everything, then decode everything"):

  1. ``num_llm_workers`` Ray actors, one vLLM engine per GPU, each consume a
     round-robin shard of prompts. vLLM's own continuous batching overlaps
     prompts of different lengths inside each engine and maximizes throughput.
  2. Once all tokens are generated, the completed token streams are sharded to
     ``num_snac_workers`` SNAC actors (sharing the dedicated SNAC GPU) which run
     ``batch_decode_audio`` — one batched ``snac.decode`` per microbatch — and
     write 24 kHz WAVs to disk.

Output: ``<output_dir>/audio/<basename>.wav`` (falls back to ``NNNN_<sha8>.wav``)
plus ``<output_dir>/manifest.jsonl`` (base ``AudioRow`` columns). Resume-aware:
rows whose target WAV already exists are skipped unless ``--force-regenerate``.

Defaults can come from a YAML file via ``--config configs/tts/infer/offline_vllm.yaml``;
explicit CLI flags always override the config values.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from bodhan_genai.tts.engine.types import SamplingConfig
from bodhan_genai.tts.inference.audio_io import (
    SNAC_SAMPLE_RATE,
    AudioRow,
    audio_filename,
    write_manifest_atomic,
    write_wav_24k,
)
from bodhan_genai.tts.inference.prompts import load_prompts_jsonl, resolve_snac_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SAMPLING_DEFAULTS = SamplingConfig()


# ---------------------------------------------------------------------------
# vLLM engine / sampling config (shared with the streaming variant)
# ---------------------------------------------------------------------------


def vllm_engine_kwargs(
    args: argparse.Namespace, checkpoint_path: str, tokenizer_path: str
) -> dict[str, Any]:
    """Build the kwarg dict accepted by both ``vllm.LLM`` and ``vllm.EngineArgs``.

    Each engine lives inside a Ray actor pinned to a single GPU, so
    ``tensor_parallel_size`` stays 1 and the engine runs in-process (the actor
    sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0``) — that keeps vLLM on the GPU Ray
    assigned via CUDA_VISIBLE_DEVICES instead of spawning its own placement.
    """
    return dict(
        model=checkpoint_path,
        tokenizer=tokenizer_path,
        dtype=str(args.dtype),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
        enforce_eager=bool(args.enforce_eager),
        enable_prefix_caching=bool(args.enable_prefix_caching),
        disable_log_stats=not bool(args.benchmark),
        trust_remote_code=True,
        seed=int(args.seed),
    )


# ---------------------------------------------------------------------------
# Ray actors
# ---------------------------------------------------------------------------


def _make_llm_worker_cls():
    """GPU share is set by the caller via ``.options(num_gpus=...)`` so 1-GPU
    boxes can split the card between the vLLM engine and the SNAC decoders
    (e.g. 0.8 LLM + 0.2 SNAC)."""
    import ray

    @ray.remote
    class VLLMWorker:
        """One vLLM engine per GPU. ``process`` generates every token for its
        shard via vLLM's high-level ``LLM.generate`` (continuous batching)."""

        def __init__(
            self,
            checkpoint_path: str,
            tokenizer_path: str,
            engine_kwargs: dict[str, Any],
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
            repetition_penalty: float = 1.0,
            benchmark: bool = False,
        ):
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams, TokensPrompt

            self._benchmark = bool(benchmark)
            self._TokensPrompt = TokensPrompt

            tok = AutoTokenizer.from_pretrained(tokenizer_path)
            self._ids = resolve_snac_ids(tok)
            del tok

            logger.info(
                "[VLLMWorker] Loading vLLM engine: %s (tokenizer %s)",
                checkpoint_path,
                tokenizer_path,
            )
            self._llm = LLM(**engine_kwargs)
            self._sp = SamplingParams(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                max_tokens=int(max_new_tokens),
                stop_token_ids=[self._ids["end_of_audio_id"], self._ids["eos_token_id"]],
                detokenize=False,
            )

        def ids_for_snac(self) -> dict[str, int]:
            return {
                "start_of_audio_id": self._ids["start_of_audio_id"],
                "end_of_audio_id": self._ids["end_of_audio_id"],
                "audio_token_base_id": self._ids["audio_token_base_id"],
            }

        def process(self, items: list[tuple[int, list[int]]]) -> list[dict[str, Any]]:
            if not items:
                return []
            prompts = [self._TokensPrompt(prompt_token_ids=list(ids)) for _, ids in items]
            t0 = time.time()
            outs = self._llm.generate(prompts, self._sp, use_tqdm=False)
            shard_wall = time.time() - t0
            results: list[dict[str, Any]] = []
            for (row_idx, ids), out in zip(items, outs, strict=False):
                gen = list(out.outputs[0].token_ids)
                row: dict[str, Any] = {
                    "row_idx": int(row_idx),
                    "generated_tokens": gen,
                    "error": None,
                }
                if self._benchmark:
                    row["bench"] = {
                        "prompt_len": len(ids),
                        "n_gen_tokens": len(gen),
                        "shard_wall_sec": float(shard_wall),
                    }
                results.append(row)
            return results

        def shutdown(self) -> None:
            return None

    return VLLMWorker


def make_snac_worker_cls():
    """Batched SNAC decode + WAV writer. Instantiate with
    ``.options(num_gpus=...)`` so callers control GPU sharing (the two-phase
    driver may run several sharing the dedicated GPU; the streaming driver runs
    one that owns it)."""
    import ray

    @ray.remote
    class SNACBatchWorker:
        def __init__(
            self,
            snac_model_path: str,
            audio_token_base_id: int,
            start_of_audio_id: int,
            end_of_audio_id: int,
            output_dir: str,
            decode_batch_size: int = 32,
            vocos: str = "true",
        ):
            import torch

            from bodhan_genai.tts.codec.snac import (
                SNAC_NUM_CODEBOOKS,
                batch_decode_audio,
                load_snac_model,
            )

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "[SNACBatchWorker] Loading SNAC from %s on %s", snac_model_path, self._device
            )
            self._snac = load_snac_model(snac_model_path, device=self._device)
            from bodhan_genai.tts.codec.vocos import resolve_decoder

            self._snac = resolve_decoder(self._snac, str(vocos), device=self._device)
            self._batch_decode = batch_decode_audio
            self._num_codebooks = SNAC_NUM_CODEBOOKS
            self._base_id = int(audio_token_base_id)
            self._start_id = int(start_of_audio_id)
            self._end_id = int(end_of_audio_id)
            self._output_dir = Path(output_dir)
            self._decode_batch_size = max(1, int(decode_batch_size))
            self._output_dir.mkdir(parents=True, exist_ok=True)

            try:
                warm = [self._base_id] * (self._num_codebooks * 4)
                _ = self._batch_decode(
                    self._snac,
                    [warm],
                    self._base_id,
                    device=self._device,
                    max_batch_size=self._decode_batch_size,
                )
            except Exception as e:
                logger.warning("[SNACBatchWorker] warmup failed (non-fatal): %s", e)

        def decode_and_save_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Each item: ``{row_idx, generated_tokens, target_path, error}``.
            Returns ``{row_idx, ok, error, gen_duration_sec, gen_wall_sec}``."""
            from bodhan_genai.tts.inference.prompts import extract_audio_tokens

            out: list[dict[str, Any]] = []
            token_lists: list[list[int]] = []
            meta: list[tuple[int, str]] = []

            for item in items:
                row_idx = int(item["row_idx"])
                gen_tokens = list(item.get("generated_tokens") or [])
                target_path = item["target_path"]
                if not gen_tokens:
                    out.append(
                        {
                            "row_idx": row_idx,
                            "ok": False,
                            "error": item.get("error") or "empty_generation",
                            "gen_duration_sec": 0.0,
                            "gen_wall_sec": 0.0,
                        }
                    )
                    continue
                tok_ids = extract_audio_tokens(gen_tokens, self._start_id, self._end_id)
                if len(tok_ids) < self._num_codebooks:
                    out.append(
                        {
                            "row_idx": row_idx,
                            "ok": False,
                            "error": "no_audio_tokens",
                            "gen_duration_sec": 0.0,
                            "gen_wall_sec": 0.0,
                        }
                    )
                    continue
                token_lists.append(tok_ids)
                meta.append((row_idx, target_path))

            if token_lists:
                t0 = time.time()
                audios = self._batch_decode(
                    self._snac,
                    token_lists,
                    self._base_id,
                    device=self._device,
                    max_batch_size=self._decode_batch_size,
                )
                per_row_wall = (time.time() - t0) / len(token_lists)
                for (row_idx, target_path), audio_bytes in zip(meta, audios, strict=False):
                    if audio_bytes is None:
                        out.append(
                            {
                                "row_idx": row_idx,
                                "ok": False,
                                "error": "decode_failed",
                                "gen_duration_sec": 0.0,
                                "gen_wall_sec": per_row_wall,
                            }
                        )
                        continue
                    try:
                        audio = (
                            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                        )
                        write_wav_24k(self._output_dir / target_path, audio)
                        out.append(
                            {
                                "row_idx": row_idx,
                                "ok": True,
                                "error": None,
                                "gen_duration_sec": float(len(audio)) / SNAC_SAMPLE_RATE,
                                "gen_wall_sec": per_row_wall,
                            }
                        )
                    except Exception as e:
                        out.append(
                            {
                                "row_idx": row_idx,
                                "ok": False,
                                "error": str(e),
                                "gen_duration_sec": 0.0,
                                "gen_wall_sec": per_row_wall,
                            }
                        )
            return out

    return SNACBatchWorker


# ---------------------------------------------------------------------------
# Sharding / Ray / worker helpers (absorbed from eval/generate_audios.py)
# ---------------------------------------------------------------------------


def _shard_round_robin(items: list, n_buckets: int) -> list[list]:
    buckets: list[list] = [[] for _ in range(n_buckets)]
    for i, x in enumerate(items):
        buckets[i % n_buckets].append(x)
    return buckets


def _build_prompts(jsonl_path: str, tokenizer_path: str, max_rows: int | None) -> list[dict]:
    """Use the shared prompt builder in bodhan_genai.tts.inference.prompts so the
    prompts match exactly what the training pipeline produced."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    items = load_prompts_jsonl(jsonl_path, tokenizer=tokenizer, max_rows=max_rows)
    del tokenizer
    return items


def _validate_positive_worker_count(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value})")
    return value


def _safe_ray_get(ray_module, refs, *, timeout_sec: float, stage_name: str):
    try:
        if timeout_sec and timeout_sec > 0:
            return ray_module.get(refs, timeout=float(timeout_sec))
        return ray_module.get(refs)
    except ray_module.exceptions.GetTimeoutError as e:
        raise RuntimeError(f"{stage_name} timed out after {float(timeout_sec):.1f}s") from e


def _cleanup_workers(ray_module, llm_workers: list, snac_workers: list) -> None:
    for worker in llm_workers:
        with contextlib.suppress(Exception):
            ray_module.get(worker.shutdown.remote())
    for worker in llm_workers + snac_workers:
        with contextlib.suppress(Exception):
            ray_module.kill(worker)


def _mark_failed_rows(
    rows: list[dict[str, Any]],
    gen_results: dict[int, dict[str, Any]],
    error: str,
) -> None:
    for row in rows:
        rid = int(row["row_idx"])
        if rid in gen_results:
            continue
        gen_results[rid] = {
            "row_idx": rid,
            "ok": False,
            "error": error,
            "gen_duration_sec": 0.0,
            "gen_wall_sec": 0.0,
        }


def _write_final_manifest(
    output_dir: Path,
    plan: list[dict],
    gen_results: dict[int, dict[str, Any]],
) -> None:
    """Build ``manifest.jsonl`` from plan + gen results, atomically."""
    rows: list[AudioRow] = []
    for p in plan:
        rid = int(p["row_idx"])
        target_rel = p["target_rel"]
        if p["skip"]:
            # Already on disk — manifest marks it ok=True so Phase B picks it up.
            rows.append(
                AudioRow(
                    row_idx=rid,
                    audio_filepath=p["audio_filepath"],
                    text=p["text"],
                    language=p["language"],
                    speaker_id=p["speaker_id"],
                    gen_audio_path=target_rel,
                    ok=True,
                    error=None,
                )
            )
            continue
        r = gen_results.get(rid)
        if r is None:
            rows.append(
                AudioRow(
                    row_idx=rid,
                    audio_filepath=p["audio_filepath"],
                    text=p["text"],
                    language=p["language"],
                    speaker_id=p["speaker_id"],
                    gen_audio_path="",
                    ok=False,
                    error="no_llm_result",
                )
            )
            continue
        rows.append(
            AudioRow(
                row_idx=rid,
                audio_filepath=p["audio_filepath"],
                text=p["text"],
                language=p["language"],
                speaker_id=p["speaker_id"],
                gen_audio_path=target_rel if r["ok"] else "",
                ok=bool(r["ok"]),
                error=r.get("error"),
                gen_duration_sec=float(r.get("gen_duration_sec", 0.0)),
                gen_wall_sec=float(r.get("gen_wall_sec", 0.0)),
            )
        )
    rows.sort(key=lambda x: x.row_idx)
    manifest_path = output_dir / "manifest.jsonl"
    write_manifest_atomic(manifest_path, rows)
    logger.info("Manifest written: %s (%d rows)", manifest_path, len(rows))


# ---------------------------------------------------------------------------
# Shared plan / manifest helpers
# ---------------------------------------------------------------------------


def dedup_items(items: list[dict]) -> tuple[list[dict], int]:
    """Drop duplicate rows, keeping the first occurrence. Identity is the
    ``audio_filepath`` (falls back to text+language+speaker when absent). Big
    eval manifests (e.g. Rasa) repeat each sample several times — deduping here
    avoids regenerating + rescoring identical rows."""
    seen: set = set()
    out: list[dict] = []
    dropped = 0
    for it in items:
        afp = (it.get("audio_filepath") or "").strip()
        key = afp or (it.get("text", ""), it.get("language", ""), it.get("speaker_id", ""))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(it)
    return out, dropped


def _output_basename(audio_filepath: str, row_idx: int, text: str) -> str:
    """Output WAV name = the input audio's basename (``foo.wav``). Falls back to
    the deterministic ``NNNN_<sha8>.wav`` scheme when ``audio_filepath`` is empty."""
    afp = (audio_filepath or "").strip()
    if afp:
        stem = Path(afp).stem
        if stem:
            return stem + ".wav"
    return audio_filename(row_idx, audio_filepath, text)


def build_plan(
    items: list[dict],
    output_dir: Path,
    force_regenerate: bool,
    split_by_language: bool = False,
) -> list[dict]:
    """Per-row metadata: target WAV path + resume skip flag.

    The WAV is named after the input ``audio_filepath`` basename. With
    ``split_by_language`` it lands under ``audio/<language>/`` instead of a flat
    ``audio/`` dir (subdir created on first write). The relative path is recorded
    in the manifest's ``gen_audio_path``, so Phase B is unaffected.

    Distinct rows that would collide on the same target path (different text, same
    basename) are disambiguated with a ``__<row_idx>`` suffix and a warning, so no
    sample is silently overwritten.
    """
    plan: list[dict] = []
    skipped = 0
    taken: dict[str, str] = {}  # target_rel -> text that owns it
    for it in items:
        rid = int(it["_row_idx"])
        fname = _output_basename(it["audio_filepath"], rid, it["text"])
        if split_by_language:
            lang = (str(it.get("language") or "unknown").strip() or "unknown").replace("/", "_")
            prefix = f"audio/{lang}"
        else:
            prefix = "audio"
        rel = f"{prefix}/{fname}"
        owner = taken.get(rel)
        if owner is not None and owner != it["text"]:
            stem = Path(fname).stem
            rel = f"{prefix}/{stem}__{rid}.wav"
            logger.warning(
                "basename collision on %s/%s with differing text; writing %s", prefix, fname, rel
            )
        taken[rel] = it["text"]
        already = (output_dir / rel).exists()
        plan.append(
            {
                "row_idx": rid,
                "audio_filepath": it["audio_filepath"],
                "text": it["text"],
                "language": it["language"],
                "speaker_id": it["speaker_id"],
                "input_ids": it["input_ids"],
                "target_rel": rel,
                "skip": (already and not force_regenerate),
            }
        )
        if already and not force_regenerate:
            skipped += 1
    logger.info("Resume: %d / %d rows already have audio (skipping)", skipped, len(plan))
    return plan


def write_manifests(
    output_dir: Path,
    plan: list[dict],
    gen_results: dict[int, dict[str, Any]],
    split_by_language: bool = False,
) -> None:
    """Write the combined ``manifest.jsonl`` (always), and when
    ``split_by_language`` also a per-language ``manifest_<lang>.jsonl``.

    The per-language files are produced by reading the combined manifest back and
    filtering by ``language`` — so they stay byte-for-byte consistent with it and
    Phase B can run on the whole set or one language (``gen_audio_path`` stays
    relative to ``output_dir`` either way)."""
    _write_final_manifest(output_dir, plan, gen_results)
    if not split_by_language:
        return
    from bodhan_genai.tts.inference.audio_io import read_manifest

    rows = read_manifest(output_dir / "manifest.jsonl")
    by_lang: dict[str, list] = {}
    for r in rows:
        lang = (str(r.language or "unknown").strip() or "unknown").replace("/", "_")
        by_lang.setdefault(lang, []).append(r)
    for lang, lrows in sorted(by_lang.items()):
        write_manifest_atomic(output_dir / f"manifest_{lang}.jsonl", lrows)
    logger.info("Wrote %d per-language manifests (manifest_<lang>.jsonl)", len(by_lang))


def write_benchmark_summary(
    output_dir: Path,
    llm_results: list[dict[str, Any]],
    llm_wall_sec: float,
    num_workers: int,
    mode: str,
) -> None:
    import json

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, round(p / 100.0 * (len(s) - 1))))
        return float(s[idx])

    def agg_int(xs: list[int]) -> dict[str, float]:
        fxs = [float(x) for x in xs]
        return {
            "mean": (sum(fxs) / len(fxs)) if fxs else 0.0,
            "p50": pct(fxs, 50),
            "p90": pct(fxs, 90),
            "p99": pct(fxs, 99),
            "min": float(min(xs)) if xs else 0.0,
            "max": float(max(xs)) if xs else 0.0,
        }

    prompt_lens: list[int] = []
    n_gens: list[int] = []
    errors = 0
    for row in llm_results:
        if row.get("error"):
            errors += 1
        b = row.get("bench")
        if not b:
            continue
        prompt_lens.append(int(b["prompt_len"]))
        n_gens.append(int(b["n_gen_tokens"]))

    total_gen = sum(n_gens)
    n_req = len(prompt_lens)
    summary = {
        "mode": mode,
        "n_requests_timed": n_req,
        "n_errors": errors,
        "num_llm_workers": int(num_workers),
        "llm_stage_wall_sec": float(llm_wall_sec),
        "requests_per_sec": (n_req / llm_wall_sec) if llm_wall_sec > 0 else 0.0,
        "tokens_per_sec_aggregate": (total_gen / llm_wall_sec) if llm_wall_sec > 0 else 0.0,
        "prompt_len": agg_int(prompt_lens),
        "n_gen_tokens": agg_int(n_gens),
    }
    out_path = output_dir / "benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Benchmark summary written: %s", out_path)
    logger.info(
        "  req/s=%.2f  tok/s(agg)=%.1f  n_gen p50/p99=%.0f/%.0f",
        summary["requests_per_sec"],
        summary["tokens_per_sec_aggregate"],
        summary["n_gen_tokens"]["p50"],
        summary["n_gen_tokens"]["p99"],
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)

    if int(args.num_llm_workers) <= 0:
        # Auto topology: all GPUs but one for vLLM engines, the last for SNAC.
        gpu_count = int(ray.cluster_resources().get("GPU", 0))
        args.num_llm_workers = max(1, gpu_count - 1)
        logger.info(
            "[run] auto num_llm_workers=%d (cluster GPUs=%d)", args.num_llm_workers, gpu_count
        )
    args.num_llm_workers = _validate_positive_worker_count(
        "--num_llm_workers", args.num_llm_workers
    )
    args.num_snac_workers = _validate_positive_worker_count(
        "--num_snac_workers", args.num_snac_workers
    )
    logger.info(
        "vLLM two-phase policy: %d engines, dtype=%s, max_model_len=%d, max_num_seqs=%d, "
        "enforce_eager=%s, prefix_caching=%s, temperature=%.2f",
        args.num_llm_workers,
        args.dtype,
        args.max_model_len,
        args.max_num_seqs,
        args.enforce_eager,
        args.enable_prefix_caching,
        args.temperature,
    )

    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    items = _build_prompts(
        args.jsonl_path,
        tokenizer_path,
        args.max_rows if args.max_rows and args.max_rows > 0 else None,
    )
    if not items:
        logger.error("No usable rows in %s", args.jsonl_path)
        return 1
    if args.dedup:
        items, dropped = dedup_items(items)
        logger.info("Dedup: dropped %d duplicate rows, %d unique remain", dropped, len(items))
    logger.info(
        "Loaded %d prompts; %d LLM x %d SNAC workers",
        len(items),
        args.num_llm_workers,
        args.num_snac_workers,
    )

    output_dir = Path(args.output_dir).resolve()
    (output_dir / "audio").mkdir(parents=True, exist_ok=True)

    plan = build_plan(items, output_dir, args.force_regenerate, args.split_by_language)
    to_generate = [p for p in plan if not p["skip"]]
    if not to_generate:
        logger.info("All rows already generated; writing manifest only.")
        write_manifests(output_dir, plan, gen_results={}, split_by_language=args.split_by_language)
        return 0

    engine_kwargs = vllm_engine_kwargs(args, args.checkpoint_path, tokenizer_path)

    llm_workers: list[Any] = []
    snac_workers: list[Any] = []
    llm_results: list[dict[str, Any]] = []
    snac_results: list[dict[str, Any]] = []
    stage_name = "setup"
    failure_error: str | None = None

    try:
        stage_name = "spawn_llm_workers"
        VLLMWorker = _make_llm_worker_cls()
        llm_gpus = float(args.llm_gpu_fraction)
        llm_workers = [
            VLLMWorker.options(num_gpus=llm_gpus).remote(
                checkpoint_path=args.checkpoint_path,
                tokenizer_path=tokenizer_path,
                engine_kwargs=engine_kwargs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                benchmark=args.benchmark,
            )
            for _ in range(args.num_llm_workers)
        ]
        snac_ids = _safe_ray_get(
            ray,
            llm_workers[0].ids_for_snac.remote(),
            timeout_sec=float(args.ray_timeout_sec),
            stage_name="SNAC token-id bootstrap",
        )

        stage_name = "spawn_snac_workers"
        SNACWorker = make_snac_worker_cls()
        snac_gpus = float(args.snac_gpu_fraction)
        snac_workers = [
            SNACWorker.options(num_gpus=snac_gpus).remote(
                snac_model_path=args.snac_model_path,
                audio_token_base_id=snac_ids["audio_token_base_id"],
                start_of_audio_id=snac_ids["start_of_audio_id"],
                end_of_audio_id=snac_ids["end_of_audio_id"],
                output_dir=str(output_dir),
                decode_batch_size=args.snac_decode_batch_size,
                vocos=str(args.vocos),
            )
            for _ in range(args.num_snac_workers)
        ]

        # --- Phase 1: generate all tokens ---
        stage_name = "llm"
        t0 = time.time()
        llm_shards = _shard_round_robin(
            [(p["row_idx"], p["input_ids"]) for p in to_generate],
            args.num_llm_workers,
        )
        llm_futures = [llm_workers[i].process.remote(shard) for i, shard in enumerate(llm_shards)]
        for shard_out in _safe_ray_get(
            ray,
            llm_futures,
            timeout_sec=float(args.ray_timeout_sec),
            stage_name="LLM stage",
        ):
            llm_results.extend(shard_out)
        llm_wall = time.time() - t0
        logger.info(
            "LLM stage: %d / %d rows in %.1fs (avg %.2f s/row)",
            len(llm_results),
            len(to_generate),
            llm_wall,
            llm_wall / max(1, len(to_generate)),
        )
        if args.benchmark:
            try:
                write_benchmark_summary(
                    output_dir, llm_results, llm_wall, args.num_llm_workers, "vllm_two_phase"
                )
            except Exception as e:
                logger.warning("benchmark summary failed (non-fatal): %s", e, exc_info=True)

        # --- Phase 2: batched SNAC decode + save ---
        stage_name = "snac"
        t1 = time.time()
        plan_by_rid = {p["row_idx"]: p for p in plan}
        snac_items = [
            {
                "row_idx": int(r["row_idx"]),
                "generated_tokens": r.get("generated_tokens") or [],
                "target_path": plan_by_rid[int(r["row_idx"])]["target_rel"],
                "error": r.get("error"),
            }
            for r in llm_results
        ]
        snac_shards = _shard_round_robin(snac_items, args.num_snac_workers)
        snac_futures = [
            snac_workers[i].decode_and_save_batch.remote(shard)
            for i, shard in enumerate(snac_shards)
        ]
        for shard_out in _safe_ray_get(
            ray,
            snac_futures,
            timeout_sec=float(args.ray_timeout_sec),
            stage_name="SNAC stage",
        ):
            snac_results.extend(shard_out)
        logger.info(
            "SNAC stage: %d rows in %.1fs (%d ok, %d err)",
            len(snac_results),
            time.time() - t1,
            sum(1 for r in snac_results if r["ok"]),
            sum(1 for r in snac_results if not r["ok"]),
        )
    except Exception as e:
        failure_error = str(e)
        logger.exception("Phase A failed during %s", stage_name)
    finally:
        _cleanup_workers(ray, llm_workers, snac_workers)

    gen_results = {int(r["row_idx"]): r for r in snac_results}
    if failure_error is not None:
        _mark_failed_rows(to_generate, gen_results, failure_error)
        write_manifests(output_dir, plan, gen_results, split_by_language=args.split_by_language)
        return 1

    write_manifests(output_dir, plan, gen_results, split_by_language=args.split_by_language)
    return 0


def add_common_args(p: argparse.ArgumentParser) -> None:
    """CLI shared by the vLLM offline variants."""
    p.add_argument("--checkpoint_path", default="bodhan-ai/indic-speak")
    p.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer dir; defaults to --checkpoint_path. Point at the extended "
        "audio-tokenizer dir when step-checkpoints lack tokenizer files.",
    )
    p.add_argument("--jsonl-path", dest="jsonl_path", required=True)
    p.add_argument("--dataset_name", default="tts_eval")
    p.add_argument("--training_stage", default="pt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--snac_model_path", default="hubertsiuzdak/snac_24khz")
    p.add_argument(
        "--vocos",
        default="true",
        help="Decoder: 'true' = fine-tuned Vocos decoder (default), 'false' = SNAC's "
        "own decoder, or a path to a local vocos .pt checkpoint.",
    )
    p.add_argument("--max_new_tokens", type=int, default=_SAMPLING_DEFAULTS.max_new_tokens)
    p.add_argument(
        "--num_llm_workers",
        type=int,
        default=-1,
        help="vLLM engine actors; -1 = auto (max(1, GPU count - 1), reserving one "
        "GPU for the SNAC decoders).",
    )
    p.add_argument(
        "--llm_gpu_fraction",
        type=float,
        default=1.0,
        help="Ray num_gpus per vLLM engine actor. On 1-GPU boxes set e.g. 0.8 "
        "(with snac_gpu_fraction 0.2 and gpu_memory_utilization lowered) so "
        "the LLM and SNAC share the card.",
    )
    # vLLM engine knobs
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument(
        "--max_num_seqs", type=int, default=256, help="vLLM continuous-batching width per engine."
    )
    p.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graphs (slower decode; use only for debugging).",
    )
    p.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Share KV for common prompt prefixes (helps when many prompts share a "
        "system/speaker preamble).",
    )
    p.add_argument("--seed", type=int, default=0)
    # sampling — defaults come from the shared SamplingConfig; pass
    # --temperature 0.0 for deterministic (greedy) output in evals.
    p.add_argument(
        "--temperature",
        type=float,
        default=_SAMPLING_DEFAULTS.temperature,
        help="0.0 == greedy / deterministic (useful for evals).",
    )
    p.add_argument("--top_p", type=float, default=_SAMPLING_DEFAULTS.top_p)
    p.add_argument("--top_k", type=int, default=_SAMPLING_DEFAULTS.top_k)
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=_SAMPLING_DEFAULTS.repetition_penalty,
        help="vLLM repetition_penalty (>1.0 penalizes repeated tokens; 1.0 = off). "
        "For codec TTS, ~1.1-1.3 can curb repetition/looping; too high can "
        "distort naturally-repeated audio tokens.",
    )
    # SNAC
    p.add_argument(
        "--snac_decode_batch_size",
        type=int,
        default=32,
        help="Sequences per batched snac.decode call.",
    )
    # resume / misc
    p.add_argument(
        "--split_by_language",
        action="store_true",
        help="Write WAVs under audio/<language>/ instead of a flat audio/ dir "
        "(handy for big mixed-language manifests). The manifest's "
        "gen_audio_path records the per-language path either way.",
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Drop duplicate rows (by audio_filepath) before generating, keeping "
        "the first. Big manifests repeat samples; this avoids redundant work.",
    )
    p.add_argument("--force-regenerate", dest="force_regenerate", action="store_true")
    p.add_argument("--max_rows", type=int, default=0, help="0 = no limit")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable vLLM stats + write benchmark.json (throughput / token dists).",
    )
    p.add_argument("--ray-timeout-sec", type=float, default=3600.0)


def _apply_yaml_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load a YAML config (schema: configs/tts/infer/offline_vllm.yaml) and install its
    values as argparse defaults. Nested sections (topology/engine/sampling/io) are
    flattened one level onto the argparse dests, so explicit CLI flags always
    override config values. Unknown keys fail loudly to keep YAML and CLI in sync."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(flat) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    parser.set_defaults(**flat)
    # A non-empty config value satisfies required flags (CLI may still override).
    for action in parser._actions:
        if action.required and flat.get(action.dest) not in (None, ""):
            action.required = False


def main(argv: list[str] | None = None) -> int:
    # Pre-scan for --config so its values can become defaults of the real parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    cfg_ns, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML defaults file (see configs/tts/infer/offline_vllm.yaml); explicit "
        "CLI flags override its values.",
    )
    add_common_args(p)
    p.add_argument(
        "--num_snac_workers",
        type=int,
        default=3,
        help="Batched-SNAC actors sharing the dedicated SNAC GPU.",
    )
    p.add_argument(
        "--snac_gpu_fraction",
        type=float,
        default=0.33,
        help="Ray num_gpus per SNAC actor; num_snac_workers * fraction <= 1 keeps "
        "them all on the one dedicated GPU.",
    )
    if cfg_ns.config:
        _apply_yaml_config_defaults(p, cfg_ns.config)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
