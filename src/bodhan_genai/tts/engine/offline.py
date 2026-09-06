"""IndicTTSEngine — offline (non-serving) TTS engine with pluggable synthesis backends.

One resident engine object owns: tokenizer + template ids (cheap, at ctor), a
synthesis backend (vLLM offline engine or plain HF ``model.generate``), and a
lazily-loaded SNAC codec model. ``synthesize`` / ``synthesize_batch`` run the
full text -> prompt -> token generation -> SNAC decode -> float32 audio flow
and return :class:`~bodhan_genai.tts.engine.types.TTSResult` rows.

Backends implement the tiny :class:`SynthesisBackend` protocol, so tests (and
alternative engines) can inject a fake and exercise the whole pipeline without
GPUs, vLLM or transformers.

Heavy imports (torch / vllm / transformers / peft / snac / soundfile) live
inside methods — this module imports clean on a bare CPU box.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from bodhan_genai.tts.engine.types import SamplingConfig, TTSResult
from bodhan_genai.tts.inference import prompts as _prompts
from bodhan_genai.tts.templates.chat import get_template_ids

logger = logging.getLogger(__name__)

_DEFAULT_SNAC_MODEL = "hubertsiuzdak/snac_24khz"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SynthesisBackend(Protocol):
    """Token-level generation backend: prompt ids in, generated ids out."""

    def generate(
        self,
        prompt_ids_list: list[list[int]],
        sc: SamplingConfig,
        stop_ids: list[int],
    ) -> list[list[int]]:
        """Generate continuations for each prompt. Returns one generated-id
        list per prompt (prompt tokens excluded), aligned with the input."""
        ...

    def close(self) -> None:
        """Release model / GPU resources. Idempotent best-effort."""
        ...


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


class _VllmBackend:
    """Offline vLLM engine (single process, single GPU by default).

    Engine kwargs mirror ``offline_vllm.vllm_engine_kwargs`` so an engine built
    here behaves like one Ray-actor engine from the batch pipeline; caller
    ``engine_kwargs`` merge last and win.
    """

    def __init__(
        self,
        model: str,
        tokenizer_path: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        enforce_eager: bool = False,
        seed: int = 0,
        engine_kwargs: dict[str, Any] | None = None,
    ):
        # Must be set before vLLM is imported: keeps the engine in-process so it
        # respects CUDA_VISIBLE_DEVICES instead of spawning its own workers.
        if "vllm" in sys.modules and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") is None:
            logger.warning(
                "vllm was already imported without VLLM_ENABLE_V1_MULTIPROCESSING=0; "
                "the in-process engine setting may not take effect."
            )
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # Also before the import: vLLM >= 0.26 samples through a flashinfer kernel it
        # JIT-compiles at engine warm-up, which needs nvcc. Nodes with a runtime-only
        # CUDA install have none, and the build failure surfaces as the generic
        # "EngineCore failed to start" rather than anything naming flashinfer. The
        # native sampler is equivalent for our top_p/top_k settings. Override by
        # exporting VLLM_USE_FLASHINFER_SAMPLER=1 where a toolkit is present.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import vllm

        self._vllm = vllm
        kw: dict[str, Any] = dict(
            model=model,
            tokenizer=tokenizer_path,
            dtype=str(dtype),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            enforce_eager=bool(enforce_eager),
            disable_log_stats=True,
            trust_remote_code=True,
            seed=int(seed),
        )
        if engine_kwargs:
            kw.update(engine_kwargs)  # user kwargs win
        logger.info("Loading vLLM engine: %s (tokenizer %s)", model, tokenizer_path)
        self._llm = vllm.LLM(**kw)

    def generate(
        self,
        prompt_ids_list: list[list[int]],
        sc: SamplingConfig,
        stop_ids: list[int],
    ) -> list[list[int]]:
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(
            temperature=float(sc.temperature),
            top_p=float(sc.top_p),
            top_k=int(sc.top_k),
            repetition_penalty=float(sc.repetition_penalty),
            max_tokens=int(sc.max_new_tokens),
            stop_token_ids=list(stop_ids),
            detokenize=False,
        )
        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_ids_list]
        outputs = self._llm.generate(prompts, sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def close(self) -> None:
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:  # best-effort across vLLM versions
            engine = getattr(llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
            elif hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("vLLM shutdown raised (ignored): %s", e)
        self._llm = None
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------


class _HfBackend:
    """Plain ``transformers`` generate backend (one prompt at a time) with
    optional PEFT adapter — the sample-quality / debugging path."""

    def __init__(
        self,
        model: str,
        *,
        adapter_dir: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(str(dtype), torch.bfloat16)
        logger.info("Loading HF model %s on %s (%s)", model, device, dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch_dtype, device_map=device
        )
        if adapter_dir:
            from peft import PeftModel

            logger.info("Applying PEFT adapter from %s", adapter_dir)
            self._model = PeftModel.from_pretrained(self._model, adapter_dir)
        self._model.eval()

    def generate(
        self,
        prompt_ids_list: list[list[int]],
        sc: SamplingConfig,
        stop_ids: list[int],
    ) -> list[list[int]]:
        import torch

        do_sample = sc.temperature > 0
        gen_kwargs: dict[str, Any] = dict(
            do_sample=do_sample,
            repetition_penalty=float(sc.repetition_penalty),
            max_new_tokens=int(sc.max_new_tokens),
            eos_token_id=list(stop_ids),
            pad_token_id=int(stop_ids[-1]),
        )
        if do_sample:  # transformers-5 rejects sampling knobs when do_sample=False
            gen_kwargs["temperature"] = float(sc.temperature)
            gen_kwargs["top_p"] = float(sc.top_p)

        results: list[list[int]] = []
        for prompt_ids in prompt_ids_list:
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._model.device)
            with torch.inference_mode():
                output = self._model.generate(input_ids, **gen_kwargs)
            results.append(output[0][len(prompt_ids) :].tolist())
        return results

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is None:
            return
        self._model = None
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class IndicTTSEngine:
    """Resident offline TTS engine: text -> audio.

    ``backend`` selects token generation: ``"vllm"`` (throughput), ``"hf"``
    (sample-quality / adapters), or any :class:`SynthesisBackend` instance
    (tests, custom engines). The SNAC codec model is loaded lazily on first
    decode; ``tokenizer_loader`` / ``snac_loader`` are test seams.
    """

    def __init__(
        self,
        model: str = "bodhan-ai/indic-speak",
        *,
        backend: str | SynthesisBackend = "vllm",
        tokenizer: Any | None = None,
        snac_model_path: str = _DEFAULT_SNAC_MODEL,
        adapter_dir: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        sampling: SamplingConfig | None = None,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 64,
        enforce_eager: bool = False,
        seed: int = 0,
        engine_kwargs: dict[str, Any] | None = None,
        tokenizer_loader: Any | None = None,
        snac_loader: Any | None = None,
        vocos: str | bool = True,
    ):
        if adapter_dir and backend == "vllm":
            raise ValueError(
                "adapter_dir is only supported with backend='hf' "
                "(merge the adapter into the checkpoint for vLLM)."
            )

        self._model_path = str(model)
        self._snac_model_path = snac_model_path
        self._device = device  # None -> resolved lazily (cuda if available)
        self._snac_loader = snac_loader
        self._snac_model: Any | None = None
        # True -> fine-tuned Vocos decoder from the Hub; str -> local checkpoint;
        # False -> SNAC's own decoder.
        self._vocos = vocos
        self._sampling = sampling if sampling is not None else SamplingConfig()
        self._closed = False

        # Tokenizer + template ids (cheap, needed for every prompt build).
        if tokenizer is not None and not isinstance(tokenizer, (str, os.PathLike)):
            self._tokenizer = tokenizer
        else:
            tokenizer_path = str(tokenizer) if tokenizer is not None else self._model_path
            if tokenizer_loader is not None:
                self._tokenizer = tokenizer_loader(tokenizer_path)
            else:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._tmpl = get_template_ids(self._tokenizer)
        self._ids = _prompts.resolve_snac_ids(self._tokenizer)
        self._stop_ids = [self._ids["end_of_audio_id"], self._ids["eos_token_id"]]

        # Backend.
        if isinstance(backend, str):
            tokenizer_path = (
                str(tokenizer)
                if tokenizer is not None and isinstance(tokenizer, (str, os.PathLike))
                else self._model_path
            )
            if backend == "vllm":
                self._backend: SynthesisBackend = _VllmBackend(
                    self._model_path,
                    tokenizer_path,
                    dtype=dtype,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                    enforce_eager=enforce_eager,
                    seed=seed,
                    engine_kwargs=engine_kwargs,
                )
            elif backend == "hf":
                self._backend = _HfBackend(
                    self._model_path,
                    adapter_dir=adapter_dir,
                    device=self._resolve_device(),
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    f"Unknown backend {backend!r} (expected 'vllm', 'hf' or an instance)"
                )
        else:
            self._backend = backend

    # -- infrastructure -----------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Output sample rate (SNAC 24 kHz)."""
        from bodhan_genai.tts.inference.audio_io import SNAC_SAMPLE_RATE

        return SNAC_SAMPLE_RATE

    def _resolve_device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _snac(self):
        """SNAC codec model, loaded on first use and kept resident."""
        if self._snac_model is None:
            device = self._resolve_device()
            if self._snac_loader is not None:
                self._snac_model = self._snac_loader(
                    self._snac_model_path, device=device, compile_model=False
                )
            else:
                from bodhan_genai.tts.codec.snac import load_snac_model

                self._snac_model = load_snac_model(
                    self._snac_model_path, device=device, compile_model=False
                )
            from bodhan_genai.tts.codec.vocos import resolve_decoder

            self._snac_model = resolve_decoder(self._snac_model, self._vocos, device=device)
        return self._snac_model

    def close(self) -> None:
        """Release backend + SNAC resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.close()
        finally:
            self._snac_model = None
            gc.collect()

    def __enter__(self) -> IndicTTSEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- synthesis ----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        speaker: str = "",
        style: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> TTSResult:
        """Synthesize one utterance; raises ``RuntimeError`` on failure."""
        result = self.synthesize_batch(
            [text],
            speakers=speaker,
            styles=style,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )[0]
        if result.error:
            raise RuntimeError(result.error)
        return result

    def synthesize_batch(
        self,
        texts: list[str] | str,
        *,
        speakers: list[str] | str = "",
        styles: list[str] | str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[TTSResult]:
        """Synthesize a batch. One backend.generate call for all valid prompts
        and one batched SNAC decode; failed rows come back with ``error`` set
        instead of aborting the batch."""
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(speakers, str):
            speakers = [speakers] * len(texts)
        if len(speakers) != len(texts):
            raise ValueError(f"speakers ({len(speakers)}) must match texts ({len(texts)})")
        if isinstance(styles, str):
            styles = [styles] * len(texts)
        if len(styles) != len(texts):
            raise ValueError(f"styles ({len(styles)}) must match texts ({len(texts)})")

        sc = self._sampling.merged(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        results = [TTSResult(sample_rate=self.sample_rate) for _ in texts]

        # Build prompts; rows that fail here get an error and are excluded.
        valid_idx: list[int] = []
        prompt_ids_list: list[list[int]] = []
        for i, (text, speaker, style) in enumerate(zip(texts, speakers, styles, strict=False)):
            try:
                prompt_ids = _prompts.build_prompt_ids(
                    text, speaker, self._tokenizer, tmpl=self._tmpl, style=style
                )
            except ValueError as e:
                results[i].error = f"prompt build failed: {e}"
                continue
            if not prompt_ids:
                results[i].error = "empty prompt (blank text?)"
                continue
            results[i].prompt_tokens = len(prompt_ids)
            valid_idx.append(i)
            prompt_ids_list.append(prompt_ids)
        return self._run_generation(results, valid_idx, prompt_ids_list, sc)

    def synthesize_conversation_batch(
        self,
        conversations: list[list[dict]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> list[TTSResult]:
        """Synthesize a batch of conversations, ONE continuous sample each.

        Mirrors ``synthesize_batch``: one backend.generate call for every
        valid conversation prompt and one batched SNAC decode; a conversation
        whose prompt cannot be built comes back with ``error`` set instead of
        aborting the batch."""
        sc = self._sampling.merged(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        results = [TTSResult(sample_rate=self.sample_rate) for _ in conversations]

        # Build prompts; rows that fail here get an error and are excluded.
        valid_idx: list[int] = []
        prompt_ids_list: list[list[int]] = []
        for i, msgs in enumerate(conversations):
            try:
                prompt_ids = _prompts.build_conversation_prompt_ids(
                    msgs, self._tokenizer, tmpl=self._tmpl
                )
            except ValueError as e:
                results[i].error = f"prompt build failed: {e}"
                continue
            if not prompt_ids:
                results[i].error = "empty conversation prompt (no messages?)"
                continue
            results[i].prompt_tokens = len(prompt_ids)
            valid_idx.append(i)
            prompt_ids_list.append(prompt_ids)
        return self._run_generation(results, valid_idx, prompt_ids_list, sc)

    def synthesize_conversation(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
    ) -> TTSResult:
        """Synthesize a multi-turn conversation as ONE continuous audio sample.

        ``messages`` is a chat-style list of ``{"speaker": ..., "text": ...}``
        dicts, rendered through the conversation chat template
        (``<|speaker>NAME<speaker|>`` tags inline in the text, no per-utterance
        metadata prefix). Multi-speaker output follows the speaker tags the
        model saw in conversation training data. Raises on failure."""
        (result,) = self.synthesize_conversation_batch(
            [messages],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        if result.error:
            if result.error.startswith("prompt build failed: "):
                # Surface prompt-construction problems as the original ValueError.
                raise ValueError(result.error.removeprefix("prompt build failed: "))
            if result.error == "empty conversation prompt (no messages?)":
                raise ValueError(result.error)
            raise RuntimeError(result.error)
        return result

    def _run_generation(
        self,
        results: list[TTSResult],
        valid_idx: list[int],
        prompt_ids_list: list[list[int]],
        sc,
    ) -> list[TTSResult]:
        """Shared backend.generate -> extract -> batched-SNAC-decode pipeline;
        fills ``results`` in place (rows keep ``error`` on failure)."""
        if not prompt_ids_list:
            return results

        # One generation call for the whole batch.
        t0 = time.time()
        generated = self._backend.generate(prompt_ids_list, sc, list(self._stop_ids))
        gen_time = time.time() - t0
        per_row_gen = gen_time / len(prompt_ids_list)

        # Extract audio tokens; collect decodable rows.
        decode_idx: list[int] = []
        token_lists: list[list[int]] = []
        for i, gen in zip(valid_idx, generated, strict=False):
            results[i].generated_tokens = len(gen)
            results[i].gen_time_s = per_row_gen
            audio_tokens = _prompts.extract_audio_tokens(
                gen, self._ids["start_of_audio_id"], self._ids["end_of_audio_id"]
            )
            if not audio_tokens:
                results[i].error = "no audio tokens in generation (missing <|start_of_speech|>)"
                continue
            results[i].audio_tokens = len(audio_tokens)
            decode_idx.append(i)
            token_lists.append(audio_tokens)
        if not token_lists:
            return results

        # One batched SNAC decode for every good row.
        from bodhan_genai.tts.codec.snac import batch_decode_audio

        t1 = time.time()
        audio_bytes_list = batch_decode_audio(
            self._snac(),
            token_lists,
            self._ids["audio_token_base_id"],
            device=self._resolve_device(),
        )
        decode_time = time.time() - t1
        per_row_decode = decode_time / len(token_lists)

        for i, audio_bytes in zip(decode_idx, audio_bytes_list, strict=False):
            results[i].decode_time_s = per_row_decode
            if audio_bytes is None:
                results[i].error = "SNAC decode failed (no valid audio frames)"
                continue
            results[i].audio = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            )
        return results
