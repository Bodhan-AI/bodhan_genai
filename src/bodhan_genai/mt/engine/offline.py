"""Resident offline MT engine: text in, translation out.

``IndicMTEngine`` owns the prompt contract and delegates token generation to a
:class:`TranslationBackend` — ``"vllm"`` for throughput, ``"hf"`` for a
dependency-light reference run or to exercise a PEFT adapter without merging it.

All heavy imports (torch / vllm / transformers) live inside methods, so importing
this module stays free — asserted by ``tests/mt/test_mt_lazy_import.py``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol, runtime_checkable

from bodhan_genai.mt.engine.backends import find_api_key, resolve_backend_name
from bodhan_genai.mt.engine.types import (
    BackendOutput,
    MTRequest,
    MTResult,
    MTSamplingConfig,
)
from bodhan_genai.mt.templates.prompt import (
    STOP_STRINGS,
    render_conversations,
    resolve_target_name,
)

logger = logging.getLogger("mt.engine.offline")

#: The published checkpoint, so IndicMTEngine() matches IndicTTSEngine(), IndicOCR() and
#: IndicASREngine() in taking no required argument. Public on the Hub. A caller on the
#: hosted API needs no checkpoint at all; ``model`` is then ignored.
DEFAULT_MODEL_REPO = "bodhan-ai/indic-translate"

#: The context window the released checkpoint is validated at. The architecture
#: allows up to 131072, but only 32768 has been measured end to end.
DEFAULT_MAX_MODEL_LEN = 32_768


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class TranslationBackend(Protocol):
    """Request-level generation backend: requests in, outputs out.

    Request-level and not conversation-level because the hosted API never
    receives a prompt — it builds one server-side — so a rendered conversation
    would have to be un-rendered to reach the wire.

    Local backends are conversation-level internally and reach this protocol
    through :class:`PromptedBackend`, which is the single place an
    :class:`MTRequest` becomes a prompt.
    """

    def generate(self, requests: list[MTRequest], sc: MTSamplingConfig) -> list[BackendOutput]:
        """Generate one output per request, aligned with the input."""
        ...

    def close(self) -> None:
        """Release model / GPU resources. Idempotent best-effort."""
        ...


@runtime_checkable
class ConversationBackend(Protocol):
    """Internal protocol for backends that consume rendered prompts."""

    def generate_conversations(
        self, conversations: list[list[dict[str, str]]], sc: MTSamplingConfig
    ) -> list[BackendOutput]:
        """Generate one output per conversation, aligned with the input."""
        ...

    def close(self) -> None:
        """Release model / GPU resources. Idempotent best-effort."""
        ...


class PromptedBackend:
    """Adapts a :class:`ConversationBackend` to the request-level protocol.

    This exists so prompt construction has exactly ONE call site. Letting each
    local backend render its own prompts would put two copies of the contract in
    the package, and N copies of the contract drifting apart is the documented
    failure mode this whole module is organised against.
    """

    def __init__(self, inner: ConversationBackend) -> None:
        self._inner = inner

    def generate(self, requests: list[MTRequest], sc: MTSamplingConfig) -> list[BackendOutput]:
        return self._inner.generate_conversations(render_conversations(requests), sc)

    def close(self) -> None:
        self._inner.close()

    # The adapter must not hide what the inner backend can do. Both of these are
    # optional on the protocol, so they are forwarded only when present.
    def validate(self, task: str, script: str) -> None:
        inner = getattr(self._inner, "validate", None)
        if inner is not None:
            inner(task, script)

    def health(self) -> bool:
        inner = getattr(self._inner, "health", None)
        return True if inner is None else bool(inner())


# --------------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------------- #


class _VllmBackend:
    """vLLM ``LLM.chat`` backend — the throughput path.

    Stock vLLM >= 0.20 registers ``Gemma4ForConditionalGeneration`` and loads the
    published checkpoint unpatched: it ships zeroed ``k_norm`` tensors for its 18
    KV-shared layers, which is what vLLM's weight loader expects. A checkpoint you
    trained yourself needs ``python -m bodhan_genai.mt.tools.vllm_ready`` first.

    ``enforce_eager`` defaults on: that is the configuration the released scores
    were measured with, and CUDA-graph capture buys little for translation-length
    generations.
    """

    def __init__(
        self,
        model: str,
        *,
        dtype: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        engine_kwargs: dict[str, Any] | None,
        llm_factory: Any | None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": model,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            # Required for the Gemma 4 processor; there is no remote python in the
            # checkpoint, so this executes nothing of the model's own.
            "trust_remote_code": True,
        }
        kwargs.update(engine_kwargs or {})

        if llm_factory is not None:
            self._llm = llm_factory(kwargs)
        else:
            # Must be set before vLLM is imported: vLLM >= 0.26 samples through a
            # flashinfer kernel it JIT-compiles at engine warm-up, which needs nvcc.
            # Nodes with a runtime-only CUDA install have none, and the build failure
            # surfaces as the generic "EngineCore failed to start" rather than anything
            # naming flashinfer. The native sampler is equivalent for greedy decoding.
            # Override by exporting VLLM_USE_FLASHINFER_SAMPLER=1 where nvcc exists.
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            # Same reason, different kernel: vLLM probes vllm.third_party.deep_gemm, whose import
            # asserts on _find_cuda_home(). Without nvcc that assertion fails and vLLM logs a
            # twenty-line traceback ending in a bare AssertionError, then carries on -- so a
            # perfectly healthy engine start looks like a crash. IndicOCR's recognizer has always
            # set this; the TTS and MT engines did not, which made the noise look modality-specific.
            os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

            from vllm import LLM

            logger.info("loading vLLM engine: %s", model)
            self._llm = LLM(**kwargs)

    def generate_conversations(
        self, conversations: list[list[dict[str, str]]], sc: MTSamplingConfig
    ) -> list[BackendOutput]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=sc.temperature,
            # Passing a top_p/seed alongside greedy decoding is contradictory;
            # vLLM ignores them, but keeping the request honest keeps logs readable.
            top_p=1.0 if sc.greedy else sc.top_p,
            repetition_penalty=sc.repetition_penalty,
            max_tokens=sc.max_new_tokens,
            stop=STOP_STRINGS,
            seed=None if sc.greedy else sc.seed,
        )
        # vLLM applies the chat template shipped in the checkpoint and schedules
        # the whole list as one continuous batch.
        outputs = self._llm.chat(conversations, sampling_params=sampling)
        return [
            BackendOutput(
                text=out.outputs[0].text.strip(),
                # "length" is the only signal that the budget was hit; a truncated
                # translation is otherwise fluent and simply stops.
                truncated=out.outputs[0].finish_reason == "length",
                prompt_tokens=len(out.prompt_token_ids or ()),
                generated_tokens=len(out.outputs[0].token_ids or ()),
            )
            for out in outputs
        ]

    def close(self) -> None:
        self._llm = None


# --------------------------------------------------------------------------- #
# HuggingFace backend
# --------------------------------------------------------------------------- #


class _HfBackend:
    """HF ``generate()`` backend — the reference path.

    Loads through ``AutoModelForImageTextToText`` + ``AutoProcessor``: the
    checkpoint is a multimodal ``Gemma4ForConditionalGeneration`` and that pair is
    its documented entry point. (Training loads the text-only
    ``AutoModelForCausalLM`` instead — see ``bodhan_genai.mt.training``.)

    Two settings here are load-bearing and easy to lose:

    * ``padding_side = "left"`` so every sequence in a batch ends flush against
      the generation boundary.
    * ``generation_config.use_cache = True`` — the checkpoint was trained with the
      KV cache disabled, and without re-enabling it every token re-runs the full
      forward pass.
    """

    def __init__(
        self,
        model: str,
        *,
        adapter_dir: str | None,
        dtype: str,
        device: str,
        attn_implementation: str,
        processor_loader: Any | None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch

        logger.info("loading processor: %s", model)
        loader = processor_loader or AutoProcessor.from_pretrained
        self.processor = loader(model)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

        logger.info("loading weights (%s, attn=%s)", dtype, attn_implementation)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            dtype=getattr(torch, dtype),
            device_map=device,
            attn_implementation=attn_implementation,
        )
        if adapter_dir:
            from peft import PeftModel

            logger.info("attaching PEFT adapter: %s", adapter_dir)
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)

        self.model.eval()
        self.model.generation_config.use_cache = True
        self.device = next(self.model.parameters()).device
        logger.info("ready on %s", self.device)

    def generate_conversations(
        self, conversations: list[list[dict[str, str]]], sc: MTSamplingConfig
    ) -> list[BackendOutput]:
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": sc.max_new_tokens,
            "use_cache": True,
            # Belt and braces: <turn|> is already in eos_token_id, but an explicit
            # stop string survives a template change.
            "stop_strings": STOP_STRINGS,
            "tokenizer": self.processor.tokenizer,
        }
        if sc.greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(do_sample=True, temperature=sc.temperature, top_p=sc.top_p)
        if sc.repetition_penalty and sc.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = sc.repetition_penalty

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        completions = output_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(completions, skip_special_tokens=True)
        # No finish_reason here: hitting the budget exactly is the signal.
        truncated = completions.shape[-1] >= sc.max_new_tokens
        return [
            BackendOutput(
                text=d.strip(),
                truncated=truncated,
                prompt_tokens=prompt_len,
                generated_tokens=int(completions.shape[-1]),
            )
            for d in decoded
        ]

    def close(self) -> None:
        self.model = None
        self.processor = None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class IndicMTEngine:
    """Resident offline MT engine: text -> translation.

    ``backend`` selects generation: ``"vllm"`` (throughput), ``"hf"``
    (reference / adapters), or any :class:`TranslationBackend` instance (tests,
    custom engines).

    The prompt contract is enforced here, not by the caller: every path goes
    through ``build_conversation``, so the target language is always named and the
    source language never is.

        engine = IndicMTEngine("/path/to/checkpoint")
        engine.translate("Hello world", tgt_lang="hin_Deva").text
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_REPO,
        *,
        backend: str | TranslationBackend | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        adapter_dir: str | None = None,
        dtype: str = "bfloat16",
        sampling: MTSamplingConfig | None = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = 0.90,
        enforce_eager: bool = True,
        attn_implementation: str = "sdpa",
        device: str = "auto",
        engine_kwargs: dict[str, Any] | None = None,
        processor_loader: Any | None = None,
        llm_factory: Any | None = None,
    ) -> None:
        self.model_path = model
        self.sampling = sampling or MTSamplingConfig()

        if not isinstance(backend, str) and backend is not None:
            self.backend_name = "custom"
            self._backend: TranslationBackend = backend
            return

        # None means "decide": the hosted API when a token is present, else vLLM.
        # An explicit name always wins, so nothing that names a backend changes.
        name = resolve_backend_name(backend)
        self.backend_name = name

        if adapter_dir and name != "hf":
            raise ValueError(
                f"adapter_dir is only supported with backend='hf' (got {name!r}). "
                "Merge the adapter into the checkpoint first: "
                "python -m bodhan_genai.mt.training.merge."
            )

        if name == "api":
            from bodhan_genai.mt.engine.api import DEFAULT_BASE_URL, ApiBackend

            # Conversation-level, like the local backends: the service takes a
            # rendered chat conversation, so it goes behind PromptedBackend and
            # the prompt contract keeps exactly one call site. That is also what
            # gives the hosted path full capability parity -- Roman, code-mix and
            # transliteration all ride in the prompt.
            self._backend = PromptedBackend(
                ApiBackend(
                    api_key or find_api_key() or "",
                    base_url=api_base_url or DEFAULT_BASE_URL,
                )
            )
        elif name == "vllm":
            self._backend = _VllmBackend(
                model,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                engine_kwargs=engine_kwargs,
                llm_factory=llm_factory,
            )
            self._backend = PromptedBackend(self._backend)
        else:
            self._backend = PromptedBackend(
                _HfBackend(
                    model,
                    adapter_dir=adapter_dir,
                    dtype=dtype,
                    device=device,
                    attn_implementation=attn_implementation,
                    processor_loader=processor_loader,
                )
            )

    # -- public API -------------------------------------------------------- #

    def translate(
        self,
        text: str,
        *,
        tgt_lang: str,
        src_lang: str | None = None,
        script: str = "native",
        **overrides,
    ) -> MTResult:
        """Translate one segment. Never raises for generation failure — see
        ``MTResult.error``."""
        return self.translate_batch(
            [text], tgt_lang=tgt_lang, src_lang=src_lang, script=script, **overrides
        )[0]

    def translate_batch(
        self,
        texts: list[str],
        *,
        tgt_lang: str,
        src_lang: str | None = None,
        script: str = "native",
        task: str = "translate",
        **overrides,
    ) -> list[MTResult]:
        """Translate a list of segments in one batched call.

        The language, script and task are resolved before anything is generated,
        so a typo fails immediately rather than after a 16 GB model load and a
        full batch. ``script`` selects the output rendering (``native``,
        ``roman``, ``code_mixed``).
        """
        tgt_name = resolve_target_name(tgt_lang, script=script, task=task)
        # Transport-level fail-fast, for a backend that cannot carry some
        # (task, script) pair. Saying so here beats a 4xx after the batch is
        # already in flight. No current backend restricts anything.
        validate = getattr(self._backend, "validate", None)
        if validate is not None:
            validate(task, script)
        sc = self.sampling.merged(**overrides)

        results = [MTResult(source=t, src_lang=src_lang, tgt_lang=tgt_name) for t in texts]
        if not texts:
            return results

        requests = [
            MTRequest(text=t, tgt_lang=tgt_lang, script=script, task=task, src_lang=src_lang)
            for t in texts
        ]

        started = time.perf_counter()
        try:
            outputs = self._backend.generate(requests, sc)
        except Exception as exc:
            logger.exception("generation failed for a batch of %d", len(texts))
            for r in results:
                r.error = f"{type(exc).__name__}: {exc}"
            return results
        elapsed = time.perf_counter() - started

        if len(outputs) != len(texts):
            raise RuntimeError(
                f"backend returned {len(outputs)} completions for {len(texts)} "
                f"prompts; results would be misaligned"
            )

        per_request = elapsed / len(texts)
        for r, out in zip(results, outputs, strict=True):
            r.text = out.text
            r.truncated = out.truncated
            r.prompt_tokens = out.prompt_tokens
            r.generated_tokens = out.generated_tokens
            # A per-request backend reports the truth; a batched one cannot
            # attribute time to one row, so the batch elapsed is split evenly.
            r.gen_time_s = per_request if out.gen_time_s is None else out.gen_time_s
        return results

    def transliterate(
        self, text: str, *, lang: str, script: str = "roman", **overrides
    ) -> MTResult:
        """Rewrite one text in another script, without translating it.

        Same language in, same language out: ``script="roman"`` goes native to
        Latin, ``script="native"`` comes back.
        """
        return self.transliterate_batch([text], lang=lang, script=script, **overrides)[0]

    def transliterate_batch(
        self, texts: list[str], *, lang: str, script: str = "roman", **overrides
    ) -> list[MTResult]:
        """Transliterate a list of segments in one batched call."""
        return self.translate_batch(
            texts, tgt_lang=lang, script=script, task="transliterate", **overrides
        )

    def translate_document(
        self,
        text: str,
        *,
        tgt_lang: str,
        src_lang: str | None = None,
        script: str = "native",
        **overrides,
    ) -> MTResult:
        """Translate a whole document as ONE request, preserving its structure.

        Raise ``max_new_tokens`` accordingly (~8192 for a full document) — the
        default of 512 is sized for sentences and will truncate.
        """
        return self.translate(
            text, tgt_lang=tgt_lang, src_lang=src_lang, script=script, **overrides
        )

    # -- health ------------------------------------------------------------ #

    def health(self) -> bool:
        """True when the backend is reachable and serving the expected model.

        Only the hosted backend can meaningfully answer this; an in-process
        backend is up by construction, so it reports ``True`` rather than
        forcing callers to special-case it.
        """
        probe = getattr(self._backend, "health", None)
        return True if probe is None else bool(probe())

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Release backend resources. Idempotent."""
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend.close()
            self._backend = None  # type: ignore[assignment]

    def __enter__(self) -> IndicMTEngine:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
