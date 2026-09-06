"""The vLLM recognizer, isolated.

Kept out of ``engine.recognizer`` so that module imports no vLLM at all -- not even inside a
method. The Hub's ``trust_remote_code`` path vendors the engine, and transformers scans vendored
files with a regex that does not care whether an import sits inside a function: a single
``from vllm import ...`` anywhere would make vLLM a hard requirement of the quickstart path.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from bodhan_genai.ocr.engine.recognizer import CropRequest
from bodhan_genai.ocr.engine.types import RecognizerConfig

logger = logging.getLogger("ocr.engine.recognizer_vllm")


def _flashinfer_cache_is_usable(root: Path) -> bool:
    """Whether flashinfer can write under ``root``.

    Checking the root alone is not enough: it is commonly writable while a per-version
    subdirectory underneath it is not.
    """
    if not root.exists():
        return True  # it will be created under a writable parent
    if not os.access(root, os.W_OK | os.X_OK):
        return False
    return all(os.access(child, os.W_OK | os.X_OK) for child in root.iterdir() if child.is_dir())


def _prepare_runtime() -> None:
    """Environment that must be set before vLLM is imported.

    Two things, both of which otherwise fail deep inside engine startup with an error that does
    not name its cause:

    * ``ninja`` -- the recognizer JIT-compiles a GDN kernel on first use, and ninja lives in the
      venv's bin/, which is not on PATH when the venv python is launched by its full path.
    * flashinfer's JIT cache -- flashinfer opens a log under ``$HOME/.cache/flashinfer`` at
      import. On a shared machine that path is often a symlink into a directory owned by another
      user (or populated by a root container), and the resulting PermissionError surfaces only
      as "Engine core initialization failed". Redirect to a private directory when the default
      is not writable; an explicit FLASHINFER_WORKSPACE_BASE always wins.
    """
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    if not os.environ.get("FLASHINFER_WORKSPACE_BASE") and not _flashinfer_cache_is_usable(
        Path.home() / ".cache" / "flashinfer"
    ):
        fallback = Path(tempfile.gettempdir()) / f"flashinfer-{os.getuid()}"
        (fallback / ".cache" / "flashinfer").mkdir(parents=True, exist_ok=True)
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(fallback)
        logger.warning(
            "$HOME/.cache/flashinfer is not writable; using %s instead. Kernels will be "
            "recompiled on first use. Point FLASHINFER_WORKSPACE_BASE at shared, persistent "
            "storage to keep the cache between runs.",
            fallback,
        )


class VllmRecognizer:
    """In-process vLLM engine. All blocks of a page go through one continuous batch."""

    def __init__(self, ckpt: str | None = None, config: RecognizerConfig | None = None) -> None:
        _prepare_runtime()

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

        self.config = config or RecognizerConfig()
        self.ckpt = resolve_ckpt("recognizer", ckpt)
        self.processor = AutoProcessor.from_pretrained(self.ckpt)
        self.llm = LLM(
            model=self.ckpt,
            trust_remote_code=True,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=self.config.enforce_eager,
        )
        self.sampling = SamplingParams(
            temperature=self.config.temperature, max_tokens=self.config.max_tokens
        )

    def _chat(self, prompt: str) -> str:
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        payload = [
            {"prompt": self._chat(r.prompt), "multi_modal_data": {"image": r.image}}
            for r in requests
        ]

        texts: list[str] = []
        # Bounded sub-batches: one giant generate() over tens of thousands of multi-modal
        # requests wedges the vLLM V1 scheduler at 100% util with no progress.
        for i in range(0, len(payload), self.config.batch_size):
            outputs = self.llm.generate(payload[i : i + self.config.batch_size], self.sampling)
            texts.extend(out.outputs[0].text for out in outputs)
        return texts

    def close(self) -> None:
        self.llm = None
        self.processor = None
