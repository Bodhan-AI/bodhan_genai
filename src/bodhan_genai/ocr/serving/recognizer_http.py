"""A ``RecognizerBackend`` that talks to a stock ``vllm serve`` endpoint.

The endpoint serves IndicBlockOCR alone. Layout runs client-side, so this backend receives the
same ``CropRequest`` list the in-process backends do and returns one transcription per crop.

The messages built here render, through the recognizer's chat template, to exactly the string
``VllmRecognizer`` builds locally: the template branches on ``'image' in item or 'image_url' in
item``, so OpenAI-style content and the local ``{"type": "image"}`` form produce the same prompt.
``tests/ocr/test_serving_client.py`` asserts that against the shipped template.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from bodhan_genai.ocr.engine.types import RecognizerConfig

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

    from bodhan_genai.ocr.engine.recognizer import CropRequest

logger = logging.getLogger("ocr.serving.recognizer_http")

DEFAULT_MODEL = "indic_ocr"


def encode_crop(image: Image) -> str:
    """A crop as a PNG data URI. PNG because the crops are already lossy-free page regions."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(image: Image, prompt: str) -> list[dict[str, Any]]:
    """The single user turn for one crop. Image first, then text, as the local path builds it."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_crop(image)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


class HttpRecognizer:
    """Transcribe crops through an OpenAI-compatible endpoint.

    ``base_url`` is the ``/v1`` root. vLLM ignores the API key, so the placeholder is fine.

    Crops are sent concurrently so the server's continuous batching sees them together;
    ``num_workers`` is the number in flight, not a batch size. Results are reassembled by index,
    because stage 2 matches transcriptions back to blocks by position.

    With ``strict``, one failed crop fails the page. Turn it off for a long batch and failures
    become empty transcriptions instead, which are indistinguishable in the output from a block
    that was deliberately not transcribed -- so the count is logged either way.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        timeout: float = 300.0,
        config: RecognizerConfig | None = None,
        num_workers: int = 32,
        strict: bool = True,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config or RecognizerConfig()
        self.num_workers = max(1, num_workers)
        self.strict = strict
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            # vLLM ignores the key when the server was started without --api-key, so the
            # "EMPTY" placeholder stays valid for an open server. When OCR_API_KEY is exported
            # the client picks it up, instead of every caller having to wire it through.
            key = api_key or os.environ.get("OCR_API_KEY") or "EMPTY"
            self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)

    def _one(self, request: CropRequest) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(request.image, request.prompt),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        if not requests:
            return []

        texts: list[str] = [""] * len(requests)
        failures: list[str] = []

        def run(index: int) -> None:
            try:
                texts[index] = self._one(requests[index])
            except Exception as exc:  # transport, timeout, server error
                failures.append(f"crop {index}: {type(exc).__name__}: {exc}")

        workers = min(self.num_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so exceptions inside the pool surface here rather than being dropped.
            list(pool.map(run, range(len(requests))))

        if failures:
            summary = f"{len(failures)}/{len(requests)} crops failed; first: {failures[0]}"
            if self.strict:
                raise RuntimeError(summary)
            logger.warning("%s -- those blocks carry empty text", summary)

        return texts

    def close(self) -> None:
        self._client = None
