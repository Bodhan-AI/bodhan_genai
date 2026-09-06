"""bodhan_genai.ocr.serving — client side of a stock ``vllm serve`` deployment.

There is no server module here on purpose. The endpoint holds IndicBlockOCR on unmodified vLLM
(``scripts/ocr/serve.sh`` is a wrapper, not a service). What is owned in-process is the half vLLM
cannot serve: layout detection, cropping, and reassembly.

``HttpRecognizer`` implements the same ``RecognizerBackend`` protocol as the in-process backends,
so ``IndicBlockOCR`` runs unchanged and served output follows the offline path exactly.

The re-exports below resolve on attribute access rather than at import time. Eagerly importing
``.client`` here would put it in ``sys.modules`` before
``python -m bodhan_genai.ocr.serving.client`` -- the invocation the docs print -- re-executes it as
``__main__``, and Python warns about the double import on every run.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhan_genai.ocr.serving.client import OCRClient
    from bodhan_genai.ocr.serving.recognizer_http import DEFAULT_MODEL, HttpRecognizer

__all__ = ["DEFAULT_MODEL", "HttpRecognizer", "OCRClient"]


def __getattr__(name: str):
    if name == "OCRClient":
        from bodhan_genai.ocr.serving import client

        return client.OCRClient
    if name in ("DEFAULT_MODEL", "HttpRecognizer"):
        from bodhan_genai.ocr.serving import recognizer_http

        return getattr(recognizer_http, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
