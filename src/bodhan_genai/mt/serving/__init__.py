"""bodhan_genai.mt.serving — client side of a stock ``vllm serve`` deployment.

There is no server module here on purpose. IndicTranslate runs on unmodified vLLM
(``scripts/mt/serve.sh`` is a wrapper, not a service), so the only thing worth
owning in-process is the prompt contract — which is what ``MTClient`` does.

Importing this pulls in ``openai`` lazily via the client module's constructor, so
the import itself stays cheap.

The re-export below is resolved on attribute access rather than at import time.
Eagerly importing ``.client`` here would put it in ``sys.modules`` before
``python -m bodhan_genai.mt.serving.client`` — the invocation the docs and
``serve.sh`` both print — re-executes it as ``__main__``, and Python warns about
the double import on every run.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bodhan_genai.mt.serving.client import DEFAULT_MODEL, MTClient

__all__ = ["DEFAULT_MODEL", "MTClient"]


def __getattr__(name: str):
    if name in __all__:
        from bodhan_genai.mt.serving import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
