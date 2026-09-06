"""Typed client for a IndicTranslate server.

IndicTranslate serves on stock ``vllm serve`` — there is no custom server to run, and no
custom wire protocol. What there *is* is a prompt contract, and a client that
hand-rolls its own ``messages`` payload will silently get worse translations. This
client owns that contract so callers never see it:

    from bodhan_genai.mt.serving import MTClient

    client = MTClient("http://localhost:8000/v1")
    print(client.translate("Hello world", tgt_lang="hin_Deva").text)

Start a server with ``scripts/mt/serve.sh``.

Also runnable directly::

    python -m bodhan_genai.mt.serving.client --tgt-lang Hindi --text "Hello world"
    python -m bodhan_genai.mt.serving.client --tgt-lang Tamil \\
        --input-file segments.txt --output-file out.jsonl --num-workers 32
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bodhan_genai.mt.engine.types import MTResult, MTSamplingConfig
from bodhan_genai.mt.templates.prompt import (
    STOP_STRINGS,
    build_conversation,
    resolve_language,
)

logger = logging.getLogger("mt.serving.client")

DEFAULT_MODEL = "indic_translate"


class MTClient:
    """Translate through an OpenAI-compatible IndicTranslate endpoint.

    ``base_url`` is the ``/v1`` root, e.g. ``http://localhost:8000/v1``. vLLM does
    not check the API key, so the default placeholder is fine.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        sampling: MTSamplingConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.sampling = sampling or MTSamplingConfig()
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    # -- single ------------------------------------------------------------ #

    def translate(
        self,
        text: str,
        *,
        tgt_lang: str,
        src_lang: str | None = None,
        **overrides,
    ) -> MTResult:
        """Translate one segment. Transport errors land in ``MTResult.error``."""
        tgt_name = resolve_language(tgt_lang)
        sc = self.sampling.merged(**overrides)
        result = MTResult(source=text, src_lang=src_lang, tgt_lang=tgt_name)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": build_conversation(text, tgt_lang),
            "temperature": sc.temperature,
            "max_tokens": sc.max_new_tokens,
            # <turn|> is already an EOS; sending it as a stop is harmless and
            # slightly more robust against a template change on the server.
            "extra_body": {"stop": STOP_STRINGS},
        }
        if not sc.greedy:
            body["top_p"] = sc.top_p
            if sc.seed is not None:
                body["seed"] = sc.seed

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**body)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        result.gen_time_s = time.perf_counter() - started

        result.text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        if usage is not None:
            result.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            result.generated_tokens = getattr(usage, "completion_tokens", 0) or 0
        return result

    # -- batch ------------------------------------------------------------- #

    def translate_batch(
        self,
        texts: list[str],
        *,
        tgt_lang: str,
        src_lang: str | None = None,
        num_workers: int = 32,
        **overrides,
    ) -> list[MTResult]:
        """Translate many segments concurrently, preserving input order.

        The server batches internally; concurrency here is only about keeping its
        queue fed. Results stay aligned with ``texts`` regardless of completion
        order, so a failed row never shifts the ones after it.
        """
        resolve_language(tgt_lang)  # fail fast, once, before fanning out
        if not texts:
            return []

        def _one(text: str) -> MTResult:
            return self.translate(text, tgt_lang=tgt_lang, src_lang=src_lang, **overrides)

        with ThreadPoolExecutor(max_workers=max(1, min(num_workers, len(texts)))) as pool:
            return list(pool.map(_one, texts))

    # -- health ------------------------------------------------------------ #

    def health(self) -> bool:
        """True when the endpoint is up and serving ``self.model``.

        Checks the served model name, not just that something answered: on a
        shared box the port may belong to someone else's server.
        """
        try:
            models = self._client.models.list()
        except Exception as exc:
            logger.warning("health check failed: %s", exc)
            return False
        return any(getattr(m, "id", None) == self.model for m in models.data)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    from bodhan_genai.mt.inference.common import (
        add_io_args,
        add_language_args,
        add_sampling_args,
        print_languages,
        read_inputs,
        sampling_from_args,
        write_results,
    )

    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.serving.client",
        description="Translate through a running IndicTranslate server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", default="http://localhost:8000/v1", help="OpenAI API root")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--num-workers", type=int, default=32)
    add_language_args(p)
    add_io_args(p)
    add_sampling_args(p)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # One INFO line per HTTP request buries the output on a large batch.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_languages:
        print_languages()
        return 0
    if not args.tgt_lang:
        p.error("--tgt-lang is required (or use --list-languages)")

    client = MTClient(
        args.url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        sampling=sampling_from_args(args),
    )
    if not client.health():
        logger.error(
            "no server serving %r at %s — start one with scripts/mt/serve.sh",
            args.model,
            args.url,
        )
        return 1

    texts = read_inputs(args)
    results = client.translate_batch(
        texts,
        tgt_lang=args.tgt_lang,
        src_lang=args.src_lang,
        num_workers=args.num_workers,
    )
    write_results(results, args.output_file)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
