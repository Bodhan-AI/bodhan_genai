"""Parse pages against a running IndicBlockOCR server.

    python -m bodhan_genai.ocr.serving.client pages/ -o out/

The server (``scripts/ocr/serve.sh``) is stock ``vllm serve`` holding the recognizer. Layout runs
here instead: it is a 33 MB-parameter detector that takes about a second per page on CPU, so the
GPU box only does the part that needs a GPU.

Everything between the two stages -- cropping, nested-equation dedup, matching transcriptions back
to blocks, markdown assembly -- is the same code the offline path runs. Only the recognizer
backend differs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bodhan_genai.ocr.engine.types import (
    CropConfig,
    DedupConfig,
    LayoutConfig,
    RecognizerConfig,
)
from bodhan_genai.ocr.serving.recognizer_http import DEFAULT_MODEL, HttpRecognizer

if TYPE_CHECKING:  # pragma: no cover
    from bodhan_genai.ocr.engine.types import PageResult

logger = logging.getLogger("ocr.serving.client")


def resolve_device(device: str = "auto") -> str:
    """``auto`` means the GPU when there is one. CPU is the fallback, not the goal.

    The layout model is small enough to run on CPU (~1s/page), which is what lets the client
    live away from the GPU box -- but a client that has a GPU should use it.
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class OCRClient:
    """Both stages, with the recognizer on the far end of an HTTP endpoint.

    The layout stage runs on the GPU when the client has one and falls back to CPU otherwise,
    which is what lets the client live away from the GPU box. Override with
    ``layout_config=LayoutConfig(device=...)``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        *,
        model: str = DEFAULT_MODEL,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        layout_ckpt: str | None = None,
        layout_config: LayoutConfig | None = None,
        recognizer_config: RecognizerConfig | None = None,
        dedup: DedupConfig | None = None,
        crop: CropConfig | None = None,
        num_workers: int = 32,
        strict: bool = True,
        client: Any | None = None,
    ) -> None:
        from bodhan_genai.ocr.engine.offline import IndicBlockOCR, IndicDocLayout

        self.recognizer_config = recognizer_config or RecognizerConfig()
        self.backend = HttpRecognizer(
            base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            config=self.recognizer_config,
            num_workers=num_workers,
            strict=strict,
            client=client,
        )
        if layout_config is None:
            layout_config = LayoutConfig(device=resolve_device())
        logger.info("layout stage on %s", layout_config.device)
        self.layout = IndicDocLayout(ckpt=layout_ckpt, config=layout_config, dedup=dedup)
        self.ocr = IndicBlockOCR(
            backend=self.backend,
            config=self.recognizer_config,
            dedup=dedup,
            crop=crop,
        )

    def detect(self, image_path: str) -> PageResult:
        """Stage 1 only. Touches no server."""
        return self.layout.detect(image_path)

    def parse(self, image_path: str) -> PageResult:
        return self.ocr.run(image_path, self.layout.detect(image_path))

    def parse_batch(self, image_paths: list[str]) -> list[PageResult]:
        """Pages in sequence; the crops within each page go out concurrently."""
        return [self.parse(p) for p in image_paths]

    def health(self) -> bool:
        """True when the endpoint is up and serving this model.

        Checks the served name rather than merely that something answered: on a shared box the
        port may belong to someone else's server.
        """
        try:
            models = self.backend._client.models.list()
        except Exception as exc:
            logger.warning("health check failed: %s", exc)
            return False
        return any(getattr(m, "id", None) == self.backend.model for m in models.data)

    def close(self) -> None:
        self.ocr.close()
        self.layout.close()

    def __enter__(self) -> OCRClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    from bodhan_genai.ocr.inference.common import collect_images, out_path, write_json

    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.serving.client",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("images", nargs="+", help="an image, or a folder of images")
    p.add_argument("-o", "--out-dir", default=None, help="default: beside each input")
    p.add_argument("--url", default="http://localhost:8000/v1", help="OpenAI API root")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--num-workers", type=int, default=32, help="crops in flight per page")
    p.add_argument("--layout-ckpt", default=None)
    p.add_argument(
        "--device", default="auto", help="device for the layout stage; auto uses a GPU if present"
    )
    p.add_argument("--save-layout", action="store_true", help="also write <name>.layout.json")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--table-format",
        choices=["html", "markdown"],
        default="html",
        help="html preserves merged cells; markdown cannot express them",
    )
    p.add_argument(
        "--best-effort",
        action="store_true",
        help="a failed crop yields empty text instead of failing the page",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # One INFO line per HTTP request buries the output on a large batch.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from bodhan_genai.ocr.templates.contract import TableFormat

    client = OCRClient(
        args.url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        layout_ckpt=args.layout_ckpt,
        layout_config=LayoutConfig(device=resolve_device(args.device)),
        recognizer_config=RecognizerConfig(
            max_tokens=args.max_tokens, table_format=TableFormat(args.table_format)
        ),
        num_workers=args.num_workers,
        strict=not args.best_effort,
    )

    if not client.health():
        return _no_server(args.url, args.model)

    import sys

    with client:
        for image in collect_images(args.images):
            page = client.layout.detect(str(image))
            if args.save_layout:
                path = out_path(image.stem, args.out_dir, ".layout.json", image.parent)
                write_json(path, page.as_record())
                print(
                    f"[layout] {image.name}: {len(page.blocks)} blocks -> {path}", file=sys.stderr
                )

            result = client.ocr.run(str(image), page)
            md = out_path(image.stem, args.out_dir, ".md", image.parent)
            js = out_path(image.stem, args.out_dir, ".json", image.parent)
            md.write_text(result.markdown or "", encoding="utf-8")
            write_json(js, result.as_record())
            transcribed = sum(1 for b in result.blocks if b.text)
            print(
                f"[ocr] {image.name}: {transcribed}/{len(result.blocks)} transcribed -> {md}, {js}",
                file=sys.stderr,
            )
    return 0


def _no_server(url: str, model: str) -> int:
    import sys

    print(f"no server serving {model!r} at {url}", file=sys.stderr)
    print("start one with: scripts/ocr/serve.sh", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
