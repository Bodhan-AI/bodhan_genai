"""IndicOCR -- page image in, reading-ordered Markdown and per-block JSON out.

    import sys
    from huggingface_hub import snapshot_download

    repo = snapshot_download("bodhan-ai/indic-ocr")
    sys.path.insert(0, repo)                       # the code ships in the repo
    from indic_ocr import IndicOCR

    parser = IndicOCR.from_pretrained(repo)
    print(parser("page.png"))                      # markdown

``sys.path.insert`` is needed because ``snapshot_download`` returns a cache directory, which is
not importable on its own. With the path added, this is ordinary Python -- no
``trust_remote_code``, and nothing of ours to install.

The pipeline is two models, both under ``weights/`` here, and either can be used on its own.
Nothing needs ``trust_remote_code``.

The recognizer is a stock Qwen3.5 checkpoint:

    AutoModelForImageTextToText.from_pretrained(repo, subfolder="weights/ocr")

The detector loads through ``AutoModelForObjectDetection`` for inference. To finetune it, use
``PPDocLayoutV3Trainable`` instead: the reading-order loss lives on that subclass, and the stock
class trains without it.

    from iocr_model_ppdoc import PPDocLayoutV3Trainable   # once repo is on sys.path
    PPDocLayoutV3Trainable.from_pretrained(f"{repo}/weights/layout")
"""

from __future__ import annotations

from pathlib import Path

from iocr_offline import IndicBlockOCR as _BlockOCR
from iocr_offline import IndicDocLayout as _DocLayout
from iocr_types import CropConfig, DedupConfig, LayoutConfig, RecognizerConfig, TableFormat

__all__ = ["IndicOCR"]


class IndicOCR:
    """Both stages. Construct with :meth:`from_pretrained`, then call it on a page image."""

    def __init__(
        self,
        path: str | Path,
        device: str = "cuda",
        table_format: str = "html",
        dedup_mode: str = "both",
        contain: float = 0.90,
        min_px_side: int = 256,
        max_new_tokens: int = 2048,
    ) -> None:
        self.path = Path(path)
        self._device = device
        self._layout_cfg = LayoutConfig(device=device)
        self._dedup = DedupConfig(mode=dedup_mode, contain=contain)
        self._crop = CropConfig(min_px_side=min_px_side)
        self._rec_cfg = RecognizerConfig(
            max_tokens=max_new_tokens, table_format=TableFormat(table_format)
        )
        # Both stages are built on first use, so detect() never loads the 1.7 GB recognizer.
        self._layout = None
        self._ocr = None

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None, **kwargs) -> IndicOCR:
        """Load from a downloaded snapshot. Defaults to the directory this file lives in, which
        is the snapshot itself -- so ``IndicOCR.from_pretrained()`` also works."""
        return cls(Path(path) if path else Path(__file__).resolve().parent, **kwargs)

    # -- stages ------------------------------------------------------------ #

    @property
    def layout(self):
        if self._layout is None:
            self._layout = _DocLayout(
                ckpt=str(self.path / "weights" / "layout"),
                config=self._layout_cfg,
                dedup=self._dedup,
            )
        return self._layout

    @property
    def recognizer(self):
        if self._ocr is None:
            from iocr_recognizer import HfRecognizer

            self._ocr = _BlockOCR(
                backend=HfRecognizer(
                    ckpt=str(self.path / "weights" / "ocr"),
                    config=self._rec_cfg,
                    device=self._device,
                ),
                config=self._rec_cfg,
                dedup=self._dedup,
                crop=self._crop,
            )
        return self._ocr

    # -- public API -------------------------------------------------------- #

    def detect(self, image_path: str) -> dict:
        """Stage 1 only -- blocks, labels, reading order. Loads no recognizer."""
        return self.layout.detect(image_path).as_record()

    def parse(self, image_path: str) -> dict:
        """Both stages -> ``{image, width, height, blocks, markdown}``."""
        page = self.recognizer.run(image_path, self.layout.detect(image_path))
        return {**page.as_record(), "markdown": page.markdown}

    def __call__(self, image_path: str) -> str:
        """The markdown for a page. ``parse()`` if you also want the per-block JSON."""
        return self.parse(image_path)["markdown"]
