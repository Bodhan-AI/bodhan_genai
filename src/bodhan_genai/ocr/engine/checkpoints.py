"""Locating the two sets of weights.

Both live in one Hub repo, under ``weights/layout`` and ``weights/ocr``. One repo because the two
models are co-validated -- the published scores describe the pair plus a recipe, so a revision
should be a coherent snapshot of the whole pipeline rather than two artifacts that can drift apart.

Either can still be used or finetuned alone, via ``subfolder=``:

    AutoModelForImageTextToText.from_pretrained(REPO, subfolder="weights/ocr")     # no flag needed
    AutoModelForObjectDetection.from_pretrained(REPO, subfolder="weights/layout",
                                                trust_remote_code=True)

First hit wins: explicit path -> environment variable -> bundled ``weights/<sub>/`` -> download.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_REPO = "bodhan-ai/indic-ocr"
HF_REPO_ENV = "BODHAN_OCR_HF_REPO"

#: Published layout of the repo -- also the layout of a local ``weights/`` directory.
SUBDIRS = {"layout": "layout", "recognizer": "ocr"}

#: Per-stage path override, so a finetune of one model can be swapped in on its own.
ENV_VARS = {"layout": "BODHAN_OCR_LAYOUT_CKPT", "recognizer": "BODHAN_OCR_RECOGNIZER_CKPT"}

_REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_ckpt(stage: str, explicit: str | None = None) -> str:
    """Weights directory for ``stage`` ("layout" or "recognizer").

    Nothing is downloaded until this is called, so import and ``--help`` never touch the network.
    """
    if stage not in SUBDIRS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(SUBDIRS)}")
    if explicit:
        return explicit

    override = os.environ.get(ENV_VARS[stage])
    if override:
        # Fail loudly rather than silently downloading 1.7 GB because a path had a typo.
        if not os.path.isdir(override):
            raise FileNotFoundError(
                f"{ENV_VARS[stage]} points at {override!r}, which is not a directory"
            )
        return override

    sub = SUBDIRS[stage]
    bundled = _REPO_ROOT / "weights" / sub
    if bundled.is_dir():
        return str(bundled)

    from huggingface_hub import snapshot_download

    # Only this stage's weights: pulling the layout model should not drag in 1.7 GB of recognizer.
    root = snapshot_download(
        os.environ.get(HF_REPO_ENV, DEFAULT_HF_REPO), allow_patterns=[f"weights/{sub}/*"]
    )
    return os.path.join(root, "weights", sub)
