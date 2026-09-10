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

#: Per-stage path override, so a finetune of one model can be swapped in on its own. Honoured
#: ONLY inside a deployment image -- see :func:`_in_deployment_image`.
ENV_VARS = {"layout": "BODHAN_OCR_LAYOUT_CKPT", "recognizer": "BODHAN_OCR_RECOGNIZER_CKPT"}

#: Set by docker/*/Dockerfile*, and nowhere else. Its presence means "weights are bundled or
#: mounted into this image, and the environment is how they are addressed".
DEPLOYMENT_ENV = "BODHAN_GENAI_DEPLOYMENT"

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _in_deployment_image() -> bool:
    """True only inside one of this repo's deployment images.

    Checkpoint paths must not move because of an inherited environment variable, or because a
    ``weights/`` directory happens to exist. A stale ``BODHAN_OCR_LAYOUT_CKPT`` silently loading
    different weights is a correctness bug that presents as a model regression, so outside a
    deployment image neither the environment nor a bundled directory is consulted: the explicit
    argument decides, or the published default does.

    Inside the image both are how mounted weights at ``/models`` are addressed, which is why the
    marker exists rather than a blanket removal.

    Deliberately duplicated in ``bodhan_genai.asr.checkpoints``: ``bodhan_genai`` is a bare
    PEP 420 namespace with no shared module, and no modality imports another.
    """
    return os.environ.get(DEPLOYMENT_ENV) == "1"


def resolve_ckpt(stage: str, explicit: str | None = None) -> str:
    """Weights directory for ``stage`` ("layout" or "recognizer").

    Nothing is downloaded until this is called, so import and ``--help`` never touch the network.

    Resolution order is ``explicit -> published default``. The two implicit routes -- the
    per-stage environment variables and a bundled ``weights/`` directory -- apply **only** inside
    a deployment image, so the same arguments always load the same weights everywhere else.
    """
    if stage not in SUBDIRS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(SUBDIRS)}")
    if explicit:
        return explicit

    sub = SUBDIRS[stage]
    repo = DEFAULT_HF_REPO

    if _in_deployment_image():
        override = os.environ.get(ENV_VARS[stage])
        if override:
            # Fail loudly rather than silently downloading 1.7 GB because a path had a typo.
            if not os.path.isdir(override):
                raise FileNotFoundError(
                    f"{ENV_VARS[stage]} points at {override!r}, which is not a directory"
                )
            return override

        bundled = _REPO_ROOT / "weights" / sub
        if bundled.is_dir():
            return str(bundled)

        repo = os.environ.get(HF_REPO_ENV, DEFAULT_HF_REPO)

    from huggingface_hub import snapshot_download

    # Only this stage's weights: pulling the layout model should not drag in 1.7 GB of recognizer.
    root = snapshot_download(repo, allow_patterns=[f"weights/{sub}/*"])
    return os.path.join(root, "weights", sub)
