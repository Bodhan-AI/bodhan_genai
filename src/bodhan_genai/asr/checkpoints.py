# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Resolving an IndicTranscribe checkpoint, and the files inside it.

``IndicTranscribeForConditionalGeneration`` inherits the transformers resolver and accepts a Hub
repo id for free. The tokenizer and feature extractor do not: they read files with
``sentencepiece`` and ``safetensors``, both of which are pure filesystem. ``resolve_file`` closes
that gap so all three ``from_pretrained`` classmethods accept a directory *or* a repo id, which
is the convention callers already expect.

Nothing here touches the network until a file is actually missing locally, so importing this
module — and therefore ``--help`` — stays offline.

Deliberately NOT under ``engine/``: that subpackage's ``__init__`` imports the engines eagerly,
so anything living there drags torch in. This sits directly under ``bodhan_genai.asr``, whose
``__init__`` is a PEP 562 lazy table.
"""

from __future__ import annotations

import os

#: The published checkpoint. Public on the Hub, so it resolves without credentials.
DEFAULT_HF_REPO = "bodhan-ai/indic-transcribe-core"

#: Repo-level override, mirroring BODHAN_OCR_HF_REPO on the OCR side. Honoured ONLY inside a
#: deployment image -- see :func:`_in_deployment_image`.
HF_REPO_ENV = "BODHAN_ASR_HF_REPO"

#: Set by docker/*/Dockerfile*, and nowhere else. Its presence means "weights are bundled or
#: mounted into this image, and the environment is how they are addressed".
DEPLOYMENT_ENV = "BODHAN_GENAI_DEPLOYMENT"


def _in_deployment_image() -> bool:
    """True only inside one of this repo's deployment images.

    Checkpoint paths must not move because of an inherited environment variable. A stale
    ``BODHAN_ASR_HF_REPO`` in someone's shell silently loading different weights is a
    correctness bug that looks like a model regression, so outside a deployment image the
    environment is not consulted at all: the explicit argument decides, or the default does.

    Deliberately duplicated in ``bodhan_genai.ocr.engine.checkpoints``: ``bodhan_genai`` is a
    bare PEP 420 namespace with no shared module, and no modality imports another.
    """
    return os.environ.get(DEPLOYMENT_ENV) == "1"


def resolve_ckpt(explicit: str | None = None) -> str:
    """Pick the checkpoint identifier: explicit -> the published default.

    Returns a directory path or a Hub repo id, whichever was selected — it does not download,
    and does not check existence. Loaders resolve individual files through :func:`resolve_file`.

    The environment is consulted **only** inside a deployment image. Everywhere else the same
    arguments always resolve to the same weights.
    """
    if explicit:
        return explicit
    if _in_deployment_image():
        from_env = os.environ.get(HF_REPO_ENV)
        if from_env:
            return from_env
    return DEFAULT_HF_REPO


def resolve_file(repo_or_dir: str, filename: str, **kwargs) -> str:
    """Local path to ``filename``, whether ``repo_or_dir`` is a directory or a Hub repo id.

    Delegates to ``transformers.utils.cached_file`` — the same resolver
    ``PretrainedConfig._get_config_dict`` uses — so the tokenizer and feature extractor inherit
    exactly what the model class gets for free: the HF cache, ``revision``, ``subfolder``,
    ``token``, proxies, and ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``. ``kwargs`` are
    forwarded, so a caller can pin a revision the same way they would anywhere else in the
    transformers API.

    Raises:
        FileNotFoundError: if the file cannot be resolved. The message names the likely cause,
            because the Hub answers 404 rather than 403 for a private repo a token cannot read
            and the default checkpoint is private.
    """
    from transformers.utils import cached_file

    try:
        return cached_file(repo_or_dir, filename, **kwargs)
    except Exception as exc:
        raise FileNotFoundError(
            f"could not resolve {filename!r} from {repo_or_dir!r}. If that is a directory it is "
            f"an incomplete checkpoint. If it is a Hub repo id, note the bodhan-ai repos are "
            f"private and the Hub returns 404 (not 403) for a repo your token cannot read, so "
            f"this is usually a missing or unauthorised HF_TOKEN. Pass a local directory, or "
            f"set {HF_REPO_ENV}."
        ) from exc
