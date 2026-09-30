"""Which backend to use, and how that decision is made.

Three backends translate: the hosted Bodhan API (``"api"``), in-process vLLM
(``"vllm"``) and HuggingFace ``generate()`` (``"hf"``).

The default is **token-driven**: a caller who has a Bodhan API key gets the
hosted service and needs no GPU, no checkpoint and no extras; a caller who does
not falls back to vLLM. An explicit ``backend=`` always wins over both, so
nothing that names a backend today changes behaviour.

stdlib-only imports — resolving a backend must not drag in vllm, torch or an HTTP
library. Asserted by ``tests/mt/test_mt_lazy_import.py``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Environment variables consulted, in order. The first one holding a non-empty
#: value selects the hosted API.
#:
#: Two spellings are accepted on purpose. The published contract's examples use
#: ``BODHAN_API_KEY``; ``BODHAN_API_TOKEN`` is the spelling people reach for
#: anyway, and silently falling back to vLLM because someone exported the other
#: one is a bad failure — it looks like a missing GPU, not a missing variable.
API_KEY_ENV_VARS = ("BODHAN_API_KEY", "BODHAN_API_TOKEN")

#: Every backend name the engine accepts.
BACKEND_NAMES = ("api", "vllm", "hf")

#: Used when no backend is named and no API token is present.
FALLBACK_BACKEND = "vllm"


def find_api_key(env: Mapping[str, str] | None = None) -> str | None:
    """Return the Bodhan API key from the environment, or ``None``.

    ``env`` is injectable so tests never touch the ambient environment.

    An exported-but-empty variable counts as absent. That case is the whole
    reason this is a function: ``BODHAN_API_KEY=""`` in a CI job would otherwise
    select the hosted backend and then fail every request on a 401, which reads
    as a broken endpoint rather than an unset secret.
    """
    environ = os.environ if env is None else env
    for name in API_KEY_ENV_VARS:
        value = environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_backend_name(
    explicit: str | None = None, *, env: Mapping[str, str] | None = None
) -> str:
    """Resolve the backend to use.

    Order: an explicit name wins; else the hosted API when a key is present; else
    :data:`FALLBACK_BACKEND`.

    >>> resolve_backend_name("hf", env={"BODHAN_API_KEY": "k"})
    'hf'
    >>> resolve_backend_name(env={"BODHAN_API_KEY": "k"})
    'api'
    >>> resolve_backend_name(env={})
    'vllm'
    >>> resolve_backend_name(env={"BODHAN_API_KEY": ""})
    'vllm'
    """
    if explicit is not None:
        if explicit not in BACKEND_NAMES:
            raise ValueError(
                f"Unknown backend {explicit!r} "
                f"(expected {', '.join(repr(n) for n in BACKEND_NAMES)} or an instance)"
            )
        return explicit
    return "api" if find_api_key(env) is not None else FALLBACK_BACKEND
