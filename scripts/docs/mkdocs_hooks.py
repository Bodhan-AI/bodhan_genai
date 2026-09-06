"""MkDocs hooks.

Silences one griffe warning class so the docs can build under ``--strict``.

``--strict`` promotes every warning to an error, which is what we want for broken
links and missing nav entries. It also catches griffe's "No type or annotation for
parameter" on the ``tokenizer`` arguments in ``bodhan_genai.tts.templates``. Those
are unannotated deliberately: the only honest annotation is a transformers type,
and that module has to stay importable without the GPU stack. Suppress exactly
that message and nothing else, so a real problem still fails the build.

The filter is attached to the *handlers* rather than to a logger. A
``logging.Filter`` on a logger only sees records logged through that logger, not
records propagated up from its children — and griffe logs through a child. MkDocs
implements ``--strict`` with a counting handler on the root logger, so filtering
there is both where the records actually arrive and where the strict-mode tally is
kept.
"""

from __future__ import annotations

import logging

_SUPPRESSED = "No type or annotation for parameter"


class _DropMissingAnnotation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _SUPPRESSED not in record.getMessage()


def _install(filt: logging.Filter) -> None:
    for logger in (logging.getLogger(), logging.getLogger("mkdocs")):
        for handler in logger.handlers:
            handler.addFilter(filt)


def on_startup(**_kwargs) -> None:
    _install(_DropMissingAnnotation())


def on_config(config, **_kwargs):
    # MkDocs installs its counting handler after on_startup, so re-attach here.
    _install(_DropMissingAnnotation())
    return config
