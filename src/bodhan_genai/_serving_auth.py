# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""HTTP Basic auth for the Ray Serve deployments, shared by IndicSpeak and IndicTranscribe.

Lives at the namespace root rather than in either modality because duplicated access control
is a liability: a fix applied to one copy and forgotten in the other is exactly the bug you do
not want in the thing standing between a network and eight H100s.

That placement is deliberate and does not weaken the package layout. ``bodhan_genai`` stays a
bare PEP 420 namespace — this adds a module, not an ``__init__.py`` — and this is not a
modality, so the enforced boundary (no modality imports another) is untouched.

Off by default -- see :func:`install_basic_auth`. This is a reference serving setup meant to be
copied and extended, so a server starts with no arguments; opting in is one environment variable.

The two vLLM-backed servers do **not** use this. IndicTranslate and IndicOCR run stock
``vllm serve``, which has its own bearer-token auth (``VLLM_API_KEY`` / ``--api-key``), equally
optional. Reimplementing it here would mean sitting a second auth layer in front of one that
already works.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets

logger = logging.getLogger("bodhan_genai.serving.auth")


class BasicAuthMiddleware:
    """HTTP Basic auth over every route except the health probe.

    Written as raw ASGI rather than a BaseHTTPMiddleware subclass because the latter only sees
    ``http`` scopes -- a streaming websocket would sail straight past it.
    """

    def __init__(self, app, user: str, password: str, exempt=("/health",), realm: str = "Bodhan"):
        self.app = app
        creds = f"{user}:{password}".encode()
        # bytes, not str: compare_digest rejects non-ASCII str, so a header carrying any byte
        # >= 0x80 raised TypeError and surfaced as a 500 -- an unauthenticated caller could
        # tell the middleware apart from a genuine 401 that way.
        self._expected = b"Basic " + base64.b64encode(creds)
        self.exempt = frozenset(exempt)
        self._realm = realm

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("path") in self.exempt:
            return await self.app(scope, receive, send)

        offered = b""
        for key, value in scope.get("headers", ()):
            if key == b"authorization":
                offered = value
                break
        # compare_digest, not ==, so a wrong password cannot be recovered by timing how far the
        # comparison got.
        if secrets.compare_digest(offered, self._expected):
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            await receive()  # drain websocket.connect before refusing
            # Closing before accept is the only refusal available pre-handshake. The client
            # never sees this code: an ASGI server (and Ray Serve's proxy) turns it into an
            # HTTP 403 on the connect, which is what a websocket client reports.
            return await send({"type": "websocket.close", "code": 1008})

        body = b"unauthorized\n"
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"www-authenticate", f'Basic realm="{self._realm}"'.encode()),
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install_basic_auth(app, *, prefix: str, realm: str) -> None:
    """Add HTTP Basic auth to ``app`` **if** the operator asked for it.

    Opt-in, and off by default. The servers here are a minimal reference setup: the documented
    main path is offline inference, and a server you can start with no arguments is the point.
    Requiring a credential to run ``scripts/tts/serve.sh`` made the first thing a new user sees
    an error message about an environment variable.

    So: set ``{prefix}_AUTH_FILE`` to a readable ``user:password`` file and every route except
    ``/health`` requires it; leave it unset and the server is open. Keep the credential outside
    the repo -- it is then never committed and never passed through Ray's ``runtime_env``, only
    the path travels.

    A *malformed* or unreadable file is still fatal, because that is someone trying to enable
    auth and failing. Silently serving open in that case is the one behaviour nobody wants.

    Being open is a real exposure, not a footnote: these deployments bind ``0.0.0.0``. Basic auth
    is also only as private as the transport -- base64 is not encryption, so on plain HTTP the
    credential is readable by anything on the path. Put a reverse proxy with TLS in front, or
    bind to localhost and tunnel, for anything beyond a trusted network.

    Call this from the deployment builder, never at module scope: importing a service module
    should not read the environment, or every reader of that module -- tooling, ``python -c``, an
    editor -- inherits a check that only matters when something is actually served.
    """
    file_var = f"{prefix}_AUTH_FILE"

    path = os.environ.get(file_var, "").strip()
    if not path:
        logger.info(
            "[%s.auth] %s is unset - endpoints are OPEN (set it to a 'user:password' file "
            "to require authentication)",
            prefix.lower(),
            file_var,
        )
        return
    try:
        with open(path) as fh:
            user, _, password = fh.read().strip().partition(":")
    except OSError as e:
        raise RuntimeError(f"{file_var}={path!r} is unreadable: {e}") from e
    if not user or not password:
        raise RuntimeError(f"{file_var}={path!r} is not 'user:password'")
    app.add_middleware(BasicAuthMiddleware, user=user, password=password, realm=realm)
    logger.info("[%s.auth] basic auth enabled for user %r", prefix.lower(), user)
