"""Optional auth on the IndicSpeak deployment.

Off by default — a reference server starts with no arguments. The mechanism is shared with
IndicTranscribe (``bodhan_genai._serving_auth``) precisely so it cannot drift; these tests pin
the IndicSpeak half of the contract — unset is open, a broken credential file is still fatal,
and it covers the websocket routes, which is the whole reason it is raw ASGI rather than a
FastAPI dependency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

from bodhan_genai._serving_auth import BasicAuthMiddleware
from bodhan_genai.tts.serving.service import _install_basic_auth


def test_importing_the_service_module_needs_no_auth_env(monkeypatch):
    """Importing must be free; only *serving* requires a credential.

    If the check ran at module scope, every reader of the module — tooling, ``python -c``, an
    editor — would break for something that only matters when a server is started.
    """
    monkeypatch.delenv("TTS_AUTH_FILE", raising=False)
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-c", "import bodhan_genai.tts.serving.service"],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(repo_root / "src"),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
        },
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr


def test_unset_means_open(monkeypatch):
    """The default is an open server, with no middleware attached at all."""
    monkeypatch.delenv("TTS_AUTH_FILE", raising=False)
    app = FastAPI()
    _install_basic_auth(app)
    assert not any("BasicAuth" in str(m) for m in app.user_middleware)


def test_a_malformed_credential_file_is_rejected(monkeypatch, tmp_path):
    bad = tmp_path / "creds"
    bad.write_text("no-colon-here")
    monkeypatch.setenv("TTS_AUTH_FILE", str(bad))
    with pytest.raises(RuntimeError, match="not 'user:password'"):
        _install_basic_auth(FastAPI())


def test_an_unreadable_credential_file_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_AUTH_FILE", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="unreadable"):
        _install_basic_auth(FastAPI())


@pytest.mark.parametrize("scope_type", ["http", "websocket"])
def test_unauthenticated_requests_never_reach_the_app(scope_type):
    """Both transports must be refused.

    A BaseHTTPMiddleware subclass only sees ``http`` scopes, so ``WS /tts`` would sail past it —
    which is why the middleware is raw ASGI. This asserts the websocket half specifically.
    """
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": f"{scope_type}.request" if scope_type == "http" else "websocket.connect"}

    async def app(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("unauthenticated request reached the app")

    mw = BasicAuthMiddleware(app, user="u", password="p")
    scope = {"type": scope_type, "path": "/tts", "headers": []}

    import asyncio

    asyncio.run(mw(scope, receive, send))

    if scope_type == "websocket":
        assert sent and sent[0]["type"] == "websocket.close"
        assert sent[0]["code"] == 1008
    else:
        assert sent and sent[0]["status"] == 401
        assert any(k == b"www-authenticate" for k, _ in sent[0]["headers"])


def test_health_is_exempt_so_probes_still_work():
    """An orchestrator's liveness probe cannot carry credentials."""
    reached = []

    async def app(scope, receive, send):
        reached.append(scope["path"])

    async def send(message):  # pragma: no cover - not reached for an exempt path
        raise AssertionError("exempt path produced a response from the middleware")

    async def receive():
        return {"type": "http.request"}

    import asyncio

    mw = BasicAuthMiddleware(app, user="u", password="p")
    asyncio.run(mw({"type": "http", "path": "/health", "headers": []}, receive, send))
    assert reached == ["/health"]


def test_a_non_ascii_header_is_401_not_500():
    """compare_digest rejects non-ASCII str, so a header with any byte >= 0x80 used to raise
    TypeError — a 500 that told an unauthenticated caller the middleware was there."""
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    async def app(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("unauthenticated request reached the app")

    import asyncio

    mw = BasicAuthMiddleware(app, user="u", password="p")
    scope = {"type": "http", "path": "/tts/offline", "headers": [(b"authorization", b"Basic \xff")]}
    asyncio.run(mw(scope, receive, send))
    assert sent and sent[0]["status"] == 401


def test_the_correct_credential_is_let_through(tmp_path):
    import base64

    reached = []

    async def app(scope, receive, send):
        reached.append(scope["path"])

    async def send(message):  # pragma: no cover - authorised, so no 401 is produced
        raise AssertionError("an authorised request produced a middleware response")

    async def receive():
        return {"type": "http.request"}

    import asyncio

    mw = BasicAuthMiddleware(app, user="u", password="p")
    header = b"Basic " + base64.b64encode(b"u:p")
    scope = {"type": "http", "path": "/tts/offline", "headers": [(b"authorization", header)]}
    asyncio.run(mw(scope, receive, send))
    assert reached == ["/tts/offline"]


def test_the_installer_wires_the_file_credential_and_the_indicspeak_realm(monkeypatch, tmp_path):
    """The installer half, driven through a real middleware stack.

    The tests above exercise the shared mechanism; this one pins what IndicSpeak asks of it --
    the credential comes from the file, and the realm names this model rather than the default.
    """
    from fastapi.testclient import TestClient

    creds = tmp_path / "creds"
    creds.write_text("alice:s3cret\n")
    monkeypatch.setenv("TTS_AUTH_FILE", str(creds))

    app = FastAPI()
    _install_basic_auth(app)
    client = TestClient(app)

    refused = client.get("/tts/offline")
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"] == 'Basic realm="IndicSpeak"'

    # 404, not 401: no such route on a bare app, which is the point -- it got past auth.
    assert client.get("/tts/offline", auth=("alice", "s3cret")).status_code == 404
