"""The hosted Bodhan API backend — translation without a GPU.

The service exposes an **OpenAI-compatible** surface::

    GET  {base_url}/models             -> indic-translate, indic-transcribe, ...
    POST {base_url}/chat/completions   -> standard OpenAI request/response
    Authorization: Bearer <key>

Because it takes a rendered conversation, this backend is conversation-level like
the vLLM and HF ones: it reaches :class:`TranslationBackend` through
``PromptedBackend``, so the prompt contract still has exactly one call site and
the hosted path gets **every** capability the local ones have — native, Roman,
code-mix, transliteration, and script-qualified names.

.. note:: A flat REST route (``POST /translate`` taking ``{sentence, lang}``) was
   live earlier and returned 404 the same day, having never matched the published
   contract either. It was also strictly weaker: translation only, bare language
   names, no decoding controls and no truncation signal. This backend targets the
   OpenAI surface, which is the one ``/v1/models`` advertises.

Deliberately stdlib-only over ``urllib.request``. Neither ``requests`` nor
``httpx`` is a declared dependency of this package — both arrive transitively —
and the point of this path is that it works on a bare core install with no
extras, no GPU and no vLLM. The sibling ASR and TTS clients POST JSON the same
way. All HTTP imports live inside functions, so importing this module stays free.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from bodhan_genai.mt.engine.types import BackendOutput, MTSamplingConfig
from bodhan_genai.mt.templates.prompt import STOP_STRINGS

logger = logging.getLogger("mt.engine.api")

#: The OpenAI-compatible base. Includes the version segment, as the service
#: advertises it.
DEFAULT_BASE_URL = "https://console.bodhan.ai/v1"

#: The model id the service serves this package's checkpoint under.
DEFAULT_MODEL = "indic-translate"

CHAT_ROUTE = "/chat/completions"
MODELS_ROUTE = "/models"

#: Error codes a retry can never fix. The service may also signal a bad request
#: as a bare 4xx with a ``detail`` body and no code, so :func:`is_retryable`
#: treats 4xx-except-429 as terminal regardless.
_TERMINAL_CODES = frozenset(
    {"invalid_request_error", "unsupported_language", "invalid_script", "payload_too_large"}
)


class ApiError(RuntimeError):
    """A structured error from the hosted service.

    ``request_id`` is carried because it is the only field that makes a support
    report actionable, and it is the first thing lost if the message is flattened
    to a string too early.
    """

    def __init__(self, message: str, *, status: int, code: str = "", request_id: str = "") -> None:
        detail = f"HTTP {status}"
        if code:
            detail += f" {code}"
        if request_id:
            detail += f" (request_id={request_id})"
        super().__init__(f"{detail}: {message}")
        self.status = status
        self.code = code
        self.request_id = request_id
        self.message = message


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to retry, and how long to wait between attempts.

    ``max_attempts`` counts *attempts*, not retries: 4 means one try and three
    more.
    """

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0
    timeout_s: float = 120.0


def is_retryable(status: int, code: str, *, key_has_succeeded: bool) -> bool:
    """Whether a failed request is worth sending again.

    The rules, and why each is not the obvious default:

    * ``429`` and ``5xx`` — yes, with backoff. Standard.
    * ``401``/``403`` — **yes, but only if this key has already succeeded once.**
      A hosted deployment of this model has been observed returning auth errors
      for a *valid* key under high parallelism, and a client that retries only on
      429 records those as hard failures. A key that has never worked is simply
      wrong, and retrying it just burns the quota faster.
    * every other ``4xx`` — never. The request is malformed, unsupported or too
      big, and sending it again changes nothing.
    """
    if code in _TERMINAL_CODES:
        return False
    if status in (401, 403):
        return key_has_succeeded
    if status == 429:
        return True
    if 400 <= status < 500:
        return False
    return status >= 500


class ApiBackend:
    """Conversation-level backend that calls the hosted OpenAI-compatible service.

    ``post`` is injectable — any callable taking ``(url, body, headers, timeout)``
    and returning a decoded dict — so tests exercise the wire format without a
    socket, and a caller can swap the transport without touching this class.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        retry: RetryPolicy | None = None,
        num_workers: int = 32,
        post: Callable[..., dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "the hosted API needs a key. Export BODHAN_API_KEY, or pass "
                "backend='vllm' / 'hf' to run locally."
            )
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retry = retry or RetryPolicy()
        self.num_workers = max(1, num_workers)
        self._post = post or _post_json
        self._sleep = sleep
        # Drives the 401-is-retryable rule; see is_retryable.
        self._key_has_succeeded = False

    # -- request construction ---------------------------------------------- #

    def body_for(self, conversation: list[dict[str, str]], sc: MTSamplingConfig) -> dict[str, Any]:
        """Build the chat-completions body for one rendered conversation."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": sc.temperature,
            "max_tokens": sc.max_new_tokens,
            # <turn|> is already an EOS; sending it explicitly survives a
            # server-side template change and costs nothing.
            "stop": list(STOP_STRINGS),
        }
        # Passing a top_p/seed alongside greedy decoding is contradictory; the
        # server ignores them, but an honest request keeps logs readable.
        if not sc.greedy:
            body["top_p"] = sc.top_p
            if sc.seed is not None:
                body["seed"] = sc.seed
        return body

    # -- generation -------------------------------------------------------- #

    def generate_conversations(
        self, conversations: list[list[dict[str, str]]], sc: MTSamplingConfig
    ) -> list[BackendOutput]:
        """One output per conversation, in input order."""
        if not conversations:
            return []
        workers = min(self.num_workers, len(conversations))
        if workers == 1:
            return [self._one(c, sc) for c in conversations]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # pool.map preserves input order regardless of completion order.
            return list(pool.map(lambda c: self._one(c, sc), conversations))

    def _one(self, conversation: list[dict[str, str]], sc: MTSamplingConfig) -> BackendOutput:
        url = f"{self.base_url}{CHAT_ROUTE}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        payload = self._send_with_retry(url, self.body_for(conversation, sc), headers)
        elapsed = time.perf_counter() - started

        choices = payload.get("choices")
        if not choices:
            # A 200 with no choices is a FAILURE, not an empty translation.
            # Defaulting to "" hands back a blank string with ok=True and no
            # error -- the silent failure this package exists to avoid, and one a
            # well-behaved stub never reproduces.
            raise ApiError(
                f"200 response carried no choices (keys: {sorted(payload)})",
                status=200,
                code="malformed_response",
            )
        choice = choices[0]
        usage = payload.get("usage") or {}
        return BackendOutput(
            text=((choice.get("message") or {}).get("content") or "").strip(),
            # "length" is the only signal the budget was hit; a truncated
            # translation is otherwise fluent and simply stops.
            truncated=choice.get("finish_reason") == "length",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            generated_tokens=int(usage.get("completion_tokens") or 0),
            gen_time_s=elapsed,
        )

    def _send_with_retry(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retry.max_attempts):
            try:
                payload = self._post(url, body, headers, self.retry.timeout_s)
            except ApiError as exc:
                last = exc
                retryable = is_retryable(
                    exc.status, exc.code, key_has_succeeded=self._key_has_succeeded
                )
                if not retryable or attempt == self.retry.max_attempts - 1:
                    raise
                self._sleep(self._delay(attempt, getattr(exc, "retry_after", None)))
                continue
            self._key_has_succeeded = True
            return payload
        raise last if last else RuntimeError("unreachable")

    def _delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with jitter, honouring ``Retry-After`` when given.

        The jitter is load-bearing, not decoration: at 32-way concurrency an
        unjittered exponential backoff resynchronises every worker into one
        thundering herd against the endpoint that just rate-limited us.
        """
        if retry_after is not None:
            return min(float(retry_after), self.retry.max_delay_s)
        raw = self.retry.base_delay_s * (2**attempt)
        return min(raw, self.retry.max_delay_s) * (0.5 + random.random())

    # -- health ------------------------------------------------------------ #

    def health(self) -> bool:
        """True when the service lists :attr:`model`.

        Checks the served model name, not just a 200, so another model behind the
        same gateway cannot look like success.
        """
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}{MODELS_ROUTE}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                listed = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError) as exc:
            logger.warning("health check failed: %s", exc)
            return False
        return any(m.get("id") == self.model for m in listed.get("data", []))

    def close(self) -> None:
        """No persistent resources to release."""


def _post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """POST JSON and decode the reply, raising :class:`ApiError` on failure."""
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        message, code, request_id = raw[:500], "", ""
        try:
            payload = json.loads(raw)
            err = payload.get("error")
            if isinstance(err, dict):  # OpenAI-shaped error
                message = err.get("message", message)
                code = err.get("code") or err.get("type") or ""
                request_id = err.get("request_id", "")
            elif isinstance(err, str):
                message = err
            elif "detail" in payload:  # FastAPI-shaped error
                detail = payload["detail"]
                message = detail if isinstance(detail, str) else json.dumps(detail)[:500]
        except (ValueError, AttributeError):
            pass  # a non-JSON error body is still an error; keep the raw text
        api_error = ApiError(message, status=exc.code, code=code, request_id=request_id)
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            with contextlib.suppress(ValueError):
                api_error.retry_after = float(retry_after)  # type: ignore[attr-defined]
        raise api_error from None
    except urllib.error.URLError as exc:
        # Connection-level failure: no status, so treat it as a retryable 5xx.
        raise ApiError(str(exc.reason), status=503, code="internal_error") from None
