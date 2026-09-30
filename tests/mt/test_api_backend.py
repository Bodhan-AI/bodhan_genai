"""The hosted Bodhan API backend, against a stub transport.

These pin the wire format of the **live** service: an OpenAI-compatible
`POST {base}/chat/completions` carrying a rendered conversation. Every shape
asserted here was checked against the real endpoint first.

They exist because most mistakes this backend can make are silent — a dropped
stop string, a lost truncation flag, or a retried 4xx all come back looking like
ordinary output. One such bug did get through, precisely because an earlier stub
was too well-behaved: see `test_a_200_without_choices_is_an_error`.
"""

from __future__ import annotations

import pytest

from bodhan_genai.mt.engine.api import (
    CHAT_ROUTE,
    DEFAULT_MODEL,
    ApiBackend,
    ApiError,
    RetryPolicy,
    is_retryable,
)
from bodhan_genai.mt.engine.offline import PromptedBackend, TranslationBackend
from bodhan_genai.mt.engine.types import MTRequest, MTSamplingConfig
from bodhan_genai.mt.templates.prompt import STOP_STRINGS, render_conversations

SC = MTSamplingConfig()


def reply(text="ok", finish="stop", prompt_tokens=11, completion_tokens=7):
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "model": DEFAULT_MODEL,
    }


class StubTransport:
    """Records requests and replies from a script."""

    def __init__(self, replies=None):
        self.calls: list[dict] = []
        self._replies = list(replies or [])

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        r = self._replies.pop(0) if self._replies else reply()
        if isinstance(r, Exception):
            raise r
        return r


def backend(replies=None, **kw):
    kw.setdefault("sleep", lambda _s: None)  # no wall-clock in retry tests
    return ApiBackend("sk-test", post=StubTransport(replies), **kw)


def convs(*requests):
    return render_conversations(requests or (MTRequest("Hello.", "hin_Deva"),))


# --------------------------------------------------------------------------- #
# The wire format
# --------------------------------------------------------------------------- #


def test_route_and_body():
    b = backend()
    b.generate_conversations(convs(), SC)
    call = b._post.calls[0]
    assert call["url"].endswith(CHAT_ROUTE)
    assert call["body"]["model"] == DEFAULT_MODEL
    assert call["body"]["messages"] == [
        {"role": "user", "content": "Translate the following text into Hindi:\n\nHello."}
    ]


def test_greedy_envelope_is_sent():
    """temperature 0, the stop string, and NO top_p/seed under greedy decoding."""
    b = backend()
    b.generate_conversations(convs(), SC)
    body = b._post.calls[0]["body"]
    assert body["temperature"] == 0.0
    assert body["stop"] == list(STOP_STRINGS)
    assert body["max_tokens"] == SC.max_new_tokens
    assert "top_p" not in body and "seed" not in body


def test_sampled_decoding_sends_top_p_and_seed():
    b = backend()
    b.generate_conversations(convs(), MTSamplingConfig(temperature=0.7, top_p=0.9, seed=3))
    body = b._post.calls[0]["body"]
    assert body["top_p"] == 0.9 and body["seed"] == 3


def test_bearer_token_is_sent():
    b = backend()
    b.generate_conversations(convs(), SC)
    assert b._post.calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_the_prompt_carries_the_task_and_script():
    """Roman, code-mix and transliteration ride in the prompt, so the hosted path
    reaches every capability the local backends do."""
    b = backend()
    b.generate_conversations(
        convs(
            MTRequest("Hello.", "hin_Deva"),
            MTRequest("Hello.", "hin_Deva", script="roman"),
            MTRequest("Hello.", "hin_Deva", script="code_mixed"),
            MTRequest("नमस्ते", "hin_Deva", script="roman", task="transliterate"),
            MTRequest("Hello.", "kas_Deva"),
        ),
        SC,
    )
    sent = [c["body"]["messages"][0]["content"].splitlines()[0] for c in b._post.calls]
    assert sent == [
        "Translate the following text into Hindi:",
        "Translate the following text into Hindi (Roman):",
        "Translate the following text into Hindi (code-mixed):",
        "Transliterate the following text into Hindi (Roman):",
        # The script qualifier survives -- the flat route could not express it.
        "Translate the following text into Kashmiri (Devanagari script):",
    ]


def test_the_prompt_never_names_the_source_language():
    b = backend()
    b.generate_conversations(convs(MTRequest("Hello.", "hin_Deva", src_lang="eng_Latn")), SC)
    content = b._post.calls[0]["body"]["messages"][0]["content"]
    assert "English" not in content
    assert len(b._post.calls[0]["body"]["messages"]) == 1  # one user turn, no system turn


# --------------------------------------------------------------------------- #
# Reading the reply
# --------------------------------------------------------------------------- #


def test_text_and_usage_are_read_back():
    b = backend([reply("नमस्ते", prompt_tokens=44, completion_tokens=41)])
    out = b.generate_conversations(convs(), SC)[0]
    assert out.text == "नमस्ते"
    assert out.prompt_tokens == 44 and out.generated_tokens == 41
    assert out.gen_time_s is not None


def test_finish_reason_length_means_truncated():
    b = backend([reply("half a sen", finish="length")])
    assert b.generate_conversations(convs(), SC)[0].truncated is True


def test_finish_reason_stop_is_not_truncated():
    assert backend([reply("whole")]).generate_conversations(convs(), SC)[0].truncated is False


def test_a_200_without_choices_is_an_error():
    """The bug an over-obliging stub hid.

    A 200 whose body carries no usable choice must raise, not yield "" with
    ok=True. The service really does answer 200 with an error payload under rate
    limiting, and defaulting to "" made a failed request look like a successful
    empty translation.
    """
    for bad in ({"usage": {}}, {"choices": []}):
        b = backend([bad])
        with pytest.raises(ApiError, match="no choices"):
            b.generate_conversations(convs(), SC)


def test_batch_order_is_preserved():
    replies = [reply(f"t{i}") for i in range(5)]
    b = backend(replies, num_workers=4)
    outs = b.generate_conversations(convs(*[MTRequest(f"s{i}", "hin_Deva") for i in range(5)]), SC)
    assert [o.text for o in outs] == ["t0", "t1", "t2", "t3", "t4"]


def test_empty_batch_sends_nothing():
    b = backend()
    assert b.generate_conversations([], SC) == []
    assert b._post.calls == []


# --------------------------------------------------------------------------- #
# It is a conversation backend, like the local ones
# --------------------------------------------------------------------------- #


def test_it_satisfies_the_protocol_behind_the_adapter():
    assert isinstance(PromptedBackend(backend()), TranslationBackend)


def test_the_adapter_forwards_health():
    """PromptedBackend must not hide what the inner backend can do."""
    inner = backend()
    inner.health = lambda: False  # type: ignore[method-assign]
    assert PromptedBackend(inner).health() is False


def test_an_empty_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="BODHAN_API_KEY"):
        ApiBackend("")


# --------------------------------------------------------------------------- #
# Retry policy — a pure truth table, then the loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "code", "succeeded", "expected"),
    [
        (429, "rate_limit_exceeded_error", False, True),
        (500, "internal_error", False, True),
        (503, "internal_error", False, True),
        (400, "invalid_request_error", False, False),
        (404, "", True, False),
        (413, "payload_too_large", True, False),
        (422, "", True, False),
        # The documented transient: a VALID key can return auth errors under
        # high parallelism. Retry only once the key has proven itself.
        (401, "invalid_api_key_error", True, True),
        (401, "invalid_api_key_error", False, False),
        (403, "invalid_api_key_error", True, True),
        (403, "invalid_api_key_error", False, False),
    ],
)
def test_is_retryable_truth_table(status, code, succeeded, expected):
    assert is_retryable(status, code, key_has_succeeded=succeeded) is expected


def test_429_is_retried_and_then_succeeds():
    b = backend(
        [ApiError("slow down", status=429, code="rate_limit_exceeded_error"), reply("made it")]
    )
    assert b.generate_conversations(convs(), SC)[0].text == "made it"
    assert len(b._post.calls) == 2


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_client_errors_are_never_retried(status):
    b = backend([ApiError("nope", status=status)])
    with pytest.raises(ApiError):
        b.generate_conversations(convs(), SC)
    assert len(b._post.calls) == 1


def test_retries_are_capped_by_max_attempts():
    err = ApiError("boom", status=500, code="internal_error")
    b = backend([err] * 10, retry=RetryPolicy(max_attempts=3))
    with pytest.raises(ApiError):
        b.generate_conversations(convs(), SC)
    # max_attempts counts ATTEMPTS, not retries: 3 means one try and two more.
    assert len(b._post.calls) == 3


def test_401_is_retried_only_after_the_key_has_worked():
    auth = ApiError("bad key", status=401, code="invalid_api_key_error")
    cold = backend([auth, reply("never reached")])
    with pytest.raises(ApiError):
        cold.generate_conversations(convs(), SC)
    assert len(cold._post.calls) == 1, "an unproven key must not be retried"

    warm = backend([reply("first"), auth, reply("second")], num_workers=1)
    warm.generate_conversations(convs(), SC)  # proves the key
    assert warm.generate_conversations(convs(), SC)[0].text == "second"
    assert len(warm._post.calls) == 3


def test_error_carries_the_request_id():
    """It is the only field that makes a support report actionable."""
    err = ApiError("upstream fell over", status=500, code="internal_error", request_id="req_abc123")
    assert "req_abc123" in str(err) and err.request_id == "req_abc123"


def test_backoff_is_bounded_and_jittered():
    b = backend()
    delays = [b._delay(a) for a in range(6) for _ in range(20)]
    assert all(0 < d <= b.retry.max_delay_s for d in delays)
    # Jitter is load-bearing: without it, N concurrent workers resynchronise into
    # one thundering herd against the endpoint that just rate-limited them.
    assert len({b._delay(0) for _ in range(30)}) > 1


def test_retry_after_header_is_honoured_and_clamped():
    b = backend(retry=RetryPolicy(max_delay_s=5.0))
    assert b._delay(0, retry_after=2.0) == 2.0
    assert b._delay(0, retry_after=9999.0) == 5.0


# --------------------------------------------------------------------------- #
# Error-body shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"message": "bad language", "code": "unsupported_language"}}, "bad language"),
        ({"detail": "unsupported language 'bho'"}, "unsupported language 'bho'"),
        ({"error": "flat string error"}, "flat string error"),
    ],
)
def test_both_error_body_shapes_are_parsed(body, expected, monkeypatch):
    """OpenAI-shaped `{"error": {...}}` and FastAPI-shaped `{"detail": ...}` are
    both live behaviours of this gateway."""
    import json as _json
    import urllib.error
    import urllib.request

    from bodhan_genai.mt.engine.api import _post_json

    class Err(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 422, "Unprocessable", {}, None)

        def read(self):
            return _json.dumps(body).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(Err()))
    with pytest.raises(ApiError, match=expected.split()[0]):
        _post_json("http://x/chat/completions", {}, {}, 5.0)
