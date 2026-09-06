"""The HTTP recognizer backend: prompt parity with the offline path, and order safety.

No network and no server. The OpenAI client is stubbed at the seam ``HttpRecognizer`` takes it,
the same injection point ``MTClient`` uses.
"""

from __future__ import annotations

import base64
import io
import threading

import pytest

from bodhan_genai.ocr.engine.recognizer import CropRequest, RecognizerBackend
from bodhan_genai.ocr.engine.types import RecognizerConfig
from bodhan_genai.ocr.serving.recognizer_http import (
    HttpRecognizer,
    build_messages,
    encode_crop,
)

pytest.importorskip("PIL", reason="pillow is an ocr extra, absent on a bare CPU runner")

PROMPT = "Output only the LaTeX for this equation image."


def crop(width: int = 40, height: int = 20, colour: str = "white"):
    from PIL import Image

    return Image.new("RGB", (width, height), colour)


def request(prompt: str = PROMPT) -> CropRequest:
    return CropRequest(image=crop(), prompt=prompt)


def prompt_of(body) -> str:
    """The prompt HttpRecognizer put in this request body."""
    return body["messages"][0]["content"][1]["text"]


class StubCompletions:
    """Returns a canned answer per call, recording the bodies it was given.

    ``replies`` is either a list, consumed in ARRIVAL order, or a mapping from a
    request's prompt to its answer. Use the mapping in any test that asserts on
    positions: ``transcribe`` sends crops through a thread pool, so arrival order is
    not request order, and a list binds answers to whichever thread got there first.
    That is a property of the stub, not of the backend -- ``transcribe`` reassembles
    by index and is order-correct either way.
    """

    def __init__(self, replies, on_call=None):
        self._replies = replies if isinstance(replies, dict) else list(replies)
        self.bodies = []
        self._on_call = on_call
        # `index = len(bodies)` then `bodies.append(...)` is a read-modify-write; without
        # this two concurrent calls can take the same index and lose a body.
        self._lock = threading.Lock()

    def create(self, **body):
        with self._lock:
            index = len(self.bodies)
            self.bodies.append(body)
        if self._on_call is not None:
            self._on_call(index, body)
        if isinstance(self._replies, dict):
            text = self._replies[prompt_of(body)]
        else:
            text = self._replies[index] if index < len(self._replies) else ""
        if isinstance(text, Exception):
            raise text
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text})()})()]},
        )()


class StubClient:
    def __init__(self, replies, on_call=None):
        self.chat = type("Chat", (), {"completions": StubCompletions(replies, on_call)})()

    @property
    def bodies(self):
        return self.chat.completions.bodies


def backend(replies, **kwargs) -> HttpRecognizer:
    return HttpRecognizer("http://stub/v1", client=StubClient(replies), **kwargs)


def ordered(replies, on_call=None, **kwargs):
    """A backend plus matching requests, with each answer bound to its own request.

    Returns ``(backend, requests)``. Every request carries a distinct prompt and the
    stub answers by prompt, so the assertion holds no matter what order the pool's
    threads reach the client in -- which is the only way a positional assertion here
    can be deterministic.
    """
    prompts = [f"{PROMPT} #{i}" for i in range(len(replies))]
    stub = StubClient(dict(zip(prompts, replies, strict=True)), on_call=on_call)
    return (
        HttpRecognizer("http://stub/v1", client=stub, **kwargs),
        [request(p) for p in prompts],
    )


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_satisfies_the_recognizer_protocol():
    """It must be substitutable for the in-process backends without the engine caring."""
    assert isinstance(backend([]), RecognizerBackend)


def test_no_requests_means_no_calls():
    b = backend([])
    assert b.transcribe([]) == []


# --------------------------------------------------------------------------- #
# Prompt parity -- the reason this approach is safe
# --------------------------------------------------------------------------- #


def _render(template, messages) -> str:
    import json

    from jinja2.exceptions import TemplateError

    template.environment.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(
        TemplateError(m)
    )
    template.environment.filters.setdefault("tojson", lambda o, **k: json.dumps(o))
    return template.render(messages=messages, add_generation_prompt=True)


def test_served_messages_render_like_the_offline_ones_on_a_template_that_branches_on_image_url():
    """A template written the way the recognizer's is must collapse both content forms.

    This is a property test against the branch that makes the whole approach safe --
    ``'image' in item or 'image_url' in item``. It does NOT prove the shipped template does
    this; ``test_prompt_parity_against_the_shipped_template`` does, when weights are present.
    """
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
        "{%- for item in m.content %}"
        "{%- if 'image' in item or 'image_url' in item or item.type == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|>"
        "{%- elif item.type == 'text' %}{{ item.text }}{%- endif %}"
        "{%- endfor %}<|im_end|>\n{%- endfor %}"
    )
    local = _render(
        template,
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    )
    assert _render(template, build_messages(crop(), PROMPT)) == local


def test_prompt_parity_against_the_shipped_template():
    """The real check: the served prompt must equal the offline one on the actual checkpoint.

    Skips without weights, so CI does not need them. Run it wherever a checkpoint exists:
        BODHAN_OCR_RECOGNIZER_CKPT=/path/to/weights/ocr pytest tests/ocr/test_serving_client.py
    """
    import os
    from pathlib import Path

    jinja2 = pytest.importorskip("jinja2")

    ckpt = os.environ.get("BODHAN_OCR_RECOGNIZER_CKPT")
    if not ckpt:
        pytest.skip("set BODHAN_OCR_RECOGNIZER_CKPT to check against the shipped chat template")
    path = Path(ckpt) / "chat_template.jinja"
    if not path.is_file():
        pytest.skip(f"no chat_template.jinja under {ckpt}")

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(path.read_text(encoding="utf-8"))
    local = _render(
        template,
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
    )
    served = _render(template, build_messages(crop(), PROMPT))
    assert served == local, "served and offline prompts diverge on the shipped template"


def test_message_shape_is_image_then_text():
    """Order matters: the local path builds [image, text] and the template emits them in order."""
    messages = build_messages(crop(), PROMPT)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    kinds = [part["type"] for part in messages[0]["content"]]
    assert kinds == ["image_url", "text"]
    assert messages[0]["content"][1]["text"] == PROMPT


def test_crop_encodes_to_a_png_data_uri_that_decodes_back():
    from PIL import Image

    original = crop(37, 19, "red")
    uri = encode_crop(original)
    assert uri.startswith("data:image/png;base64,")

    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.size == original.size
    assert decoded.format == "PNG"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_follows_the_recognizer_config():
    """Greedy, and the config's token cap -- otherwise served output drifts from offline."""
    b = backend(["x"], config=RecognizerConfig(max_tokens=123))
    b.transcribe([request()])
    body = b._client.bodies[0]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 123


# --------------------------------------------------------------------------- #
# Order -- load-bearing, stage 2 matches transcriptions back by position
# --------------------------------------------------------------------------- #


def test_results_follow_request_order_not_completion_order():
    """Responses that finish out of order must still land against their own crop."""
    import time

    def stagger(index: int, body) -> None:
        # Key the delay to the REQUEST, not to arrival order: make request #0 the
        # slowest, so it completes last however the threads are scheduled.
        time.sleep(0.05 if prompt_of(body).endswith("#0") else 0.0)

    b, reqs = ordered(["first", "second", "third"], on_call=stagger, num_workers=4)
    assert b.transcribe(reqs) == ["first", "second", "third"]


def test_one_result_per_request_even_when_a_reply_is_empty():
    """Stage 2 raises on a count mismatch, so the list length is part of the contract."""
    b, reqs = ordered(["a", "", "c"])
    assert b.transcribe(reqs) == ["a", "", "c"]


def test_whitespace_is_stripped_like_the_offline_backends():
    assert backend(["  padded \n"]).transcribe([request()]) == ["padded"]


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


def test_a_failed_crop_fails_the_page_by_default():
    b, reqs = ordered(["ok", RuntimeError("boom"), "ok"])
    with pytest.raises(RuntimeError, match="1/3 crops failed"):
        b.transcribe(reqs)


def test_best_effort_keeps_going_and_leaves_that_block_empty(caplog):
    b, reqs = ordered(["ok", RuntimeError("boom"), "fine"], strict=False)
    with caplog.at_level("WARNING"):
        texts = b.transcribe(reqs)
    assert texts == ["ok", "", "fine"]
    assert "1/3 crops failed" in caplog.text, "a silently empty block must at least be logged"
