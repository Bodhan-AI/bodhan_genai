"""The training chat template must render byte-identically to the served one.

This is the highest-risk seam in the MT pipeline. Training rewrites
``tokenizer.chat_template`` to :data:`GEMMA4_TRL_TEMPLATE` — which carries the
``{% generation %}`` markers ``assistant_only_loss`` needs — while inference uses
the canonical template shipped in the checkpoint. If the two disagree by even one
newline, the model is trained on a prefix it never sees at serve time, and nothing
in the loss curve says so.

The reference implementation this was ported from had exactly that bug: its
template was written across plain lines with no jinja whitespace control and
rendered ``'\\n<bos>\\n<|turn>user\\n…<turn|>\\n\\n<|turn>model\\n\\n'`` where the
canonical template renders ``'<bos><|turn>user\\n…<turn|>\\n<|turn>model\\n'``.
(``trim_blocks`` does not rescue it: that strips the newline after ``{% %}`` block
tags, not after ``{{ }}`` expressions.)

Two layers of checking:

*   :func:`test_renders_golden_prefix` and friends run **everywhere**, including on
    a CPU CI runner with no model files, by driving the template through jinja2
    with a minimal stand-in for the ``generation`` tag.
*   The token-level comparison against the real shipped template runs only when a
    checkpoint is available — point ``BODHAN_MT_CHECKPOINT`` at one.
"""

from __future__ import annotations

import os
from typing import ClassVar

import pytest

from bodhan_genai.mt.templates.trl_chat import GEMMA4_TRL_TEMPLATE

jinja2 = pytest.importorskip("jinja2", reason="jinja2 backs every chat template")
# Submodules are not imported by `import jinja2` alone.
pytest.importorskip("jinja2.ext")
pytest.importorskip("jinja2.sandbox")

USER = "Translate the following text into Hindi:\n\nHello world."
ASSISTANT = "नमस्ते दुनिया।"

# What the checkpoint's own chat_template.jinja produces. Verified against
# transformers 5.13.1 and the published checkpoint.
GOLDEN_PROMPT = "<bos><|turn>user\nTranslate the following text into Hindi:\n\nHello world.<turn|>\n<|turn>model\n"
GOLDEN_FULL = GOLDEN_PROMPT + "नमस्ते दुनिया।<turn|>\n"


class _GenerationTag(jinja2.ext.Extension):
    """Minimal stand-in for transformers' ``{% generation %}`` extension.

    Emits the body unchanged and records where it landed, which is all this test
    needs: we are checking the rendered text and the span boundaries, not
    transformers' mask-building. Keeping it local also means the test does not
    depend on which transformers version is installed.
    """

    tags: ClassVar[set[str]] = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_mark", []), [], [], body).set_lineno(
            lineno
        )

    def _mark(self, caller):
        rendered = caller()
        self.environment.generation_spans.append(rendered)  # type: ignore[attr-defined]
        return rendered


def _render(
    messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> tuple[str, list[str]]:
    """Render GEMMA4_TRL_TEMPLATE the way transformers would, plus the loss spans."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationTag]
    )
    env.generation_spans = []  # type: ignore[attr-defined]
    template = env.from_string(GEMMA4_TRL_TEMPLATE)
    text = template.render(
        messages=messages,
        bos_token="<bos>",
        add_generation_prompt=add_generation_prompt,
    )
    return text, env.generation_spans  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Always-on: whitespace and structure
# --------------------------------------------------------------------------- #


def test_renders_golden_prefix():
    """The generation prompt must match the served prefix byte for byte."""
    text, _ = _render([{"role": "user", "content": USER}], add_generation_prompt=True)
    assert text == GOLDEN_PROMPT


def test_renders_golden_full_turn():
    text, _ = _render(
        [{"role": "user", "content": USER}, {"role": "assistant", "content": ASSISTANT}],
        add_generation_prompt=False,
    )
    assert text == GOLDEN_FULL


def test_no_stray_leading_or_doubled_newlines():
    """The specific regression from the reference implementation."""
    text, _ = _render(
        [{"role": "user", "content": USER}, {"role": "assistant", "content": ASSISTANT}],
        add_generation_prompt=False,
    )
    assert text.startswith("<bos><|turn>user\n"), f"leading whitespace: {text[:24]!r}"
    assert "\n\n<|turn>" not in text, "doubled newline before a turn marker"
    assert "<bos>\n" not in text, "newline after bos"
    assert not text.endswith("\n\n"), "trailing blank line"


def test_generation_span_is_exactly_the_assistant_content_plus_stop():
    """The loss span must include the terminating <turn|>.

    Excluding it trains a model that never emits its own stop token; including
    anything before the content trains it on the prompt.
    """
    _, spans = _render(
        [{"role": "user", "content": USER}, {"role": "assistant", "content": ASSISTANT}],
        add_generation_prompt=False,
    )
    assert spans == [f"{ASSISTANT}<turn|>"]


def test_template_uses_whitespace_control_everywhere():
    """Guard the mechanism, not just the outcome: a tag added later without the
    ``-`` modifiers reintroduces the exact bug this file exists for."""
    import re

    for tag in re.findall(r"\{%.*?%\}|\{\{.*?\}\}", GEMMA4_TRL_TEMPLATE):
        assert tag[2] == "-" and tag[-3] == "-", f"tag lacks whitespace control: {tag!r}"


def test_content_is_trimmed_so_ragged_corpus_rows_render_identically():
    text_a, _ = _render([{"role": "user", "content": USER}], add_generation_prompt=True)
    text_b, _ = _render([{"role": "user", "content": f"  {USER}\n\n"}], add_generation_prompt=True)
    assert text_a == text_b


# --------------------------------------------------------------------------- #
# Opt-in: real token-level parity against the shipped template
# --------------------------------------------------------------------------- #

CHECKPOINT = os.environ.get("BODHAN_MT_CHECKPOINT")


@pytest.mark.skipif(
    not CHECKPOINT,
    reason="set BODHAN_MT_CHECKPOINT to a IndicTranslate checkpoint to run real-tokenizer parity",
)
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_token_parity_with_shipped_template(add_generation_prompt):
    """Same token ids from the canonical template and the training template."""
    from transformers import AutoTokenizer

    messages = [{"role": "user", "content": USER}]
    if not add_generation_prompt:
        messages = [*messages, {"role": "assistant", "content": ASSISTANT}]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    canonical = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True
    )
    canonical_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )

    tokenizer.chat_template = GEMMA4_TRL_TEMPLATE
    training = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True
    )
    training_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )

    assert training_text == canonical_text
    assert list(training) == list(canonical)
