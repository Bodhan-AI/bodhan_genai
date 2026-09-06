"""Training-time chat template carrying ``{% generation %}`` loss markers.

TRL's ``assistant_only_loss=True`` needs the chat template to mark which span the
loss is computed over, via the ``{% generation %}`` / ``{% endgeneration %}``
extension. The checkpoint's shipped ``chat_template.jinja`` (the Gemma 4 canonical
template — tool calling, thinking channels, ~18 KB) carries no such markers, so
training swaps in the compact template below.

**The contract: this template must render byte-identically to the canonical one.**
If it does not, training teaches the model a different prefix than inference sends
it, and the model degrades in a way no loss curve shows.
``tests/mt/test_trl_template_parity.py`` asserts string *and* token equality for
both the prompt-only and full-turn renderings; treat a failure there as blocking.

Two whitespace notes, both learned the hard way:

*   Every tag uses the ``{%- -%}`` / ``{{- -}}`` trim modifiers. Without them the
    literal newlines between tags land in the output — a template written across
    plain lines renders ``'\\n<bos>\\n<|turn>user\\n…<turn|>\\n\\n<|turn>model\\n\\n'``
    instead of ``'<bos><|turn>user\\n…<turn|>\\n<|turn>model\\n'``. Note that
    ``trim_blocks`` does not save you: it strips the newline after ``{% %}`` block
    tags but not after ``{{ }}`` expressions.
*   The generation span deliberately includes the trailing ``<turn|>`` so the model
    is trained to emit its own stop token. Excluding it trains a model that never
    terminates.

stdlib-only imports.
"""

from __future__ import annotations

#: Renders exactly:
#:   prompt    ``<bos><|turn>user\n{instruction}<turn|>\n<|turn>model\n``
#:   full turn ``<bos><|turn>user\n{instruction}<turn|>\n<|turn>model\n{target}<turn|>\n``
#: with the loss span covering ``{target}<turn|>``.
GEMMA4_TRL_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{- '<|turn>user\n' + message['content'] | trim + '<turn|>\n' -}}"
    "{%- elif message['role'] == 'assistant' -%}"
    "{{- '<|turn>model\n' -}}"
    "{%- generation -%}"
    "{{- message['content'] | trim + '<turn|>' -}}"
    "{%- endgeneration -%}"
    "{{- '\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|turn>model\n' -}}"
    "{%- endif -%}"
)

__all__ = ["GEMMA4_TRL_TEMPLATE"]
