"""Instruction-phrasing variants for building training corpora.

:mod:`bodhan_genai.mt.templates.prompt` carries the ONE canonical instruction the
model is served with. This module carries the ~12 phrasing variants a *training*
corpus is rendered with, so the model sees paraphrase diversity rather than a
single memorised string.

Two flavours, kept API-compatible and one-for-one by index:

``target_only``
    Names only the target language ("into Hindi"). This is the released contract
    — index 0 is exactly ``prompt.TEMPLATE``. Use it whenever the corpus mixes
    directions, which is the common case.

``with_source``
    Names both ("from English to Hindi"). Useful only for a single fixed-direction
    finetune, where stating the source is unambiguous and closer to how the model
    will actually be prompted. It is *not* the served contract: a model trained
    this way must be evaluated and served with the same variant.

``EVAL_TEMPLATE_INDEX = 0`` in both, so evaluation uses the plainest variant while
training sees the whole distribution.

Rendering uses plain ``str.replace`` (never ``str.format``) so source text
containing ``{`` or ``}`` never breaks formatting.

stdlib-only imports.
"""

from __future__ import annotations

import random

from bodhan_genai.mt.templates.prompt import LANGUAGE_NAMES, resolve_language

# --------------------------------------------------------------------------- #
# Template banks
# --------------------------------------------------------------------------- #

#: Variety spans phrasing (imperative / polite / question), verb choice,
#: punctuation, and the position of the source text (before vs after the
#: instruction). Index 0 is the canonical served instruction.
TARGET_ONLY_TEMPLATES: list[str] = [
    "Translate the following text into {tgt}:\n\n{text}",
    "Translate into {tgt}:\n\n{text}",
    "Please translate the text below into {tgt}.\n\n{text}",
    "Convert the following text to {tgt}:\n\n{text}",
    "Render the following passage in {tgt}:\n\n{text}",
    "{text}\n\nTranslate the above into {tgt}.",
    "Give the {tgt} translation of:\n\n{text}",
    "What is the following in {tgt}?\n\n{text}",
    "Provide a {tgt} translation for the text below.\n\n{text}",
    "Translate to {tgt}.\n\nText: {text}",
    "I need this translated into {tgt}:\n\n{text}",
    "Rewrite the following text in {tgt}:\n\n{text}",
]

#: Mirrors TARGET_ONLY_TEMPLATES one-for-one by index, with ``{src}`` added.
WITH_SOURCE_TEMPLATES: list[str] = [
    "Translate the following text from {src} to {tgt}:\n\n{text}",
    "Translate from {src} into {tgt}:\n\n{text}",
    "Please translate the text below from {src} to {tgt}.\n\n{text}",
    "Convert the following {src} text to {tgt}:\n\n{text}",
    "Render the following {src} passage in {tgt}:\n\n{text}",
    "{text}\n\nTranslate the above from {src} into {tgt}.",
    "Give the {tgt} translation of this {src} text:\n\n{text}",
    "What is the following {src} text in {tgt}?\n\n{text}",
    "Provide a {tgt} translation for the {src} text below.\n\n{text}",
    "Translate from {src} to {tgt}.\n\nText: {text}",
    "I need this {src} text translated into {tgt}:\n\n{text}",
    "Rewrite the following {src} text in {tgt}:\n\n{text}",
]

VARIANTS: dict[str, list[str]] = {
    "target_only": TARGET_ONLY_TEMPLATES,
    "with_source": WITH_SOURCE_TEMPLATES,
}

#: Index of the canonical variant. Eval renders with this one so evaluation
#: prompts sit inside the training distribution rather than beside it.
EVAL_TEMPLATE_INDEX = 0


def get_templates(variant: str) -> list[str]:
    """Return the template bank for ``variant``, failing loudly on a typo."""
    try:
        return VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"unknown template variant {variant!r}; expected one of {sorted(VARIANTS)}"
        ) from None


# --------------------------------------------------------------------------- #
# Language display names
# --------------------------------------------------------------------------- #


def display_name(lang_code: str, extra: dict[str, str] | None = None) -> str:
    """Return the display name for a language code.

    ``extra`` lets a corpus add languages outside the served set (a finetune on a
    language the base model does not cover) without editing the frozen
    :data:`~bodhan_genai.mt.templates.prompt.LANGUAGE_NAMES` contract.

    Unknown codes raise: a silently mislabelled language trains the wrong thing,
    and the failure is invisible in the loss curve.
    """
    if extra and lang_code in extra:
        return extra[lang_code]
    if lang_code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[lang_code]
    raise ValueError(
        f"unknown language code {lang_code!r}. Pass it via `extra_languages` in the "
        f"render config if this corpus covers a language outside the served set."
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_instruction(template: str, tgt_name: str, text: str, src_name: str = "") -> str:
    """Fill a template with language names and source text.

    Plain string replacement, so braces in ``text`` are left untouched.
    ``src_name`` is ignored by target-only templates (they carry no ``{src}``).
    """
    return template.replace("{tgt}", tgt_name).replace("{src}", src_name).replace("{text}", text)


def choose_template(rng: random.Random, variant: str = "target_only") -> int:
    """Pick a template index using the provided (seeded) RNG."""
    return rng.randrange(len(get_templates(variant)))


def eval_instruction(
    tgt_name: str, text: str, src_name: str = "", variant: str = "target_only"
) -> str:
    """Render the canonical eval instruction for ``variant``.

    For ``target_only`` this is identical to
    :func:`bodhan_genai.mt.templates.prompt.build_instruction` — asserted by
    ``tests/mt/test_prompt_contract.py``.
    """
    return render_instruction(get_templates(variant)[EVAL_TEMPLATE_INDEX], tgt_name, text, src_name)


__all__ = [
    "EVAL_TEMPLATE_INDEX",
    "TARGET_ONLY_TEMPLATES",
    "VARIANTS",
    "WITH_SOURCE_TEMPLATES",
    "choose_template",
    "display_name",
    "eval_instruction",
    "get_templates",
    "render_instruction",
    "resolve_language",
]
