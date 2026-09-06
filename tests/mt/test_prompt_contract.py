"""The prompt contract is frozen. These tests are the freeze.

Every rule here fails *silently* in production — a wrong prompt still returns
fluent text, just measurably worse text — so it has to fail loudly in CI instead.
"""

from __future__ import annotations

import pytest

from bodhan_genai.mt.templates.prompt import (
    DEFAULT_SCRIPT,
    LANGUAGE_NAMES,
    STOP_STRINGS,
    TEMPLATE,
    build_conversation,
    build_instruction,
    resolve_language,
)
from bodhan_genai.mt.templates.variants import (
    EVAL_TEMPLATE_INDEX,
    TARGET_ONLY_TEMPLATES,
    WITH_SOURCE_TEMPLATES,
    display_name,
    eval_instruction,
    get_templates,
    render_instruction,
)

# The golden instruction, byte for byte. If this line has to change, the model
# has been retrained and the published scores no longer apply.
GOLDEN = "Translate the following text into Hindi:\n\nHello world."


def test_template_is_frozen():
    assert TEMPLATE == "Translate the following text into {tgt}:\n\n{text}"
    assert STOP_STRINGS == ["<turn|>"]


def test_instruction_matches_golden():
    assert build_instruction("Hello world.", "hin_Deva") == GOLDEN
    # A name and a code must render identically.
    assert build_instruction("Hello world.", "Hindi") == GOLDEN


def test_conversation_is_exactly_one_user_turn():
    """One user turn, no system turn. An extra or empty system turn changes the
    rendered prefix and degrades output."""
    convo = build_conversation("Hello world.", "hin_Deva")
    assert len(convo) == 1
    assert convo[0]["role"] == "user"
    assert convo[0]["content"] == GOLDEN
    assert not any(m["role"] == "system" for m in convo)


def test_prompt_never_names_the_source_language():
    """The source language is inferred, never stated. The historical bug this
    guards against rendered 'Translate the following English text into Hindi'."""
    instruction = build_instruction("Le chat dort.", "hin_Deva")
    for source_name in ("English", "French", "from"):
        assert source_name not in instruction, f"prompt leaked a source language: {source_name}"


@pytest.mark.parametrize("code", sorted(LANGUAGE_NAMES))
def test_every_code_renders_with_its_own_name(code):
    """All 25 language-script combinations render, and each names its own language."""
    name = LANGUAGE_NAMES[code]
    instruction = build_instruction("probe", code)
    assert instruction == f"Translate the following text into {name}:\n\nprobe"


def test_language_set_size():
    """22 Eighth-Schedule languages + English = 25 language-script combinations."""
    assert len(LANGUAGE_NAMES) == 25


def test_braces_in_source_text_survive():
    """Rendering is str.replace, never str.format: a literal brace in the source
    must not be interpreted as a placeholder."""
    text = "Use {tgt} and {text} and {} and {0}."
    instruction = build_instruction(text, "Tamil")
    assert instruction.endswith(text)
    assert "{tgt}" in instruction and "{text}" in instruction


def test_resolve_language_accepts_code_bare_and_qualified():
    assert resolve_language("hin_Deva") == "Hindi"
    assert resolve_language("hindi") == "Hindi"
    assert resolve_language("  Hindi  ") == "Hindi"
    assert resolve_language("mni_Beng") == "Manipuri (Bengali script)"
    assert resolve_language("Manipuri (Bengali script)") == "Manipuri (Bengali script)"


@pytest.mark.parametrize(
    ("bare", "expected_code"),
    [("kashmiri", "kas_Arab"), ("manipuri", "mni_Mtei"), ("sindhi", "snd_Deva")],
)
def test_multi_script_bare_names_resolve_to_the_documented_default(bare, expected_code):
    """The script qualifier is functional, not decoration: prompting bare 'Sindhi'
    instead of 'Sindhi (Devanagari script)' measured 29.9 chrF++ worse."""
    assert DEFAULT_SCRIPT[bare] == expected_code
    assert resolve_language(bare) == LANGUAGE_NAMES[expected_code]


def test_unknown_language_raises_and_lists_the_valid_codes():
    with pytest.raises(ValueError, match="unsupported language") as excinfo:
        resolve_language("klingon")
    assert "hin_Deva" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Training variants must agree with the served contract
# --------------------------------------------------------------------------- #


def test_eval_template_index_zero_is_the_served_contract():
    """Index 0 of the target-only bank IS the served instruction, so eval prompts
    sit inside the training distribution instead of beside it."""
    assert TARGET_ONLY_TEMPLATES[EVAL_TEMPLATE_INDEX] == TEMPLATE
    assert eval_instruction("Hindi", "Hello world.") == GOLDEN
    assert eval_instruction("Hindi", "Hello world.") == build_instruction(
        "Hello world.", "hin_Deva"
    )


def test_template_banks_are_index_aligned():
    """The two banks mirror each other one-for-one, so EVAL_TEMPLATE_INDEX means
    'the plainest variant' in both."""
    assert len(TARGET_ONLY_TEMPLATES) == len(WITH_SOURCE_TEMPLATES) == 12


@pytest.mark.parametrize("variant", ["target_only", "with_source"])
def test_every_variant_template_uses_both_placeholders(variant):
    for template in get_templates(variant):
        assert "{tgt}" in template
        assert "{text}" in template
        if variant == "with_source":
            assert "{src}" in template


def test_target_only_templates_never_mention_a_source():
    for template in TARGET_ONLY_TEMPLATES:
        assert "{src}" not in template


def test_render_instruction_leaves_braces_alone():
    out = render_instruction(WITH_SOURCE_TEMPLATES[0], "Hindi", "keep {this}", "English")
    assert "keep {this}" in out
    assert "from English to Hindi" in out


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown template variant"):
        get_templates("nonsense")


def test_display_name_supports_out_of_set_languages_and_fails_loudly_otherwise():
    """A corpus can add a language without editing the frozen contract, but an
    unknown code must raise — silently mislabelling a language trains the wrong
    thing and never shows up in the loss."""
    assert display_name("hin_Deva") == "Hindi"
    assert display_name("xyz_Deva", {"xyz_Deva": "Some Language"}) == "Some Language"
    with pytest.raises(ValueError, match="unknown language code"):
        display_name("xyz_Deva")
