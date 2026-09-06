"""IndicTranslate — the prompt contract.

Single source of truth for the request format. Every path that talks to the model
(the HF backend, the vLLM backend, the served client, the eval harness, and the
training-data renderer) goes through :func:`build_conversation`, so no two
runtimes can drift apart.

Two rules matter, and both fail *silently* — a wrong prompt still yields fluent
output, just measurably worse output:

1.  **The prompt names only the TARGET language.** The source language is never
    stated; the model infers it. Do not write "from English to Hindi".
2.  **Exactly one ``user`` turn, and no ``system`` turn.** An empty or extra
    system turn changes the rendered prefix.

Rendered through the checkpoint's chat template with ``add_generation_prompt=True``
the result is byte-exactly::

    <bos><|turn>user\\nTranslate the following text into Hindi:\\n\\nHello world.<turn|>\\n<|turn>model\\n

stdlib-only imports — safe to import without torch / vllm / transformers.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Instruction template
# --------------------------------------------------------------------------- #

#: The canonical instruction. ``{tgt}`` is the target-language prompt name and
#: ``{text}`` is the source text. Rendered by plain string replacement, never
#: ``str.format``, so braces inside the source text are left untouched.
TEMPLATE = "Translate the following text into {tgt}:\n\n{text}"

#: The turn terminator. ``<turn|>`` (id 106) is already an EOS alongside ``<eos>``
#: (id 1), but passing it as an explicit stop string survives template changes and
#: costs nothing.
STOP_STRINGS = ["<turn|>"]


# --------------------------------------------------------------------------- #
# Language names
# --------------------------------------------------------------------------- #

#: Language code -> the exact string to substitute for ``{tgt}``.
#: Keys are FLORES-200 style ``<iso639-3>_<script>`` codes. Where a language is
#: supported in more than one script the script is spelled out in the name, and
#: that qualification is what selects the output script -- it is a functional
#: part of the prompt, not decoration. (Measured: prompting bare "Sindhi" instead
#: of "Sindhi (Devanagari script)" cost 29.9 chrF++.)
LANGUAGE_NAMES: dict[str, str] = {
    "eng_Latn": "English",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "gom_Deva": "Konkani",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri (Perso-Arabic script)",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Beng": "Manipuri (Bengali script)",
    "mni_Mtei": "Manipuri (Meitei script)",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Arab": "Sindhi (Perso-Arabic script)",
    "snd_Deva": "Sindhi (Devanagari script)",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}

#: For the languages carried in more than one script, the script a bare language
#: name resolves to. Pass the explicit code (or the fully qualified name) to pick
#: the other script.
DEFAULT_SCRIPT: dict[str, str] = {
    "kashmiri": "kas_Arab",
    "manipuri": "mni_Mtei",
    "sindhi": "snd_Deva",
}

_BY_NAME = {name.casefold(): name for name in LANGUAGE_NAMES.values()}


def resolve_language(value: str) -> str:
    """Resolve a code, a bare name, or a qualified name to its prompt name.

    >>> resolve_language("hin_Deva")
    'Hindi'
    >>> resolve_language("hindi")
    'Hindi'
    >>> resolve_language("Manipuri")
    'Manipuri (Meitei script)'
    >>> resolve_language("mni_Beng")
    'Manipuri (Bengali script)'
    """
    key = value.strip()
    if key in LANGUAGE_NAMES:  # FLORES-style code
        return LANGUAGE_NAMES[key]
    folded = key.casefold()
    if folded in _BY_NAME:  # exact, possibly qualified, name
        return _BY_NAME[folded]
    if folded in DEFAULT_SCRIPT:  # bare name of a two-script language
        return LANGUAGE_NAMES[DEFAULT_SCRIPT[folded]]
    raise ValueError(
        f"unsupported language {value!r}.\nSupported codes: {', '.join(sorted(LANGUAGE_NAMES))}"
    )


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def build_instruction(text: str, tgt_lang: str) -> str:
    """Render the user-turn instruction for one translation request.

    ``tgt_lang`` may be a FLORES-style code or a language name; it is resolved
    through :func:`resolve_language`.
    """
    tgt_name = resolve_language(tgt_lang)
    return TEMPLATE.replace("{tgt}", tgt_name).replace("{text}", text)


def build_conversation(text: str, tgt_lang: str) -> list[dict[str, str]]:
    """Build the single-turn conversation to hand to the chat template.

    Returns a one-element list holding a lone ``user`` turn. Rendered with
    ``add_generation_prompt=True`` this is exactly the request format the model
    expects.
    """
    return [{"role": "user", "content": build_instruction(text, tgt_lang)}]
