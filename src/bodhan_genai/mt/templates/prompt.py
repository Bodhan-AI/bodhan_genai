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

The model carries three output renderings of every target language — native
script, Roman (Latin letters), and code-mixed (English mixed into the target) —
plus a second task, transliteration, which changes a text's script without
translating it. Both axes are selected by :func:`resolve_target_name`; see
``script=`` and ``task=`` below.

Rendered through the checkpoint's chat template with ``add_generation_prompt=True``
the result is byte-exactly::

    <bos><|turn>user\\nTranslate the following text into Hindi:\\n\\nHello world.<turn|>\\n<|turn>model\\n

stdlib-only imports — safe to import without torch / vllm / transformers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps templates/ free of engine/
    from collections.abc import Iterable

    from bodhan_genai.mt.engine.types import MTRequest

# --------------------------------------------------------------------------- #
# Instruction template
# --------------------------------------------------------------------------- #

#: The canonical instruction. ``{tgt}`` is the target-language prompt name and
#: ``{text}`` is the source text. Rendered by plain string replacement, never
#: ``str.format``, so braces inside the source text are left untouched.
TEMPLATE = "Translate the following text into {tgt}:\n\n{text}"

#: The transliteration instruction. Its OWN family, deliberately not :data:`TEMPLATE`:
#: "Translate the following text into Hindi" is *wrong* when the source is already
#: Hindi in another script -- it would teach the model that translating to Hindi
#: sometimes means leaving the content alone.
TRANSLITERATE_TEMPLATE = "Transliterate the following text into {tgt}:\n\n{text}"

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
    "kas_Deva": "Kashmiri (Devanagari script)",
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

#: Short (ISO 639-1 style) code -> FLORES-style code. The hosted API accepts these
#: alongside the FLORES codes, so the shared resolver accepts them too: one table
#: and one resolver, or the served path and the local path drift.
#:
#: 23 entries -- the 22 Indic languages plus English. ``kas_Deva``, ``mni_Beng``
#: and ``snd_Arab`` have no short code of their own; the short code of a
#: multi-script language resolves to that language's documented default script,
#: and these agree with :data:`DEFAULT_SCRIPT` by construction (asserted in the
#: test suite). Pass the FLORES code to select the other script.
SHORT_CODES: dict[str, str] = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "en": "eng_Latn",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "kok": "gom_Deva",
    "ks": "kas_Arab",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Mtei",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
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

# --------------------------------------------------------------------------- #
# Roman and code-mixed target names
# --------------------------------------------------------------------------- #

#: Native code -> the ``{tgt}`` name that asks for **Roman** (Latin-letter) output.
#:
#: This is a table and not ``LANGUAGE_NAMES[code] + " (Roman)"`` on purpose. That
#: formula is wrong exactly where it matters: it would produce
#: ``"Sindhi (Perso-Arabic script) (Roman)"``, which the model was never trained
#: on. A language written in more than one native script has only ONE Roman name,
#: so both Kashmiri codes, both Manipuri codes and both Sindhi codes collapse onto
#: a single entry here. 25 Indic codes - 3 collapses = 22 distinct Roman names.
#:
#: ``eng_Latn`` is deliberately absent: English is already Latin script, so a
#: Roman target for it is a contract error, not a no-op.
ROMAN_NAMES: dict[str, str] = {
    "asm_Beng": "Assamese (Roman)",
    "ben_Beng": "Bengali (Roman)",
    "brx_Deva": "Bodo (Roman)",
    "doi_Deva": "Dogri (Roman)",
    "gom_Deva": "Konkani (Roman)",
    "guj_Gujr": "Gujarati (Roman)",
    "hin_Deva": "Hindi (Roman)",
    "kan_Knda": "Kannada (Roman)",
    "kas_Arab": "Kashmiri (Roman)",
    "kas_Deva": "Kashmiri (Roman)",
    "mai_Deva": "Maithili (Roman)",
    "mal_Mlym": "Malayalam (Roman)",
    "mar_Deva": "Marathi (Roman)",
    "mni_Beng": "Manipuri (Roman)",
    "mni_Mtei": "Manipuri (Roman)",
    "npi_Deva": "Nepali (Roman)",
    "ory_Orya": "Odia (Roman)",
    "pan_Guru": "Punjabi (Roman)",
    "san_Deva": "Sanskrit (Roman)",
    "sat_Olck": "Santali (Roman)",
    "snd_Arab": "Sindhi (Roman)",
    "snd_Deva": "Sindhi (Roman)",
    "tam_Taml": "Tamil (Roman)",
    "tel_Telu": "Telugu (Roman)",
    "urd_Arab": "Urdu (Roman)",
}

#: Native code -> the ``{tgt}`` name that asks for **code-mixed** output: English
#: mixed into the target language within a sentence.
#:
#: 24 keys over **22 distinct targets**, which reconciles as: 25 Indic codes, less
#: ``kas_Deva`` (the one code with no code-mix counterpart), less the Manipuri and
#: Sindhi collapses -- 25 - 1 - 2 = 22. Kashmiri does not collapse here because
#: only ``kas_Arab`` survives the first subtraction. ``eng_Latn`` is absent too:
#: code-mixing English into English is not a thing.
CODEMIX_NAMES: dict[str, str] = {
    "asm_Beng": "Assamese (code-mixed)",
    "ben_Beng": "Bengali (code-mixed)",
    "brx_Deva": "Bodo (code-mixed)",
    "doi_Deva": "Dogri (code-mixed)",
    "gom_Deva": "Konkani (code-mixed)",
    "guj_Gujr": "Gujarati (code-mixed)",
    "hin_Deva": "Hindi (code-mixed)",
    "kan_Knda": "Kannada (code-mixed)",
    "kas_Arab": "Kashmiri (code-mixed)",
    "mai_Deva": "Maithili (code-mixed)",
    "mal_Mlym": "Malayalam (code-mixed)",
    "mar_Deva": "Marathi (code-mixed)",
    "mni_Beng": "Manipuri (code-mixed)",
    "mni_Mtei": "Manipuri (code-mixed)",
    "npi_Deva": "Nepali (code-mixed)",
    "ory_Orya": "Odia (code-mixed)",
    "pan_Guru": "Punjabi (code-mixed)",
    "san_Deva": "Sanskrit (code-mixed)",
    "sat_Olck": "Santali (code-mixed)",
    "snd_Arab": "Sindhi (code-mixed)",
    "snd_Deva": "Sindhi (code-mixed)",
    "tam_Taml": "Tamil (code-mixed)",
    "tel_Telu": "Telugu (code-mixed)",
    "urd_Arab": "Urdu (code-mixed)",
}

#: The three output renderings a target language can be asked for.
SCRIPTS = ("native", "roman", "code_mixed")

#: The two tasks. ``translate`` changes the language; ``transliterate`` changes
#: only the script and leaves the language alone.
TASKS = ("translate", "transliterate")

_SCRIPT_TABLES = {"roman": ROMAN_NAMES, "code_mixed": CODEMIX_NAMES}


def resolve_language(value: str) -> str:
    """Resolve a code, a bare name, or a qualified name to its prompt name.

    >>> resolve_language("hin_Deva")
    'Hindi'
    >>> resolve_language("hindi")
    'Hindi'
    >>> resolve_language("hi")
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
    if folded in SHORT_CODES:  # short code, as the hosted API accepts
        return LANGUAGE_NAMES[SHORT_CODES[folded]]
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


def resolve_code(value: str) -> str:
    """Resolve a code, a bare name, or a qualified name to its FLORES-style code.

    The inverse of :func:`resolve_language`. Needed wherever the *code* rather
    than the prompt name goes on the wire.

    >>> resolve_code("hindi")
    'hin_Deva'
    >>> resolve_code("hin_Deva")
    'hin_Deva'
    >>> resolve_code("Manipuri")
    'mni_Mtei'
    """
    name = resolve_language(value)
    key = value.strip()
    if key in LANGUAGE_NAMES:
        return key
    if key.casefold() in SHORT_CODES:
        return SHORT_CODES[key.casefold()]
    # `name` is unique per code except where two codes share a display name,
    # which LANGUAGE_NAMES never does -- every multi-script language spells its
    # script out. So the reverse lookup is unambiguous.
    for code, display in LANGUAGE_NAMES.items():
        if display == name:
            return code
    raise AssertionError(f"unreachable: {name!r} resolved but has no code")


def resolve_target_name(tgt_lang: str, *, script: str = "native", task: str = "translate") -> str:
    """Resolve the exact ``{tgt}`` string for a language, script and task.

    ``script`` picks the output rendering -- ``"native"``, ``"roman"`` or
    ``"code_mixed"``. The parenthetical it produces is **functional**: it selects
    what the model writes, and dropping it is a silent quality loss, not a
    cosmetic difference.

    >>> resolve_target_name("hin_Deva")
    'Hindi'
    >>> resolve_target_name("hin_Deva", script="roman")
    'Hindi (Roman)'
    >>> resolve_target_name("snd_Arab", script="roman")
    'Sindhi (Roman)'
    """
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}. Supported: {', '.join(TASKS)}")
    if script not in SCRIPTS:
        raise ValueError(f"unsupported script {script!r}. Supported: {', '.join(SCRIPTS)}")

    if script == "native":
        if task == "transliterate" and resolve_code(tgt_lang) == "eng_Latn":
            raise ValueError(
                "transliteration is not defined for English: it is already in Latin "
                "script. Pass an Indic language, or use task='translate'."
            )
        return resolve_language(tgt_lang)

    if task == "transliterate" and script == "code_mixed":
        raise ValueError(
            "script='code_mixed' is a translation target, not a transliteration one: "
            "code-mixing changes the lexicon, not the script."
        )

    code = resolve_code(tgt_lang)
    table = _SCRIPT_TABLES[script]
    try:
        return table[code]
    except KeyError:
        if code == "eng_Latn":
            raise ValueError(
                f"script={script!r} is not available for English: English is already "
                "written in Latin letters. Use script='native'."
            ) from None
        raise ValueError(
            f"script={script!r} is not available for {code!r}. "
            f"Supported codes: {', '.join(sorted(table))}"
        ) from None


def build_instruction(
    text: str, tgt_lang: str, *, script: str = "native", task: str = "translate"
) -> str:
    """Render the user-turn instruction for one request.

    ``tgt_lang`` may be a FLORES-style code or a language name; it is resolved
    through :func:`resolve_target_name`. The defaults reproduce the original
    two-argument call byte-for-byte, so existing callers are unaffected.
    """
    tgt_name = resolve_target_name(tgt_lang, script=script, task=task)
    template = TRANSLITERATE_TEMPLATE if task == "transliterate" else TEMPLATE
    return template.replace("{tgt}", tgt_name).replace("{text}", text)


def build_conversation(
    text: str, tgt_lang: str, *, script: str = "native", task: str = "translate"
) -> list[dict[str, str]]:
    """Build the single-turn conversation to hand to the chat template.

    Returns a one-element list holding a lone ``user`` turn. Rendered with
    ``add_generation_prompt=True`` this is exactly the request format the model
    expects.
    """
    return [
        {"role": "user", "content": build_instruction(text, tgt_lang, script=script, task=task)}
    ]


def render_conversations(requests: Iterable[MTRequest]) -> list[list[dict[str, str]]]:
    """Render a batch of requests into chat conversations.

    **The single place an ``MTRequest`` becomes a prompt.** Every local backend
    goes through here, so there is exactly one call site that knows the contract
    and no two runtimes can drift apart.
    """
    return [build_conversation(r.text, r.tgt_lang, script=r.script, task=r.task) for r in requests]
