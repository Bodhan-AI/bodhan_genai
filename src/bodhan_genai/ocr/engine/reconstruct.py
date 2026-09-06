"""Assemble transcribed blocks into the page's markdown.

stdlib-only -- importable with no GPU stack and no PIL.
"""

from __future__ import annotations

import re

from bodhan_genai.ocr.engine.types import Block
from bodhan_genai.ocr.templates.contract import DROP_TYPES

# Dashes as codepoints: several of the seven are indistinguishable in a source file.
# 002D HYPHEN-MINUS | 2010-2014 HYPHEN..EM DASH | 2212 MINUS SIGN
_HYPHEN_BREAK = re.compile("(\\w)[-\\u2010-\\u2014\\u2212]\\n[ \\t]*(\\w)")
_MATH_DELIMITERS = ("$", "\\[", "\\(")

# A math span: $$...$$ (display) or $...$ (inline).
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", re.S)

# Runs of the scripts IndicBlockOCR transcribes: Arabic (Urdu, Kashmiri, Sindhi),
# Devanagari..Malayalam, and Ol Chiki (Santali). Intervening spaces and the ZW(N)J joiners that
# Indic shaping relies on are kept inside the run so one run does not fragment into many.
# The run must both start and end on a script character, so a trailing space stays outside the
# \text{} and keeps separating it from what follows.
_NON_LATIN_RUN = re.compile(
    "[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f]"
    "(?:[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f \\u200c\\u200d]*"
    "[\\u0600-\\u06ff\\u0900-\\u0d7f\\u1c50-\\u1c7f])?"
)
_TEXT_CMD = re.compile(r"\\text\{[^{}]*\}")


def dehyphenate(text: str) -> str:
    """Rejoin words split by a hyphen at a line break.

    Iterates to a fixpoint: re.sub matches non-overlappingly, so in "a-\\nb-\\nc" the first pass
    consumes the "b" the second break needs.
    """
    previous = None
    while previous != text:
        previous = text
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return text


def _repair_expression(tex: str, display: bool) -> str:
    """Make one transcribed expression valid LaTeX.

    Two repairs, both for things the recognizer emits that no LaTeX engine accepts:

    * **Indic script in math mode.** ``$$প্রোটন = 9$$`` is invalid -- math mode has no glyphs for
      those codepoints, so KaTeX, MathJax and a real TeX run all fail on it. Each run is wrapped
      in ``\\text{}``, which is what the recognizer itself does when it gets it right. Runs
      already inside ``\\text{}`` are left alone.
    * **Bare newlines in display math**, which are a syntax error; ``\\\\`` is the row separator.
    """
    protected: list[str] = []

    def stash(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    tex = _TEXT_CMD.sub(stash, tex)
    tex = _NON_LATIN_RUN.sub(lambda m: f"\\text{{{m.group(0)}}}", tex)
    tex = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], tex)

    if display:
        tex = re.sub(r"\s*\n\s*", r" \\\\ ", tex.strip())
    return tex


def repair_math(text: str) -> str:
    """Repair every math span in a markdown string. Prose outside ``$`` is untouched."""

    def fix(match: re.Match) -> str:
        display = match.group(1) is not None
        inner = _repair_expression(match.group(1) or match.group(2), display)
        return f"$${inner}$$" if display else f"${inner}$"

    return _MATH_SPAN.sub(fix, text)


def reconstruct(blocks: list[Block], repair: bool = True) -> str:
    """Reading-ordered markdown. Blocks with no text contribute nothing but are not removed --
    they still appear in the JSON with text "" .

    ``repair=False`` emits the recognizer's math verbatim, including expressions no LaTeX engine
    can render. Only useful for comparing byte-for-byte against output produced before the repair
    existed.
    """
    kept = sorted((b for b in blocks if b.type not in DROP_TYPES), key=lambda b: b.order)

    parts = []
    for block in kept:
        text = (block.text or "").strip()
        if not text:
            continue
        # Bare LaTeX would render as literal source, so wrap it -- but only when the recognizer
        # supplied no delimiters at all. Checking just the first character is not enough: an
        # Equation block often comes back as a prose prefix followed by already-delimited math
        # ("বা, $\\frac{a}{b}$"), and wrapping that produces "$$...$$$", which nothing can parse.
        if block.type == "Equation" and "$" not in text and not text.startswith(_MATH_DELIMITERS):
            text = f"$${text}$$"
        parts.append(text)

    markdown = dehyphenate("\n\n".join(parts))
    return repair_math(markdown) if repair else markdown
