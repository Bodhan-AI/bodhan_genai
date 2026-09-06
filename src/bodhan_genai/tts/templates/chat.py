"""
Sequence builder for Llama-TTS SFT training — simple full-loss templates
(loss on all tokens; labels are a straight copy of input_ids).

  Basic TTS:
  <|start_of_human|><bos>{metadata_prefix}{text}<|eot_id|><|end_of_human|>
  <|start_of_ai|><|start_of_speech|>{audio}<|end_of_speech|><|end_of_ai|>

  Conversation (is_conversation=True in entry):
  <|start_of_human|><bos><|speaker>S1<speaker|>\nturn1\n\n<|speaker>S2<speaker|>\nturn2...<|eot_id|><|end_of_human|>
  <|start_of_ai|><|start_of_speech|>{audio}<|end_of_speech|><|end_of_ai|>

Input data structure fields:
  token_ids        — main audio SNAC tokens
  text             — transcript / conversation text
  is_conversation  — multi-turn conversation flag (optional)
  speaker_id       — speaker identifier (optional)
  style            — optional style label (canonical or free-form)
  accent           — optional free-form accent label
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _get_token_id(tokenizer, name: str) -> int:
    """Look up a required special token id; raise loudly if the tokenizer
    doesn't have it. Without this, missing specials silently map to
    unk_token_id and downstream training trains on garbage.
    """
    tid = tokenizer.convert_tokens_to_ids(name)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(
            f"Required special token {name!r} not found in tokenizer "
            f"({tokenizer.name_or_path}). Run extend_tokenizer.py to add it."
        )
    return tid


def get_template_ids(tokenizer) -> dict[str, list[int]]:
    """Pre-compute Llama token IDs for the simple speech-chat template."""
    return {
        "start_of_human": [_get_token_id(tokenizer, "<|start_of_human|>")],
        "end_of_human": [_get_token_id(tokenizer, "<|end_of_human|>")],
        "start_of_ai": [_get_token_id(tokenizer, "<|start_of_ai|>")],
        "end_of_ai": [_get_token_id(tokenizer, "<|end_of_ai|>")],
        "start_of_speech": [_get_token_id(tokenizer, "<|start_of_speech|>")],
        "end_of_speech": [_get_token_id(tokenizer, "<|end_of_speech|>")],
        "speaker_start": [_get_token_id(tokenizer, "<|speaker>")],
        "speaker_end": [_get_token_id(tokenizer, "<speaker|>")],
        "style_start": [_get_token_id(tokenizer, "<|style>")],
        "style_end": [_get_token_id(tokenizer, "<style|>")],
        "newline": tokenizer.encode("\n", add_special_tokens=False),
        "end_of_text": [_get_token_id(tokenizer, "<|eot_id|>")],
    }


def _build_metadata_prefix_ids(
    tmpl: dict,
    tokenizer,
    speaker_id: str = "",
    style: str = "",
    accent: str = "",
) -> list[int]:
    """
    Build a structured metadata prefix for Llama SFT prompts.

    Order is always: speaker → style → accent, followed by a trailing newline
    before the transcript text when at least one metadata block is present.
    """
    blocks: list[list[int]] = []

    speaker_value = speaker_id.strip()
    if speaker_value:
        speaker_ids = tokenizer.encode(speaker_value, add_special_tokens=False)
        blocks.append(tmpl["speaker_start"] + speaker_ids + tmpl["speaker_end"])

    style_value = style.strip()
    if style_value:
        style_ids = tokenizer.encode(style_value, add_special_tokens=False)
        blocks.append(tmpl["style_start"] + style_ids + tmpl["style_end"])

    accent_value = accent.strip()
    if accent_value:
        accent_ids = tokenizer.encode(accent_value, add_special_tokens=False)
        blocks.append(tmpl["style_start"] + accent_ids + tmpl["style_end"])

    if not blocks:
        return []

    ids: list[int] = []
    for i, block in enumerate(blocks):
        if i > 0:
            ids += tmpl["newline"]
        ids += block
    ids += tmpl["newline"]
    return ids


# ---------------------------------------------------------------------------
# SFT builders
# ---------------------------------------------------------------------------


def _build_llama_sft_tts(
    text: str,
    audio_token_ids: list[int],
    tmpl: dict,
    tokenizer,
    speaker_id: str = "",
    style: str = "",
    accent: str = "",
) -> tuple[list[int], int]:
    """Simple Llama chat template with full loss on the whole sequence.

    Returns ``(ids, prompt_end)`` where ``prompt_end`` is the slice index
    inference callers must use: everything up to and **including**
    ``<|start_of_ai|>`` so the model's first emitted token is
    ``<|start_of_speech|>``. ``ids[prompt_end:]`` is the training-only audio
    target (start_of_speech → audio → end_of_speech → end_of_ai).
    """
    metadata_prefix_ids = _build_metadata_prefix_ids(
        tmpl,
        tokenizer,
        speaker_id=speaker_id,
        style=style,
        accent=accent,
    )
    text_ids = [tokenizer.bos_token_id]
    text_ids += metadata_prefix_ids
    text_ids += tokenizer.encode(text, add_special_tokens=False)
    text_ids += tmpl["end_of_text"]

    prompt_segment = tmpl["start_of_human"] + text_ids + tmpl["end_of_human"] + tmpl["start_of_ai"]
    prompt_end = len(prompt_segment)
    ids = (
        prompt_segment
        + tmpl["start_of_speech"]
        + audio_token_ids
        + tmpl["end_of_speech"]
        + tmpl["end_of_ai"]
    )
    return ids, prompt_end


def _build_llama_sft_tts_conversation(
    conversation_text: str,
    audio_token_ids: list[int],
    tmpl: dict,
    tokenizer,
) -> tuple[list[int], int]:
    """Llama SFT for multi-turn conversations.

    Same layout as _build_llama_sft_tts but without a metadata prefix — speaker
    labels are already embedded in conversation_text via <|speaker>...<speaker|>.

    Layout:
      <|start_of_human|>
        <bos> conversation_text <|eot_id|>
      <|end_of_human|>
      <|start_of_ai|>
        <|start_of_speech|> audio_tokens <|end_of_speech|>
      <|end_of_ai|>
    """
    text_ids = [tokenizer.bos_token_id]
    text_ids += tokenizer.encode(conversation_text, add_special_tokens=False)
    text_ids += tmpl["end_of_text"]

    prompt_segment = tmpl["start_of_human"] + text_ids + tmpl["end_of_human"] + tmpl["start_of_ai"]
    prompt_end = len(prompt_segment)
    ids = (
        prompt_segment
        + tmpl["start_of_speech"]
        + audio_token_ids
        + tmpl["end_of_speech"]
        + tmpl["end_of_ai"]
    )
    return ids, prompt_end


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_sequence(
    entry: dict,
    tokenizer,
    tmpl: dict | None = None,
    return_prompt_end: bool = False,
) -> dict | None:
    """
    Build a Llama SFT training sequence from a tokenized dataset row.

    Args:
        entry: Tokenized row with fields: token_ids (audio), text,
               is_conversation (optional),
               speaker_id or speaker (optional), style / accent (optional).
        tokenizer: Extended tokenizer with audio special tokens.
        tmpl: Pre-computed template token IDs from get_template_ids().
              Pass this for efficiency when processing many entries.
        return_prompt_end: If True, the returned dict also contains
            ``prompt_end`` — the index where the user turn ends and the
            model turn begins. Inference callers slice ``input_ids[:prompt_end]``
            to get a generation prompt.

    Returns:
        dict with keys: input_ids (list[int]), labels (list[int]), length (int).
        Labels are a full copy of input_ids (full-sequence loss).
        When return_prompt_end=True, also contains prompt_end (int).
        Returns None if required data is missing.
    """
    if tmpl is None:
        tmpl = get_template_ids(tokenizer)

    audio_token_ids = entry.get("token_ids") or []
    text = entry.get("text", "") or ""
    speaker_id = entry.get("speaker_id", "") or entry.get("speaker", "") or ""
    style = entry.get("style", "") or ""
    accent = entry.get("accent", "") or ""
    has_text = bool(text.strip())
    has_audio = bool(audio_token_ids)

    if not (has_text and has_audio):
        logger.warning(
            f"Llama SFT requires text+audio rows (has_text={has_text}, has_audio={has_audio})."
        )
        return None

    if entry.get("is_conversation"):
        # Llama multi-turn conversation: speaker labels embedded in text.
        ids, prompt_end = _build_llama_sft_tts_conversation(
            conversation_text=text,
            audio_token_ids=audio_token_ids,
            tmpl=tmpl,
            tokenizer=tokenizer,
        )
    else:
        ids, prompt_end = _build_llama_sft_tts(
            text=text,
            audio_token_ids=audio_token_ids,
            tmpl=tmpl,
            tokenizer=tokenizer,
            speaker_id=speaker_id,
            style=style,
            accent=accent,
        )

    # Llama SFT is full-loss: labels are a straight copy of the input ids.
    labels = ids[:]
    result = {"input_ids": ids, "labels": labels, "length": len(ids)}

    if return_prompt_end:
        result["prompt_end"] = int(prompt_end)

    return result
