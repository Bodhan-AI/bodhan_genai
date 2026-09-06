"""Prompt building + audio-token extraction helpers shared by every inference path
(offline vLLM, single-GPU HF generate, serving).

Consolidates the pieces that used to live in eval/run_eval.py (extract_audio_tokens,
load_eval_jsonl), eval/generate_audios_vllm.py (_resolve_snac_ids) and
serving/replica.py (build_prompt_ids). Llama-only: prompts are produced by the
training-path SFT chat template (``build_sequence`` -> slice ``[:prompt_end]``) so
inference prompts match exactly what the model saw during training.

Pure-python module: imports only ``bodhan_genai.tts.templates.chat`` — safe to import
without torch / vllm / ray.
"""

from __future__ import annotations

import json
import logging

from bodhan_genai.tts.templates.chat import build_sequence, get_template_ids

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio-token extraction
# ---------------------------------------------------------------------------


def extract_audio_tokens(
    generated_ids: list[int],
    start_of_audio_id: int,
    end_of_audio_id: int,
) -> list[int]:
    """Extract the audio token sequence between the last <|start_of_speech|> and
    <|end_of_speech|>. Returns ``[]`` if the start marker is missing."""
    start_pos = None
    for i in range(len(generated_ids) - 1, -1, -1):
        if generated_ids[i] == start_of_audio_id:
            start_pos = i
            break
    if start_pos is None:
        return []
    end_pos = None
    for i in range(start_pos + 1, len(generated_ids)):
        if generated_ids[i] == end_of_audio_id:
            end_pos = i
            break
    if end_pos is None:
        return generated_ids[start_pos + 1 :]
    return generated_ids[start_pos + 1 : end_pos]


# ---------------------------------------------------------------------------
# Token-id resolution
# ---------------------------------------------------------------------------


def resolve_snac_ids(tokenizer) -> dict[str, int]:
    """Audio sentinel + base IDs, matching the prompt builders in
    bodhan_genai.tts.templates.chat. Always resolved from the tokenizer — never
    hardcoded — so a re-extended vocab keeps working."""
    tmpl = get_template_ids(tokenizer)
    start_id = int(tmpl["start_of_speech"][0])
    end_id = int(tmpl["end_of_speech"][0])
    base_id = int(tokenizer.convert_tokens_to_ids("<|snac_0|>"))
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else end_id
    return {
        "start_of_audio_id": start_id,
        "end_of_audio_id": end_id,
        "audio_token_base_id": base_id,
        "eos_token_id": eos_id,
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_prompt_ids(
    text: str,
    speaker: str,
    tokenizer,
    tmpl: dict | None = None,
    style: str = "",
) -> list[int]:
    """Build the SFT generation prompt input_ids from text+speaker(+style) the same
    way load_prompts_jsonl does (build_sequence -> slice to prompt_end)."""
    text = (text or "").strip()
    if not text:
        return []
    entry = {
        "text": text,
        "speaker_id": str(speaker or ""),
        "style": str(style or ""),
        "token_ids": [0],  # dummy audio (we only need the prompt half)
    }
    result = build_sequence(entry=entry, tokenizer=tokenizer, tmpl=tmpl, return_prompt_end=True)
    if result is None or "prompt_end" not in result:
        return []
    return list(result["input_ids"][: int(result["prompt_end"])])


def build_conversation_prompt_ids(
    messages: list[dict], tokenizer, tmpl: dict | None = None
) -> list[int]:
    """Conversation prompt: render a chat-style message list through the
    conversation template and slice to prompt_end.

    ``messages`` is ``[{"speaker": "S1", "text": "..."}, ...]``; turns are
    serialized to ``<|speaker>S1<speaker|>\\ntext`` blocks (see
    ``templates.conversation.format_messages``) and the whole conversation is
    synthesized as ONE continuous audio sample. Unlike the single-utterance
    prompt there is no per-utterance metadata prefix — speakers live inline in
    the conversation text. Returns ``[]`` for an empty message list; raises
    ``ValueError`` on a malformed turn."""
    from bodhan_genai.tts.templates.conversation import format_messages

    conversation_text = format_messages(messages)
    if not conversation_text:
        return []
    entry = {
        "text": conversation_text,
        "token_ids": [0],  # dummy audio (we only need the prompt half)
        "is_conversation": True,
    }
    result = build_sequence(entry=entry, tokenizer=tokenizer, tmpl=tmpl, return_prompt_end=True)
    if result is None or "prompt_end" not in result:
        return []
    return list(result["input_ids"][: int(result["prompt_end"])])


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d in %s: %s", lineno, path, e)
    return rows


def load_prompts_jsonl(
    jsonl_path: str,
    tokenizer,
    max_rows: int | None = None,
) -> list[dict]:
    """Build inference rows from a JSONL manifest by running the training-path SFT
    chat template and slicing ``input_ids[:prompt_end]``. Used by offline vLLM
    Phase A."""
    raw = _load_jsonl(jsonl_path)
    if max_rows is not None:
        raw = raw[: int(max_rows)]
    tmpl = get_template_ids(tokenizer)
    dummy_audio = [0]

    out: list[dict] = []
    skipped = 0
    for row in raw:
        text = (row.get("text") or "").strip()
        audio_filepath = row.get("audio_filepath") or ""
        language = (row.get("language") or "").strip()
        speaker = row.get("speaker_id") or row.get("speaker") or ""
        if not text:
            skipped += 1
            continue
        entry = {
            "text": text,
            "speaker_id": str(speaker) if speaker else "",
            "token_ids": dummy_audio,
        }
        result = build_sequence(
            entry=entry,
            tokenizer=tokenizer,
            tmpl=tmpl,
            return_prompt_end=True,
        )
        if result is None or "prompt_end" not in result:
            skipped += 1
            continue
        prompt_ids = result["input_ids"][: int(result["prompt_end"])]
        if not prompt_ids:
            skipped += 1
            continue
        out.append(
            {
                "_row_idx": len(out),
                "input_ids": prompt_ids,
                "text": text,
                "audio_filepath": audio_filepath,
                "language": language,
                "speaker_id": str(speaker) if speaker else "",
            }
        )
    if skipped:
        logger.warning("Skipped %d / %d JSONL rows", skipped, len(raw))
    return out
