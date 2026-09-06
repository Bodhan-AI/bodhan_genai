"""Helpers for formatting multi-turn conversation records into TTS training text.

Each record is expected to come from a conversation JSONL (e.g. manifest_full_data.jsonl)
with the following relevant fields:

  turns               — list of {role, text, ...}
  tts_generation      — dict with speaker_voices: {"Teacher": "Charon", "Student": "Kore"}

Example output of format_conversation_text:

  <|speaker>Charon<speaker|>
  Teacher text here

  <|speaker>Kore<speaker|>
  Student text here

"""

from __future__ import annotations


def format_messages(messages: list[dict]) -> str:
    """Format a chat-style message list into the tagged conversation string the
    conversation template consumes.

    Each message is ``{"speaker": "S1", "text": "..."}``; output is
    ``<|speaker>S1<speaker|>\\ntext`` per turn, turns separated by a blank line —
    the exact serialization the model saw for conversation rows in training.
    Raises ``ValueError`` on a missing/blank speaker or text so a malformed turn
    can't silently collapse into the previous one."""
    if not messages:
        return ""
    parts = []
    for i, message in enumerate(messages):
        speaker = str(message.get("speaker", "") or "").strip()
        text = str(message.get("text", "") or "").strip()
        if not speaker:
            raise ValueError(f"messages[{i}] has no speaker")
        if not text:
            raise ValueError(f"messages[{i}] has no text")
        parts.append(f"<|speaker>{speaker}<speaker|>\n{text}")
    return "\n\n".join(parts)


def _speaker_voices(record: dict) -> dict[str, str]:
    tts = record.get("tts_generation") or {}
    voices = tts.get("speaker_voices") or {}
    return {k.lower(): v for k, v in voices.items()}


def format_conversation_text(record: dict) -> str:
    """Serialize a conversation record into a tagged text string.

    Each turn is formatted as:
        <|speaker>VoiceName<speaker|>\nturn_text

    Turns are separated by a blank line (\\n\\n).
    Falls back to the role name if tts_generation.speaker_voices is absent.
    """
    voices = _speaker_voices(record)
    parts = []
    for turn in record.get("turns", []):
        role = turn.get("role", "")
        voice = voices.get(role.lower(), role)
        text = turn.get("text", "")
        parts.append(f"<|speaker>{voice}<speaker|>\n{text}")
    return "\n\n".join(parts)


def expand_conversation_turns(record: dict) -> list[dict]:
    """Expand a conversation record into per-turn rows.

    Each returned dict contains:
      text               — turn text (translated)
      speaker            — voice name (e.g. "Charon")
      role               — "teacher" or "student"
      turn_index         — 0-based position in the conversation
      sample_id          — from the parent record
      language           — target_language code (e.g. "ta")
      target_language_name — human-readable language name (e.g. "Tamil")
    """
    voices = _speaker_voices(record)
    meta = {
        "sample_id": record.get("sample_id", ""),
        "language": record.get("target_language", ""),
        "target_language_name": record.get("target_language_name", ""),
    }
    rows = []
    for i, turn in enumerate(record.get("turns", [])):
        role = turn.get("role", "")
        voice = voices.get(role.lower(), role)
        rows.append(
            {
                "text": turn.get("text", ""),
                "speaker": voice,
                "role": role,
                "turn_index": i,
                **meta,
            }
        )
    return rows
