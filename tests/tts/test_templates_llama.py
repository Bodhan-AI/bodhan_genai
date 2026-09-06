"""Tests for the Llama basic-TTS chat-template builder's metadata prefix."""

from __future__ import annotations

from bodhan_genai.tts.templates.chat import _build_llama_sft_tts


class _StubTokenizer:
    """Minimal tokenizer that emits 1 id per word — enough for the builder."""

    unk_token_id = 0
    bos_token_id = 99

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        return [hash(w) % 30000 + 100 for w in s.split() if w]

    def convert_tokens_to_ids(self, tok):
        return 99999  # never used when tmpl is supplied


def _make_llama_tmpl() -> dict:
    """Mimic the Llama tmpl shape get_template_ids produces (1-token lists)."""
    return {
        "start_of_human": [1],
        "end_of_human": [2],
        "start_of_ai": [3],
        "end_of_ai": [4],
        "start_of_speech": [5],
        "end_of_speech": [6],
        "speaker_start": [8],
        "speaker_end": [9],
        "style_start": [10],
        "style_end": [11],
        "newline": [13],
        "end_of_text": [7],
    }


def test_tts_uses_structured_speaker_and_style_metadata():
    tmpl = _make_llama_tmpl()
    tok = _StubTokenizer()

    ids, prompt_end = _build_llama_sft_tts(
        text="hello",
        audio_token_ids=[10, 11],
        tmpl=tmpl,
        tokenizer=tok,
        speaker_id="alice",
        style="happy",
        accent="indian english",
    )

    human_end = ids.index(2)
    user_segment = ids[1:human_end]

    assert 8 in user_segment and 9 in user_segment
    first_style_open = user_segment.index(10)
    first_style_close = user_segment.index(11)
    second_style_open = user_segment.index(10, first_style_close + 1)
    second_style_close = user_segment.index(11, second_style_open + 1)
    assert user_segment[first_style_open + 1 : first_style_close] == tok.encode(
        "happy", add_special_tokens=False
    )
    assert user_segment[second_style_open + 1 : second_style_close] == tok.encode(
        "indian english", add_special_tokens=False
    )
    assert ids[prompt_end] == 5
