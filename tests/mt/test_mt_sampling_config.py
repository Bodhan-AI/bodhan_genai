"""MTSamplingConfig / MTResult semantics."""

from __future__ import annotations

import dataclasses

import pytest

from bodhan_genai.mt.engine.types import MTResult, MTSamplingConfig


def test_defaults_are_greedy():
    """Greedy is the recommended setting for translation and the only one that is
    reproducible run to run."""
    sc = MTSamplingConfig()
    assert sc.temperature == 0.0
    assert sc.top_p == 1.0
    assert sc.repetition_penalty == 1.0
    assert sc.max_new_tokens == 512
    assert sc.seed is None
    assert sc.greedy


@pytest.mark.parametrize(
    ("temperature", "expected_greedy"),
    [(0.0, True), (0.7, False), (1.0, False)],
)
def test_greedy_property(temperature, expected_greedy):
    assert MTSamplingConfig(temperature=temperature).greedy is expected_greedy


def test_config_is_frozen():
    sc = MTSamplingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sc.temperature = 0.9  # type: ignore[misc]


def test_merged_applies_overrides_and_returns_a_new_object():
    base = MTSamplingConfig()
    derived = base.merged(temperature=0.7, max_new_tokens=2048)

    assert derived.temperature == 0.7
    assert derived.max_new_tokens == 2048
    assert base.temperature == 0.0, "base config was mutated"
    assert derived is not base


def test_merged_ignores_none_so_cli_kwargs_pass_straight_through():
    """Per-request kwargs arrive with unset values as None; the config must treat
    those as 'no override' rather than as a value."""
    base = MTSamplingConfig(max_new_tokens=2048)
    merged = base.merged(temperature=None, max_new_tokens=None, top_p=0.9)

    assert merged.max_new_tokens == 2048
    assert merged.temperature == 0.0
    assert merged.top_p == 0.9


def test_merged_rejects_unknown_fields():
    with pytest.raises(TypeError, match="Unknown MTSamplingConfig field\\(s\\): top_k"):
        MTSamplingConfig().merged(top_k=50)


def test_merged_lists_every_unknown_field():
    with pytest.raises(TypeError, match="beam_size, top_k"):
        MTSamplingConfig().merged(top_k=50, beam_size=4)


# --------------------------------------------------------------------------- #
# MTResult
# --------------------------------------------------------------------------- #


def test_result_ok_reflects_error():
    assert MTResult(text="hi").ok
    assert not MTResult(error="boom").ok


def test_as_record_matches_the_shipped_jsonl_schema():
    record = MTResult(
        text="नमस्ते", source="hello", src_lang="eng_Latn", tgt_lang="Hindi"
    ).as_record()
    assert record == {
        "source": "hello",
        "translation": "नमस्ते",
        "src_lang": "eng_Latn",
        "tgt_lang": "Hindi",
    }


def test_as_record_carries_the_error_only_when_there_is_one():
    assert "error" not in MTResult(text="ok").as_record()
    assert MTResult(error="boom").as_record()["error"] == "boom"
