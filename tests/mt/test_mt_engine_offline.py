"""IndicMTEngine behaviour, driven through an injected fake backend.

No model is loaded anywhere here: the engine's job is to own the prompt contract,
batch alignment and failure isolation, and all three are testable on CPU with a
stand-in backend.
"""

from __future__ import annotations

import pytest

from bodhan_genai.mt.engine.offline import IndicMTEngine, TranslationBackend
from bodhan_genai.mt.engine.types import MTSamplingConfig


class FakeBackend:
    """Records what it was asked for and echoes a deterministic completion."""

    def __init__(self, completions: list[str] | None = None) -> None:
        self.calls: list[tuple[list[list[dict[str, str]]], MTSamplingConfig]] = []
        self._completions = completions
        self.closed = False

    def generate(self, conversations, sc):
        self.calls.append((conversations, sc))
        if self._completions is not None:
            return self._completions
        return [f"translated:{c[0]['content'][-12:]}" for c in conversations]

    def close(self):
        self.closed = True


class BoomBackend:
    def generate(self, conversations, sc):
        raise RuntimeError("CUDA out of memory")

    def close(self):
        pass


def test_fake_backend_satisfies_the_protocol():
    assert isinstance(FakeBackend(), TranslationBackend)


# --------------------------------------------------------------------------- #
# Prompt contract is applied by the engine, not the caller
# --------------------------------------------------------------------------- #


def test_engine_builds_the_contract_conversation():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    engine.translate("Hello world.", tgt_lang="hin_Deva")

    conversations, _ = backend.calls[0]
    assert conversations == [
        [
            {
                "role": "user",
                "content": "Translate the following text into Hindi:\n\nHello world.",
            }
        ]
    ]


def test_src_lang_is_recorded_but_never_prompted():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    result = engine.translate("Bonjour.", tgt_lang="Hindi", src_lang="fra_Latn")

    assert result.src_lang == "fra_Latn"
    assert result.as_record()["src_lang"] == "fra_Latn"
    content = backend.calls[0][0][0][0]["content"]
    assert "fra" not in content and "French" not in content


def test_tgt_lang_is_resolved_to_the_display_name_in_the_result():
    engine = IndicMTEngine("dummy", backend=FakeBackend())
    assert engine.translate("x", tgt_lang="mni_Beng").tgt_lang == "Manipuri (Bengali script)"
    assert engine.translate("x", tgt_lang="sindhi").tgt_lang == "Sindhi (Devanagari script)"


def test_bad_language_fails_before_the_backend_is_touched():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    with pytest.raises(ValueError, match="unsupported language"):
        engine.translate("x", tgt_lang="klingon")
    assert backend.calls == [], "backend was invoked despite an invalid language"


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_batch_is_one_backend_call_and_stays_aligned():
    backend = FakeBackend(completions=["A", "B", "C"])
    engine = IndicMTEngine("dummy", backend=backend)
    results = engine.translate_batch(["one", "two", "three"], tgt_lang="Tamil")

    assert len(backend.calls) == 1, "batch should be a single generate call"
    assert [r.text for r in results] == ["A", "B", "C"]
    assert [r.source for r in results] == ["one", "two", "three"]


def test_empty_batch_short_circuits():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    assert engine.translate_batch([], tgt_lang="Hindi") == []
    assert backend.calls == []


def test_misaligned_backend_output_raises_rather_than_mispairing():
    """Silently zipping mismatched lists would attach translations to the wrong
    sources — worse than failing."""
    backend = FakeBackend(completions=["only one"])
    engine = IndicMTEngine("dummy", backend=backend)
    with pytest.raises(RuntimeError, match="misaligned"):
        engine.translate_batch(["a", "b"], tgt_lang="Hindi")


# --------------------------------------------------------------------------- #
# Sampling overrides
# --------------------------------------------------------------------------- #


def test_per_call_overrides_reach_the_backend():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    engine.translate("x", tgt_lang="Hindi", max_new_tokens=4096, temperature=0.7)

    _, sc = backend.calls[0]
    assert sc.max_new_tokens == 4096
    assert sc.temperature == 0.7
    assert not sc.greedy


def test_constructor_sampling_is_the_default_and_is_not_mutated():
    base = MTSamplingConfig(max_new_tokens=2048)
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend, sampling=base)

    engine.translate("x", tgt_lang="Hindi")
    engine.translate("x", tgt_lang="Hindi", max_new_tokens=8192)

    assert backend.calls[0][1].max_new_tokens == 2048
    assert backend.calls[1][1].max_new_tokens == 8192
    assert base.max_new_tokens == 2048, "frozen config was mutated"


def test_unknown_override_raises():
    engine = IndicMTEngine("dummy", backend=FakeBackend())
    with pytest.raises(TypeError, match="Unknown MTSamplingConfig field"):
        engine.translate("x", tgt_lang="Hindi", top_k=50)


def test_translate_document_is_a_single_request():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    engine.translate_document("para one\n\npara two", tgt_lang="Tamil", max_new_tokens=8192)

    conversations, sc = backend.calls[0]
    assert len(conversations) == 1, "a document must not be split into segments"
    assert "para one\n\npara two" in conversations[0][0]["content"]
    assert sc.max_new_tokens == 8192


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #


def test_backend_failure_lands_in_error_instead_of_raising():
    engine = IndicMTEngine("dummy", backend=BoomBackend())
    results = engine.translate_batch(["a", "b"], tgt_lang="Hindi")

    assert len(results) == 2
    for r in results:
        assert not r.ok
        assert "CUDA out of memory" in r.error
        assert r.text == ""
        assert "error" in r.as_record()


# --------------------------------------------------------------------------- #
# Construction guards
# --------------------------------------------------------------------------- #


def test_adapter_dir_with_vllm_is_rejected_with_the_fix_in_the_message():
    with pytest.raises(ValueError, match="backend='hf'") as excinfo:
        IndicMTEngine("dummy", backend="vllm", adapter_dir="/some/adapter")
    assert "merge" in str(excinfo.value)


def test_unknown_backend_string_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        IndicMTEngine("dummy", backend="tensorrt")


def test_context_manager_closes_the_backend():
    backend = FakeBackend()
    with IndicMTEngine("dummy", backend=backend) as engine:
        engine.translate("x", tgt_lang="Hindi")
    assert backend.closed


def test_close_is_idempotent():
    backend = FakeBackend()
    engine = IndicMTEngine("dummy", backend=backend)
    engine.close()
    engine.close()
    assert backend.closed
