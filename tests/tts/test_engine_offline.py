"""Tests for IndicTTSEngine (bodhan_genai.tts.engine.offline).

Everything runs CPU-only through a FakeBackend implementing SynthesisBackend
and the shared MockSNACModel — no vllm, no transformers, no CUDA. Token ids
come from the frozen llama3-TTS layout (see conftest.FROZEN_SPECIALS).
"""

from __future__ import annotations

import numpy as np
import pytest

from bodhan_genai.tts.engine.offline import IndicTTSEngine, SynthesisBackend
from bodhan_genai.tts.engine.types import SamplingConfig, TTSResult
from bodhan_genai.tts.inference.prompts import build_prompt_ids

# Frozen-contract ids (must match conftest.FROZEN_SPECIALS).
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
SNAC_BASE = 128266
EOT_ID = 128009


def snac_token_ids(n_frames: int, seed: int = 0) -> list[int]:
    """Offset-correct flattened SNAC ids: base + (i % 7) * 4096 + code."""
    return [SNAC_BASE + (i % 7) * 4096 + ((seed + i) % 4096) for i in range(7 * n_frames)]


def scripted_generation(n_frames: int = 3, seed: int = 0) -> list[int]:
    return [START_OF_SPEECH, *snac_token_ids(n_frames, seed), END_OF_SPEECH]


class FakeBackend:
    """Records every generate() call; returns scripted outputs."""

    def __init__(self, outputs: list[list[int]] | None = None, n_frames: int = 3):
        self.calls: list[tuple[list[list[int]], SamplingConfig, list[int]]] = []
        self.closed = False
        self._outputs = outputs
        self._n_frames = n_frames

    def generate(self, prompt_ids_list, sc, stop_ids):
        self.calls.append(([list(p) for p in prompt_ids_list], sc, list(stop_ids)))
        if self._outputs is not None:
            assert len(self._outputs) >= len(prompt_ids_list)
            return [list(o) for o in self._outputs[: len(prompt_ids_list)]]
        return [scripted_generation(self._n_frames, seed=i) for i in range(len(prompt_ids_list))]

    def close(self):
        self.closed = True


def make_engine(frozen_tokenizer, mock_snac_model, backend=None, **kwargs):
    backend = backend if backend is not None else FakeBackend()
    # The codec is stubbed here, so there is no real decoder to swap and no reason
    # to reach the Hub for one. Vocos itself is covered by test_vocos_codec.py.
    kwargs.setdefault("vocos", False)
    engine = IndicTTSEngine(
        "fake-model",
        backend=backend,
        tokenizer=frozen_tokenizer,
        device="cpu",
        snac_loader=lambda *a, **k: mock_snac_model,
        **kwargs,
    )
    return engine, backend


class TestPromptRouting:
    def test_basic_prompt_matches_build_prompt_ids(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.synthesize("hello there", speaker="spk1")
        expected = build_prompt_ids("hello there", "spk1", frozen_tokenizer)
        assert expected  # sanity: builder produced a real prompt
        ((prompt_ids_list, _, _),) = backend.calls
        assert prompt_ids_list == [expected]

    def test_stop_ids_resolved_from_tokenizer(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.synthesize("hi")
        ((_, _, stop_ids),) = backend.calls
        assert stop_ids == [END_OF_SPEECH, EOT_ID]


class TestSamplingMerge:
    def test_ctor_sampling_and_per_call_override(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(
            frozen_tokenizer,
            mock_snac_model,
            sampling=SamplingConfig(temperature=0.3, max_new_tokens=512),
        )
        engine.synthesize("hi", top_p=0.5)
        ((_, sc, _),) = backend.calls
        assert sc.temperature == 0.3  # from ctor sampling
        assert sc.max_new_tokens == 512  # from ctor sampling
        assert sc.top_p == 0.5  # per-call override
        assert sc.repetition_penalty == 1.1  # untouched default

    def test_defaults_used_when_nothing_overridden(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.synthesize("hi")
        ((_, sc, _),) = backend.calls
        assert sc == SamplingConfig()


class TestFullPipeline:
    def test_synthesize_returns_audio_result(self, frozen_tokenizer, mock_snac_model, tmp_path):
        n_frames = 4
        engine, _ = make_engine(
            frozen_tokenizer, mock_snac_model, backend=FakeBackend(n_frames=n_frames)
        )
        result = engine.synthesize("hello world", speaker="spk1")

        assert isinstance(result, TTSResult)
        assert result.error is None
        assert result.sample_rate == 24_000 == engine.sample_rate
        assert result.audio.dtype == np.float32
        assert result.audio.size > 0
        assert float(np.abs(result.audio).max()) <= 1.0
        assert result.audio_tokens == 7 * n_frames
        assert result.generated_tokens == 7 * n_frames + 2  # + start/end markers
        assert result.prompt_tokens == len(
            build_prompt_ids("hello world", "spk1", frozen_tokenizer)
        )
        assert result.gen_time_s >= 0.0 and result.decode_time_s >= 0.0

        out = result.save(tmp_path / "out.wav")
        assert out.exists()
        import soundfile as sf

        audio, sr = sf.read(out)
        assert sr == 24_000
        assert len(audio) == result.audio.size

    def test_batch_makes_exactly_one_generate_call(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        texts = ["first sentence", "second sentence", "third sentence"]
        results = engine.synthesize_batch(texts, speakers="spk1")
        assert len(results) == 3
        assert len(backend.calls) == 1
        assert len(backend.calls[0][0]) == 3  # all three prompts in the one call
        assert all(r.error is None for r in results)
        assert all(r.audio.dtype == np.float32 and r.audio.size > 0 for r in results)

    def test_speaker_broadcast_and_mismatch(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.synthesize_batch(["a b", "c d"], speakers="spk9")
        prompts = backend.calls[0][0]
        assert prompts[0] == build_prompt_ids("a b", "spk9", frozen_tokenizer)
        assert prompts[1] == build_prompt_ids("c d", "spk9", frozen_tokenizer)
        with pytest.raises(ValueError):
            engine.synthesize_batch(["a", "b"], speakers=["only-one"])

    def test_empty_text_row_errors_without_generate(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        results = engine.synthesize_batch(["", "real text"])
        assert results[0].error is not None
        assert results[1].error is None
        # only the valid prompt reached the backend
        assert len(backend.calls) == 1
        assert len(backend.calls[0][0]) == 1


class TestErrorPaths:
    def test_missing_start_marker_sets_error_and_raises(self, frozen_tokenizer, mock_snac_model):
        # generation with no <|start_of_speech|> at all
        bad = [*snac_token_ids(2), END_OF_SPEECH]
        engine, _ = make_engine(
            frozen_tokenizer, mock_snac_model, backend=FakeBackend(outputs=[bad])
        )
        results = engine.synthesize_batch(["hello"])
        assert results[0].error is not None
        assert results[0].audio.size == 0
        with pytest.raises(RuntimeError):
            engine.synthesize("hello")

    def test_bad_row_does_not_poison_batch(self, frozen_tokenizer, mock_snac_model):
        outputs = [scripted_generation(2), snac_token_ids(2), scripted_generation(3)]
        engine, _ = make_engine(
            frozen_tokenizer, mock_snac_model, backend=FakeBackend(outputs=outputs)
        )
        results = engine.synthesize_batch(["one two", "three four", "five six"])
        assert results[0].error is None and results[0].audio.size > 0
        assert results[1].error is not None
        assert results[2].error is None and results[2].audio.size > 0

    def test_adapter_dir_with_vllm_backend_raises(self):
        with pytest.raises(ValueError, match="adapter_dir"):
            IndicTTSEngine("fake-model", backend="vllm", adapter_dir="/some/adapter")


class TestLifecycle:
    def test_close_calls_backend_close(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.close()
        assert backend.closed
        engine.close()  # idempotent

    def test_context_manager_closes(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        with engine as e:
            assert e is engine
            e.synthesize("hi")
        assert backend.closed

    def test_fake_backend_satisfies_protocol(self):
        assert isinstance(FakeBackend(), SynthesisBackend)


class TestConversation:
    """synthesize_conversation: one continuous sample from a message list."""

    def test_prompt_matches_conversation_builder(self, frozen_tokenizer, mock_snac_model):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        msgs = [
            {"speaker": "S1", "text": "hello there"},
            {"speaker": "S2", "text": "hi back"},
        ]
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        result = engine.synthesize_conversation(msgs)
        expected = build_conversation_prompt_ids(msgs, frozen_tokenizer)
        assert expected
        ((prompt_ids_list, _, _),) = backend.calls
        assert prompt_ids_list == [expected]
        assert result.error is None
        assert result.audio.size > 0
        assert result.prompt_tokens == len(expected)

    def test_sampling_overrides_reach_backend(self, frozen_tokenizer, mock_snac_model):
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        engine.synthesize_conversation(
            [{"speaker": "S1", "text": "hi"}], temperature=0.2, max_new_tokens=64
        )
        ((_, sc, _),) = backend.calls
        assert sc.temperature == 0.2
        assert sc.max_new_tokens == 64

    def test_empty_messages_raises(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_engine(frozen_tokenizer, mock_snac_model)
        with pytest.raises(ValueError, match="empty conversation"):
            engine.synthesize_conversation([])

    def test_blank_speaker_raises(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_engine(frozen_tokenizer, mock_snac_model)
        with pytest.raises(ValueError, match="no speaker"):
            engine.synthesize_conversation([{"speaker": "", "text": "hi"}])


class TestConversationBatch:
    """synthesize_conversation_batch: ONE generate call for many conversations."""

    def test_batched_conversations_single_generate_call(self, frozen_tokenizer, mock_snac_model):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        convs = [
            [{"speaker": "S1", "text": "hello there"}, {"speaker": "S2", "text": "hi back"}],
            [{"speaker": "S1", "text": "a second conversation"}],
            [{"speaker": "S2", "text": "third one"}, {"speaker": "S1", "text": "yes indeed"}],
        ]
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        results = engine.synthesize_conversation_batch(convs)
        assert len(results) == 3
        assert len(backend.calls) == 1  # ONE batched generate for all conversations
        expected = [build_conversation_prompt_ids(c, frozen_tokenizer) for c in convs]
        assert all(expected)  # sanity: builder produced real prompts
        assert backend.calls[0][0] == expected
        assert all(r.error is None for r in results)
        assert all(r.audio.dtype == np.float32 and r.audio.size > 0 for r in results)
        for r, ids in zip(results, expected, strict=True):
            assert r.prompt_tokens == len(ids)

    def test_bad_row_does_not_poison_batch(self, frozen_tokenizer, mock_snac_model):
        valid = [{"speaker": "S1", "text": "hello there"}]
        engine, backend = make_engine(frozen_tokenizer, mock_snac_model)
        results = engine.synthesize_conversation_batch([valid, [], valid])
        assert results[0].error is None and results[0].audio.size > 0
        assert results[1].error is not None
        assert results[2].error is None and results[2].audio.size > 0
        # only the two valid prompts reached the backend, in one call
        assert len(backend.calls) == 1
        assert len(backend.calls[0][0]) == 2
