"""ChunkedIndicStreamingTTS over both engines, CPU-only via the existing fakes."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_engine_offline import FakeBackend
from test_engine_offline import make_engine as make_offline_engine
from test_engine_streaming import (
    FakeAsyncLLM,
    make_codes,
    make_token_stream,
    uneven_chunks,
)
from test_engine_streaming import (
    make_engine as make_stream_engine,
)

from bodhan_genai.tts.engine.chunked import ChunkedIndicStreamingTTS

LONG_TEXT = (
    "This is the first sentence of a long paragraph. "
    "Here comes a second sentence with more words in it. "
    "And finally a third sentence to close the paragraph."
)


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------


class TestOfflineLong:
    def make(self, frozen_tokenizer, mock_snac_model, **kw):
        engine, backend = make_offline_engine(frozen_tokenizer, mock_snac_model)
        return ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, **kw), backend

    def test_single_generate_call_with_all_chunks(self, frozen_tokenizer, mock_snac_model):
        chunked, backend = self.make(frozen_tokenizer, mock_snac_model)
        n_chunks = len(chunked.chunk(LONG_TEXT))
        assert n_chunks >= 2
        result = chunked.synthesize_long(LONG_TEXT, speaker="spk1")
        assert len(backend.calls) == 1  # ONE batched generate for all chunks
        (prompt_ids_list, _, _) = backend.calls[0]
        assert len(prompt_ids_list) == n_chunks
        assert result.audio.dtype == np.float32
        assert result.sample_rate == 24_000

    def test_gap_arithmetic_without_normalize(self, frozen_tokenizer, mock_snac_model):
        chunked, _ = self.make(frozen_tokenizer, mock_snac_model, normalize=False, gap_ms=250.0)
        merged, chunks = chunked.synthesize_long(LONG_TEXT, return_chunks=True)
        gap = int(0.250 * 24_000)
        expected = sum(c.audio.size for c in chunks) + gap * (len(chunks) - 1)
        assert merged.audio.size == expected
        assert merged.audio_tokens == sum(c.audio_tokens for c in chunks)
        assert merged.prompt_tokens == sum(c.prompt_tokens for c in chunks)

    def test_failed_chunk_raises_with_index(self, frozen_tokenizer, mock_snac_model):
        # Pre-script one bad row: chunk 1 produces no start-of-speech marker.
        probe = ChunkedIndicStreamingTTS(
            make_offline_engine(frozen_tokenizer, mock_snac_model)[0], max_chunk_chars=60
        )
        n = len(probe.chunk(LONG_TEXT))
        from test_engine_offline import scripted_generation

        outputs = [scripted_generation(3, seed=i) for i in range(n)]
        outputs[1] = [999]  # no <|start_of_speech|> -> extract_audio_tokens fails
        engine, _ = make_offline_engine(
            frozen_tokenizer, mock_snac_model, backend=FakeBackend(outputs=outputs)
        )
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60)
        with pytest.raises(RuntimeError, match="chunk 1"):
            chunked.synthesize_long(LONG_TEXT)

    def test_empty_text_raises(self, frozen_tokenizer, mock_snac_model):
        chunked, _ = self.make(frozen_tokenizer, mock_snac_model)
        with pytest.raises(ValueError, match="empty text"):
            chunked.synthesize_long("   ")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

GAP_BYTES = int(0.250 * 24_000) * 2


class MultiCallAsyncLLM(FakeAsyncLLM):
    """FakeAsyncLLM whose generate() serves the SAME scripted chunk stream on
    every call (one call per text chunk in stream_long)."""


def run_stream_long(chunked, text, **kw):
    async def _run():
        frames = []
        async for msg in chunked.stream_long(text, **kw):
            frames.append(msg)
        await chunked._engine.shutdown()
        return frames

    return asyncio.run(_run())


class TestStreamingLong:
    def build(self, frozen_tokenizer, n_frames=6, normalize=True, **kw):
        codes = make_codes(n_frames, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, normalize=normalize, **kw)
        return chunked, fake

    def test_structure_without_normalize(self, frozen_tokenizer):
        chunked, fake = self.build(frozen_tokenizer, normalize=False)
        n_chunks = len(chunked.chunk(LONG_TEXT))
        assert n_chunks >= 2
        msgs = run_stream_long(chunked, LONG_TEXT, speaker="spk1")
        # one generate call per text chunk
        assert len(fake.generate_calls) == n_chunks
        # exactly n_chunks-1 gap messages of zero PCM, in between audio
        gaps = [m for m in msgs if len(m) == GAP_BYTES and not any(m)]
        assert len(gaps) == n_chunks - 1
        audio_bytes = b"".join(m for m in msgs if not (len(m) == GAP_BYTES and not any(m)))
        # per-chunk audio is identical (same script) -> total divides evenly
        assert len(audio_bytes) % n_chunks == 0

    def test_prefetched_chunks_hit_exact_loudness(self, frozen_tokenizer):
        chunked, _ = self.build(frozen_tokenizer, normalize=True)
        n_chunks = len(chunked.chunk(LONG_TEXT))
        msgs = run_stream_long(chunked, LONG_TEXT)
        # split messages back into per-chunk segments at gap markers
        segments, current = [], []
        for m in msgs:
            if len(m) == GAP_BYTES and not any(m):
                segments.append(b"".join(current))
                current = []
            else:
                current.append(m)
        segments.append(b"".join(current))
        assert len(segments) == n_chunks

        # Raw (unnormalized) reference: same fakes, normalize off.
        chunked_raw, _ = self.build(frozen_tokenizer, normalize=False)
        raw_msgs = run_stream_long(chunked_raw, LONG_TEXT)
        raw_seg = next(m for m in raw_msgs if not (len(m) == GAP_BYTES and not any(m)))

        # Chunk 1 was prefetched during chunk 0's live emission -> exact path:
        # normalization changed the bytes vs the raw reference.
        assert segments[1] != raw_seg
        # Whole pipeline is deterministic: a second identical run produces
        # byte-identical output (the volume treatment is consistent, whichever
        # hybrid path each chunk took).
        chunked2, _ = self.build(frozen_tokenizer, normalize=True)
        msgs2 = run_stream_long(chunked2, LONG_TEXT)
        assert b"".join(msgs2) == b"".join(msgs)
        # every emitted sample respects the headroom/limiter bounds
        for seg in segments:
            y = np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32767.0
            assert 0 < np.max(np.abs(y)) <= 0.986

    def test_producer_error_raises_with_chunk_index(self, frozen_tokenizer):
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks, fail_at=1)  # every call fails mid-stream
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60)
        with pytest.raises(RuntimeError, match="chunk 0/"):
            run_stream_long(chunked, LONG_TEXT)

    def test_empty_text_yields_nothing(self, frozen_tokenizer):
        chunked, fake = self.build(frozen_tokenizer)
        assert run_stream_long(chunked, "   ") == []
        assert fake.generate_calls == []

    def test_stream_long_sync_matches_async(self, frozen_tokenizer):
        chunked_a, _ = self.build(frozen_tokenizer, normalize=False)
        async_msgs = run_stream_long(chunked_a, LONG_TEXT)

        chunked_b, _ = self.build(frozen_tokenizer, normalize=False)
        sync_msgs = list(chunked_b.stream_long_sync(LONG_TEXT))
        asyncio.run(chunked_b._engine.shutdown())
        assert b"".join(sync_msgs) == b"".join(async_msgs)


def rms_of(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if y.size else 0.0


class FailingSecondCallLLM(FakeAsyncLLM):
    """First generate call streams normally; the second raises immediately."""

    def __init__(self, chunks):
        super().__init__(chunks)
        self._call = 0

    async def generate(self, prompt, sp, rid):
        self._call += 1
        if self._call >= 2:
            self.generate_calls.append((prompt, sp, rid))
            raise RuntimeError("scripted failure on chunk 1")
            yield  # pragma: no cover - makes this an async generator
        async for out in super().generate(prompt, sp, rid):
            yield out


class TestReviewFindings:
    """Regression tests for the adversarial-review fixes."""

    def test_live_first_message_ships_alone(self, frozen_tokenizer):
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, normalize=False)

        msgs = run_stream_long(chunked, LONG_TEXT, frames_per_message=4)
        # chunk 0 is live: its first message must be exactly ONE frame (low
        # time-to-first-audio), not frames_per_message frames.
        assert len(msgs[0]) == 4096

    def test_failed_chunk_emits_no_preceding_gap(self, frozen_tokenizer):
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = FailingSecondCallLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, normalize=False)

        collected: list[bytes] = []

        async def _run():
            try:
                async for m in chunked.stream_long(LONG_TEXT):
                    collected.append(m)
            finally:
                await engine.shutdown()

        with pytest.raises(RuntimeError, match="chunk 1/"):
            asyncio.run(_run())
        # chunk 0 audio arrived, but NO trailing gap message before the error
        assert collected, "chunk 0 audio should have been emitted"
        assert not (len(collected[-1]) == GAP_BYTES and not any(collected[-1]))

    def test_max_chunks_guard(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_offline_engine(frozen_tokenizer, mock_snac_model)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, max_chunks=2)
        assert len(chunked.chunk(LONG_TEXT)) > 2
        with pytest.raises(ValueError, match="max_chunks"):
            chunked.synthesize_long(LONG_TEXT)

    def test_sync_bridge_reuse_with_engine_stream_sync(self, frozen_tokenizer):
        """Mixing engine.stream_sync and stream_long_sync on ONE engine must
        work: the wrapper reuses the engine's bridge loop (review finding)."""
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)

        single = b"".join(engine.stream_sync("hello there friend"))
        assert single  # engine bridge now owns the loop

        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60, normalize=False)
        long = b"".join(chunked.stream_long_sync(LONG_TEXT))
        assert long
        asyncio.run(engine.shutdown())


# ---------------------------------------------------------------------------
# Dialogue-long (turn-boundary chunking end-to-end)
# ---------------------------------------------------------------------------

# Four turns, ~44-46 chars each -> serialized per-turn cost ~67-69; with
# max_chunk_chars=150 the plan packs exactly 2 turns per chunk (2 chunks).
DIALOGUE = [
    {"speaker": "S1", "text": "Hello there, how are you doing this evening?"},
    {"speaker": "S2", "text": "I am doing quite well, thanks for asking me."},
    {"speaker": "S1", "text": "Did you finish the report for the big meeting?"},
    {"speaker": "S2", "text": "Yes, I sent it over to the whole team already."},
]


def run_stream_conversation_long(chunked, messages, **kw):
    async def _run():
        frames = []
        async for msg in chunked.stream_conversation_long(messages, **kw):
            frames.append(msg)
        await chunked._engine.shutdown()
        return frames

    return asyncio.run(_run())


class TestDialogueLong:
    def test_offline_single_generate_call_with_plan_prompts(
        self, frozen_tokenizer, mock_snac_model
    ):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        engine, backend = make_offline_engine(frozen_tokenizer, mock_snac_model)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=150)
        plan = chunked.chunk_dialogue(DIALOGUE)
        assert len(plan) >= 2
        result = chunked.synthesize_conversation_long(DIALOGUE)
        assert len(backend.calls) == 1  # ONE batched generate for all chunks
        expected = [build_conversation_prompt_ids(c, frozen_tokenizer) for c in plan]
        assert backend.calls[0][0] == expected
        assert result.audio.dtype == np.float32
        assert result.sample_rate == 24_000

    def test_offline_gap_arithmetic_without_normalize(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_offline_engine(frozen_tokenizer, mock_snac_model)
        chunked = ChunkedIndicStreamingTTS(
            engine, max_chunk_chars=150, normalize=False, gap_ms=250.0
        )
        n_plan = len(chunked.chunk_dialogue(DIALOGUE))
        assert n_plan >= 2
        merged, chunks = chunked.synthesize_conversation_long(DIALOGUE, return_chunks=True)
        assert len(chunks) == n_plan
        gap = int(0.250 * 24_000)
        expected = sum(c.audio.size for c in chunks) + gap * (len(chunks) - 1)
        assert merged.audio.size == expected
        assert merged.audio_tokens == sum(c.audio_tokens for c in chunks)
        assert merged.prompt_tokens == sum(c.prompt_tokens for c in chunks)

    def test_offline_empty_conversation_raises(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_offline_engine(frozen_tokenizer, mock_snac_model)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=150)
        with pytest.raises(ValueError, match="empty conversation"):
            chunked.synthesize_conversation_long([])

    def test_streaming_structure_and_per_chunk_prompts(self, frozen_tokenizer):
        from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=150, normalize=False)
        plan = chunked.chunk_dialogue(DIALOGUE)
        n_plan = len(plan)
        assert n_plan >= 2

        msgs = run_stream_conversation_long(chunked, DIALOGUE)
        assert msgs  # frames arrived
        gaps = [m for m in msgs if len(m) == GAP_BYTES and not any(m)]
        assert len(gaps) == n_plan - 1
        # one generate call per dialogue chunk, with the chunk's conversation prompt
        assert len(fake.generate_calls) == n_plan
        for k in range(n_plan):
            prompt, _, _ = fake.generate_calls[k]
            assert prompt == build_conversation_prompt_ids(plan[k], frozen_tokenizer)

    def test_producer_error_raises_with_chunk_index(self, frozen_tokenizer):
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks, fail_at=1)  # every call fails mid-stream
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=150)
        with pytest.raises(RuntimeError, match="chunk 0/"):
            run_stream_conversation_long(chunked, DIALOGUE)

    def test_stream_conversation_long_sync_matches_async(self, frozen_tokenizer):
        def build():
            codes = make_codes(6, seed=5)
            chunks = uneven_chunks(make_token_stream(codes), seed=2)
            fake = MultiCallAsyncLLM(chunks)
            engine = make_stream_engine(frozen_tokenizer, fake)
            return ChunkedIndicStreamingTTS(engine, max_chunk_chars=150, normalize=False)

        chunked_a = build()
        async_msgs = run_stream_conversation_long(chunked_a, DIALOGUE)

        chunked_b = build()
        sync_msgs = list(chunked_b.stream_conversation_long_sync(DIALOGUE))
        asyncio.run(chunked_b._engine.shutdown())
        assert b"".join(sync_msgs) == b"".join(async_msgs)


class TestStreamStatsAndBudgets:
    def test_last_stream_stats_counts_paths(self, frozen_tokenizer):
        codes = make_codes(6, seed=5)
        chunks = uneven_chunks(make_token_stream(codes), seed=2)
        fake = MultiCallAsyncLLM(chunks)
        engine = make_stream_engine(frozen_tokenizer, fake)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=60)
        n = len(chunked.chunk(LONG_TEXT, ramp=True))
        run_stream_long(chunked, LONG_TEXT)
        stats = chunked.last_stream_stats
        assert stats is not None
        assert stats["chunks"] == n
        assert stats["exact"] + stats["live"] == n
        assert stats["live"] >= 1  # chunk 0 always starts live

    def test_seconds_budgets_convert_via_text_rate(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_offline_engine(frozen_tokenizer, mock_snac_model)
        # 10s max budget on pure-Latin text (14 chars/s) -> ~140-char chunks
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_seconds=10.0)
        text = ("Seven words are in this exact sentence. " * 12).strip()
        plan = chunked.chunk(text)
        assert all(len(c) <= 140 for c in plan)
        assert len(plan) >= 3

    def test_ramp_preview_via_chunk(self, frozen_tokenizer, mock_snac_model):
        engine, _ = make_offline_engine(frozen_tokenizer, mock_snac_model)
        chunked = ChunkedIndicStreamingTTS(engine, max_chunk_chars=200, first_chunk_chars=60)
        text = ("One two three four five six seven. " * 20).strip()
        assert len(chunked.chunk(text, ramp=True)[0]) <= 60
        assert len(chunked.chunk(text)[0]) > 60  # offline preview: no ramp
