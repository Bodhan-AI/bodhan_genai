"""Tests for bodhan_genai.tts.engine.streaming.IndicStreamingTTSEngine.

A fake AsyncLLM (scripted DELTA chunks) and a fake window decoder are injected
through the engine's factory seams; the REAL StreamingWindower and REAL
SnacMicroBatcher run in between, all the way to the END sentinel. No vllm, no
GPU. Async paths are driven via asyncio.run() inside plain sync tests (no
pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import numpy as np
import pytest

from bodhan_genai.tts.codec.snac import SNAC_CODEBOOK_SIZE, SNAC_NUM_CODEBOOKS
from bodhan_genai.tts.engine.streaming import IndicStreamingTTSEngine
from bodhan_genai.tts.serving.windower import StreamingWindower

# Frozen llama token contract (tests only; the engine resolves ids from the tokenizer).
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
SNAC_BASE = 128266
NB = SNAC_NUM_CODEBOOKS  # 7
SAMPLES_PER_FRAME = 2048
FRAME_BYTES = SAMPLES_PER_FRAME * 2  # int16

SNAC_IDS = {
    "start_of_audio_id": START_OF_SPEECH,
    "end_of_audio_id": END_OF_SPEECH,
    "audio_token_base_id": SNAC_BASE,
}


# ---------------------------------------------------------------------------
# Fakes + helpers
# ---------------------------------------------------------------------------


class FakeAsyncLLM:
    """Scripted stand-in for vLLM's AsyncLLM: generate() yields DELTA chunks
    (optionally raising mid-stream) and abort() calls are recorded."""

    def __init__(self, chunks: list[list[int]], fail_at: int | None = None):
        self._chunks = chunks
        self._fail_at = fail_at  # raise instead of yielding chunk index N
        self.aborted: list[str] = []
        self.generate_calls: list[tuple] = []
        self.shutdown_calls = 0
        self.errored = False

    async def generate(self, prompt, sp, rid):
        self.generate_calls.append((prompt, sp, rid))
        last = len(self._chunks) - 1
        for i, ids in enumerate(self._chunks):
            if self._fail_at is not None and i == self._fail_at:
                raise RuntimeError("scripted engine failure")
            yield SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=list(ids))], finished=(i == last)
            )
            await asyncio.sleep(0)

    async def abort(self, rid):
        self.aborted.append(rid)

    def shutdown(self):
        self.shutdown_calls += 1


class FakeWindowDecoder:
    """Duck-type of InProcessSnacDecoder: (n, wf*7) int32 -> deterministic
    (n, 2048) int16 where row i is a position-weighted hash of its codes (so
    every distinct window maps to a distinct, order-checkable payload)."""

    batch_size = 4

    def decode(self, window_codes: np.ndarray) -> np.ndarray:
        arr = np.asarray(window_codes, dtype=np.int32)
        n, width = arr.shape
        out = np.zeros((n, SAMPLES_PER_FRAME), dtype=np.int16)
        weights = np.arange(1, width + 1, dtype=np.int64)
        for i in range(n):
            out[i, :] = int((arr[i].astype(np.int64) * weights).sum() % 30000)
        return out


def make_codes(n_frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, SNAC_CODEBOOK_SIZE, size=(n_frames, NB), dtype=np.int64)


def make_token_stream(codes: np.ndarray) -> list[int]:
    """start_of_speech, offset-correct snac ids, end_of_speech."""
    flat = codes.reshape(-1)
    pos = np.arange(flat.size) % NB
    ids = (SNAC_BASE + flat + pos * SNAC_CODEBOOK_SIZE).tolist()
    return [START_OF_SPEECH, *ids, END_OF_SPEECH]


def uneven_chunks(seq: list[int], seed: int = 1) -> list[list[int]]:
    """Split into uneven DELTA chunks (1..9 tokens) like real engine steps."""
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(seq):
        n = rng.randint(1, 9)
        out.append(seq[i : i + n])
        i += n
    return out


def expected_frames(chunks: list[list[int]], window_frames: int) -> tuple[list[bytes], int]:
    """Reference run: real windower + FakeWindowDecoder, frame bytes in emit order."""
    w = StreamingWindower("ref", SNAC_IDS, window_frames=window_frames)
    jobs = []
    last = len(chunks) - 1
    for i, c in enumerate(chunks):
        jobs.extend(w.push(c, finished=(i == last)))
    jobs.extend(w.flush_tail())
    dec = FakeWindowDecoder()
    return [dec.decode(j.codes[None, :])[0].tobytes() for j in jobs], w.total_emits


def make_engine(
    tok, fake_llm, *, window_frames: int = 3, queue_max: int = 64, request_factory=None
) -> IndicStreamingTTSEngine:
    return IndicStreamingTTSEngine(
        "dummy-checkpoint",
        snac_window_frames=window_frames,
        snac_flush_interval_ms=1.0,
        per_request_queue_max=queue_max,
        llm_factory=lambda kw: fake_llm,
        decoder_factory=lambda: FakeWindowDecoder(),
        request_factory=request_factory or (lambda ids, sc, stop: (list(ids), sc)),
        tokenizer_loader=lambda path: tok,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stream_emits_all_frames_in_order_with_grouping(frozen_tokenizer):
    codes = make_codes(10)
    chunks = uneven_chunks(make_token_stream(codes))
    frames, total = expected_frames(chunks, window_frames=3)
    assert total == 10
    fake = FakeAsyncLLM(chunks)
    eng = make_engine(frozen_tokenizer, fake)

    async def go():
        msgs = [m async for m in eng.stream("hello world", frames_per_message=3)]
        await eng.shutdown()
        return msgs

    msgs = asyncio.run(go())
    # every frame arrives, byte payloads in emit order
    payload = b"".join(msgs)
    assert payload == b"".join(frames)
    assert len(payload) // FRAME_BYTES == total
    # first message ships exactly one frame; then exact frames_per_message groups
    assert [len(m) // FRAME_BYTES for m in msgs] == [1, 3, 3, 3]


def test_sampling_merge_and_stop_ids_reach_request_factory(frozen_tokenizer):
    chunks = uneven_chunks(make_token_stream(make_codes(3)))
    seen = {}

    def rf(ids, sc, stop):
        seen["sc"], seen["stop"] = sc, list(stop)
        return list(ids), sc

    eng = make_engine(frozen_tokenizer, FakeAsyncLLM(chunks), request_factory=rf)

    async def go():
        _ = [m async for m in eng.stream("hello world", temperature=0.9, top_k=50)]
        await eng.shutdown()

    asyncio.run(go())
    sc = seen["sc"]
    # overrides applied, None-kwargs fall back to SamplingConfig defaults
    assert (sc.temperature, sc.top_k) == (0.9, 50)
    assert (sc.top_p, sc.repetition_penalty, sc.max_new_tokens) == (0.95, 1.1, 2048)
    # stop ids resolved from the tokenizer: end_of_speech + eos
    assert seen["stop"] == [END_OF_SPEECH, 128009]


def test_failure_mid_generate_yields_partial_frames_then_raises(frozen_tokenizer):
    chunks = uneven_chunks(make_token_stream(make_codes(12, seed=3)), seed=7)
    fail_at = max(2, len(chunks) // 2)
    fake = FakeAsyncLLM(chunks, fail_at=fail_at)
    eng = make_engine(frozen_tokenizer, fake)

    async def go():
        got = []
        with pytest.raises(RuntimeError, match="mid-generation"):
            async for m in eng.stream("hello world"):
                got.append(m)
        await eng.shutdown()
        return got

    got = asyncio.run(go())
    # partial frames (everything windowed before the crash) were still delivered
    partial, total = expected_frames(chunks[:fail_at], window_frames=3)
    assert total > 0
    assert b"".join(got) == b"".join(partial)
    # a scripted RuntimeError is not an EngineDeadError -> engine stays alive
    assert eng.engine_dead is False
    eng.check_health()


def test_backpressure_overflow_aborts_request(frozen_tokenizer):
    chunks = uneven_chunks(make_token_stream(make_codes(30, seed=5)), seed=5)
    fake = FakeAsyncLLM(chunks)
    eng = make_engine(frozen_tokenizer, fake, queue_max=1)

    async def go():
        got = []
        with pytest.raises(RuntimeError):
            async for m in eng.stream("hello world"):
                got.append(m)
                await asyncio.sleep(0.05)  # slow consumer -> out queue (max 1) overflows
        await eng.shutdown()
        return got

    got = asyncio.run(go())
    assert "r1" in fake.aborted  # overflow policy aborted the request
    assert len(got) < 30  # stream was cut short


def test_check_health_raises_when_llm_errored(frozen_tokenizer):
    fake = FakeAsyncLLM([])
    eng = make_engine(frozen_tokenizer, fake)
    eng.check_health()  # healthy at rest
    assert eng.engine_dead is False
    fake.errored = True
    with pytest.raises(RuntimeError, match="dead"):
        eng.check_health()


def test_shutdown_is_idempotent(frozen_tokenizer):
    chunks = uneven_chunks(make_token_stream(make_codes(4)))
    fake = FakeAsyncLLM(chunks)
    eng = make_engine(frozen_tokenizer, fake)

    async def go():
        _ = [m async for m in eng.stream("hello world")]
        await eng.shutdown()
        await eng.shutdown()  # double await must be a no-op

    asyncio.run(go())
    assert fake.shutdown_calls == 1


def test_stream_sync_matches_async(frozen_tokenizer):
    codes = make_codes(8, seed=11)
    chunks = uneven_chunks(make_token_stream(codes), seed=11)

    eng_async = make_engine(frozen_tokenizer, FakeAsyncLLM(chunks))

    async def go():
        msgs = [m async for m in eng_async.stream("hello world", frames_per_message=2)]
        await eng_async.shutdown()
        return msgs

    async_msgs = asyncio.run(go())

    # plain sync function: no running event loop here
    eng_sync = make_engine(frozen_tokenizer, FakeAsyncLLM(chunks))
    sync_msgs = list(eng_sync.stream_sync("hello world", frames_per_message=2))
    asyncio.run(eng_sync.shutdown())  # tears down the bridge loop too

    assert sync_msgs == async_msgs
    assert [len(m) // FRAME_BYTES for m in sync_msgs] == [1, 2, 2, 2, 1]


def test_empty_text_yields_nothing(frozen_tokenizer):
    fake = FakeAsyncLLM([[START_OF_SPEECH], [END_OF_SPEECH]])
    eng = make_engine(frozen_tokenizer, fake)

    async def go():
        a = [m async for m in eng.stream("")]
        b = [m async for m in eng.stream("   ")]
        await eng.shutdown()
        return a, b

    a, b = asyncio.run(go())
    assert a == [] and b == []
    assert fake.generate_calls == []  # never reached the LLM


def test_stream_conversation_emits_frames_and_uses_conversation_prompt(frozen_tokenizer):
    from bodhan_genai.tts.inference.prompts import build_conversation_prompt_ids

    codes = make_codes(6, seed=11)
    chunks = uneven_chunks(make_token_stream(codes), seed=3)
    fake = FakeAsyncLLM(chunks)
    engine = make_engine(frozen_tokenizer, fake)
    msgs = [
        {"speaker": "S1", "text": "hello there"},
        {"speaker": "S2", "text": "hi back"},
    ]

    async def run():
        frames = [f async for f in engine.stream_conversation(msgs)]
        await engine.shutdown()
        return frames

    frames = asyncio.run(run())
    ref_frames, total_emits = expected_frames(chunks, window_frames=3)
    assert len(frames) == total_emits  # frames_per_message=1 -> one frame per message
    assert b"".join(frames) == b"".join(ref_frames)
    prompt, _, _ = fake.generate_calls[0]
    assert prompt == build_conversation_prompt_ids(msgs, frozen_tokenizer)


def test_stream_conversation_empty_messages_yields_nothing(frozen_tokenizer):
    fake = FakeAsyncLLM([[END_OF_SPEECH]])
    engine = make_engine(frozen_tokenizer, fake)

    async def run():
        frames = [f async for f in engine.stream_conversation([])]
        await engine.shutdown()
        return frames

    assert asyncio.run(run()) == []
    assert fake.generate_calls == []
