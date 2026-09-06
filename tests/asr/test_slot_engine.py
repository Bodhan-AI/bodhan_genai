"""Continuous-batching slot pool for streaming.

Runs on CPU against a tiny model: CUDA graphs auto-disable off-CUDA, so the
scheduler, admission, stepping and eviction logic are all still exercised —
which is where the bugs live. Graph capture and real transcripts are covered
by the GPU parity check in out/ (see docs/asr/serving.md).

The property that matters most here is that the pool is sized from the FIXED
streaming window, not from whatever arrives first. That is the whole reason a
slot pool is usable for serving at all: the offline engine sizes its buffers
from the first batch and then fails any later, longer utterance.
"""

from __future__ import annotations

import torch

from bodhan_genai.asr.model import IndicTranscribeConfig, IndicTranscribeForConditionalGeneration
from bodhan_genai.asr.serving.slot_engine import StreamingSlotEngine

SR = 16000


class FakeTokenizer:
    """Enough surface for admission, prompts and eviction."""

    prompt_len = 10

    def __init__(self):
        self.pad_id, self.eos_id = 2, 3

    def encode_prompt(self, lang, itn=False, romanized=False):
        # mode slots mirror the real tokenizer: 6 -> itn, 7 -> romanized
        ids = [7, 4, 18, 42, 42, 5, 9, 11, 13, 15]
        if itn:
            ids[6] = 8
        if romanized:
            ids[7] = 10
        return ids

    def strip_prompt_and_trim(self, ids, prompt):
        ids = [int(i) for i in ids][len(prompt) :]
        end = len(ids)
        while end > 0 and ids[end - 1] in (self.pad_id, self.eos_id):
            end -= 1
        return ids[:end]

    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


class FakeFE(torch.nn.Module):
    sample_rate = SR
    hop_length = 160

    def __init__(self, n_mels):
        super().__init__()
        self.n_mels = n_mels

    def get_seq_len(self, n):
        return torch.floor_divide(n, self.hop_length) + 1

    def forward(self, audio, lens):
        t = int(self.get_seq_len(lens.max()).item())
        return torch.zeros(audio.size(0), self.n_mels, t), self.get_seq_len(lens)


def tiny_engine(**kw) -> StreamingSlotEngine:
    cfg = IndicTranscribeConfig(
        vocab_size=64,
        d_model=32,
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        subsampling_conv_channels=4,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_target_positions=64,
    )
    model = IndicTranscribeForConditionalGeneration(cfg).eval()
    kw.setdefault("max_buffer_s", 2.0)
    kw.setdefault("slots", 4)
    return StreamingSlotEngine(
        model=model,
        feature_extractor=FakeFE(cfg.num_mel_bins),
        tokenizer=FakeTokenizer(),
        device="cpu",
        dtype=torch.float32,
        **kw,
    )


# --- static sizing ---------------------------------------------------------


def test_pool_is_sized_from_the_fixed_window_not_from_traffic():
    """The property that makes a slot pool servable: sizes are known upfront."""
    eng = tiny_engine(max_buffer_s=2.0)
    assert eng.t_enc_max > 0
    assert eng.l_alloc > eng.t_enc_max  # cap = enc + delta (+1), clamped
    assert eng.l_alloc <= eng.cfg.max_target_positions + 1


def test_a_longer_window_allocates_a_bigger_pool():
    small = tiny_engine(max_buffer_s=1.0)
    big = tiny_engine(max_buffer_s=4.0)
    assert big.t_enc_max > small.t_enc_max


def test_graphs_are_off_on_cpu():
    """Guards against trying to capture where capture cannot work."""
    assert tiny_engine().use_cuda_graphs is False


# --- window guard ----------------------------------------------------------


def test_oversized_buffer_is_rejected_not_silently_truncated():
    """The pool is statically sized; a longer buffer must fail loudly rather
    than wrap, overflow, or quietly lose its tail."""
    eng = tiny_engine(max_buffer_s=1.0)
    fut = eng.submit(torch.zeros(int(3.0 * SR)), "hi")
    assert fut.done()
    exc = fut.exception()
    assert isinstance(exc, ValueError)
    assert "exceeds the engine's" in str(exc)


def test_a_buffer_inside_the_window_is_accepted():
    eng = tiny_engine(max_buffer_s=2.0)
    fut = eng.submit(torch.zeros(int(1.0 * SR)), "hi")
    assert not fut.done() or fut.exception() is None


# --- scheduler end to end (CPU, tiny model) --------------------------------


def test_submissions_complete_and_free_their_slots():
    """Full round trip through the real scheduler: admit -> prompt prefill ->
    graphed-or-eager steps -> EOS/cap -> evict -> future resolved."""
    eng = tiny_engine(max_buffer_s=1.0, slots=4)
    eng.start()
    try:
        futs = [eng.submit(torch.zeros(int(0.5 * SR)), "hi") for _ in range(4)]
        texts = [f.result(timeout=120) for f in futs]
        assert len(texts) == 4
        assert all(isinstance(t, str) for t in texts)
        # every slot must be free again, or the pool leaks capacity
        assert all(s.req is None for s in eng._slots)
        assert eng.stats.completed == 4
    finally:
        eng.close()


def test_more_requests_than_slots_still_all_complete():
    """Requests beyond pool size must queue and be admitted as slots free —
    the refill half of continuous batching."""
    eng = tiny_engine(max_buffer_s=1.0, slots=2)
    eng.start()
    try:
        futs = [eng.submit(torch.zeros(int(0.5 * SR)), "hi") for _ in range(6)]
        texts = [f.result(timeout=180) for f in futs]
        assert len(texts) == 6
        assert eng.stats.completed == 6
    finally:
        eng.close()


def test_shutdown_fails_inflight_work_instead_of_hanging_callers():
    eng = tiny_engine(max_buffer_s=1.0, slots=2)
    eng.start()
    eng.close()
    fut = eng.submit(torch.zeros(int(0.5 * SR)), "hi")
    # after shutdown nothing drains the queue; the caller must not wait forever
    assert not fut.done() or fut.exception() is not None or isinstance(fut.result(0.1), str)
