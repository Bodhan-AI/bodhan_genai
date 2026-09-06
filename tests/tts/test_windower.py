"""Tests for bodhan_genai.tts.serving.windower.StreamingWindower.

Pure-numpy state machine: feed vLLM-style DELTA token chunks (start_of_speech,
N*7 audio ids, optional straggler partial frame, end_of_speech) and check the
emitted decode windows frame-by-frame. No torch, no vllm, no ray.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from bodhan_genai.tts.codec.snac import SNAC_CODEBOOK_SIZE, SNAC_NUM_CODEBOOKS
from bodhan_genai.tts.serving.windower import StreamingWindower, WindowJob

# Frozen llama token contract (tests only; runtime resolves ids from the tokenizer).
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
SNAC_BASE = 128266

SNAC_IDS = {
    "start_of_audio_id": START_OF_SPEECH,
    "end_of_audio_id": END_OF_SPEECH,
    "audio_token_base_id": SNAC_BASE,
}

NB = SNAC_NUM_CODEBOOKS  # 7


def make_token_ids(codes: np.ndarray) -> list[int]:
    """Offset raw codes (F, 7) -> flat audio token ids: base + code + (pos%7)*4096."""
    flat = codes.reshape(-1).astype(np.int64)
    pos = np.arange(flat.size) % NB
    return (SNAC_BASE + flat + pos * SNAC_CODEBOOK_SIZE).tolist()


def random_chunks(seq: list[int], rng: random.Random) -> list[list[int]]:
    """Split a token sequence into random-sized DELTA chunks (1..9 tokens)."""
    chunks, i = [], 0
    while i < len(seq):
        n = rng.randint(1, 9)
        chunks.append(seq[i : i + n])
        i += n
    return chunks


def drive(stream: list[int], window_frames: int, seed: int = 0):
    """Feed `stream` to a StreamingWindower in random-sized chunks; return
    (jobs, windower). Last chunk is marked finished=True like a final DELTA."""
    rng = random.Random(seed)
    w = StreamingWindower("req0", SNAC_IDS, window_frames=window_frames)
    jobs: list[WindowJob] = []
    chunks = random_chunks(stream, rng)
    for ci, chunk in enumerate(chunks):
        jobs.extend(w.push(chunk, finished=(ci == len(chunks) - 1)))
    jobs.extend(w.flush_tail())
    return jobs, w


@pytest.mark.parametrize("window_frames", [3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_windows_match_frames(window_frames, seed):
    rng = np.random.default_rng(100 + seed)
    F = int(rng.integers(5, 40))
    codes = rng.integers(0, SNAC_CODEBOOK_SIZE, size=(F, NB), dtype=np.int64)
    straggler = rng.integers(0, SNAC_CODEBOOK_SIZE, size=int(rng.integers(1, NB))).tolist()
    stream = (
        [128000, 128261]  # non-audio prefix tokens: ignored
        + [START_OF_SPEECH]
        + make_token_ids(codes)
        + [SNAC_BASE + int(s) for s in straggler]  # partial frame: must be dropped
        + [END_OF_SPEECH]
        + [128009]  # post-end tokens: ignored
    )
    jobs, w = drive(stream, window_frames, seed=seed)

    # one window per frame, in order
    assert len(jobs) == F
    assert [j.emit_index for j in jobs] == list(range(F))
    assert w.total_emits == F

    for k, job in enumerate(jobs):
        assert job.request_id == "req0"
        assert job.codes.shape == (window_frames * NB,)
        assert job.codes.dtype == np.int32
        assert job.codes.min() >= 0 and job.codes.max() < SNAC_CODEBOOK_SIZE
        # window covers frames [k-1, k, ..., k+window_frames-2]; out-of-range = zeros
        for slot in range(window_frames):
            f = k - 1 + slot
            got = job.codes[slot * NB : (slot + 1) * NB]
            if 0 <= f < F:
                assert np.array_equal(got, codes[f]), f"frame {k} slot {slot}"
            else:
                assert not got.any(), f"frame {k} slot {slot} should be zero-padded"


def test_left_and_right_padding():
    """First window is left-padded; flush_tail windows are right-padded."""
    F, wf = 6, 4
    codes = np.arange(F * NB, dtype=np.int64).reshape(F, NB) % SNAC_CODEBOOK_SIZE
    stream = [START_OF_SPEECH, *make_token_ids(codes), END_OF_SPEECH]
    jobs, _ = drive(stream, wf, seed=7)
    assert len(jobs) == F
    # frame 0: slot 0 (frame -1) zero-padded
    assert not jobs[0].codes[:NB].any()
    # last frame F-1: slots for frames F and F+1 zero-padded
    assert not jobs[-1].codes[2 * NB :].any()
    assert np.array_equal(jobs[-1].codes[NB : 2 * NB], codes[F - 1])


def test_codes_clamped_to_codebook_range():
    """Out-of-range ids (wrong position offset / stray ids) clamp to [0, 4096)."""
    w = StreamingWindower("clamp", SNAC_IDS, window_frames=3)
    # frame of 7 tokens with NO position offsets: positions 1..6 underflow -> clamp to 0
    low_frame = [SNAC_BASE + 5] * NB
    # frame of 7 tokens all offset as position 6: positions 0..5 overflow -> clamp to 4095
    high_frame = [SNAC_BASE + 6 * SNAC_CODEBOOK_SIZE + 4095] * NB
    jobs = list(w.push([START_OF_SPEECH, *low_frame, *high_frame, END_OF_SPEECH]))
    jobs += list(w.flush_tail())
    assert len(jobs) == 2
    lo = jobs[0].codes[NB : 2 * NB]  # emitted frame k sits at slot index 1
    hi = jobs[1].codes[NB : 2 * NB]
    assert lo[0] == 5 and not lo[1:].any()  # underflow -> 0
    assert hi[NB - 1] == 4095 and (hi[: NB - 1] == 4095).all()  # overflow -> 4095
    assert all(j.codes.min() >= 0 and j.codes.max() < SNAC_CODEBOOK_SIZE for j in jobs)


def test_incremental_emission_before_finish():
    """push() emits frame k as soon as frame k+wf-2 exists — before end_of_speech."""
    wf = 3
    F = 10
    codes = np.zeros((F, NB), dtype=np.int64)
    w = StreamingWindower("inc", SNAC_IDS, window_frames=wf)
    assert list(w.push([START_OF_SPEECH])) == []
    ids = make_token_ids(codes)
    emitted = []
    for f in range(F):
        emitted.extend(w.push(ids[f * NB : (f + 1) * NB]))
        # frame k emitted once F_seen >= k + wf - 1 -> cumulative = max(0, F_seen - wf + 2)
        assert len(emitted) == max(0, (f + 1) - wf + 2)
    # tail: remaining wf-2 frames come out on flush
    tail = list(w.flush_tail())
    assert len(emitted) + len(tail) == F
    assert w.total_emits == F


def test_no_audio_tokens_yields_nothing():
    """No start_of_speech (or empty generation) -> zero windows."""
    w = StreamingWindower("empty", SNAC_IDS, window_frames=3)
    assert list(w.push([128000, 128009, END_OF_SPEECH])) == []
    assert list(w.flush_tail()) == []
    assert w.total_emits == 0

    w2 = StreamingWindower("empty2", SNAC_IDS, window_frames=3)
    assert list(w2.push([START_OF_SPEECH, END_OF_SPEECH])) == []
    assert list(w2.flush_tail()) == []
    assert w2.total_emits == 0
