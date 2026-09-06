"""VAD-endpointed streaming state machine.

Driven by a scripted fake `transcribe` over synthetic speech/silence, so the
whole algorithm is exercised on CPU with no model. The invariants here are what
a live caption UI depends on: a span is decoded exactly once, committed text is
append-only and never contradicted, provisional text is always replaceable, and
`max_segment_s` is a hard bound nothing can exceed.
"""

from __future__ import annotations

from itertools import pairwise

import torch

from bodhan_genai.asr.serving.streaming import StreamUpdate, VadStream

SR = 16000


def speech(seconds: float, amp: float = 0.2) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(int(seconds * SR), generator=g) * amp


def silence(seconds: float) -> torch.Tensor:
    return torch.zeros(int(seconds * SR))


class ScriptedASR:
    """Returns the next scripted hypothesis on each call, recording span lengths."""

    def __init__(self, *hypotheses: str):
        self.hypotheses = list(hypotheses) or ["text"]
        self.calls = 0
        self.spans: list[float] = []

    def __call__(self, wav: torch.Tensor) -> str:
        self.spans.append(wav.numel() / SR)
        h = self.hypotheses[min(self.calls, len(self.hypotheses) - 1)]
        self.calls += 1
        return h


def make(asr, **kw) -> VadStream:
    kw.setdefault("sample_rate", SR)
    kw.setdefault("endpoint_silence_s", 0.3)
    kw.setdefault("max_segment_s", 1000.0)  # forced cuts off unless a test wants them
    kw.setdefault("partial_interval_s", None)  # partials off unless a test wants them
    return VadStream(transcribe=asr, **kw)


def feed(stream: VadStream, wav: torch.Tensor, packet_s: float = 0.1) -> list[StreamUpdate]:
    """Push `wav` in realistic small packets, collecting the updates produced."""
    out = []
    step = int(packet_s * SR)
    for i in range(0, wav.numel(), step):
        upd = stream.push(wav[i : i + step])
        if upd is not None:
            out.append(upd)
    return out


# --- endpointing ------------------------------------------------------------


def test_endpoint_closes_span_and_commits_once():
    asr = ScriptedASR("hello world")
    s = make(asr)
    upds = feed(s, torch.cat([speech(1.0), silence(0.6), speech(0.2)]))
    assert len(upds) == 1
    assert upds[0].is_final
    assert upds[0].committed_delta == "hello world"
    assert s.committed_text == "hello world"
    assert asr.calls == 1, "a span must be decoded exactly once"


def test_short_pause_does_not_close_a_span():
    """A pause below endpoint_silence_s is not an endpoint."""
    asr = ScriptedASR("a")
    s = make(asr, endpoint_silence_s=0.5)
    upds = feed(s, torch.cat([speech(1.0), silence(0.2), speech(1.0)]))
    assert upds == []
    assert asr.calls == 0


def test_silence_is_not_fed_to_the_model():
    """The closing pause belongs to neither span."""
    asr = ScriptedASR("x")
    s = make(asr, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(1.0), silence(1.5), speech(0.2)]))
    assert asr.spans, "expected one span"
    # ~1.0 s of speech, not 1.0 + 1.5 -- generous bound for frame quantisation
    assert asr.spans[0] < 1.3, f"span carried the pause: {asr.spans[0]:.2f}s"


def test_pause_is_not_carried_into_the_next_span():
    """Leftover silence would delay the NEXT span's own endpoint."""
    asr = ScriptedASR("one", "two")
    s = make(asr, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(1.0), silence(1.0), speech(1.0), silence(0.6)]))
    assert asr.calls == 2
    assert asr.spans[1] < 1.3, f"second span began with stale silence: {asr.spans[1]:.2f}s"


def test_multiple_spans_accumulate_in_order():
    asr = ScriptedASR("first", "second", "third")
    s = make(asr)
    for _ in range(3):
        feed(s, torch.cat([speech(0.8), silence(0.6)]))
    assert s.committed_text == "first second third"


def test_span_shorter_than_min_segment_is_not_emitted():
    asr = ScriptedASR("noise")
    s = make(asr, min_segment_s=1.0)
    feed(s, torch.cat([speech(0.3), silence(1.0)]))
    assert asr.calls == 0
    assert s.committed_text == ""


# --- forced cut / max_segment bound ----------------------------------------


def test_max_segment_forces_a_cut_without_any_pause():
    asr = ScriptedASR("forced")
    s = make(asr, max_segment_s=2.0)
    upds = feed(s, speech(2.5))
    assert len(upds) == 1 and upds[0].is_final
    assert upds[0].committed_delta == "forced"


def test_no_span_ever_exceeds_max_segment():
    """The hard bound the statically sized slot pool depends on."""
    asr = ScriptedASR("x")
    s = make(asr, max_segment_s=1.5, endpoint_silence_s=99.0)  # endpoint can never fire
    feed(s, speech(9.0))
    assert asr.spans, "expected forced cuts"
    assert max(asr.spans) <= 1.5 + 0.2, f"span exceeded the bound: {max(asr.spans):.2f}s"


def test_forced_cut_prefers_a_silence_over_a_hard_cut():
    """Cutting mid-word makes the model invent a word, so a real pause wins."""
    asr = ScriptedASR("x")
    # A 0.25 s pause: too short to endpoint at 0.5 s, but usable for a forced cut.
    s = make(asr, max_segment_s=2.0, endpoint_silence_s=0.5)
    feed(s, torch.cat([speech(1.0), silence(0.25), speech(1.2)]))
    assert asr.spans
    assert asr.spans[0] < 1.4, f"ignored the available pause: {asr.spans[0]:.2f}s"


def test_pause_free_audio_still_makes_progress():
    """Continuous speech must not stall forever waiting for an endpoint."""
    asr = ScriptedASR("a", "b", "c")
    s = make(asr, max_segment_s=1.0, endpoint_silence_s=99.0)
    feed(s, speech(3.2))
    assert s.committed_text.startswith("a b")


# --- partials ---------------------------------------------------------------


def test_partial_is_provisional_and_commits_nothing():
    asr = ScriptedASR("interim")
    s = make(asr, partial_interval_s=0.5, max_segment_s=1000.0)
    upds = feed(s, speech(0.8))
    assert upds, "expected a partial"
    assert not upds[-1].is_final
    assert upds[-1].provisional == "interim"
    assert upds[-1].committed_delta == ""
    assert s.committed_text == ""


def test_partial_is_replaced_not_appended():
    asr = ScriptedASR("one", "one two", "one two three")
    s = make(asr, partial_interval_s=0.5, max_segment_s=1000.0)
    upds = feed(s, speech(2.0))
    provisionals = [u.provisional for u in upds if not u.is_final]
    assert len(provisionals) >= 2
    assert provisionals[-1] == "one two three"
    assert s.committed_text == "", "partials must never commit"


def test_final_clears_the_provisional_tail():
    asr = ScriptedASR("guess", "final")
    s = make(asr, partial_interval_s=0.5, endpoint_silence_s=0.3)
    upds = feed(s, torch.cat([speech(1.0), silence(0.6), speech(0.1)]))
    final = [u for u in upds if u.is_final]
    assert final and final[-1].provisional == ""
    assert s.provisional_text == ""


def test_partials_disabled_costs_one_decode_per_span():
    """The cheapest setting: partial_interval_s=None means finals only."""
    asr = ScriptedASR("x")
    s = make(asr, partial_interval_s=None, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(2.0), silence(0.6), speech(0.1)]))
    assert asr.calls == 1


def test_partial_interval_controls_decode_count():
    """Each partial is a full re-decode -- the capacity/latency trade, measured."""
    a2 = ScriptedASR("x")
    a1 = ScriptedASR("x")
    feed(make(a2, partial_interval_s=2.0, max_segment_s=1000.0), speech(4.0))
    feed(make(a1, partial_interval_s=1.0, max_segment_s=1000.0), speech(4.0))
    assert a1.calls > a2.calls


# --- finalize ---------------------------------------------------------------


def test_finalize_commits_the_open_span():
    asr = ScriptedASR("tail")
    s = make(asr)
    feed(s, speech(1.0))
    upd = s.finalize()
    assert upd.is_final
    assert upd.committed_delta == "tail"
    assert s.committed_text == "tail"


def test_finalize_drops_a_sliver_rather_than_hallucinating():
    asr = ScriptedASR("invented")
    s = make(asr, min_segment_s=1.0)
    feed(s, speech(0.2))
    upd = s.finalize()
    assert asr.calls == 0
    assert upd.committed_delta == ""
    assert upd.is_final


def test_finalize_on_empty_stream_is_safe():
    s = make(ScriptedASR("x"))
    upd = s.finalize()
    assert upd.committed_total == ""
    assert upd.committed_delta == ""


def test_finalize_after_a_span_does_not_re_emit_it():
    asr = ScriptedASR("first", "second")
    s = make(asr, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(1.0), silence(0.6), speech(1.0)]))
    s.finalize()
    assert s.committed_text == "first second"
    assert s.committed_text.count("first") == 1


# --- bookkeeping ------------------------------------------------------------


def test_audio_seconds_counts_everything_pushed():
    s = make(ScriptedASR("x"))
    feed(s, torch.cat([speech(1.0), silence(0.5)]))
    assert abs(s.audio_seconds - 1.5) < 0.05


def test_empty_text_is_not_committed_as_a_blank():
    """A span the model transcribes as nothing must not add an empty word."""
    asr = ScriptedASR("", "real")
    s = make(asr, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(1.0), silence(0.6), speech(1.0), silence(0.6), speech(0.1)]))
    assert s.committed_text == "real"


def test_committed_text_is_append_only_across_a_session():
    asr = ScriptedASR("a", "b", "c")
    s = make(asr, endpoint_silence_s=0.3)
    seen = []
    for _ in range(3):
        feed(s, torch.cat([speech(0.8), silence(0.6)]))
        seen.append(s.committed_text)
    for earlier, later in pairwise(seen):
        assert later.startswith(earlier), f"{earlier!r} -> {later!r} was not append-only"


def test_transcribe_receives_a_1d_float_tensor():
    got = {}

    def spy(wav: torch.Tensor) -> str:
        got["ndim"], got["dtype"] = wav.ndim, wav.dtype
        return "x"

    s = make(spy, endpoint_silence_s=0.3)
    feed(s, torch.cat([speech(1.0), silence(0.6), speech(0.1)]))
    assert got == {"ndim": 1, "dtype": torch.float32}


def test_noise_floor_survives_a_silent_session_start():
    """A percentile over a short opening span used to make everything look
    silent (or nothing), closing a span on the very first packet."""
    asr = ScriptedASR("x")
    s = make(asr, endpoint_silence_s=0.3, min_segment_s=0.6)
    upds = feed(s, silence(0.5))
    assert upds == []
    assert asr.calls == 0
