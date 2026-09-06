"""Incremental Orpheus sliding-window state machine for streaming TTS.

Factored out of ``eval/generate_audios_vllm_streaming.py`` (the inline
``deoffset_clamp`` / ``window_for`` / frame-emit logic) so the serving path can
consume vLLM **DELTA** token output incrementally — feeding only the *new* tokens
each step instead of re-scanning the cumulative list (the O(seq_len·concurrency)
cost that bottlenecked the offline path).

Pure: numpy only, no torch / no Ray. One instance per in-flight request.

Decode recipe (matches bodhan_genai.tts.codec.snac.decode_window_batch): per emitted
audio frame ``k`` we decode the 4-frame window ``[k-1, k, k+1, k+2]`` (zero-padded
at the edges) and keep only that window's middle frame — so windows must be
emitted in order, one per frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from bodhan_genai.tts.codec.snac import (
    SNAC_CODEBOOK_SIZE,
    SNAC_NUM_CODEBOOKS,
    SNAC_WINDOW_FRAMES,
)


class WindowJob(NamedTuple):
    """One 4-frame decode window. ``codes`` is int32 shape (28,), raw codes in
    [0, 4096) (offsets stripped, zero-padded at sequence edges)."""

    request_id: str
    emit_index: int
    codes: np.ndarray


class StreamingWindower:
    """Consume DELTA token ids for one request; yield decode windows in order.

    Usage per request:
        w = StreamingWindower(request_id, snac_ids)
        async for out in engine.generate(... output_kind=DELTA ...):
            for job in w.push(out.outputs[0].token_ids, finished=out.finished):
                submit(job)
        for job in w.flush_tail():      # right-padded tail frames
            submit(job)
        total = w.total_emits
    """

    def __init__(self, request_id: str, snac_ids: dict, window_frames: int = SNAC_WINDOW_FRAMES):
        self.request_id = str(request_id)
        self._base = int(snac_ids["audio_token_base_id"])
        self._start_id = int(snac_ids["start_of_audio_id"])
        self._end_id = int(snac_ids["end_of_audio_id"])
        self._nb = SNAC_NUM_CODEBOOKS
        self._wf = max(2, int(window_frames))  # frames per decode window; kept frame at index 1
        self._started = False
        self._ended = False
        self._pending: list[int] = []  # audio token ids not yet forming a full frame
        self._codes: list[int] = []  # raw codes [0,4096), frame-aligned
        self._dispatched = 0  # frames already emitted
        self.total_emits = 0

    # -- internal -----------------------------------------------------------
    def _ingest(self, new_ids: list[int]) -> None:
        """Find start_of_speech once, then collect audio token ids until
        end_of_speech (which is excluded; it can appear in the raw output)."""
        for t in new_ids:
            if self._ended:
                return
            if not self._started:
                if int(t) == self._start_id:
                    self._started = True
                continue
            if int(t) == self._end_id:
                self._ended = True
                return
            self._pending.append(int(t))

    def _drain_frames(self) -> None:
        """Convert complete groups of 7 pending audio tokens into raw codes.
        Pending always starts on a frame boundary (we only ever consume whole
        frames), so position = index % 7."""
        ncomplete = (len(self._pending) // self._nb) * self._nb
        if ncomplete == 0:
            return
        arr = np.asarray(self._pending[:ncomplete], dtype=np.int64)
        del self._pending[:ncomplete]
        pos = np.arange(ncomplete) % self._nb
        codes = arr - self._base - pos * SNAC_CODEBOOK_SIZE
        np.clip(codes, 0, SNAC_CODEBOOK_SIZE - 1, out=codes)
        self._codes.extend(int(x) for x in codes)

    def _window_for(self, k: int, F: int) -> np.ndarray:
        # W frames starting at k-1: [k-1, k, ..., k+W-2]; emitted frame k is at index 1.
        w = np.zeros(self._wf * self._nb, dtype=np.int32)
        for slot, f in enumerate(range(k - 1, k - 1 + self._wf)):
            if 0 <= f < F:
                w[slot * self._nb : (slot + 1) * self._nb] = self._codes[
                    f * self._nb : (f + 1) * self._nb
                ]
        return w

    # -- public -------------------------------------------------------------
    def push(self, new_ids, finished: bool = False) -> Iterator[WindowJob]:
        """Feed the NEW tokens from one DELTA step; yield any fully-contexted
        windows now decodable (frame k needs frame k+2 -> emit up to F-3)."""
        self._ingest(list(new_ids))
        self._drain_frames()
        F = len(self._codes) // self._nb
        hi = F - self._wf + 1  # frame k needs frame k+W-2 -> emit fully-contexted frames only
        if hi >= self._dispatched:
            for k in range(self._dispatched, hi + 1):
                yield WindowJob(self.request_id, k, self._window_for(k, F))
            self._dispatched = hi + 1

    def flush_tail(self) -> Iterator[WindowJob]:
        """Emit the remaining tail frames (right-padded windows) once generation
        is done. Sets ``total_emits`` to the final frame count."""
        F = len(self._codes) // self._nb
        for k in range(self._dispatched, F):
            yield WindowJob(self.request_id, k, self._window_for(k, F))
        self._dispatched = F
        self.total_emits = F
