"""Engine-facing plain-data types shared by every TTS engine implementation.

``SamplingConfig`` is the engine-agnostic sampling knob set (each engine maps it
onto its own generate/SamplingParams API); ``TTSResult`` is the uniform output
record every engine returns: float32 audio in [-1, 1] plus token/timing
accounting.

numpy-only module imports — safe to import without torch / vllm / transformers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    """Engine-agnostic sampling parameters (frozen; use ``merged`` to derive)."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    repetition_penalty: float = 1.1
    max_new_tokens: int = 2048

    def merged(self, **overrides) -> SamplingConfig:
        """Return a new config with non-None ``overrides`` applied.

        ``None`` values are ignored (so per-request kwargs can be passed
        straight through without filtering). Unknown keys raise ``TypeError``.
        """
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"Unknown SamplingConfig field(s): {', '.join(unknown)}")
        updates = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **updates)


@dataclass
class TTSResult:
    """One synthesized utterance: float32 audio in [-1, 1] plus accounting.

    ``error`` is set (and ``audio`` left empty) when generation or decode
    failed for this request — batch callers can keep going.
    """

    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sample_rate: int = 24_000
    prompt_tokens: int = 0
    generated_tokens: int = 0
    audio_tokens: int = 0
    gen_time_s: float = 0.0
    decode_time_s: float = 0.0
    error: str | None = None

    @property
    def duration_s(self) -> float:
        """Audio duration in seconds (0.0 for empty audio)."""
        if self.sample_rate <= 0:
            return 0.0
        return float(np.asarray(self.audio).size) / float(self.sample_rate)

    @property
    def rtf(self) -> float:
        """Real-time factor: (generate + decode) wall time / audio duration.

        < 1.0 means faster than real time. 0.0 when there is no audio.
        """
        dur = self.duration_s
        if dur <= 0.0:
            return 0.0
        return (self.gen_time_s + self.decode_time_s) / dur

    def save(self, path: str | Path) -> Path:
        """Write the audio as a 24 kHz PCM-16 WAV; returns the path written.

        Import deferred so this module stays numpy-only at import time.
        """
        from bodhan_genai.tts.inference.audio_io import write_wav_24k

        out = Path(path)
        write_wav_24k(out, np.asarray(self.audio, dtype=np.float32))
        return out
