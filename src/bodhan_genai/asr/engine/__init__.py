"""Public ASR engine API: ``IndicASREngine`` (batch transcription) plus the
waveform-input helpers it shares with callers that already hold decoded audio.

Light at import time in the same sense as ``bodhan_genai.tts.engine``: the
heavy deps (torch / transformers / torchaudio) come in via the model
subpackage, so importing this module is an explicit opt-in.
"""

from bodhan_genai.asr.engine.audio_input import (
    collate_waveforms,
    read_slice,
    read_span_and_slice,
)
from bodhan_genai.asr.engine.chunker import ChunkConfig, chunk_audio, split_points
from bodhan_genai.asr.engine.continuous_batching import (
    EngineStats,
    IndicTranscribeEngine,
    Utterance,
)
from bodhan_genai.asr.engine.engine import IndicASREngine
from bodhan_genai.asr.engine.lid import (
    detect_language,
    language_token_map,
    lid_from_encoder_states,
)

__all__ = [
    "ChunkConfig",
    "EngineStats",
    "IndicASREngine",
    "IndicTranscribeEngine",
    "Utterance",
    "chunk_audio",
    "collate_waveforms",
    "detect_language",
    "language_token_map",
    "lid_from_encoder_states",
    "read_slice",
    "read_span_and_slice",
    "split_points",
]
