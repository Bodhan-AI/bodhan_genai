"""ASR serving: Ray Serve app with buffered streaming, offline transcription,
and language ID.

Import-light on purpose — ``VadStream`` is pure logic and safe to import
anywhere, while the Ray/FastAPI machinery lives behind
``service.build_deployment()`` and is only paid when a server actually starts.
"""

from bodhan_genai.asr.serving.streaming import StreamUpdate, VadStream

__all__ = ["StreamUpdate", "VadStream"]
