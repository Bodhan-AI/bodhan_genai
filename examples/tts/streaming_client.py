"""Example: stream TTS audio from a running bodhan-genai server.

Connects a WebSocket to the server, sends one JSON synthesis request, receives
raw int16 PCM frames as they are decoded, reports TTFP (time to first PCM
frame), and writes the full utterance to a 24 kHz WAV.

Start a server first (see scripts/serve.sh), then:

    python examples/tts/streaming_client.py --url ws://localhost:8000/tts \
        --text "Hello from bodhan." --speaker spk1 --out hello.wav

This is a thin wrapper over ``bodhan_genai.tts.serving.client`` — see that module
for the actual connect / send / receive-PCM / write-WAV loop.
"""

from __future__ import annotations

from bodhan_genai.tts.serving.client import main

if __name__ == "__main__":
    raise SystemExit(main())
