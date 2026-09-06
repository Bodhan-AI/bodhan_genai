"""Offline batch ASR inference.

``python -m bodhan_genai.asr.inference.transcribe`` — sharded, resumable
manifest transcription. Kept import-light: the entry point imports torch and
the engine only once it actually runs.
"""
