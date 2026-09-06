# TTS API reference

Everything below is generated from the source by
[mkdocstrings](https://mkdocstrings.github.io/), read statically — nothing on this page
required importing torch or vLLM to produce.

`bodhan_genai.tts` exports lazily, so `import bodhan_genai.tts` stays free of the GPU stack until
an engine is actually constructed.

## Engines

::: bodhan_genai.tts.engine.offline.IndicTTSEngine

::: bodhan_genai.tts.engine.streaming.IndicStreamingTTSEngine

::: bodhan_genai.tts.engine.chunked.ChunkedIndicStreamingTTS

## Types

::: bodhan_genai.tts.engine.types

## Text segmentation

The chunker's sentence scanner, exported from `bodhan_genai.tts`. `max_chars` is a strict bound on
every chunk; `min_chars` is best-effort merging only.

::: bodhan_genai.tts.engine.chunked.split_sentences

::: bodhan_genai.tts.engine.chunked.chunk_text

::: bodhan_genai.tts.engine.chunked.plan_dialogue_chunks

::: bodhan_genai.tts.engine.chunked.estimate_speech_seconds

## Templates

::: bodhan_genai.tts.templates.chat

::: bodhan_genai.tts.templates.conversation

## Codec

::: bodhan_genai.tts.codec.snac

::: bodhan_genai.tts.codec.vocos

## Serving

::: bodhan_genai.tts.serving.config

::: bodhan_genai.tts.serving.protocol
