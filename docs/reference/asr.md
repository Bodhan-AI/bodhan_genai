# ASR API reference

Generated from the source by [mkdocstrings](https://mkdocstrings.github.io/), read statically —
nothing on this page required importing torch to produce.

IndicTranscribe is an attention encoder-decoder ported from NeMo. Two things shape the whole API and
are easy to miss:

- **It is language-conditioned and the transcription path has no LID of its own.** You supply
  `lang`, and a wrong label yields *confidently wrong script* rather than visible errors.
- **Streaming is buffered, not frame-synchronous.** The latency floor is one decode interval. That
  follows from the architecture, not from this implementation.

## Engine

The public entry point. `transcribe_batch` carries the two output-mode flags (`itn`, `romanized`),
which map to slots 6 and 7 of the frozen 10-token prompt; both default to `False`, which is the
historical native-script behaviour.

::: bodhan_genai.asr.engine.engine

## Language identification

One decoder step over encoder states the transcription already computed — not a second pass.

Read the accuracy analysis in this module before relying on it: the widely-quoted 96.9% is
*agreement with NeMo*, while measured top-1 is 0.864 / 0.779 and as low as 0.047 for `bho`.

::: bodhan_genai.asr.engine.lid

## Continuous batching

::: bodhan_genai.asr.engine.continuous_batching

## Audio input

::: bodhan_genai.asr.engine.audio_input

## Inference

::: bodhan_genai.asr.inference.transcribe

## Model

The tokenizer owns the prompt contract: prompts are exactly 10 tokens in every mode, which is what
keeps mixed-mode batches working.

::: bodhan_genai.asr.model.tokenization_indic_transcribe

::: bodhan_genai.asr.model.feature_extraction_indic_transcribe

## Serving

::: bodhan_genai.asr.serving.protocol

::: bodhan_genai.asr.serving.client
