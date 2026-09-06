# ASR usage

Read [caveats.md](caveats.md) before trusting any WER number, and note the two
things that bite first: **the model needs a language label** (a wrong one gives
confidently wrong script, not obvious garbage), and **long audio needs chunking**
(quality collapses past ~60 s).

## Install

```bash
pip install -e ".[asr-infer]"
```

## Python API

```python
import torch
from bodhan_genai.asr import IndicASREngine

engine = IndicASREngine("/path/to/indic-transcribe-hf", device="cuda", dtype=torch.bfloat16)

texts = engine.transcribe_batch(["a.wav", "b.wav"], lang="hi")
```

Output modes — native script is the default; `itn=True` gives
mixed-script/ITN output, `romanized=True` gives Latin romanization
(scalar or per-row, so one batch may mix modes):

```python
texts = engine.transcribe_batch(["a.wav"], lang="hi", romanized=True)
texts = engine.transcribe_batch(["a.wav", "b.wav"], lang="hi", itn=[True, False])
```

`transcribe_batch` also accepts in-memory waveforms or a pre-collated
`(B, S)` 16 kHz tensor with `sample_lens` — all three input styles take the
identical padding / mono / resample path, so switching between them cannot move
the output text.

**Batches must be single-language**, because the prompt encodes the language.
Group by language before batching (the CLI does this for you).

### Long-form audio

```python
text = engine.transcribe_long("interview.wav", lang="hi")  # defaults: 45 s threshold
text, chunks = engine.transcribe_long("interview.wav", lang="hi", return_chunks=True)
```

`chunk_above` (default 45 s) is a threshold, not a switch: shorter audio is
transcribed whole, because chunking it measurably hurts. `chunks` is a list of
`(start_s, end_s, text)` if you need timing.

### Language identification

```python
top = engine.detect_language(["unknown.wav"])  # [[('hi', 0.99), ('ur', 0.004), ...]]
lang = top[0][0][0]
text = engine.transcribe_batch(["unknown.wav"], lang=lang)
```

One decoder step, not a second transcription pass — the encoder result is reused.

**Pass a language if you have one.** Measured top-1 accuracy is **0.864**
(lattice) and **0.779** (VOI) over 337k clips. The 96.9% figure quoted elsewhere
is *agreement with the NeMo detector*, not accuracy: the two implementations
agree closely and are wrong together on the confusable pairs.

The average also hides a very uneven spread — `ml`/`ta` reach 0.979 while `bho`
is **0.047**, `hi` **0.258**, `mai` 0.356 and `ur` 0.490, because a close
neighbour absorbs them. A wrong label produces confidently wrong *script*, not
visible errors, so **do not use LID for hi/bho/mai/ur if you have any metadata at
all**. Filtering on the returned probability does not rescue it: accuracy on the
surviving rows rises (0.779 → 0.836 at p≥0.7) but coverage falls faster.

Full per-language analysis, and why restricting to the trained 27 languages
changes nothing, is in `bodhan_genai.asr.engine.lid`.

If you already hold encoder states (e.g. inside your own batching loop), use
`bodhan_genai.asr.engine.lid.lid_from_encoder_states` instead and skip the
second encoder pass entirely.

### Transcribing spans of a long recording

```python
from bodhan_genai.asr.engine import read_span_and_slice

slices = read_span_and_slice("episode.wav", [(10.0, 25.0), (25.0, 40.0)], engine.fe)
texts = engine.transcribe_batch(slices, lang="hi")
```

One sequential read instead of N seeks — measured at 12.8 s vs 529.7 s on a
5.8 h episode with 3,177 chunks. Keep the covering span tight: it reads
`[min(start), max(end))` whether or not the middle is wanted.

## Batch CLI

```bash
python -m bodhan_genai.asr.inference.transcribe \
    --manifest data.jsonl --model-dir /path/to/indic-transcribe-hf --out-dir out/
```

Manifest is JSONL, one object per line, with an audio path and a language:

```json
{"audio_path": "/audio/0001.wav", "language": "hi"}
```

Use `--audio-key` / `--lang-key` for different field names, or `--lang hi` to
force one language for every row. Input fields are copied through to the output
alongside `row` and `hypothesis`, so any id/metadata you carry survives.

Long-form and multi-GPU:

```bash
# mixed-script/ITN or romanized output for the whole run
#   --itn | --romanized

# segment rows over 45 s on silences
python -m bodhan_genai.asr.inference.transcribe ... --chunk-above 45

# 8 GPUs, one shard each
MODEL_DIR=/path/to/indic-transcribe-hf NUM_SHARDS=8 \
    scripts/asr/infer.sh --manifest data.jsonl --out-dir out/
```

Budget **4–8 CPU cores per GPU**: audio decode and feature-batch assembly are
CPU work, and a starved node starves the GPUs (measured: 2 CPUs for 8 GPUs took
per-shard encode time from 18 s to 437 s).

### Resume

Each shard appends to `out/hyp_shard<N>.jsonl` and skips rows already present, so
a killed run resumes by re-running the same command. Resume is keyed on the
**manifest row index**, never on an id field: in the corpus this port was built
against, 84k rows carried only 48k unique keys, and key-based resume silently
dropped the duplicates. A torn final line from a killed run is re-transcribed.

### Errors

Unreadable audio and failed batches become rows with an `error` field rather
than killing the shard, so one bad file cannot cost you a long run.

### Backends

```bash
# default: gate-verified fixed batch, and the only path with long-form chunking
python -m bodhan_genai.asr.inference.transcribe ... --backend generate --chunk-above 45

# bulk throughput: continuous batching, ~2.1x faster on a tuned comparison
python -m bodhan_genai.asr.inference.transcribe ... --backend engine --slots 256
```

Use `generate` for numbers you intend to publish. `engine` has untested
regimes (see [caveats.md §4](caveats.md)) and no chunking — the CLI warns
loudly rather than silently ignoring `--chunk-above`.

## Serving

```bash
pip install -e ".[asr-serve]"
MODEL_DIR=/path/to/indic-transcribe-hf ./scripts/asr/serve.sh
```

Buffered streaming over websocket, plus HTTP endpoints for offline
transcription and language ID. **Streaming here is buffered, not
frame-synchronous** — an AED model cannot emit per frame, so the latency floor
is one decode interval. See [serving.md](serving.md) for the protocol, the
whisper_streaming-style commit semantics, and what has not been measured yet.

## Scoring

Deliberately not included — this package writes hypotheses and stops, matching
the repo-wide convention that model-quality metrics live outside the inference
library. Before you score anything, read [caveats.md §3](caveats.md): the
scoring pitfalls there move WER by more than 15 points, far more than any model
difference you are likely to be measuring.
