# ASR serving

```bash
pip install -e ".[asr-serve]"
MODEL_DIR=/path/to/indic-transcribe-hf ./scripts/asr/serve.sh
```

One Ray Serve replica per GPU, each holding one IndicTranscribe engine.

| endpoint | transport | what |
|---|---|---|
| `/asr/stream` | websocket | buffered streaming: raw int16 PCM in, incremental JSON updates out |
| `/asr/transcribe` | HTTP POST | server-readable audio paths in, transcripts out (long-form aware) |
| `/asr/detect` | HTTP POST | language identification only |
| `/health` | HTTP GET | replica readiness |

## Read this before building on `/asr/stream`

**This model cannot stream frame-synchronously.** IndicTranscribe is an attention
encoder-decoder model: the decoder cross-attends over the *whole* encoder
output and is autoregressive, so no text exists until a chunk has been decoded
end to end. There is no per-frame emission point to hook into.

That is not an implementation gap, it is the architecture. NeMo says so
directly — `mixins.py` raises `NotImplementedError` for cache-aware streaming
on anything that is not `EncDecCTCModel` / `EncDecRNNTModel`, and Canary-style
configs train with `att_context_size: [-1, -1]` (unlimited context). If you
need true low-latency streaming ASR, you need a CTC or RNNT model; no serving
layer turns this one into one.

What this endpoint does instead is **VAD endpointing**: text is produced from
complete spans of audio cut at pauses, and each span is decoded exactly once.

1. buffer incoming audio;
2. when the buffer ends in a pause of at least `stream_endpoint_silence_s`,
   close the span there and **decode it once** — that text is final;
3. a span reaching `stream_max_segment_s` without a pause is force-cut at the
   best silence available (hard cut only if there is none);
4. between endpoints, re-decode the open span every
   `stream_partial_interval_s` for interim text.

So each update sends:

```json
{"event": "update",
 "committed_delta": "text of the span that just closed",
 "committed": "everything final so far",
 "provisional": "interim guess for the open span, may be replaced",
 "is_final": true,
 "audio_seconds": 12.5}
```

`committed` is **append-only and never revised** — safe to write to a
transcript. `provisional` is a guess that the next update replaces outright —
render it greyed out, never persist it. `is_final` distinguishes the two kinds
of update.

**`stream_max_segment_s` (default 5 s) is the latency ceiling, not the packet
size.** Sending 20 ms packets does not get you 20 ms latency. Time-to-first-text
is however long it takes the speaker to pause, bounded by `max_segment_s`.

### Why not LocalAgreement, which this replaced

The previous implementation followed `whisper_streaming`: re-decode a growing
buffer every interval and commit the longest prefix surviving N consecutive
decodes. It worked, but it was paying to redo the same audio over and over.
Measured over 778 s of audio:

| | LocalAgreement | VAD endpointing |
|---|---|---|
| audio encoded per audio-second | **4.43x** | 1.00x |
| decoder tokens generated | 8757 | ~2000 (1.00x) |
| model input | buffers truncated mid-word | complete spans |

Both phases carried the redundancy, so removing it is worth up to **4.43x** the
GPU work — larger than every other optimisation in this document combined.
Partials give some of that back, because each one is a full re-decode of the
open span:

| `partial_interval_s` | encoder redundancy | capacity vs LocalAgreement |
|---|---|---|
| off | 1.00x | **4.43x** |
| 2.5 | 1.51x | 2.94x |
| **2.0** | **2.04x** | **2.17x** (shipped) |
| 1.5 | 2.56x | 1.73x |
| 1.0 | 2.85x | 1.56x |

Endpointing also feeds the model what it was trained on. Canary is trained on
complete utterances; LocalAgreement handed it buffers cut mid-word, repeatedly.

**What it costs.** Text arrives in bursts at pauses rather than growing
continuously, and a pause-free stretch yields nothing until `max_segment_s` —
LocalAgreement's latency was bounded by its interval regardless of content.

> **The endpoint threshold is NOT tuned.** The bench corpus is entirely
> synthetic TTS output (`indic_f5_bench`, `sarvam_gen`, `gemma-tts`), whose pause
> distribution is p50 0.05 s, p99 0.56 s, max 0.95 s. Real conversational speech
> pauses for 0.5-2 s at turn boundaries. The default 0.5 s is the production
> convention, not a measured optimum: on this corpus it never fires, so every cut
> is a forced one. The capacity ratios above are unaffected (they depend on
> `max_segment_s` and `partial_interval_s`, not the endpoint), but
> `stream_endpoint_silence_s` and any streaming WER claim need real
> conversational audio before they mean anything.

### Comparison to what NeMo ships for AED

NeMo's `FrameBatchMultiTaskAED` chunks into **non-overlapping** windows (its
launcher sets `total_buffer == frame_len`) and merges with a bare
`" ".join()` — no dedup, no boundary repair, so words are lost or duplicated
at every seam. It is also batch-only: the whole file is decoded before any
text is returned, with no incremental callback. Fine for offline throughput,
which is what it is for. The LocalAgreement approach here exists because a
live stream needs incremental output *and* clean seams.

## Streaming client

```bash
python -m bodhan_genai.asr.serving.client --mode stream --audio a.wav --lang hi
```

It paces audio like a live microphone by default so the printed timings mean
something; `--no-realtime` pushes as fast as possible.

Wire protocol, if you are writing your own client:

1. connect, send one JSON `{"lang": "hi", "sample_rate": 16000}` (optionally
   `"itn": true` or `"romanized": true` to select the output mode for the
   whole stream);
2. send raw **little-endian int16 PCM, mono** frames as binary messages;
3. send `{"event": "eof"}` (or just close) when done;
4. read `start`, then `update`s, then `end` (or `error`).

That is the same PCM wire format the TTS server emits, so one can be piped
into the other without a converter.

**`detect_language` is rejected on this endpoint.** At session start no audio
has arrived, and silently guessing a language would produce confidently wrong
script rather than obvious garbage. Call `POST /asr/detect` on a sample first,
then open the stream with the answer.

## Offline transcription

```bash
curl -s localhost:8000/asr/transcribe -H 'Content-Type: application/json' -d '{
  "paths": ["/data/a.wav", "/data/b.wav"],
  "lang": "hi",
  "itn": false,
  "romanized": false,
  "chunk_above": 45
}'
```

Paths are read **by the server**, not uploaded. Rows longer than `chunk_above`
go through the silence-aware chunked path automatically; shorter ones are
decoded whole, because chunking short audio measurably hurts
(see [caveats.md](caveats.md)).

### Authentication

Every route except `GET /health` sits behind HTTP Basic auth. Point
`ASR_AUTH_FILE` at a `chmod 600` file containing `user:password` — outside the
repo, so the credential is never committed and never travels through Ray's
`runtime_env`; only the path does. Starting without it is a hard error;
`ASR_AUTH_ALLOW_OPEN=1` disables auth explicitly for localhost runs.

`/health` is exempt so the launcher and tunnel can poll readiness. It discloses
liveness only, and it returns 503 — not 200 — when the replica's slot scheduler
has died.

### Limits

| knob | default | what it bounds |
|---|---|---|
| `ASR_MAX_UPLOAD_MB` | 256 | request body on `POST /asr/upload` (413 past it) |
| `ASR_LID_MAX_SECONDS` | 120 | audio a single LID call encodes; longer files are probed and voted |
| `ASR_UPLOAD_DIR` | system temp | where upload temp files land |
| `--batch_size` | 96 | rows per encoder batch; longer `paths` lists are split |
| `MAX_PATHS` | 512 | `paths` per request |

`sample_rate` on `WS /asr/stream` is checked against a whitelist: it is the
divisor for every buffer bound in the stream, so an arbitrary value made those
bounds unreachable.

### Language

Three ways, in this order of precedence:

* **Give `lang`** — used as-is. LID never overrides it.
* **Omit `lang`** — LID fills it per row, then transcription proceeds. It costs
  no extra encoder pass: identification is one decoder step over states the
  transcription already computes. Prefer supplying `lang` when you can: measured
  LID top-1 is 0.864/0.779 and as low as 0.047 for `bho` and 0.258 for `hi`, and
  a wrong label yields confidently wrong *script* rather than visible errors
  ([caveats](caveats.md)).
* **`"detect_language": true`** — report LID *as well*, including for rows where
  you supplied the language, so a disagreement between your metadata and the
  model is visible rather than silent.

Each row of `results` carries `lang`, `lang_source` (`"explicit"` or `"lid"`),
and `lid` (the ranked distribution). The top-level `lang` is the request's
single language, or `null` when the rows disagree — LID is resolved **per row**,
so one request may legitimately span several languages.

`allowed_langs` narrows the LID candidate set: a list of codes, or the named
sets `"trained"` (the 27 this checkpoint knows) and `"recommended"` (those minus
`bgc`/`bhb`, which act as sinks — excluding them lifts `pa` from 0.62 to 0.78).
It is a HARD filter: audio genuinely in an excluded language is reassigned to
the nearest permitted one, never flagged, which is why it defaults to unset.

LID accuracy is uneven — measured top-1 is 0.86 (lattice) / 0.78 (VOI) overall,
but `ml`/`ta`/`kn` are ~0.97 while `bho` is 0.05, `hi` 0.26, `mai` 0.36 and `ur`
0.49. Those four are absorbed by close neighbours, so an LID-sourced answer for
them ships a `warning`. **If you have a language label, pass it.**

## Capacity, measured

One H100, real Indic audio (10 languages, 20–57 s, median 32 s), sessions paced
at 1× like a live mic. `rt_ok%` = sessions that held real time (rtf ≤ 1.15 and
no growing lag).

**Decode cost is linear in buffer length, ~15× realtime:**

| buffer | decode | sustains a 1 s interval? |
|---|---|---|
| 2 s | 0.14 s | yes |
| 10 s | 0.66 s | yes |
| 15 s | 1.25 s | no |
| 30 s | 1.74 s | no |

That single table governs everything, because a session keeps up only while
`C × decode(buffer) < interval` — where C collapses to 1 once sessions are
batched together (below). Buffer/interval sweep, each session decoding alone:

| interval / buffer / trim | C=1 | C=2 | C=3 | C=4 |
|---|---|---|---|---|
| 1.0 / 30 / 10 (original guess) | **0%** (drift +1.5 s) | — | — | — |
| 1.0 / 6 / 3 | 100% | 100% | 67% | 0% |
| 1.0 / 10 / 5 | 100% | 100% | — | 0% (drift +17 s) |
| **2.0 / 10 / 5 (shipped)** | **100%** | **100%** | **100%** | 0% |

### Continuous batching is the main capacity lever

A decoder step is bound by launch overhead over ~300 tiny kernels, not by
weight bandwidth (measured: 6-7 ms/step against a 0.25 ms weight-read floor;
HF `generate()` itself adds only 1.0-1.2x). CUDA-graphing the step collapses
those launches:

| batch | 8 | 16 | 32 |
|---|---|---|---|
| eager | 5.82 ms | 6.02 ms | 5.98 ms |
| graphed | 2.46 ms | 2.58 ms | 2.74 ms |

**2.3x, and nearly flat in batch size** — which is what makes a large pool
cheap. A *fixed* batch cannot exploit this: a graph needs a constant shape, so
it would have to pad or wait to fill. The slot pool
(`serving/slot_engine.py`) makes the batch dim permanently constant — every
step runs ALL slots and masks idle ones — so graphs work *and* admission never
waits. It also removes ragged waste: a fixed batch steps every row until the
longest finishes, and across a realistic cohort output lengths ran 22-70
tokens, i.e. ~33% of steps were spent on rows that had already stopped.

### Encoder/decoder split, and why it is NOT disaggregated across GPUs

Measured GPU time at the ceiling (device syncs on — without them a graphed
decode replay is async and its cost lands in whatever syncs next, which made an
early reading say "95% encoder"; that number was an artifact):

| phase | GPU time | share |
|---|---|---|
| encode | 13.68 s | 42% |
| decode | 18.65 s | 58% |

So the natural encoder:decoder ratio is about **1 : 1.4** — roughly 2 encoders
per 3 decoders. A 1-encoder-per-4-decoders split, the intuitive guess, would
give the encoder side 20% of the hardware for 42% of the work and starve it by
about 2x.

**But the two phases are not split across GPUs, and should not be.** At the
ceiling the GPU is only ~72% busy, so the limit was never average throughput —
it was that the encoder ran INLINE in the scheduler and stalled every queued
decode behind it whenever a burst of sessions came due together. Running the
encoder on a producer thread and a side CUDA stream fixes exactly that, on one
GPU, with no cross-GPU KV transfer and no ratio to tune:

| C | 24 | 28 | **32** | 36 | 38 | 40 |
|---|---|---|---|---|---|---|
| rt_ok% inline encoder | 100 | 39 | 0 | — | — | — |
| rt_ok% overlapped | 100 | 100 | **100** | 14 | 5 | 0 |

**24 -> 32 sessions per GPU (1.33x) from overlap alone.** Notably larger than
the 1.02-1.06x upstream measured for stream overlap in its fixed-batch driver,
because here the encoder is 42% of the work and was fully serialising with
decode rather than partially.

True cross-GPU disaggregation would remove SM contention that overlap only
hides, but the remaining headroom is small: the phases now overlap, the GPU is
near saturation at the ceiling, and splitting introduces cross-GPU cross-KV
transfer (~12 MB/utterance) plus a ratio that must track the workload or one
pool idles. Colocated-with-overlap is the better default; revisit only if a
profile shows the phases genuinely fighting for SMs.

Measured end to end at the shipped 2.0 s interval / 10 s window:

| C | 8 | 16 | 24 | **32** | 36 | 40 |
|---|---|---|---|---|---|---|
| rt_ok% | 100 | 100 | 100 | **100** | 14 | 0 |
| xRT | 6.2 | 12.0 | 17.6 | **23.2** | 25.6 | 28.6 |

**32 concurrent realtime streams per GPU** (~256 per 8-GPU node), TTFT p95
~4.0 s.

> **Caveat on the "before" number.** The previous fixed-batch path measured 8
> sessions, but that measurement was taken *before* a resampling bug was found
> (client audio was never converted to the model's 16 kHz rate). Fixing it
> changed how many samples a buffer contains, so 8 -> 24 conflates the slot
> pool, CUDA graphs, and that fix. The graph and ragged-waste numbers above are
> isolated and trustworthy; the end-to-end ratio is not a clean attribution.

Decode parity against the gate-verified `generate()` path on real streaming-
sized buffers: **15/16 exact text match**, mean character similarity 0.9988.
The single difference was a trailing word — consistent with the documented
batch-composition sensitivity, since NeMo's ConvSubsampling is unmasked.

### Streaming still costs far more GPU than offline

Same GPU, same audio: offline batch reached **RTFx 25–41**, streaming
aggregate **23.2 at the 32-session ceiling**. Re-decoding a rolling buffer many times over is
enormously more expensive per audio-second than decoding once. Use streaming
only where live partial text is genuinely required; anything batchable belongs
on `POST /asr/transcribe`.

## Knobs

| flag | default | notes |
|---|---|---|
| `--num_replicas` | GPU count | one engine per replica |
| `--max_ongoing_requests` | 32 | measured realtime ceiling per GPU; 36 degrades, 40 collapses |
| `--stream_endpoint_silence_s` | 0.5 | pause that closes a span; NOT tuned on real audio |
| `--stream_max_segment_s` | 5.0 | force-cut bound = ceiling on time-to-first-text |
| `--stream_partial_interval_s` | 2.0 | interim re-decode; 0 disables and is 2x cheaper |
| `--stream_min_segment_s` | 0.6 | shorter spans are dropped, not decoded |
| `--stream_slots` | 32 | continuous-batching pool size; graphed step cost is ~flat in slots |
| `--stream_admit_batch` | 16 | pending requests encoded per admission pass |
| `--stream_overlap_encode` | on | encoder on a producer thread + side stream; worth 1.33x |
| `--stream_no_cuda_graphs` | off | escape hatch; 2.3x slower |
| `--chunk_above` | 45.0 | long-form threshold for the HTTP endpoints |
| `--dtype` | bfloat16 | WER-neutral vs fp32 |

Budget **4–8 CPU cores per GPU**: audio decode and feature assembly are CPU
work, and a starved node starves the GPUs.

Reproduce with:

```bash
python -m bodhan_genai.asr.serving.loadtest --url ws://HOST:8231 \
    --jsonl audio.jsonl --concurrency 1 2 3 4
```

## Still not measured

- **WER under streaming.** Capacity is measured; transcription *quality* through
  the buffered path (vs decoding the same audio offline) is not. LocalAgreement
  commits text that two decodes agree on, and buffer trimming cuts at silences —
  both plausibly cost accuracy at seams. Measure before promising quality.
- Multi-replica scaling is assumed linear from single-GPU numbers, not observed.
- Non-24 kHz input rates, and sessions much longer than ~60 s.
