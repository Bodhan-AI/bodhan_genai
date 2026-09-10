# IndicTranscribe configs

Two surfaces: the **server** (`ServeConfig` in `bodhan_genai.asr.serving.config`, exposed by
`scripts/asr/serve.sh`) and the **batch CLI** (`bodhan_genai.asr.inference.transcribe`, exposed by
`scripts/asr/infer.sh`). Knob names match where they mean the same thing, so moving between them
needs one mental model rather than two.

Every default below is copied from the source, and most were measured rather than chosen. Where a
number came from a measurement, the measurement is given — a default you cannot justify is a
default nobody dares change.

## Server: model and topology

| flag | default | what it does |
|---|---|---|
| `--model_dir` | `bodhan-ai/indic-transcribe-core` | Local directory or Hub repo id. |
| `--dtype` | `bfloat16` | `bfloat16` or `float32`. WER-neutral against fp32 — see [caveats](caveats.md). |
| `--num_replicas` | `-1` | `-1` auto-detects one replica per visible GPU. |
| `--gpus_per_replica` | `1.0` | Fractional values let several replicas share a GPU. |
| `--max_ongoing_requests` | `32` | **Admission control**, and the most important knob here. |
| `--max_queued_requests` | `256` | Queue depth before rejection. |

!!! danger "`max_ongoing_requests` is the only protection against overload"

    32 concurrent streaming sessions hold real time on one GPU (rtf 1.02, no drift). 36 drops to
    14%, and 40 to 0%. Past the ceiling sessions do **not** degrade one at a time — they share the
    GPU and all fall behind together. Raising this knob does not buy capacity; it converts a
    bounded queue into a room full of sessions that are each too slow.

    The value was 24 before the encoder was moved off the scheduler thread.

## Server: offline and long-form

| flag | default | what it does |
|---|---|---|
| `--batch_size` | `96` | Measured throughput knee on one H100. |
| `--chunk_above` | `45.0` s | Long-form threshold. At or below it, audio is decoded whole — **chunking shorter audio measurably hurts**. |
| `--chunk_min` | `15.0` s | Lower bound on a chunk when splitting. |
| `--chunk_max` | `25.0` s | Upper bound. The checkpoint was trained with `max_duration: 30` s. |

## Server: streaming

| flag | default | what it does |
|---|---|---|
| `--stream_endpoint_silence_s` | `0.5` s | Trailing pause that closes a span and emits final text. |
| `--stream_max_segment_s` | `5.0` s | Force-cut bound — **the hard ceiling on time-to-first-text**. |
| `--stream_partial_interval_s` | `2.0` s | How often to re-decode the open span for interim text. `0` disables. |
| `--stream_min_segment_s` | `0.6` s | Never emit a span shorter than this. |
| `--stream_slots` | `32` | Continuous-batching slot pool size. |
| `--stream_admit_batch` | `16` | Pending requests encoded per admission pass. |
| `--stream_no_cuda_graphs` | off | Escape hatch if graph capture misbehaves. |
| `--stream_overlap_encode` | on | Encoder on a producer thread and side CUDA stream. |
| `--stream_sample_rate` | `16000` | Raw little-endian int16 PCM, mono — the same wire format the TTS server emits, so one can be piped into the other. |

Four of these carry costs worth knowing before you touch them:

- **`stream_max_segment_s` is the latency SLO knob**, set to the SLO (5 s) rather than to whatever
  maximises throughput. A pause-free stretch yields nothing until it elapses.
- **`stream_endpoint_silence_s` below ~0.35 s** closes spans mid-phrase on ordinary breath pauses.
  It cannot be tuned on the current bench corpus — synthetic TTS output with a pause p99 of only
  0.56 s. Tuning it needs real conversational audio.
- **Partials cost roughly half the capacity gain.** Spans are decoded once (4.43× the throughput
  of the old re-decoding scheme); adding partials at 2.0 s brings that to 2.17×, and at 1.0 s to
  1.56×. Set `0` when interim text is not displayed.
- **`stream_slots` is cheap in compute, not in memory.** Graphed step cost is nearly flat from 8
  to 32 slots (2.5–2.7 ms graphed vs 5.8–6.0 ms eager, 2.3×); what slots cost is KV memory.
- **`stream_overlap_encode` is worth 1.33×** (24 → 32 realtime sessions/GPU). The encoder is 42%
  of GPU time; run inline it stalls every queued decode behind it. Turn it off only to debug.

## Batch CLI

`scripts/asr/infer.sh` wraps `python -m bodhan_genai.asr.inference.transcribe`.

| flag | default | what it does |
|---|---|---|
| `--manifest` | **required** | JSONL manifest, one object per line. |
| `--model-dir` | `bodhan-ai/indic-transcribe-core` | Local directory or Hub repo id. |
| `--out-dir` | **required** | Shard outputs land here as `hyp_shard<N>.jsonl`. |
| `--shard` / `--num-shards` | `0` / `1` | One shard per GPU, each its own process. |
| `--max-items` | `0` | `0` = no limit; a debugging aid. |
| `--dtype` | `bfloat16` | |
| `--itn` / `--romanized` | off | Output mode for the whole run. |
| `--chunk-above` | `45.0` s | Segment rows longer than this on silences. |
| `--chunk-min` / `--chunk-max` | `15.0` / `25.0` s | |
| `--encoder-batch` | `24` | `engine` backend only. |
| `--audio-workers` | `8` | `engine` backend only. |

### Environment

| var | default | what it does |
|---|---|---|
| `MODEL_DIR` | *(unset)* | Local directory **or** Hub repo id. **Honoured only inside a deployment image**; outside one the launchers unset it and use the default — pass `--model-dir` / `--model_dir` instead. |
| `NUM_SHARDS` | `1` | One process per shard. |
| `BODHAN_ASR_HF_REPO` | *(unset)* | Override the default repo. **Honoured only inside a deployment image** (`BODHAN_GENAI_DEPLOYMENT=1`, set by `docker/*/Dockerfile*`); ignored elsewhere so a stale shell variable cannot redirect weights. |
| `INDIC_TRANSCRIBE_PROFILE` | unset | Set to `1` for phase timers that add up. |

!!! warning "Budget 4–8 CPU cores per GPU"

    Audio decode and feature-batch assembly are CPU work, and a CPU-starved node starves the GPUs.
    Measured: with 2 CPUs serving 8 GPUs, per-shard encode time went from 18 s to **437 s**.

## Backends

| backend | when |
|---|---|
| `batch` (default) | Gate-verified fixed batch. The only path with long-form chunking. |
| `engine` | `IndicTranscribeEngine` continuous batching. ≈2.1× faster on a tuned comparison (40.8 s vs 86.4 s on a 1500-row, 14-audio-hour shard) because it evicts and refills decoder slots instead of waiting for a batch's longest member. |
