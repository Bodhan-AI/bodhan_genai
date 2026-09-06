# IndicTranscribe — model and port

`bodhan_genai.asr` is a NeMo-independent, HuggingFace-style port of the
IndicTranscribe ASR model: a **Canary-2 AED architecture** — 32-layer FastConformer
encoder + 24-layer Transformer decoder, `d_model=1024`, vocab 7152 — trained for
Indic-language speech recognition.

"NeMo-independent" is the point: the model runs under stock `transformers`
(`from_pretrained`, `generate()`, `EncoderDecoderCache`) with no `nemo_toolkit`
dependency, while reproducing NeMo's numerics closely enough to be a drop-in
replacement for the production pipeline.

## Parity

Verified against the production NeMo pipeline before this port was vendored:

| dtype | exact text match | WER (production) | WER (port) | ΔWER |
|---|---|---|---|---|
| fp32 | 99.8% (1 tie) | 34.46% | 34.46% | **+0.000%** |
| bf16 | 75.8% | 34.46% | 34.36% | **−0.102%** |

fp32 ΔWER is exactly 0.000 in every one of the 10 languages tested. The single
non-exact utterance is a verified near-tie: at generated token 98 the top two
candidates sit 2.7e-3 apart in log-prob, and cross-process TF32 convolution
algorithm selection is enough to flip that ordering.

On real recordings (AI4Bharat Rasa val, human studio audio, 22 languages,
median 5.6 s) fp32 is **character-identical to NeMo on every set and every
language** — 352/352 main, 130/130 sub-1 s, 40/40 long — with zero empty
hypotheses.

**Treat bf16 as WER-neutral, not as an improvement.** Its lower exact-match rate
is a length effect: a single near-tie flip anywhere breaks exact match, and
per-language deltas move in both directions and net to ≈0.

## Layout

| module | what |
|---|---|
| `model/configuration_indic_transcribe.py` | `IndicTranscribeConfig` |
| `model/modeling_indic_transcribe.py` | `IndicTranscribeForConditionalGeneration` — standard HF contract |
| `model/feature_extraction_indic_transcribe.py` | 24k→16k resample + NeMo-exact mel front-end (GPU-capable) |
| `model/tokenization_indic_transcribe.py` | aggregate SPM tokenizer + canary2 prompt (selectable output-mode slots) |
| `engine/engine.py` | `IndicASREngine` — batch transcription, long-form, LID |
| `engine/continuous_batching.py` | `IndicTranscribeEngine` — continuous-batching slot pool (bulk throughput) |
| `engine/chunker.py` | silence-aware segmentation for long audio |
| `engine/audio_input.py` | waveform / file-slice input helpers |
| `engine/lid.py` | language identification from the same checkpoint |
| `inference/transcribe.py` | sharded, resumable offline manifest transcription |
| `serving/` | Ray Serve app: buffered streaming + offline endpoints ([serving.md](serving.md)) |

## Checkpoint

The engine expects an **already-converted HF checkpoint directory** containing
`config.json`, `model.safetensors`, `feature_extractor.safetensors`, and the two
tokenizer SPM models. Conversion from a NeMo `.nemo` file is out of scope for
this package (it is a one-time, pure key-rename operation), mirroring how
`bodhan_genai.tts` assumes an already-extended tokenizer rather than shipping
tokenizer-extension tooling.

The fp32 safetensors file is the master; pass `dtype=torch.bfloat16` at load time
for inference.

## Decoding contract

Every utterance is prompted with a fixed 10-token layout; only the two
output-mode slots (6: itn, 7: romanized) vary, selected per request. The
default — `itn=False, romanized=False` — is ids `[7, 4, 18, L, L, 5, 9, 11, 13, 15]`:

```
<|startofcontext|><|startoftranscript|><|emo:undefined|><|LANG|><|LANG|>
<|pnc|><|noitn or itn|><|noromanized or romanized|><|notimestamp|><|nodiarize|>
```

Three output modes (the `itn_romanized_posttrain` checkpoint line is trained
for all three; every API accepts `itn=`/`romanized=` flags, scalar or per-row):

| mode | flags | output |
|---|---|---|
| native (default) | — | native script; English/maths transliterated; numbers as words |
| mixed / ITN | `itn=True` | Latin loanwords + digits kept (`<|itn|>` in slot 6) |
| romanized | `romanized=True` | Latin romanization (`<|romanized|>` in slot 7) |

The prompt is exactly 10 tokens in every mode, so batching and prompt-stripping
are mode-independent, and mixed-mode batches stack like mixed-language ones.

Consequences you will see in the DEFAULT (native) output, all of them intended:

- output is **native script** — English and mathematics are transliterated;
- numbers are spelled as **words**, not digits;
- punctuation **is** emitted. (The legacy pipeline passed `punc=False`, but NeMo
  silently drops that kwarg — the slot is named `pnc` — so production has always
  run with punctuation on. The port reproduces the *actual* behaviour.)

Decoding is greedy, which is equivalent to NeMo's `beam(beam_size=1)` here: no
length penalty, temperature, or logit processor is active in the NeMo path.
Generation stops on EOS(3) or PAD(2), capped at `min(1024, enc_frames + 50) + 1`
total tokens including the prompt (the `+1` replicates a NeMo generator
off-by-one; the final token is never embedded, so the 1024-row position table is
not exceeded).

`<|itn|>` and `<|romanized|>` are selectable via the `itn=`/`romanized=` flags
(functionally verified: correct prompt ids, script-correct outputs, engine/slot
parity; **no WER/quality evaluation yet**). `<|timestamp|>`, `<|diarize|>` +
`<|spk0..15|>` exist in the vocabulary but remain **unverified** in this port.

## Documented deviations from the legacy pipeline

1. **Greedy instead of beam(1)** — same math; ties break identically in practice.
2. **≥2 timestamp tokens in a hypothesis**: production rewrites the text via
   `timestamp_utils`; this port strips and warns. Cannot trigger under
   `<|notimestamp|>` prompts.
3. `<|nospeech|>` leaks into text as a literal token — same as production.
4. Audio that fails to read becomes an error row up front, rather than being
   silently dropped (production's fault-tolerant loader dropped the cut and
   misaligned the batch zip).
5. Resume in the batch CLI is keyed on **manifest row index**, not on a key
   field — see `docs/asr/usage.md`.

## Two inference paths

| path | when |
|---|---|
| `IndicASREngine.transcribe_batch()` (`--backend generate`) | **default.** The gate-verified path — the parity numbers above are its numbers. Use it for anything you intend to publish. Also the only path with long-form chunking. |
| `IndicTranscribeEngine` (`--backend engine`) | bulk throughput. ≈2.1× faster than a *tuned* fixed batch (40.8 s vs 86.4 s on a 1500-row, 14-audio-hour shard) because it evicts and refills decoder slots instead of waiting for a batch's longest member. |

The engine passes an id-parity test against stock `generate()` and a
full-shard WER gate (ΔWER −0.132% vs production), but `logits_processor` and
long-audio (≥78 s) regimes lack dedicated gates — hence `generate` remaining
the default.

## Not vendored

Deliberately left in the source port, with reasons:

- **Checkpoint conversion** (`convert_from_nemo.py`) — one-time tooling.
- **Scoring and benchmarking** (`score_jiwer.py`, `compare_shard.py`,
  `bench*.py`, `nativize*.py`) — this repo keeps model-quality metrics outside
  the inference library, the same decision made for the TTS port. The scoring
  pitfalls those scripts address are documented in `docs/asr/caveats.md`,
  because they change WER far more than the model does.
- **Beam search** (`generate(num_beams>1)`) — cache reordering is untested.
