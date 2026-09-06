# IndicTranscribe: caveats, recommended settings, scoring pitfalls

Everything here is **measured**, not inferred, and carried over from the
validation done before this port was vendored. Read it before trusting any WER
number: the scoring pitfalls in §3 move WER further than the model does — raw
34.58% → 19.49% on one corpus once code-mixed script and Indic punctuation are
handled correctly.

---

## 1. Recommended inference settings

| setting | value | why |
|---|---|---|
| dtype | **bf16** | WER-neutral vs fp32: −0.10% on 512 TTS utts, −0.12% on 352 real recordings, 0.00% on sub-1 s clips |
| chunking | **`--chunk-above 45`** | measured knee: quality is flat to 45 s and collapses past 60 s. Chunking *below* 45 s is neutral-to-harmful |
| chunk window | **`--chunk-min 15 --chunk-max 25`** | best WER in the sweep; quality is flat across 10–30 s, so 10–15 s is a valid trade (~20% faster) |
| file batch | **`--batch-size 96`** | swept on a full shard: bs 24/48/96/160 = 170/109/84/90 s, so 96 is the knee. bs=24 halves throughput |
| chunk batch | `--chunk-batch-size 64` | chunks are ≤25 s and cost ~0.05 GiB each, so this can far exceed the file batch |
| **CPUs** | **4–8 cores per GPU** | with 2 CPUs for 8 GPUs, per-shard encode time went 18 s → 437 s and per-GPU throughput fell 2.9× |

### How long can it transcribe before quality drops?

Whole-file decoding, synthetic files with exact references:

| audio length | WER | **CER** | chunked WER |
|---|---|---|---|
| 15 s | 30.15% | **4.66%** | 32.50% |
| 25 s | 31.93% | **5.06%** | 31.65% |
| 35 s | 31.80% | **5.35%** | 30.53% |
| 45 s | 30.69% | **7.33%** | 30.74% |
| 60 s | 37.66% | **16.46%** | 30.71% |
| 90 s | 64.04% | **41.59%** | 30.03% |
| 120 s | 73.38% | **56.16%** | 29.22% |

**Flat to ~45 s; knee between 45 and 60 s; collapse past 60 s.** CER is the
sharper indicator. Chunked decoding is flat at 29–32% at every length, so
duration is the only variable.

This is why `chunk_above` defaults to 45 s and not 30 s: at 15 s chunking is
*worse* than whole-file (32.50% vs 30.15%) because it splits a file that did not
need splitting. Chunking only earns its keep past ~60 s, where it is worth −33
WER points on 90 s audio.

### Throughput is strongly length-dependent

| audio profile | port bf16 RTFx | speedup vs NeMo |
|---|---|---|
| 34 s (long-form eval) | 309 | 7.2× |
| 5.6 s median (real recordings) | 153 | 2.9× |
| sub-1 s | 49 | 1.9× |

Short-form audio loses most of the decode-side win — plan capacity from the
profile you actually have, not the headline number.

---

## 2. Model behaviour caveats

1. **Long audio fails without chunking.** Past ~60 s the decoder emits EOS early
   (a 221 s file produced 149 words against a 450-word reference — only ~370
   tokens, so *not* the 1024-position cap) and degrades into repetition.
2. **It is language-conditioned, and the transcription path has no LID.** The
   prompt carries `source_lang`/`target_lang` and you must supply the label. A
   wrong label produces **confidently wrong script**, not obvious garbage.
   `IndicASREngine.detect_language()` can supply a label from the same
   checkpoint, but read the next item before you rely on it.
3. **LID accuracy is 0.86 / 0.78, not 96.9%.** The 96.9% figure is *agreement
   with the NeMo detector* on 64 clips — the two implementations agree closely
   and are wrong together on exactly the pairs that matter. Measured top-1
   accuracy on the labelled benchmarks (337k clips) is **0.864** (lattice) and
   **0.779** (VOI), and the average hides a very uneven spread:

   | strong | | weak — a close neighbour eats them | |
   | --- | --- | --- | --- |
   | `ml` | 0.979 | `bho` | **0.047** |
   | `ta` | 0.979 | `hi` | **0.258** (VOI) / 0.428 (lattice) |
   | `kn` | 0.967 | `mai` | 0.356 |
   | `bn` | 0.964 | `ur` | 0.490 |

   **Do not use LID for hi/bho/mai/ur if you have any metadata at all.** A wrong
   label yields confidently wrong *script*, not visible errors. There is also no
   useful confidence threshold: filtering raises accuracy on what survives
   (0.779 → 0.836 at p≥0.7) but coverage falls faster, so whole-corpus accuracy
   only drops. See `bodhan_genai.asr.engine.lid` for the full analysis.
4. **`<unk>` appears in output** — roughly 1 per utterance on math-heavy text.
   NeMo emits essentially the same count on the same rows, so this is model
   behaviour, not a port artifact. It normalizes to the token `unk` and counts
   as a substitution when scored.
5. **Prompt layout is fixed; the two output-mode slots are selectable** — see
   `docs/asr/model.md`. Default is native script, number-words, punctuation on;
   `itn=True` / `romanized=True` switch to mixed-script/ITN or Latin output
   (functionally verified, not yet WER-evaluated).
6. **The encoder is not batch-composition invariant.** NeMo's ConvSubsampling is
   unmasked and `torch.stft(center=True)` reflects the batch zero-pad tail, so
   the last ~2 frames of a padded row depend on its batch-mates. Parity tests
   must freeze batch composition; expect ~1 near-tie token flip per 64 long
   utterances across processes.
7. **Encoder memory is quadratic in duration** but with a small constant: 0.030
   GiB/item at 15 s, 0.049 at 25 s, 0.117 at 60 s, 0.75 at 221 s. The real
   ceiling was torch's **32-bit Conv2d indexing** (`B × 256 × mel_frames/2 × 64
   < 2^31`, i.e. `B_max ≈ 2621 / seconds`); `IndicTranscribePreEncode._conv_split`
   splits the batch for that conv exactly as NeMo does, and the split is
   bit-exact because convolution is per-item independent.
8. **Urdu has no Indic normalizer** in the usual scoring stack, so Urdu numbers
   are not strictly comparable with the other languages.

---

## 3. Scoring pitfalls — these dominate WER more than the model does

Measured on one corpus: raw WER 34.58% → **19.49%** after fixing the first two.
CER 14.47% → **3.19%**.

1. **Code-mixed Latin script is the single biggest artifact.** References keep
   English and mathematics in Latin (`three x plus two y equals seven`) while
   decoding with `<|noromanized|>` writes them in native script. The audio is
   correct and every token still counts as a substitution: **75.3% of all
   substitutions**. References with no Latin at all scored 8.64%; code-mixed
   ones scored 24.50%.
   *Fix*: transliterate the reference's Latin token list per language (the
   corpus had only 267 distinct Latin tokens) and substitute deterministically.
   Do **not** LLM-rewrite whole texts — every call is a chance to paraphrase or
   drop content.
2. **Indic punctuation must be stripped by Unicode category.** A regex like
   `[^\wऀ-෿฀-๿]+` *keeps* U+0964 DEVANAGARI DANDA because it falls inside those
   ranges — worth ~1.5% WER on its own. Strip categories `P*`/`S*`/`C*` and
   **keep `M*` combining marks**: Indic vowel signs and viramas are marks, and
   removing them destroys the words.
3. **Never LLM-normalize the hypothesis, only the reference.** Hypotheses are
   ASR output; passing them through an LLM lets it repair genuine recognition
   errors and makes the WER optimistic.
4. **Beware scorers that dedupe by key.** One production scorer silently scored
   48,000 of 84,000 rows because its key omitted the speaker directory, so two
   speakers of the same sentence collided — and it dropped very different
   fractions per system, making the comparison table meaningless. This is the
   same hazard the batch CLI's row-indexed resume guards against.
5. **CER is computed on whitespace-stripped text** in the reference tooling, not
   jiwer's default. Expect a WER/CER ratio around 1.6×.

---

## 4. What is NOT verified

- Beam search (`generate(num_beams>1)`) — cache reordering is untested.
- WER/quality of `<|romanized|>` and `<|itn|>` decoding. Both modes are now
  reachable (`itn=`/`romanized=` flags) and functionally verified — correct
  prompt ids, script-correct outputs, generate/slot-engine parity — but no
  WER evaluation has been run against mode-matched references.
- `<|timestamp|>`, `<|diarize|>` outputs (still entirely unverified).
- **The continuous-batching engine (`--backend engine`) in two regimes**: with
  a `logits_processor`, and on long audio (≥78 s, where `cap_total` hits the
  1024 clamp). It *is* id-parity tested against stock `generate()` and passes a
  full-shard WER gate (ΔWER −0.132%), so it is not unvalidated — but
  `--backend generate` stays the default and is what the parity table above
  measures. Note the engine has no long-form chunking: `--chunk-above` is a
  generate-backend feature and the CLI warns if you combine them.
- **Serving latency and capacity.** Unlike the TTS server, `asr.serving` has
  had no goodput sweep on real hardware: `max_ongoing_requests` defaults to a
  conservative guess, and the streaming decode-interval vs WER vs
  concurrent-session tradeoff is uncharacterised. See `docs/asr/serving.md`.

### Streaming is buffered, not frame-synchronous

`WS /asr/stream` re-decodes a rolling buffer and commits text that survives
consecutive decodes (LocalAgreement). It cannot emit per frame, because an AED
decoder cross-attends over the whole encoder output — NeMo enforces the same
limit by raising `NotImplementedError` for cache-aware streaming on non-CTC/
RNNT models. Latency floor is one decode interval (default 1 s), and each
interval costs a full re-decode. Details and the comparison to NeMo's
non-overlapping `" ".join()` AED chunking are in `docs/asr/serving.md`.
