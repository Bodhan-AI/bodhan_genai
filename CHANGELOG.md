# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`bodhan_genai.ocr` — the OCR modality (IndicOCR).** Block-level document parsing for
  English and the 22 Eighth-Schedule Indian languages: page image in, reading-ordered Markdown and
  per-block JSON out. Two models with a plain-JSON handoff — **IndicDocLayout** (our PP-DocLayoutV3
  finetune with an integrated reading-order head, ~33M, torch) and **IndicBlockOCR** (Qwen3.5-0.8B
  on the Sarvam tokenizer, ~0.8B, vLLM). Lands as a sibling of `.tts` and `.mt`, with no
  import-path churn. Inference only — no data, training, eval or serving stages yet.
- `bodhan_genai.ocr.templates.contract` — the **contract**: the three prompts, the IndicDocLayout
  label to pipeline-type map, the label sets gating cropping and OCR, and the output schema. A test
  asserts `KEPT_BLOCK_TYPES` is exactly the reachable types less `DROP_TYPES`, so a dead or
  undocumented type cannot creep in.
- `bodhan_genai.ocr.engine` — PUBLIC API: `IndicOCR`, `IndicDocLayout`, `IndicBlockOCR` over
  `LayoutBackend` / `RecognizerBackend` Protocols, with `IndicDocLayoutBackend`, `JsonLayoutBackend`
  (torch-free, replays any layout of the documented schema) and `VllmRecognizer`. Plus the layout
  cleanup rules, crop conditioning, markdown assembly and the config dataclasses; all lazily
  exported from `bodhan_genai.ocr`.
- `bodhan_genai.ocr.inference` — `python -m bodhan_genai.ocr.inference.cli {layout,ocr,parse,show-contract}`,
  with YAML-as-argparse-defaults and suffix-form output filenames (`<name>.layout.json`).
- Modality-scoped extras `ocr-infer` and `ocr-serve`, aggregated as `all-ocr` and included in
  `all`. OCR installs into the one shared `./.venv`; its install is gated on `PPDocLayoutV3` being
  importable, `ninja` on PATH and CUDA actually being visible.
- `bodhan_genai.ocr.serving` — the recognizer behind stock `vllm serve`, with layout client-side.
  `HttpRecognizer` implements the same `RecognizerBackend` protocol as the in-process backends, so
  `IndicBlockOCR` runs unchanged and cropping, dedup, order matching and markdown assembly stay
  shared code. The messages it sends render to the same prompt the offline path builds, which a
  test asserts against the shipped chat template. Plus `scripts/ocr/serve.sh` and the client-side
  `ocr-serve` extra, which needs no vLLM and installs without a GPU.
- `bodhan_genai.ocr.data` — the layout data pipeline: the 37-class taxonomy and the canonical
  page parser (pure Python, no torch), per-source train/val/test manifests with `native` /
  `holdout` / `hash` policies, the packed blob cache the trainer reads, and a corpus summary that
  names classes with no examples. Class ids are baked into checkpoints, so `CLASSES` is
  append-only and a test asserts the frozen positions.
- `bodhan_genai.ocr.training` — IndicDocLayout: `PPDocLayoutV3Trainable` (transformers ships
  PP-DocLayoutV3 inference-only) with the base RT-DETR detection loss plus a locality-weighted
  GCE reading-order loss over Hungarian-matched queries, a mixed-source batch sampler that holds
  the configured ratio in *every* batch rather than in expectation, EMA with a warmed-up decay,
  and an accelerate-based entrypoint. Backbone, decoder and order head warm-start from
  PP-DocLayoutV3; only the class heads are re-initialized.
- `bodhan_genai.ocr.eval` — detection mAP (COCO-style, implemented locally rather than adding
  `ultralytics` for one function) and reading order as Kendall-tau / edit distance / pairwise
  accuracy, each against a raster baseline and on the non-raster "hard" slice. Plus an end-to-end
  olmOCR runner that defaults to the settings the published 82.9 was measured with
  (`dedup_mode="text_only"`, Markdown tables) and records them beside the predictions — the
  shipped defaults differ, and scoring with them measures a formatting mismatch.
- `configs/ocr/{data,train}/`, `scripts/ocr/{splits,pack,train,eval}.sh`, `notebooks/ocr/`,
  and the `ocr-train` / `ocr-eval` extras.
- `configs/ocr/`, `scripts/ocr/`, `examples/ocr/`, `docs/ocr/`, `tests/ocr/`.

- **`bodhan_genai.mt` — the MT modality (BodhanMT).** Translation between English and 22
  Eighth-Schedule Indian languages (25 language-script combinations, 44 directions) on a
  Gemma-4-E4B `Gemma4ForConditionalGeneration` backbone. Lands as a sibling subpackage of
  `bodhan_genai.tts`, with no import-path churn for existing code.
- `bodhan_genai.mt.templates` — the frozen **prompt contract** (`prompt.py`): target language only,
  never the source, exactly one user turn. Plus `variants.py` (12 instruction phrasings for
  rendering training corpora, index 0 = the served instruction) and `trl_chat.py`
  (`GEMMA4_TRL_TEMPLATE`, the training template carrying `{% generation %}` loss markers, verified
  to render byte- and token-identically to the checkpoint's shipped template).
- `bodhan_genai.mt.engine` — public Python API: `BodhanMTEngine` over a `TranslationBackend`
  Protocol with `vllm` (throughput) and `hf` (reference / unmerged PEFT adapter) backends, sharing
  `MTSamplingConfig` (greedy defaults) and `MTResult`; lazily exported from `bodhan_genai.mt`.
- `bodhan_genai.mt.inference` — `python -m bodhan_genai.mt.inference.cli {vllm,hf}` dispatcher,
  single-segment / batch / document / interactive modes, JSONL output, YAML-as-argparse-defaults.
- `bodhan_genai.mt.data` — `render` (bitext JSONL → instruction chat rows: seeded phrasing variety,
  optional reverse direction, global dedup on the raw directed pair, train/dev split) and `dataset`
  (rendered-chat length filter, file-locked tokenizer cache).
- `bodhan_genai.mt.training` — **provisional** single-stage 8k-context LoRA finetuning on TRL
  `SFTTrainer` + PEFT with an assistant-only loss mask, plus `merge` (adapter → standalone
  servable checkpoint). Deliberately isolated: nothing outside the subpackage imports it, so the
  trainer can be replaced without touching the rest of the package.
- `bodhan_genai.mt.serving` — `MTClient`, a typed client over stock `vllm serve` that owns the
  prompt contract. No custom server: `scripts/mt/serve.sh` wraps unmodified vLLM.
- `bodhan_genai.mt.eval` — IN22 score replication (BLEU + chrF++ with `word_order=2`), with pooled
  and macro aggregates reported separately.
- `bodhan_genai.mt.tools.vllm_ready` — adds the KV-shared `k_norm` sidecar (18 zeroed tensors,
  per-layer-type sized) that stock vLLM requires, so a self-trained checkpoint loads unpatched.
- Modality-scoped extras `mt-data` / `mt-train` / `mt-infer` / `mt-serve` / `mt-eval` and `all-mt`,
  a separate `constraints.txt` lock, and `install.sh --modality {tts,mt}` building `./.venv`
  with the cu129 → vLLM → package install order and Gemma-4-registration + CUDA-visibility gates.
- `configs/mt/`, `scripts/mt/`, `examples/mt/`, `docs/mt/`, `tests/mt/`.

- `docker/mt/Dockerfile.serve` and `scripts/mt/serve_docker.sh` — containerized MT serving, stock
  `vllm serve` with the validated flags. The entrypoint runs the launcher with `--foreground`
  because it otherwise backgrounds vLLM and exits at readiness, which would kill the container the
  moment it became useful.
- `notebooks/mt/inference.ipynb` and `notebooks/mt/training.ipynb` — walkthroughs mirroring the TTS
  pair. Training covers the `vllm_ready` step explicitly rather than hiding it inside `merge.sh`.
- `src/bodhan_genai/tts/README.md` and `src/bodhan_genai/mt/README.md` — per-package technical
  documentation.
- `install.sh --extras NAME` for a lean single-modality install, and `--no-flash-attn` to skip the
  flash-attn build on a box that only serves.
- **Documentation site** (`mkdocs.yml`, `docs/`, `.github/workflows/docs.yml`) — MkDocs Material
  with a generated API reference, built on every PR with `--strict` so a broken link fails review,
  and attached to every run as a downloadable artifact. Publishing to GitHub Pages is wired up but
  gated on an `ENABLE_PAGES` repository variable: Pages is unavailable on this repository's plan,
  and a deploy job that fails on every push is worse than none. MkDocs rather than Sphinx autodoc because
  mkdocstrings reads the source statically: ten modules here import torch / vllm / ray at module
  level, so an import-based build would need the CUDA stack on the docs runner or a mock list that
  silently rots. `pip install -e '.[docs]' && mkdocs serve` to preview.

### Changed

- **Tables are transcribed as HTML by default, not Markdown.** `colspan`, `rowspan` and in-cell
  line breaks have no GitHub-flavored-Markdown spelling, so a merged-cell table rendered as GFM
  silently loses its structure. `TableFormat.MARKDOWN` / `--table-format markdown` remains
  available for flat tables. The previous model card promised Markdown and was wrong.
- **TTS and MT now share one environment.** They were split on the belief that they could not
  co-install: TTS was pinned to `transformers 5.5.3` / `vllm 0.19.0`, below the Gemma 4 floor
  (`>=5.12` / `>=0.20`) MT needs. Those pins were a snapshot of the environment TTS was ported
  from, not a constraint — the TTS code only touches long-stable APIs
  (`AutoModelForCausalLM`, `Trainer`, `TrainingArguments`, `LLM`/`AsyncLLM`/`SamplingParams`) and
  already carried a version-tolerant `AsyncLLM` import. Verified by resolving the shared set and
  running the whole suite against it: 687 passed, 0 skipped. Consequences:
  - one venv (`./.venv`), one lock (`constraints.txt`); `constraints-mt.txt` and
    `requirements-ci-mt.txt` are removed, and `install.sh --modality` is obsolete (accepted with a
    notice so existing scripts do not hard-fail).
  - the whole repo moves to `torch 2.11.0` (cu129), `transformers 5.13.1`, `vllm 0.20.2`,
    `ray 2.57.0`; TTS previously sat on `torch 2.10` / cu128 / `ray 2.55.1`.
  - `all` now means every modality; `all-tts` and `all-mt` remain for lean installs.
  - CI runs one job over the whole suite instead of one per modality.
- **OCR joins that same environment.** It landed on a separate `./.venv-ocr` with its own
  `constraints-ocr.txt`, because its recognizer needs `vllm >= 0.26` while the shared environment
  pinned `0.20.2`. The only real conflict was `tts-serve`'s exact `fastapi==0.141.1` against vLLM
  0.26's `fastapi[standard]>=0.133,<0.137`; relaxing that pin to the range resolves all three
  modalities together. Consequences:
  - `constraints-ocr.txt` is removed and `install.sh --modality ocr` is obsolete (accepted with a
    notice, like `tts`/`mt`, so existing scripts do not hard-fail); `all` now includes `all-ocr`.
  - the whole repo moves to `vllm 0.26.0`; `torch 2.11.0` (cu129) and `transformers 5.13.1` are
    unchanged.
  - the install order **reverses** for every modality: vLLM first from its per-CUDA index, letting
    it pull its own matched torch, then torchaudio/torchvision. Pre-pinning torch leaves a
    half-installed torch behind on 0.26.
  - `docker/tts/Dockerfile.serve` and `docker/mt/Dockerfile.serve` follow that order and version,
    and both move off `uv:0.6`, which cannot fetch the CPython `.python-version` pins.
  - TTS and MT now set `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM, as OCR already did.
    vLLM 0.26 samples through a flashinfer kernel it JIT-compiles at engine warm-up, which needs
    `nvcc`; nodes with a runtime-only CUDA install have none, and the build failure surfaces as a
    bare "EngineCore failed to start" that names neither flashinfer nor the missing compiler.
    Export `VLLM_USE_FLASHINFER_SAMPLER=1` to opt back in where a toolkit is present.

### Verified

- The collapse was checked on a real H100, not only resolved: the whole suite (812 passed,
  2 skipped), all 86 TTS/MT/OCR modules imported under vLLM 0.26, `./install.sh` end to end, and
  MT re-scored against the repo's own anchor — IN22-Gen `eng_Latn -> snd_Deva`, 1024 segments,
  0 empty, **chrF++ 36.66 against the recorded 36.57**. The OCR recognizer was benchmarked on the
  same GPU at 32 pages, 0 empty blocks.
- **flash-attn is no longer pinned to a prebuilt wheel.** A pinned wheel is exact to one
  (flash-attn, torch, CUDA, cpython, arch) tuple and goes stale silently the moment any of the four
  moves — which is exactly what the torch 2.11 bump would have done. `install.sh` now builds it
  from source, last, and treats a failure as a warning: only TTS *training* needs it.
- Package description and keywords now cover both modalities. README restructured — the root README
  is a high-level overview and the technical detail moved into the per-package READMEs.

### Added — ASR


- **ASR: selectable output modes — `itn` (mixed-script) and `romanized`.** The
  `itn_romanized_posttrain` checkpoint line is trained for three output modes
  selected by canary2 prompt slots 6/7; the prompt was previously frozen to
  native script, leaving two of the three unreachable. `itn=`/`romanized=`
  flags (scalar or per-row) now thread through `encode_prompt`,
  `transcribe_batch`/`transcribe_long`, the continuous-batching engine, the
  streaming slot pool, the serving protocol/endpoints, the reference client
  (`--itn`/`--romanized`), and the offline CLI. Prompts stay exactly 10 tokens
  in every mode, so batching is unchanged and mixed-mode batches work.
  Defaults preserve prior behaviour exactly (verified output-identical).
  `PROMPT_LANGS` warm cache extended to the full language list (incl.
  `bgc`/`hne`) with graceful skip for older tokenizers. This supersedes the
  "frozen 10-token canary2 prompt" wording in [0.1.0] below.

### Changed — ASR

- **ASR joins the one environment.** The ASR line was developed against
  `transformers 5.5.3` / `vllm 0.19.0` / `ray 2.55.1` / `torch 2.10.0` on cu128 — a snapshot of the
  pre-collapse environment, not a requirement. It now follows the shared set
  (`torch 2.11.0+cu129`, `transformers 5.13.1`, `ray 2.57.0`). ASR imports **no vLLM**, so unlike
  TTS/MT/OCR it places no constraint on the vLLM line at all; the merge is one-directional.
  Extras `asr-infer` / `asr-serve`, aggregated as `all-asr` and folded into `all`.
- **The published LID accuracy was agreement, not accuracy.** `docs/asr/caveats.md` and
  `docs/asr/usage.md` quoted "96.9% top-1 agreement with the NeMo detector" — true, and routinely
  read as accuracy. Measured top-1 is **0.864** (lattice) / **0.779** (VOI) over 337k clips, and the
  spread is extreme: `ml`/`ta` 0.979 against `bho` **0.047** and `hi` **0.258**. Both docs now carry
  the measured figures and say plainly not to use LID for hi/bho/mai/ur when metadata exists.
- `requirements-ci.txt` gains `torchaudio`, `websockets` and `uvicorn`; without them seven
  `tests/asr` modules failed at **collection** rather than reporting a skip.

## [0.1.0] - 2026-07-07

Initial release: production port narrowed to the Llama-3.2-3B / SNAC (Orpheus-style) recipe.

### Added

- `bodhan_genai.tts.codec.snac` — SNAC 24 kHz encode/decode helpers (7-token frame interleave,
  duplicate-frame dedup, batched windowed decode).
- `bodhan_genai.tts.templates` — `chat.py` sequence builders (basic TTS, conversation;
  full-sequence loss) and `conversation.py` multi-turn helpers.
- `bodhan_genai.tts.data` — stage-1 GPU Ray SNAC tokenization (`python -m bodhan_genai.tts.data.tokenize`) and stage-2 sequence
  compilation (`python -m bodhan_genai.tts.data.compile`), both writing resumable sharded Parquet.
- `bodhan_genai.tts.training` — sequence-packing FSDP2 trainer (FFD sampler, static-shape collator,
  PackingTrainer, MFU callback), full-FT and LoRA entry points.
- `bodhan_genai.tts.inference` — `audio_io`, prompt building/extraction, two-phase offline vLLM batch
  inference (`python -m bodhan_genai.tts.inference.cli`).
- `bodhan_genai.tts.serving` — Ray Serve websocket streaming service (`python -m bodhan_genai.tts.serving.app`): vLLM AsyncLLM +
  in-process compiled SNAC + micro-batcher per GPU replica; example client and loadtest.
- Launchers: `scripts/train.sh`, `scripts/train_lora.sh`, `scripts/infer.sh`, `scripts/serve.sh`;
  `install.sh` with cu128/flash-attn/constraints install order and `--offline` mode.
- Configs under `configs/` (data, train, accelerate, infer) and examples
  (`examples/basic_tts.py`, `examples/streaming_client.py`).
- Docs: token layout, data pipeline, serving, config reference; CPU-only pytest suite.
- `bodhan_genai.tts.engine` — public Python engine API: `BodhanTTSEngine` (offline, vLLM/HF
  backends) and `BodhanStreamingTTSEngine` (async PCM streaming + `stream_sync`),
  sharing `SamplingConfig`/`TTSResult`; all four lazily exported from `bodhan_genai.tts`.
- Conversation synthesis on both engines: pass a chat-style `[{"speaker", "text"}]` message
  list to `synthesize_conversation` / `stream_conversation` (+ `stream_conversation_sync`) and
  the conversation chat template renders one continuous multi-speaker sample
  (`templates.conversation.format_messages`, `prompts.build_conversation_prompt_ids`).
- Walkthrough notebooks: `notebooks/inference.ipynb` (offline / batch / conversation /
  HF backend / streaming) and `notebooks/training.ipynb` (manifest -> tokenize ->
  compile -> smoke train -> synthesize).

### Changed

- HF single-prompt CLI default `max_new_tokens` raised 1200 -> 2048 (unified on the shared
  `SamplingConfig` defaults).

### Added (post-release)

- Dockerized streaming server: `docker/Dockerfile.serve` (CUDA 12.8 base, uv-managed Python,
  cu128 install order, weights mounted at run time) + `scripts/serve_docker.sh` launch wrapper;
  README "Serve with Docker" section.
- Three synthesis endpoints on one server: `WS /tts` (live streaming), `WS /tts/chunked`
  (long-form chunked streaming, routing forced), `POST /tts/offline` (complete `audio/wav`
  response accumulated server-side from the same engine); client gains
  `--mode {stream,chunked,offline}`.
- Long-form chunked synthesis: `ChunkedBodhanStreamingTTS` over both engines
  (`synthesize_long` = one batched generate + trim/LUFS/gap/peak-norm combine;
  `stream_long` = prefetch pipeline with hybrid volume treatment) plus
  `engine.loudness` utilities (`peak_normalize`, `normalize_loudness`,
  `trim_silence`, causal `StreamingLoudnessNormalizer`); serving takes
  `{"chunked": true}` per request (`--chunked_default` + `--chunk_*` server
  knobs, `--chunked` on the client).
- Sentence-terminator segmentation: `split_sentences` (exported from `bodhan_genai.tts`) — a
  rule-based scanner over maximal terminator runs (`. ! ? … । ॥ 。 ！ ？`) with
  abbreviation / initial / decimal / ellipsis / quote-attribution guards for Latin, Devanagari
  and CJK text; `chunk_text` now builds on it. **Behavior change:** clause marks
  (`, ; :`, `，；：、`) no longer split text into chunks — they serve only as the first rung of
  the oversize-sentence fallback ladder — and ellipses (`…`, `...`, `. . .`) never split.
- Chunk scheduling policy: ramped schedules (`first_chunk_chars`; served default
  `--chunk_first_chars 120`) for low time-to-first-audio with large steady-state chunks;
  script-aware `estimate_speech_seconds` (exported) and duration budgets
  (`max_chunk_seconds` / `first_chunk_seconds`) on `ChunkedBodhanStreamingTTS`;
  `last_stream_stats` exposes exact/live path counts per stream.
- Release-qualification harness `bodhan_genai.tts.bench`: 40-case golden suite, deterministic
  deployment checks (generation degeneracy, clipping/silence/DC, chunk level spread, seam
  discontinuity, protocol conformance, latency), thresholds-gated reports with baseline
  regression diffs, engine/server runners and a load gate; `docs/release.md` documents the
  five-gate release ritual. No model-based metrics by design.
- Dialogue turn-chunking: `plan_dialogue_chunks` (exported) plans a `[{"speaker", "text"}]`
  message list into chunks whose serialized conversation form fits `max_chars`, exploding turns
  longer than `long_turn_chars` (default: `max_chars`) at sentence boundaries into same-speaker
  segments (`long_turn_chars > max_chars` = keep-turns-intact escape hatch).
- Retuned streaming defaults: `ServeConfig.frames_per_message` 8 -> 2 and
  `ServeConfig.max_ongoing_requests` 256 -> 16 (per-replica), based on internal capacity
  measurement.

### Changed (post-release)

- Repo-level reorganization ahead of adding sibling modalities (ASR, MT, OCR): every top-level
  directory that was TTS-specific (`configs/`, `docker/`, `docs/`, `examples/`, `notebooks/`,
  `scripts/`, `tests/`) is now nested under a `tts/` subdirectory, so future modalities land as
  siblings (`configs/asr/`, `tests/mt/`, ...) instead of colliding in one flat namespace. No
  change to the Python package itself — `bodhan_genai` was already a bare namespace with nothing
  but `.tts` under it, so this needed no import-path changes. `pyproject.toml`'s optional-dependency
  extras are now modality-scoped (`data`/`train`/`infer`/`serve` -> `tts-data`/`tts-train`/
  `tts-infer`/`tts-serve`) for the same reason — a breaking change to install commands
  (`pip install bodhan-genai[serve]` -> `[tts-serve]`), made now while the package is pre-1.0/alpha
  and before a second modality would otherwise collide on the same extra names.

### Removed

- Voice-cloning support removed by design: the voice-clone chat template, reference-audio
  prompting (`ref_wav` / `ref_tokens` on both engines, `--ref_wav` in the HF CLI,
  `encode_reference`), and the data-pipeline pairing machinery (`prompt_token_ids` /
  `audio_prompt` columns, `speaker_prompt_sampling`), plus `examples/voice_clone.py`.

### Notes — deliberately dropped from the source repo

- **Slurm launchers** → replaced by single-node `accelerate` shell scripts; production runs target
  dedicated nodes, not a shared Slurm queue.
- **Curriculum sampler** — research-only feature; flat dataset ratio mixing covers production runs.
- **Async DCP checkpointing** — added complexity for marginal wall-clock savings at 3B scale.
- **FSDP2 multi-node save workaround** — obsolete once training is single-node.
- **Gemma backbone** — the shipped recipe is Llama-3.2-3B only; multi-backbone abstraction removed.
- **eval/ASR suite** (WER/MOS/judge metrics, Slurm eval callback) — evaluation lives in a separate
  environment with its own conflicting dependency pins.
- **Tokenizer extension tooling** — the token layout is frozen; consumers use an already-extended
  tokenizer rather than generating new ones.
