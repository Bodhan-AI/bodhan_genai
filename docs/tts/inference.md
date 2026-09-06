# Inference

Every inference path — the Python engines, the batch CLI, the single-prompt CLI and the
streaming server — resolves the same two models:

| model | default | what it does |
|---|---|---|
| LM checkpoint | `bodhan-ai/indic-speak` | Llama-3.2-3B backbone that generates SNAC audio tokens |
| decoder | fine-tuned Vocos (weights from `bodhan-ai/indic-speak`, `vocos/best.pt`) | turns SNAC codes into the 24 kHz waveform |

Both repos are public: out-of-the-box inference needs no credentials
with read access to `bodhan-ai/indic-speak`. Credentials resolve exactly as they do for any
`huggingface_hub` download; a local path anywhere a model id is accepted skips the Hub
entirely.

## Entry points

| path | command / API | use it for |
|---|---|---|
| Offline engine | `IndicTTSEngine()` | load-once synthesis in Python (vLLM or HF backend) |
| Streaming engine | `IndicStreamingTTSEngine()` | async int16-PCM streaming in Python |
| Batch CLI | `python -m bodhan_genai.tts.inference.cli vllm` (`scripts/tts/infer.sh`) | manifests: two-phase vLLM generate → batched SNAC decode |
| Single-prompt CLI | `python -m bodhan_genai.tts.inference.cli hf` | one prompt through HF `generate()` — sample-quality checks, adapters |
| Server | `scripts/tts/serve.sh` | Ray Serve: `WS /tts`, `WS /tts/chunked`, `POST /tts/offline` — see [Serving](serving.md) |

With the defaults in place, the shortest path to audio is:

```python
from bodhan_genai.tts import IndicTTSEngine

engine = IndicTTSEngine()  # bodhan-ai/indic-speak, Vocos decoder
engine.synthesize("Hello world", speaker="S1").save("hello.wav")
```

`synthesize` / `synthesize_batch` also take `style=...`, which reaches the `<|style>` slot of
the prompt template (trained style conditioning).

## Decoder selection (`vocos`)

Offline decode goes through the fine-tuned Vocos decoder **by default**. Only SNAC's decoder is
replaced — its quantizer still produces the `z_q` latent Vocos consumes, and its encoder is
untouched. The knob is the same everywhere it appears:

| value | meaning |
|---|---|
| `true` (default) | fine-tuned Vocos decoder, weights fetched from the Hub |
| a path | local `vocos` `.pt` checkpoint |
| `false` | SNAC's own decoder |

- Python: `IndicTTSEngine(..., vocos=True | False | "/path/to/best.pt")`
- Batch CLI: `--vocos {true,false,/path}` (or the `vocos:` key in
  [`configs/tts/infer/offline_vllm.yaml`](configs.md))

A decoder that cannot be loaded **raises rather than falling back** to SNAC's decoder —
silently decoding with a different vocoder than the one requested would be worse than failing.

**Streaming still decodes with stock SNAC.** `IndicStreamingTTSEngine` — and therefore the
server, whose three endpoints (including `POST /tts/offline`) all run on it — uses SNAC's own
decoder; its windowed decode is CUDA-graph
compiled with a constant batch/window shape, so swapping Vocos in there is separate work. Until
then, offline and streaming output audio from different vocoders.

## Batch CLI in one breath

```bash
scripts/tts/infer.sh --jsonl-path eval.jsonl --output_dir out/
```

Defaults come from `configs/tts/infer/offline_vllm.yaml` via `--config`; explicit flags
override config values, and unknown config keys fail loudly. Only `--jsonl-path` and
`--output_dir` are required. Phase 1 generates audio token ids for every prompt with vLLM
workers; phase 2 batch-decodes the SNAC windows to 24 kHz WAVs, so vLLM gets the whole GPU
during generation.

## Single-prompt CLI

```bash
python -m bodhan_genai.tts.inference.cli hf --text "Hello world" --speaker S1
```

`--model` defaults to `bodhan-ai/indic-speak`; `--adapter_dir` attaches a PEFT/LoRA adapter
(the vLLM paths need adapters merged into the checkpoint instead).

## API reference

Engine and codec signatures, including the Vocos classes, are in the
[TTS API reference](../reference/tts.md).
