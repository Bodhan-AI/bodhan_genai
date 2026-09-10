# Notebooks

Guided walkthroughs, one per stack. They cover the same ground as
[`examples/`](../examples/) but at length and in order — read these when you want to understand a
pipeline, and the examples when you want a snippet to copy.

Two per modality where both apply: **inference** (what the model does) and **training** (how a
checkpoint is made). IndicTranscribe has no training notebook because it has no training stage —
it is a port, and this repo does not train it.

## Before you start

```bash
./install.sh && source .venv/bin/activate
pip install jupyterlab            # not part of the package extras
jupyter lab
```

Checkpoints resolve from the public `bodhan-ai/` Hub repos, so no credentials are needed. The
inference notebooks want one GPU; the training ones assume you have more, and are written to be
*read* on a laptop and *run* on a node.

## 🔊 IndicSpeak — `tts/`

| notebook | covers |
|---|---|
| [`inference.ipynb`](tts/inference.ipynb) | Orpheus-style TTS end to end: Llama-3.2-3B → SNAC 24 kHz codec tokens → audio, through both the vLLM and HF paths, plus long-form chunking. |
| [`training.ipynb`](tts/training.ipynb) | Raw audio → SNAC tokens (stage 1) → compiled sequences (stage 2) → FSDP2 training. |

> Needs a tokenizer already extended with the frozen audio-token layout — this repo does not
> create tokenizers. See [the token layout](../docs/tts/token_layout.md), and read it before
> touching stage 2: a mismatch produces audio-shaped noise rather than an error.

## 🌏 IndicTranslate — `mt/`

| notebook | covers |
|---|---|
| [`inference.ipynb`](mt/inference.ipynb) | Gemma-4-E4B instruction-tuned for Indic translation; the prompt contract, the vLLM and HF backends, and the served client. |
| [`training.ipynb`](mt/training.ipynb) | Bitext → instruction rows → LoRA adapter → merged, servable checkpoint. |

## 📄 IndicOCR — `ocr/`

| notebook | covers |
|---|---|
| [`inference.ipynb`](ocr/inference.ipynb) | Page image in, reading-ordered Markdown and per-block JSON out — the two stages with a plain-transformers recognizer, so it runs without vLLM. |
| [`training.ipynb`](ocr/training.ipynb) | Training the **layout** model: detection and reading order learned jointly. The recognizer ships as a released checkpoint and is not trained here. |

## 🎙️ IndicTranscribe — `asr/`

| notebook | covers |
|---|---|
| [`inference.ipynb`](asr/inference.ipynb) | Batch transcription, long-form chunking, the three output modes, and language identification from the same forward pass. |

> The notebook shows LID, and also why not to lean on it: the widely-quoted 96.9% is *agreement
> with NeMo*, not accuracy. Measured top-1 is 0.864 / 0.779, and as low as **0.047 for `bho`**.
> [The caveats](../docs/asr/caveats.md) have the per-language table.

## A note on output

Cells are committed **without** output, so the notebooks stay small and diff cleanly and there
are no stale numbers pretending to be current. The consequence is that reading one on GitHub
shows you the code and the prose but no results — you have to run it to see anything.

## Related

- [`examples/`](../examples/) — the same operations as short standalone scripts
- [`docs/<modality>/end-to-end.md`](../docs/) — the command-line path, in order
- [`docs/troubleshooting.md`](../docs/troubleshooting.md) — when the environment is the problem
