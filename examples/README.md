# Examples

Runnable single-file programs, one concern each. They are the shortest path from "installed" to
"it produced something", and they are deliberately small enough to read in full before running.

For the full API and the reasoning behind the defaults, go to the package README for that
model — [IndicSpeak](../src/bodhan_genai/tts/README.md) ·
[IndicTranslate](../src/bodhan_genai/mt/README.md) ·
[IndicOCR](../src/bodhan_genai/ocr/README.md) ·
[IndicTranscribe](../src/bodhan_genai/asr/README.md).

## Before you start

```bash
./install.sh && source .venv/bin/activate
```

Every example resolves its checkpoint from the public `bodhan-ai/` Hub repos by default, so none
of them need credentials. Pass a local path to work offline. Most need a GPU; the exceptions are
noted below.

## 🔊 IndicSpeak — `tts/`

| file | what it shows |
|---|---|
| [`basic_tts.py`](tts/basic_tts.py) | The smallest useful program: one prompt, HF `generate` path, one GPU. Start here. |
| [`batch_vllm.py`](tts/batch_vllm.py) | Builds a tiny JSONL manifest and runs the offline vLLM path — the throughput route. |
| [`streaming_client.py`](tts/streaming_client.py) | Connects to a running server and writes frames as they arrive — `--mode {stream,chunked,sse,offline}`. Needs `scripts/tts/serve.sh` up in another shell, and `--auth user:password` unless it was started open. |
| [`dialogue_sample.json`](tts/dialogue_sample.json) | A multi-speaker `messages` payload, for the dialogue path rather than flat `text`. |

## 🌏 IndicTranslate — `mt/`

| file | what it shows |
|---|---|
| [`basic_translate.py`](mt/basic_translate.py) | One-shot translation with no server. The smallest useful program. |
| [`batch_vllm.py`](mt/batch_vllm.py) | A file into several languages through one resident vLLM engine — pay engine startup once. |
| [`serve_client.py`](mt/serve_client.py) | Talks to a running server over the OpenAI-compatible API. Needs `scripts/mt/serve.sh`. |

> The prompt contract is not optional. Every path here goes through `build_conversation`, and
> bypassing it fails **silently** — output stays fluent and gets measurably worse. Naming
> `"Sindhi"` instead of `"Sindhi (Devanagari script)"` measured **29.9 chrF++ worse**. See
> [the contract](../docs/mt/prompt_contract.md).

## 📄 IndicOCR — `ocr/`

| file | what it shows |
|---|---|
| [`basic_parse.py`](ocr/basic_parse.py) | One page, both stages, Markdown out. Start here. |
| [`two_stage.py`](ocr/two_stage.py) | The stages run separately, and how to substitute your own layout backend via the `LayoutBackend` protocol. |
| [`viz_layout.py`](ocr/viz_layout.py) | Draws a `*.layout.json` over its page image with per-block boxes and reading-order labels. **CPU-only** — useful for eyeballing why a page came out wrong. |

> Layout defaults to `device="cuda"`. Pass `LayoutConfig(device="cpu")` to run stage 1 without a
> GPU, which is the point of splitting it from the recognizer.

## 🎙️ IndicTranscribe — `asr/`

| file | what it shows |
|---|---|
| [`basic_asr.py`](asr/basic_asr.py) | Transcribe one or more files. Start here. |
| [`long_form_asr.py`](asr/long_form_asr.py) | A long recording with silence-aware chunking — the path for anything past ~45 s. |

> Transcription is **language-conditioned and has no LID of its own**. A wrong `--lang` yields
> confidently wrong *script*, not visible errors. If you do not know the language, detect it
> first — and read [the caveats](../docs/asr/caveats.md) before trusting the result, because
> accuracy is far below the commonly quoted figure for `hi`, `bho`, `mai` and `ur`.

## Related

- [`notebooks/`](../notebooks/) — the same ground at more length (cells ship without output; run them to see results)
- [`docs/<modality>/end-to-end.md`](../docs/) — one ordered pass per stack, data to server
- [`docs/troubleshooting.md`](../docs/troubleshooting.md) — when the environment is the problem
