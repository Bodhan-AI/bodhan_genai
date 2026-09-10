# Bodhan GenAI

Model tooling for Bodhan's generative models — data pipelines, training, inference and serving,
one modality per subpackage.

<div class="grid cards" markdown>

-   **IndicSpeak** — `bodhan_genai.tts`

    ---

    Orpheus-style LLM speech synthesis: a **Llama-3.2-3B** backbone emitting
    [SNAC 24 kHz](https://github.com/hubertsiuzdak/snac) codec tokens.

    [Overview](tts/index.md) · [API reference](reference/tts.md)

-   **IndicTranslate** — `bodhan_genai.mt`

    ---

    A **Gemma-4-E4B** instruction-tuned for English ⇄ 22 Eighth-Schedule Indian languages,
    44 directions.

    [Overview](mt/index.md) · [API reference](reference/mt.md)

-   **IndicOCR** — `bodhan_genai.ocr`

    ---

    Block-level document parsing for English and 22 Indian languages. Page image in,
    reading-ordered Markdown and per-block JSON out.

    [Overview](ocr/index.md) · [API reference](reference/ocr.md)

-   **IndicTranscribe** — `bodhan_genai.asr`

    ---

    Speech recognition for English and Indic languages, with three selectable output modes
    from one checkpoint: native script, mixed script, or romanised.

    [Overview](asr/index.md) · [API reference](reference/asr.md)

</div>

## Install

One environment covers every modality:

```bash
./install.sh && source .venv/bin/activate
```

The order the installer encodes is load-bearing, and so are the flags for lean, CPU-only,
air-gapped and non-default-CUDA installs. Rather than repeat them here where they would drift,
they live in one place: **the [repository README](https://github.com/Bodhan-AI/bodhan_genai#install)**.

If an install went wrong, go straight to [Troubleshooting](troubleshooting.md).

## Where things live

The four package READMEs are the canonical documentation for each *model* — what it is, its
evaluation numbers, its Python API. These pages go deeper on single *tasks*.

| you want to… | go to |
|---|---|
| synthesize or train speech | [IndicSpeak overview](tts/index.md) |
| translate, finetune or serve | [IndicTranslate overview](mt/index.md) |
| parse document pages to Markdown | [IndicOCR overview](ocr/index.md) |
| transcribe speech | [IndicTranscribe overview](asr/index.md) |
| walk a stack end to end, once, in order | [TTS](tts/end-to-end.md) · [MT](mt/end-to-end.md) · [OCR](ocr/end-to-end.md) · [ASR](asr/end-to-end.md) |
| get the MT prompt format exactly right | [prompt contract](mt/prompt_contract.md) |
| get the OCR prompts and taxonomy right | [the contract](ocr/contract.md) |
| trust an ASR WER or LID number | [ASR caveats](asr/caveats.md) |
| fix a broken environment | [Troubleshooting](troubleshooting.md) |
| look up a class or function | [TTS](reference/tts.md) · [MT](reference/mt.md) · [OCR](reference/ocr.md) · [ASR](reference/asr.md) |

!!! note "Four failure modes that do not announce themselves"

    The MT [prompt contract](mt/prompt_contract.md) fails silently — a wrong prompt still returns
    fluent text, just measurably worse. chrF++ requires `word_order=2`; plain chrF looks close
    enough to pass a casual review. And OCR's [labels and types](ocr/contract.md) are separate
    vocabularies: an unrecognised label resolves to `Text` rather than raising, so a typo quietly
    gets the prose prompt instead of the equation one. And ASR is language-conditioned with no LID
    on the transcription path — a wrong `lang` yields confidently wrong *script*, not garbage,
    and [LID itself](asr/caveats.md) is 0.86/0.78 accurate rather than the 96.9% figure that is
    agreement with NeMo. All four are documented because none of them raises.
