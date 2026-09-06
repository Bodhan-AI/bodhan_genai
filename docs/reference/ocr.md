# OCR API reference

Everything below is generated from the source by
[mkdocstrings](https://mkdocstrings.github.io/), read statically — nothing on this page
required importing torch or vLLM to produce. That is not incidental: `bodhan_genai.ocr` guarantees
that importing it pulls in no torch, vLLM, transformers or PIL, and
`tests/ocr/test_ocr_lazy_import.py` asserts it.

## The contract

The single source of truth for prompts and the block taxonomy. Two vocabularies live here and
mixing them fails *silently*: **labels** are what IndicDocLayout emits and are matched
case-sensitively, **types** are the coarse categories a label maps to and they select the prompt.
An unrecognised label resolves to `Text` by design rather than raising.

`KEPT_BLOCK_TYPES` is asserted to be exactly the reachable outputs of `map_label` less
`DROP_TYPES`, so a dead or undocumented type cannot creep in unnoticed.

::: bodhan_genai.ocr.templates.contract

## Engine

The three public entry points. `IndicOCR` constructs the recognizer **before** the layout
model on purpose: vLLM's `EngineCore` forks at construction and must initialise CUDA first, and
reversed, the child cannot re-init.

::: bodhan_genai.ocr.engine.offline

## Types and configuration

Plain data plus four frozen config dataclasses. These defaults *are* the shipped recipe — an
earlier version read them from the environment at import time, and three callers ended up running
three different pipelines without knowing it.

::: bodhan_genai.ocr.engine.types

## Layout backends

`LayoutBackend` is a runtime-checkable Protocol, which is what makes "bring your own layout" a
contract rather than a claim. `detect` must return blocks already cleaned and **densely ordered** —
`order` a gap-free 0-based rank — because stage 2 matches transcriptions back by `order`, and a gap
mis-assigns text.

::: bodhan_genai.ocr.engine.layout

## Recognizer backends

`HfRecognizer` is the quickstart path, not the working one: without continuous batching it is
orders of magnitude slower per block. Note also that the two backends diverge slightly — both
decode greedily, but different kernels give different logits and a near-tie flips the argmax.

::: bodhan_genai.ocr.engine.recognizer

## Layout cleanup

Pure geometry on `Block`s: no PIL, no torch, no page image. Keeping it that way is what lets the
rules deciding *what gets transcribed* be tested with none of the GPU stack installed.

::: bodhan_genai.ocr.engine.blocks

## Cropping

The one part that needs PIL, split out for exactly that reason. The area clamp is on **area**, not
on a side: pinning a side exploded elongated crops, turning a 122:1 rule line into ~32k image
tokens and wedging the engine.

::: bodhan_genai.ocr.engine.crops

## Reconstruction

Reading-ordered markdown, plus the math repairs. The recognizer emits Indic script inside math mode
(`$$প্রোটন = 9$$`), which KaTeX, MathJax and a real TeX run all reject; those runs are wrapped in
`\text{}`.

::: bodhan_genai.ocr.engine.reconstruct

## Serving

`HttpRecognizer` is the third implementation of the recognizer protocol, alongside the vLLM and
transformers backends. Because it satisfies the same interface, `IndicBlockOCR` runs unchanged and
served output follows the offline path rather than a parallel one.

::: bodhan_genai.ocr.serving.recognizer_http

::: bodhan_genai.ocr.serving.client.OCRClient

## Checkpoints

Resolution order is first-hit-wins: explicit path, then environment variable, then a bundled
`weights/<sub>/`, then a Hub download. Nothing is fetched until this is called, so import and
`--help` never touch the network.

::: bodhan_genai.ocr.engine.checkpoints

## Data pipeline

The 37-class taxonomy and the canonical page parser are pure Python — no torch — so the
label set can be used from a laptop, a test, or the recognizer side of the pipeline.
`labels_from_doc` is deliberately the ONE parser used by the packer, the trainer and the
eval, because a split that parses its labels differently from the split it is compared
against produces a number that means nothing.

**Class ids are baked into every checkpoint.** `CLASSES` is append-only; reordering it
silently relabels every model ever trained.

::: bodhan_genai.ocr.data.taxonomy

::: bodhan_genai.ocr.data.splits

::: bodhan_genai.ocr.data.blob

::: bodhan_genai.ocr.data.summarize

## Training

Only the layout model is trained here — see [Data & training](../ocr/training.md).
Nothing outside this subpackage imports from it, so the trainer can be replaced without
touching inference or serving.

::: bodhan_genai.ocr.training.order_loss

::: bodhan_genai.ocr.training.modeling

::: bodhan_genai.ocr.training.dataset

::: bodhan_genai.ocr.training.ema

::: bodhan_genai.ocr.training.config

## Evaluation

Two different questions kept apart: whether the detector finds and orders blocks, and
whether the whole pipeline turns a page into the right Markdown. AP is implemented
locally rather than pulled from `ultralytics` or `pycocotools` — both would be a
heavyweight dependency for one function, and they differ from each other in the details.

::: bodhan_genai.ocr.eval.metrics

::: bodhan_genai.ocr.eval.layout

::: bodhan_genai.ocr.eval.olmocr
