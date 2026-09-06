# MT API reference

Everything below is generated from the source by
[mkdocstrings](https://mkdocstrings.github.io/), read statically — nothing on this page
required importing torch or vLLM to produce.

## The prompt contract

The single source of truth for the request format. Every path that talks to the model — the HF
backend, the vLLM backend, the served client, the eval harness and the training-data renderer —
goes through `build_conversation`, so no two runtimes can drift apart.

Both of its rules fail *silently*: a wrong prompt still yields fluent output, just measurably
worse. See [the prompt contract](../mt/prompt_contract.md) for the measured cost.

::: bodhan_genai.mt.templates.prompt

## Engine

::: bodhan_genai.mt.engine.offline.IndicMTEngine

::: bodhan_genai.mt.engine.types

## Serving client

::: bodhan_genai.mt.serving.client.MTClient

## Evaluation

`chrF++` is `corpus_chrf(..., word_order=2)`. Dropping `word_order` silently reports plain chrF,
which looks close enough to pass a casual review.

::: bodhan_genai.mt.eval.metrics

## Data pipeline

::: bodhan_genai.mt.data.render

::: bodhan_genai.mt.data.dataset

## Tools

::: bodhan_genai.mt.tools.vllm_ready
