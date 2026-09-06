# Evaluating IndicTranslate

Translation quality on a held-out benchmark, scored the way translation is normally scored. Run this
whenever the inference path or a checkpoint changes: a prompt-contract regression shows up here and
almost nowhere else, because the output stays fluent either way.

## Run it

Needs a server:

```bash
scripts/mt/serve.sh
```

Then, narrow first:

```bash
# smoke: one direction, one language, 32 segments
scripts/mt/eval.sh --langs hin_Deva --directions en-xx --max-samples 32

# the full run: 22 languages, both directions, 1024 segments = 45,056 requests
scripts/mt/eval.sh
```

`eval.sh` reads `vllm-serve.info` for the port the server actually bound to, so it follows a
free-port fallback automatically.

Outputs, under `<output-dir>/IN22-Gen/`:

| file | contents |
|---|---|
| `metrics.json` | per-direction BLEU/chrF++, plus pooled and macro aggregates |
| `pairwise_predictions.json` | every hypothesis, keyed by direction |
| `pairwise_references.json` | every reference, keyed by direction |

Keep the prediction files. They are what makes a bad number diagnosable rather than just bad.

## Metrics

```python
sacrebleu.corpus_bleu(preds, [refs])  # 13a tokenizer, defaults
sacrebleu.corpus_chrf(preds, [refs], word_order=2)  # word_order=2 => chrF++
```

**`word_order=2` is what makes it chrF++**, not chrF. Dropping it reports a different metric that
looks close enough to pass a casual review. A test asserts the two differ, so the check cannot go
stale.

No normalization, transliteration or detokenization anywhere — scoring runs on the raw strings.
Empty predictions are kept, not filtered: a model that fails a segment should be penalised for it,
and dropping the row quietly inflates the score.

## Pooled vs macro aggregates

`metrics.json` reports both, named explicitly, because they are different numbers and mixing them up
invents a regression that is not there:

| key | meaning |
|---|---|
| `en_xx_pooled` / `xx_en_pooled` | **headline.** Every direction concatenated into one corpus, then scored |
| `en_xx_macro` / `xx_en_macro` | unweighted mean of the per-direction scores |

BLEU aggregates n-gram counts, so pooling and then scoring is not the same as averaging per-direction
scores — expect them to differ by around a point. The macro average is the more useful diagnostic:
it is unweighted, so a regression in one low-resource direction stays visible instead of being
diluted by the rest.

Into-English and out-of-English are reported separately throughout. They are very different tasks
and one number over both hides each.

## Noise floor: ±0.06 chrF++

Re-scoring the *same* checkpoint does not give the same number. vLLM's continuous batching reorders
float reductions even at temperature 0, so identical inputs move by roughly ±0.06 chrF++ (measured on
a repeat run of one direction).

Batch size matters too: the same segment translated alone and inside a batch can differ by a
character or two, because a near-tie token flips under greedy decoding. This is expected, not a bug.

Consequences:

- Do not read anything into a difference under ~0.1.
- When **selecting between checkpoints**, treat anything inside **±0.2 chrF++** as a tie and prefer
  the earlier one. Chasing a 0.1 difference is chasing float ordering.

## Reading a bad number

A **large BLEU drop with a small chrF++ drop** almost always means a handful of degenerate
generations, not a systematic regression. BLEU is brittle to a few empty or runaway outputs; chrF++
is not.

```python
from bodhan_genai.mt.eval.metrics import is_degenerate
```

flags empty predictions and runaways (output far longer than its source — the model looped instead
of terminating). Check those before concluding anything about the model.

If chrF++ moved too, it is systematic. Most likely causes, in order:

1. **The prompt.** A source language named, a system turn added, a bare multi-script language name.
   Run `pytest tests/mt/test_prompt_contract.py tests/mt/test_trl_template_parity.py`.
2. **The checkpoint.** A merge that lost the processor, or a skipped `vllm_ready` step.
3. **Versions.** An upgrade of vllm/transformers/torch is a deliberate pin edit, and it needs
   re-measuring — see `constraints.txt`.

Absolute scores vary enormously by language. Low-resource targets score far below Hindi or Urdu; a
low number there is the expected level for that language, not a bug. Compare a direction against
*itself* over time, not against another direction.

## Language set

Eval covers 22 languages + English = 44 directions, using **one script per language**
(`mni_Mtei`, `snd_Deva`). The served set has 25 language-script combinations; the extra three are
alternate scripts the benchmark does not cover.

## Notes

- The dataset is `ai4bharat/IN22-Gen`, multi-way parallel, so both directions of every pair come
  from the same rows. It is **gated** on the Hub — `hf auth login` (or `HF_TOKEN`) first.
- `--num-workers` (default 64) only controls how full the server's queue is kept; vLLM batches
  internally.
- Greedy decoding throughout. Sampling makes scores unreproducible.
