# The IndicTranslate prompt contract

**Read this before touching anything that builds a request.** Every rule here fails *silently*: a
wrong prompt still returns fluent, plausible text — just measurably worse text. There is no
exception, no crash, and no warning in the logs.

The contract lives in one place, `src/bodhan_genai/mt/templates/prompt.py`, and every path goes
through it: both inference backends, the served client, the eval harness, and the training-data
renderer. Do not hand-write a `messages` payload.

## The two rules

### 1. Name only the TARGET language

The source language is never stated. The model infers it — that is what makes one call handle
English→Hindi and Hindi→English.

```
✅  Translate the following text into Hindi:

    The committee approved the proposal.

❌  Translate the following English text into Hindi:
❌  Translate from English to Hindi:
```

This is not a style preference. The model was trained on target-only instructions because its
corpus mixes both directions per language pair, so a target-only instruction is the common
denominator. Naming the source language puts the prompt off-distribution: output stays fluent, so
the regression is easy to miss and can persist for a long time.

### 2. Exactly one `user` turn, and no `system` turn

```python
[{"role": "user", "content": "Translate the following text into Hindi:\n\nHello world."}]
```

An empty or extra system turn changes the rendered prefix. So does a second user turn, or a
prior assistant turn — this model is not conversational, and there is no multi-turn mode.

## The rendered prefix

With `add_generation_prompt=True`, byte for byte:

```
<bos><|turn>user\nTranslate the following text into Hindi:\n\nHello world.<turn|>\n<|turn>model\n
```

`tests/mt/test_prompt_contract.py` freezes this against a golden fixture for all 25 languages. If
that test has to change, the model has been retrained and the published scores no longer apply.

## Languages

25 language-script combinations: 22 Eighth-Schedule languages + English, giving 44 directions.
Codes are FLORES-200 style `<iso639-3>_<script>`.

```bash
python -m bodhan_genai.mt.inference.cli hf --list-languages
```

`resolve_language()` accepts three forms, all equivalent:

| form | example |
|---|---|
| FLORES code | `hin_Deva` |
| bare name | `hindi`, `Hindi` |
| qualified name | `Manipuri (Bengali script)` |

### The script qualifier is functional, not decoration

Three languages are carried in more than one script, and **the parenthetical in the prompt is what
selects the output script**:

| language | scripts | bare name resolves to |
|---|---|---|
| Kashmiri | `kas_Arab` (Perso-Arabic) | `kas_Arab` |
| Manipuri | `mni_Mtei` (Meitei), `mni_Beng` (Bengali) | `mni_Mtei` |
| Sindhi | `snd_Deva` (Devanagari), `snd_Arab` (Perso-Arabic) | `snd_Deva` |

Measured cost of getting this wrong: prompting bare `"Sindhi"` instead of
`"Sindhi (Devanagari script)"` scored **6.67 vs 36.57 chrF++** — a 29.9-point drop. Prompting bare
`"Kashmiri"` costs about 0.5 BLEU. Both look like a bad language, not a bad prompt.

Pass the explicit code (or the fully qualified name) whenever you want the non-default script.

## Rendering mechanics

Substitution is plain `str.replace`, **never** `str.format`:

```python
TEMPLATE = "Translate the following text into {tgt}:\n\n{text}"
```

Source text containing `{`, `}`, `{0}` or `{tgt}` is common in real corpora (code, templates,
placeholders). `str.format` would raise or silently substitute; `str.replace` leaves it alone.

## Stopping

```python
STOP_STRINGS = ["<turn|>"]
```

`<turn|>` (id 106) and `<eos>` (id 1) are both in `eos_token_id`, so generation terminates without
help. Passing the stop string as well is harmless and survives a template change — every path here
does.

## Training-time templates

Two things differ when rendering a *training* corpus, and neither relaxes the rules above.

### Phrasing variants

`templates/variants.py` carries 12 paraphrases so the model sees instruction diversity rather
than one memorised string. **Index 0 is exactly the served instruction**, and the eval harness
always uses index 0 — so evaluation prompts sit inside the training distribution rather than
beside it.

A `with_source` bank exists (`"Translate the following text from {src} to {tgt}"`) for the narrow
case of a single fixed-direction finetune, where naming the source is unambiguous. It is **not**
the served contract: a model trained that way must be evaluated and served the same way, and the
default everywhere is `target_only`.

### The `{% generation %}` chat template

TRL's `assistant_only_loss=True` needs the chat template to mark which span the loss covers, which
the checkpoint's shipped `chat_template.jinja` does not do. Training therefore swaps in
`GEMMA4_TRL_TEMPLATE` (`templates/trl_chat.py`).

**That template must render byte-identically to the shipped one.** If it does not, the model is
trained on a prefix it never sees at serve time, and no loss curve will show it.
`tests/mt/test_trl_template_parity.py` asserts string *and* token equality for both the
prompt-only and full-turn renderings — treat a failure there as blocking, not cosmetic.

The specific trap is jinja whitespace. Every tag in that template uses the `{%- -%}` / `{{- -}}`
trim modifiers. Written across plain lines without them it renders

```
'\n<bos>\n<|turn>user\n…<turn|>\n\n<|turn>model\n\n'
```

instead of

```
'<bos><|turn>user\n…<turn|>\n<|turn>model\n'
```

`trim_blocks=True` (which transformers sets) does not save you: it strips the newline after
`{% %}` block tags but not after `{{ }}` expressions. A test asserts the modifiers are present on
every tag, so a tag added later cannot quietly reintroduce this.

The loss span deliberately includes the trailing `<turn|>`, so the model learns to emit its own
stop token. Excluding it trains a model that never terminates.

## Checklist

Before shipping anything that builds a request:

- [ ] It goes through `build_conversation()` / `MTClient` / `IndicMTEngine`, not a hand-built dict
- [ ] Target language named, source language absent
- [ ] One `user` turn, no `system` turn
- [ ] Multi-script languages use the qualified name or explicit code
- [ ] `pytest tests/mt/test_prompt_contract.py tests/mt/test_trl_template_parity.py` green
