# The hosted API

IndicTranslate is reachable three ways. When a Bodhan API key is present, the hosted service is
the **default** — no GPU, no checkpoint, no extras:

```bash
export BODHAN_API_KEY=sk-...        # or BODHAN_API_TOKEN
```

```python
from bodhan_genai.mt import IndicMTEngine

with IndicMTEngine() as engine:  # note: no model path
    print(engine.translate("The meeting has been postponed.", tgt_lang="kn").text)
```

Note what is missing from that call: a checkpoint. The service picks the model.

## The endpoint

An **OpenAI-compatible** surface, which is what `/v1/models` advertises:

```
GET  https://console.bodhan.ai/v1/models             -> indic-translate, indic-transcribe, ...
POST https://console.bodhan.ai/v1/chat/completions
Authorization: Bearer <key>
```

Because it takes a rendered conversation rather than flat fields, the hosted backend is
conversation-level like the vLLM and HF ones — it sits behind the same `PromptedBackend` adapter,
so the prompt contract keeps exactly one call site and the hosted path gets **full capability
parity** with running the model yourself.

!!! note "A flat `POST /translate` route existed briefly and is gone"

    An earlier route took `{sentence, lang}` and returned `{translation, lang}`. It never matched
    the published contract in `docs/api-contracts/`, and it returned **404 the same day** it was
    tested. It was also strictly weaker: translation only, bare language names (so the script
    qualifier could not be expressed), no decoding controls, and no truncation signal. This
    backend targets the OpenAI surface instead. If you find code written against `/translate`,
    it is out of date.

## How the backend is chosen

| | |
|---|---|
| 1 | an explicit `backend="api"` / `"vllm"` / `"hf"` always wins |
| 2 | else `BODHAN_API_KEY`, then `BODHAN_API_TOKEN` → the hosted API |
| 3 | else `vllm` |

Both variable names are accepted because silently falling back to vLLM when someone exported the
other spelling reads as a missing GPU, not a missing secret. An **exported-but-empty** variable
counts as absent — otherwise `BODHAN_API_KEY=` in a CI job selects the hosted path and then fails
every request, which looks like a broken endpoint.

Nothing that names a backend today changes behaviour.

```python
IndicMTEngine(ckpt, backend="vllm")  # force local
IndicMTEngine(api_key="sk-...", api_base_url="https://console.bodhan.ai/v1")
```

## Everything the local backends do

The target language, the script and the task all ride in the prompt, so nothing is lost:

```python
engine.translate(text, tgt_lang="hin_Deva")  # native script
engine.translate(text, tgt_lang="hin_Deva", script="roman")  # Latin letters
engine.translate(text, tgt_lang="hin_Deva", script="code_mixed")  # English mixed in
engine.transliterate(text, lang="hin_Deva", script="roman")  # script only
engine.transliterate(text, lang="hin_Deva", script="native")  # and back
engine.translate(text, tgt_lang="kas_Deva")  # qualifier preserved
```

Verified against the live endpoint for native, Roman, transliteration, code-mix, and the
script-qualified names.

`tgt_lang` accepts everything the rest of the package accepts — FLORES code, short code, bare
name, qualified name — resolved by the same `resolve_target_name`, so the served and local paths
cannot drift. An unsupported language raises **before** a request is sent and is never mapped to a
neighbour: a wrong-language translation is fluent and undetectable downstream.

## Decoding and truncation

The request carries `temperature 0.0`, `max_tokens`, and `stop: ["<turn|>"]`; `top_p` and `seed`
are sent only when you ask for sampling, so a greedy request stays honest.

The response's `finish_reason` distinguishes `"stop"` from `"length"`, so truncation **is**
detectable here:

```python
r = engine.translate(long_text, tgt_lang="hin_Deva", max_new_tokens=2048)
if r.truncated:
    ...  # the output hit the budget and stopped mid-thought
```

A truncated translation comes back **fluent and simply stops** — no error, no warning. `ok` stays
`True` because it is a real partial result that should still be scored; `complete` is the strict
view. `usage` also gives real `prompt_tokens` / `generated_tokens`.

If the truncated count does **not** move when you raise the budget, the row is degenerate — the
model looped — and a bigger budget only lets the loop run longer.

## Errors and retries

| status | retried? |
|---|---|
| `429`, `5xx` | yes, exponential backoff with jitter |
| `401` / `403` | **only if this key has already succeeded** |
| any other `4xx` | never — the request is wrong, sending it again changes nothing |

Both error-body shapes this gateway produces are parsed: OpenAI's `{"error": {message, code}}` and
FastAPI's `{"detail": ...}`. Errors carry a `request_id` when the service supplies one; quote it
when reporting a problem.

The auth rule is not paranoia: a hosted deployment of this model has returned auth errors for a
*valid* key under high parallelism, and a client that retries only on `429` records those as hard
failures. A key that has never worked is simply wrong, and retrying it burns the quota faster.

The jitter matters at concurrency. Without it, N workers that all get rate-limited back off in
lockstep and return as one thundering herd against the endpoint that just throttled them.

**Rate limit.** The gateway enforced **120 requests/min** per key during testing, and a retry
counts against it. `num_workers` defaults to 32; lower it for long runs, and note the IN22 harness
at full scale is 45,056 requests.

## CLI

```bash
export BODHAN_API_KEY=sk-...

python -m bodhan_genai.mt.inference.api_translate --tgt-lang kn \
    --text "The meeting has been postponed."

python -m bodhan_genai.mt.inference.cli api --tgt-lang hin_Deva \
    --input-file segments.txt --output-file out.jsonl --concurrency 8
```

`--concurrency` is deliberately low by default: the gateway rate-limits per key.

## Health check

`engine.health()` asks `/models` and confirms the served model is listed — not just that something
answered with a 200, so another model behind the same gateway cannot look like success.
