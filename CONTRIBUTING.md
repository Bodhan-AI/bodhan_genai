# Contributing

## Dev setup

```bash
# CPU-only dev box (unit tests, docs, linting):
./install.sh --cpu

# GPU box (full stack incl. flash-attn + vllm):
./install.sh

# Hooks
pre-commit install
```

## Repo layout

Every repo-level directory is namespaced by modality — `tts/`, `mt/`, `ocr/` and `asr/`.
`bodhan_genai` is a bare PEP 420 namespace, so each modality is a sibling subpackage with no
import-path churn for the others.

```
bodhan_genai/
├── src/bodhan_genai/
│   ├── tts/                     ← package README + engine, codec, data, training, serving
│   ├── mt/                      ← package README + prompt contract, engine, training, eval
│   ├── ocr/                     ← package README + contract, engine, layout, inference, eval
│   └── asr/                     ← package README + engine, model, inference, serving
├── configs/{tts,mt,ocr}/        data / train / accelerate / infer YAMLs
├── scripts/{tts,mt,ocr,asr}/    *.sh launchers (train, infer, serve, eval, render, merge)
├── docker/{tts,mt,ocr}/         Dockerfile.serve, and Dockerfile.parse for OCR
├── notebooks/{tts,mt,ocr,asr}/  inference.ipynb, training.ipynb walkthroughs (+ README)
├── examples/{tts,mt,ocr,asr}/   runnable single-file examples (+ README)
├── docs/{tts,mt,ocr,asr}/       per-modality reference docs
└── tests/{tts,mt,ocr,asr}/      CPU-only pytest suite
```

Three boundaries are enforced by tests, not by convention:

- No modality package imports another.
- Nothing outside `bodhan_genai.mt.training` imports from it.
- Importing `bodhan_genai.ocr` pulls in no torch, vLLM, transformers or PIL —
  `tests/ocr/test_ocr_lazy_import.py`.

## Documentation layers

Each fact has exactly one home. Lower layers link up; they never restate.

| layer | holds |
|---|---|
| `README.md` | The front door, and the **only** copy of the install instructions. |
| `src/bodhan_genai/<mod>/README.md` | The canonical doc for that model: what it is, its **evaluation results**, its API, its quickstart, its own troubleshooting. |
| `docs/<mod>/*.md` | Task deep-dives — end-to-end, data pipeline, training, serving, configs. **How to reproduce** results lives here; the numbers live in the package README. |
| `docs/troubleshooting.md` | Failure modes shared across modalities. |

If you find yourself copying a paragraph between these, link instead.

## End-to-end check

The pytest suite is CPU-only and offline, so **nothing in it proves a checkpoint loads**. Before
releasing, run the four published models for real:

```bash
HF_HOME=/big/disk python scripts/e2e_public_models.py
```

Needs network and pulls ~30 GB. It takes the transformers path so it runs without a GPU, which
means it does *not* exercise the vLLM backends — treat green as "weights load and the plumbing
is connected", not as a parity or performance check.

## Serving conventions

All four launchers (`scripts/{tts,mt,ocr,asr}/serve.sh`) share one contract. Add to it in all
four, or not at all — the bugs found here were all "one modality has it, the others don't".

| every launcher must | why |
|---|---|
| search for a free port near `PORT` | shared clusters squat ports; Ray reports a collision as a bind error buried in a worker-node log |
| write `PID` / `PORT` to a per-modality `INFO_FILE` | with a port search, the caller cannot otherwise tell where the server landed. Never a shared filename — two servers on one box would clobber each other |
| announce the resolved port on stdout | |
| **start with no arguments** | these launchers are a reference setup to copy and extend. Anything a server cannot run without — a credential, a token — belongs behind an opt-in, not in front of `--help` |
| ignore checkpoint env vars unless `BODHAN_GENAI_DEPLOYMENT=1` | an inherited `CHECKPOINT` / `MODEL_DIR` silently serving different weights is a correctness bug that presents as a model regression. The marker is set only by `docker/*/Dockerfile*` |

Two differences are by design, not oversight:

- **Ray Serve (`tts`, `asr`) runs in the foreground**, so it cannot gate on readiness after
  launching — poll `/health`. **Stock `vllm serve` (`mt`, `ocr`) backgrounds itself** and gates
  on `/v1/models` advertising its own `SERVED_NAME`, because a bare 200 on a shared box could be
  someone else's server.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0` and the venv on `PATH` are set only
  by `mt` and `ocr`.** Those two launch stock `vllm serve`, which never imports our engine
  modules and so never gets their `os.environ.setdefault`. TTS goes through our engine; ASR uses
  no vLLM at all.

### Authentication

**Off by default, in all four.** These launchers are a minimal reference setup: the documented
main path is offline inference, and a server you can start with no arguments is the point. Auth
is one environment variable away when a deployment needs it, and extending it is the caller's
job, not ours.

| launcher | opt in with | mechanism |
|---|---|---|
| `tts`, `asr` | `TTS_AUTH_FILE` / `ASR_AUTH_FILE` → a `chmod 600` `user:password` file | HTTP Basic, `bodhan_genai._serving_auth` |
| `mt`, `ocr` | `MT_API_KEY` / `OCR_API_KEY` | bearer token, vLLM's native auth |

The Ray Serve deployments share a single implementation on purpose: access control copied into
two modalities is access control fixed in one of them. It is raw ASGI rather than a FastAPI
dependency because `BaseHTTPMiddleware` only sees `http` scopes, so a streaming websocket would
sail straight past it. `GET /health` is exempt — an orchestrator's liveness probe cannot carry
credentials.

`mt` and `ocr` run stock `vllm serve`, which enforces bearer tokens itself. Putting our
middleware in front of that would be a second auth layer guarding one that already works, so
they hand the token over as `VLLM_API_KEY` in the environment — **not** `--api-key` on the
command line, whose argv any account on the node can read with `ps -eo args`. `MTClient` and
OCR's `HttpRecognizer` read their variable from the environment.

Two things stay strict, because both mean somebody is *enabling* auth and getting it wrong:
an unreadable credential file and a malformed one are errors, never a silent fallback to open.

!!! warning "What this is not"
    Every launcher binds `0.0.0.0`, so the default is reachable by anyone on the network. Basic
    auth is also only as private as its transport: base64 is not encryption, so over plain HTTP
    the credential is readable in flight. There is one shared credential per service, no
    per-user identity, no rotation and no rate limiting. That is deliberate for a reference
    setup — for a real deployment, put a TLS reverse proxy in front, or bind to localhost and
    tunnel.

!!! note "Testing auth: assert it is *attached*, not merely configured"
    Both suites once stayed green with `add_middleware` deleted — the tests either asserted the
    installer raised or exercised a middleware they constructed themselves, so none of them
    noticed the endpoints were wide open. Any new auth test must drive a request through the real
    stack (`TestClient`) and assert a 401. Neuter the installer and watch it fail before you
    trust it.

## Before pushing

```bash
# CPU test suite must pass (GPU-marked tests are excluded):
pytest -m "not gpu"

# Lint + format:
ruff check .
ruff format .
```

Both run in pre-commit and CI; running them locally first saves a round trip.

## Commit style

Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` — scope optional
(`feat(serving): ...`). Keep the subject under ~72 chars; explain *why* in the body when the diff
doesn't.

## Releases

1. Bump `version` in `pyproject.toml` **and** add the matching `CHANGELOG.md` entry in the same PR.
2. After merge, tag it: `git tag v<X.Y.Z> && git push --tags`.

## Never commit artifacts

Checkpoints, parquet shards, WAVs, wandb dirs — none of it belongs in git. A pre-commit large-file
hook blocks files over the size limit; do not bypass it with `--no-verify`. Keep artifacts on the
shared filesystem and reference them by path in configs (as comments, per the config convention).
