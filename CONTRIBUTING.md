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
├── notebooks/{tts,mt,ocr,asr}/  inference.ipynb, training.ipynb walkthroughs
├── examples/{tts,mt,ocr,asr}/   runnable single-file examples
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
