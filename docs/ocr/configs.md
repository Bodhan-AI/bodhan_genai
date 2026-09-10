# OCR configuration

Every tunable is a frozen dataclass in `bodhan_genai.ocr.engine.types`, whose **defaults are the
shipped recipe**. Nothing is read from the environment at import time.

Override them explicitly through the dataclasses rather than through the environment.

## The dataclasses

| | field | default | notes |
| --- | --- | --- | --- |
| `LayoutConfig` | `conf` | `0.5` | below ~0.4 stains and page borders start scoring as blocks |
| | `img_size` | `1024` | square input resolution |
| | `device` | `cuda` | |
| `DedupConfig` | `nest` | `True` | resolve nested equation boxes |
| | `mode` | `both` | `both` \| `text_only` \| `eq_only` |
| | `contain` | `0.90` | duplicate threshold in `clean_layout` |
| | `wrap` | `0.5` | a header counts as occupied at this containment |
| | `nested` | `0.70` | nested-equation threshold |
| `CropConfig` | `min_px_side` | `256` | crops below side² are upscaled; `0` disables |
| | `max_px_side` | `1536` | token ceiling |
| | `pad_px` | `0` | margin to recover glyph edges a tight box clips |
| `RecognizerConfig` | `max_model_len` | `8192` | |
| | `max_tokens` | `2048` | per-block generation cap |
| | `temperature` | `0.0` | greedy — the only reproducible setting |
| | `gpu_memory_utilization` | `0.80` | leaves headroom for the layout model |
| | `batch_size` | `2048` | requests per `generate()` call |
| | `enforce_eager` | `True` | skips ~4 min of torch.compile |
| | `table_format` | `HTML` | `HTML` \| `MARKDOWN` |

## Two defaults worth leaving alone

**`CropConfig` clamps area, not a side.** Pinning a side exploded elongated crops — a 122:1 rule
line became roughly 32k image tokens and wedged the engine. An area clamp bounds worst-case tokens
at `max_px / 32²` for any aspect ratio.

**`RecognizerConfig.batch_size` is a ceiling, not a target.** One `generate()` call over tens of
thousands of multi-modal requests wedges the vLLM V1 scheduler at 100% utilisation with no
progress. ~2k-request chunks run cleanly and let large page-sets finish.

## `DedupConfig.mode`

IndicDocLayout over-produces equation boxes: small inline-math boxes nested inside a text
paragraph, and per-line boxes nested inside a display array. Transcribing container and children
both emits the same math twice and fragments it.

| mode | folds an equation into |
| --- | --- |
| `both` *(default)* | a text-like block **or** a larger equation |
| `text_only` | a text-like block only — a display array stays fragmented |
| `eq_only` | a larger equation only — inline math stays in its paragraph |

The published 82.9 was measured with `text_only`; `both` is the shipped default and re-measurement
is pending.

## YAML

`configs/ocr/infer/parse.yaml` and `layout.yaml` supply CLI defaults:

```bash
python -m bodhan_genai.ocr.inference.cli parse pages/ --config configs/ocr/infer/parse.yaml
```

Sections (`layout:`, `dedup:`, `crop:`, `recognizer:`) are flattened one level onto the argparse
destinations, so explicit flags always override the file. Within a section a key resolves to its
bare flag if one exists, otherwise to `<section>_<key>` — which is why `dedup: {mode: both}` reads
naturally while the flag stays unambiguous as `--dedup-mode`.

Unknown keys raise rather than being ignored, and a test asserts every key in every shipped config
resolves to a real flag. That is what stops a config and the CLI from drifting apart, which
otherwise fails by silently ignoring a setting you thought you had changed.

## Checkpoints

Resolution order, first hit wins:

1. an explicit path (constructor argument or `--layout-ckpt` / `--ocr-ckpt`);
2. the public `bodhan-ai/indic-ocr` repo, downloaded per stage.

Three further routes exist, and **all three apply only inside a deployment image** — that is,
when `BODHAN_GENAI_DEPLOYMENT=1`, which only `docker/*/Dockerfile*` sets. Between step 1 and
step 2 there, in order:

- `BODHAN_OCR_LAYOUT_CKPT` / `BODHAN_OCR_RECOGNIZER_CKPT` — how `parse_docker.sh` addresses
  weights mounted at `/models`. A path that does not exist raises, rather than silently falling
  through to a 1.7 GB download;
- `weights/{layout,ocr}/` bundled beside the repository;
- `BODHAN_OCR_HF_REPO`, replacing the default repo id.

!!! warning "Why they are gated"

    Both are *implicit* — nobody typed them at the call site. An inherited
    `BODHAN_OCR_LAYOUT_CKPT`, or a `weights/` directory left behind by an earlier experiment,
    silently loads different weights on every subsequent run. That is a correctness bug which
    presents as a model regression, and it is invisible: no flag, no log line, no diff. Inside
    the image they are the mechanism for mounted weights, so they are kept there and nowhere
    else. Everywhere else, the same arguments always load the same weights.

Nothing is downloaded until a stage is constructed, so import and `--help` never touch the network.
