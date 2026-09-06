# Troubleshooting

Failure modes shared by every modality. Anything specific to one stack lives with that stack:
[IndicSpeak](https://github.com/AshwinSankar17/bodhan_gen_ai_tools/blob/master/src/bodhan_genai/tts/README.md#troubleshooting) ·
[IndicTranslate](https://github.com/AshwinSankar17/bodhan_gen_ai_tools/blob/master/src/bodhan_genai/mt/README.md#troubleshooting) ·
[IndicOCR](https://github.com/AshwinSankar17/bodhan_gen_ai_tools/blob/master/src/bodhan_genai/ocr/README.md#troubleshooting) ·
[IndicTranscribe caveats](asr/caveats.md).

Most of these share one cause: **the environment was built in the wrong order, or against the
wrong CUDA line.** The installer encodes a required order — vLLM first from its per-CUDA index so
it pulls its own matched torch, then torchaudio/torchvision, then the package — and deviating
from it is the most common cause of a broken environment.

---

## `transformers X.Y < 5.12`, or unknown architecture `gemma4`

A stale environment predating the single-venv layout. The stack pins
`transformers==5.13.1`; the Gemma 4 architecture IndicTranslate uses needs `>= 5.12`.

```bash
rm -rf .venv && ./install.sh
```

## `torch.cuda.is_available()` is False on a GPU box

The wheels are built for a newer CUDA than the driver provides. They import cleanly and *then*
report no GPU, which is why this reads as a code problem rather than an install one.

Reinstall against a matching line:

```bash
CUDA_TAG=cu126 ./install.sh      # default is cu129
```

## `EngineCore failed to start`, with no further explanation

vLLM 0.26 samples through a flashinfer kernel it JIT-compiles at engine warm-up, and that build
needs `nvcc`. Nodes with a runtime-only CUDA install have none.

This should not reach you. The three vLLM-backed engines each disable that sampler before
importing vLLM — `bodhan_genai.tts.engine.offline`, `.tts.engine.streaming`,
`bodhan_genai.mt.engine.offline` and `bodhan_genai.ocr.engine.recognizer_vllm` all call
`os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")`, and `scripts/ocr/serve.sh` exports
the same default. (IndicTranscribe is unaffected: the ASR stack does not use vLLM.)

Because they use `setdefault`, an explicit export wins. If you exported
`VLLM_USE_FLASHINFER_SAMPLER=1`, unset it — or keep it, on a node that has a CUDA toolkit.

!!! warning "An out-of-date flashinfer breaks IndicOCR's recognizer, and the sampler flag will not save you"

    Verified both ways on an H100 node. On the **pinned** environment (`vllm 0.26.0`,
    `flashinfer-python 0.6.14`) `VllmRecognizer` works. On an older stack (`vllm 0.19.1`,
    `flashinfer-python 0.6.6`) the same call dies —

    ```
    vllm/model_executor/layers/mamba/gdn_linear_attn.py -> forward_cuda
    flashinfer/gdn_prefill.py -> get_gdn_prefill_module().build_and_load()
    RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
    ```

    The recognizer's Qwen3.5 backbone uses gated delta-net (GDN) linear attention. Older
    flashinfer has no prebuilt kernel for it and JIT-compiles at first use, which needs the CUDA
    *compiler*; `VLLM_USE_FLASHINFER_SAMPLER=0` covers a different path and does nothing here.

    **The fix is the version pin, not the toolkit.** Do not go hunting for `nvcc`: rebuild with
    `rm -rf .venv && ./install.sh`. IndicSpeak's vLLM backend is unaffected either way — a Llama
    backbone has no GDN layers.

!!! info "These nodes have the CUDA runtime, not the CUDA compiler"

    Worth knowing, because the error message above misleads. A typical node here carries
    `/usr/local/cuda-12.2` containing `include`, `lib64`, `nvml` and `targets` — headers and
    libraries, **no `bin/` and no `nvcc`** — and there is no bare `/usr/local/cuda` symlink for
    tools that look for one.

    That is enough for everything that ships precompiled kernels, which is why training and
    inference work. It is not enough for anything that compiles at install or first use: it is
    why `flash-attn` must be built elsewhere or skipped with `--no-flash-attn`, and why an
    unpinned flashinfer fails as above. Nothing in the pinned environment needs the compiler.

## 401 / 403 pulling checkpoints

The `bodhan-ai/` default checkpoints are public, so this should not happen on them. If it does — a gated fork, a private finetune, or an access change — authenticate:

```bash
export HF_TOKEN=hf_...
```

…or point the model arguments at local paths and take the Hub out of the loop entirely with
`HF_HUB_OFFLINE=1`. The Docker wrappers (`scripts/{tts,mt}/*_docker.sh`,
`scripts/ocr/parse_docker.sh`) and the training launchers (`scripts/tts/train.sh`,
`scripts/tts/train_lora.sh`, `scripts/mt/train_lora.sh`) already set it.

!!! note "A 404 usually means access, not a missing repo"

    The four default repos — `bodhan-ai/indic-speak`, `indic-translate`, `indic-ocr` and
    `indic-transcribe-core` — are **public**, so they resolve with no credentials at all.

    If you point at a private or gated repo instead, note that the Hub returns **404 rather
    than 403** for something a token cannot read. A "repository not found" is then an access
    problem, not a naming one: check the token before checking the id.

## No network for `wandb`

```bash
WANDB_MODE=offline ./scripts/tts/train.sh    # then `wandb sync <run-dir>` later
```

Read by `scripts/tts/train.sh`, `scripts/tts/train_lora.sh` and `scripts/mt/train_lora.sh`.

## flash-attn builds forever, or fails

Expected: it is compiled from source on purpose. PyPI ships
[flash-attn](https://github.com/Dao-AILab/flash-attention) as an sdist with no wheel, and pinning
a prebuilt one ties the install to a single `(flash-attn, torch, CUDA, cpython, arch)` tuple that
goes stale the moment any of the four moves. The build needs `ninja` and a CUDA toolkit.

Only **IndicSpeak training** requires it, so on a box that serves or runs inference:

```bash
./install.sh --no-flash-attn
```

## The pip resolver replaces torch with a CPU build

Install order. vLLM first (it pins torch), then torchaudio/torchvision from the per-CUDA index,
then the package under `constraints.txt`. Never install the package before torch. `./install.sh`
does this for you and asserts the resulting torch version afterwards.
