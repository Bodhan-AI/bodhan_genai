# Finetuning IndicTranslate

> ## ⚠️ This recipe is a provisional default
>
> It works and it is a reasonable starting point, but it is **not a qualified configuration**: no
> release has been trained with it from this repo, and the trainer behind it may be replaced.
> Treat the numbers below as defaults to adjust, not as settings shown to be optimal.
>
> What *is* stable is the boundary around it. Nothing outside `bodhan_genai.mt.training` imports
> from it, and the rest of the package talks to training through exactly two contracts:
>
> | contract | shape |
> |---|---|
> | **input** | rendered `messages` JSONL, from `bodhan_genai.mt.data.render` |
> | **output** | a PEFT adapter directory → `training.merge` → `tools.vllm_ready` |
>
> Any trainer that reads that JSONL and emits an adapter drops in without touching the engine, the
> prompt contract, the data pipeline, serving or eval. `tests/mt/test_training_is_replaceable.py`
> enforces this mechanically.

## What it is

A single-stage LoRA finetune at 8192-token context, on TRL's `SFTTrainer` with an assistant-only
loss mask. One config, one command.

It is **not** seq2seq training. IndicTranslate is a decoder-only causal LM, so this is instruction SFT:
the instruction is the prompt, the translation is the completion, and loss is computed on the
completion only.

## Prerequisites

MT needs its own environment — `transformers>=5.12` and `vllm>=0.20` for the Gemma 4 architecture,
which the TTS pins cannot satisfy:

```bash
./install.sh
source .venv/bin/activate
```

`scripts/mt/train_lora.sh` checks this up front and names the right venv if you are in the wrong
one.

## 1. Render the corpus

Training data is bitext JSONL — one object per line with a source and target field:

```json
{"eng": "The committee approved the proposal.", "hin": "समिति ने प्रस्ताव को मंजूरी दे दी।"}
```

Point `configs/mt/data/render.yaml` at it and run:

```bash
scripts/mt/render.sh                       # or --dry-run first, to see the mix
```

That writes `train.jsonl` + `dev.jsonl` in the `messages` format the trainer consumes, drawing one
of 12 instruction phrasings per row with a seeded RNG. Knobs worth knowing:

- **`reverse_fraction`** — bitext is normally stored one-way. `1.0` also emits every row reversed,
  making the corpus bidirectional. For a token-heavy document corpus, `0.2` is a reasonable
  asymmetry.
- **`template_variant`** — leave it `target_only` (the released contract). See
  [prompt_contract.md](prompt_contract.md).
- **`extra_languages`** — add a language outside the served 25 without editing the frozen contract.
- **A dev split is required.** Checkpoint selection is by `eval_loss` and early stopping has
  nothing to watch without one. `dev_max_rows` caps it — eval runs every few hundred steps, so a
  huge dev set just burns wall-clock.

**Order matters:** do any upsampling or corpus mixing *before* rendering. The RNG advances per row,
so N copies of a row get N different phrasings (useful augmentation); copies made afterwards would
all share one.

## 2. Train

```bash
scripts/mt/train_lora.sh configs/mt/train/lora.yaml
NUM_GPUS=4 scripts/mt/train_lora.sh              # override the GPU count
```

Re-running with the same `training.output_dir` resumes from the last checkpoint
(optimizer/scheduler/LR/step restored). Set `resume: false` to start fresh.

### The recipe

| setting | value | why |
|---|---|---|
| `max_seq_length` | 8192 | sentence corpora sit far under it; documents need it |
| LoRA `r` / `alpha` / `dropout` | 32 / 64 / 0.05 | alpha = 2×r is the usual ratio |
| `target_modules` | `"all-linear"` | Gemma 4 has `per_layer_input_gate` / `per_layer_projection` beside the usual projections; an explicit list silently misses them |
| `exclude_modules` | vision/audio towers, `lm_head` | translation never touches them |
| optimizer | AdamW, lr `1e-4` | suits LoRA; full-FT would want ~1e-5 |
| scheduler | `cosine_with_min_lr`, `min_lr 1e-5` | floors the decay instead of letting it reach 0 |
| `warmup_steps` | `0.03` | a float < 1 is a *fraction* of total steps (transformers 5; `warmup_ratio` is deprecated) |
| `max_grad_norm` | 5.0 | set explicitly — see the note below |
| batch | 1 × grad-accum 4 | memory-safe at 8192 tokens on an 80 GB card; effective batch = `1 × 4 × world_size` |
| `gradient_checkpointing` | on | trades compute for memory; needed at 8k |
| `packing` | off | one example per sequence, so the loss mask covers exactly one translation |
| `assistant_only_loss` | on | loss on the completion only |
| eval/save cadence | `eval_fraction: 0.1` | a fraction of one epoch, resolved from the real dataset size — the config transfers between corpora |
| selection | `load_best_model_at_end` on `eval_loss`, patience 5 | |

`max_length`, `packing` and `assistant_only_loss` are owned by the recipe: setting them in the
config logs a warning and is ignored.

### Distributed strategy: DDP, not FSDP

`configs/mt/accelerate/single_node.yaml` is plain DDP + bf16. Only the LoRA adapter is trainable,
so the optimizer state is small and sharding would add communication for nothing. (TTS uses FSDP2
because it trains all 3B parameters — the two are not comparable, and copying that config here
would be a mistake.)

Multi-node uses `configs/mt/accelerate/multinode.yaml`, which differs only in the rendezvous
(`c10d`, so ranks negotiate at startup) with the head address passed at launch:

```bash
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun accelerate launch \
  --config_file configs/mt/accelerate/multinode.yaml \
  --num_machines "$SLURM_NNODES" --num_processes $((SLURM_NNODES * 8)) \
  --main_process_ip "$head" --main_process_port 29500 \
  -m bodhan_genai.mt.training.train configs/mt/train/lora.yaml
```

### Two things to decide deliberately

Both are places where a default had to be chosen explicitly rather than inherited:

- **`max_grad_norm`.** Frameworks disagree on the default — some do not clip at all, HF defaults to
  `1.0`. The config sets `5.0` explicitly rather than inheriting a silent behaviour change.
- **Loss reduction.** HF averages over non-ignored tokens; a sum reduction changes the effective
  learning rate. Loss curves are therefore only comparable between runs that agree on this.
  (`final_logit_softcapping: 30.0` comes from the model config either way.)

### Continuing from an existing adapter

Two distinct operations, and the config refuses to let them be confused:

```yaml
# seed weights only — fresh optimizer, scheduler and LR
model:
  adapter_path: training_output/previous-run/checkpoint-4400
resume: false
```

versus `resume: true` with no `adapter_path`, which continues an interrupted run from
`output_dir`. Setting both raises.

## 3. Merge and serve

A LoRA adapter is ~300 MB and needs its base at load time; vLLM wants one self-contained
directory.

```bash
scripts/mt/merge.sh training_output/mt-lora-8k/checkpoint-4400 merged-ckpt-4400
```

That merges `W + (alpha/r)·B@A`, re-enables `use_cache` (training turns it off), stages the
tokenizer **and processor** from the base model, and adds the KV-shared `k_norm` sidecar. All four
matter — see [serving.md](serving.md) for why a merged checkpoint otherwise fails to load.

Then:

```bash
CHECKPOINT=merged-ckpt-4400 scripts/mt/serve.sh
python -m bodhan_genai.mt.serving.client --tgt-lang Hindi --text "Hello world"
```

## 4. Evaluate

Checkpoint selection by `eval_loss` is a proxy. Translation quality is chrF++ and BLEU on a held-out
benchmark:

```bash
scripts/mt/eval.sh --langs hin_Deva --directions en-xx --max-samples 128   # quick
scripts/mt/eval.sh                                                        # full IN22
```

The **noise floor is about ±0.06 chrF++** — vLLM's continuous batching reorders float reductions
even at temperature 0. When picking between checkpoints, treat anything inside ±0.2 as a tie and
prefer the earlier one. See [eval.md](eval.md).

## Troubleshooting

- **`transformers X < 5.12`** — you are in the TTS venv. `source .venv/bin/activate`.
- **OOM at 8192 tokens** — `gradient_accumulation_steps` up, `per_device_train_batch_size` already
  1. Confirm `gradient_checkpointing: true`. Dropping `max_seq_length` to 2048 is the big lever if
  your corpus is sentence-level.
- **`SFTConfig.__init__() got an unexpected keyword argument`** — a TRL version mismatch. The pin
  is `trl==1.6.0`; `tests/mt/test_sample_configs.py` constructs a real `SFTConfig` precisely so
  this fails in CI rather than on a node.
- **Trainable parameter count looks like the whole model** — `peft_config` was not applied. The run
  prints `print_trainable_parameters()` on rank 0; LoRA at r=32 should be a low single-digit
  percentage.
- **Dataset rebuild on every rank / a hang at startup** — the cache build is serialised with a
  `filelock`, not a distributed barrier, precisely to avoid an NCCL timeout on a large corpus. If
  you see a rebuild storm, check `data.cache_dir` is on a shared filesystem all ranks can see.
- **Every row filtered out** — the length filter measures the *rendered chat*, not the raw text.
  A corpus of long documents against `max_seq_length: 1024` legitimately empties; the loader raises
  rather than training on nothing.
