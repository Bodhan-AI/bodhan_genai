# MT configuration reference

Field-by-field for the shipped `configs/mt/` files.

Convention throughout: model and data references are written as **live HF hub ids** or repo-relative
paths. A machine-specific absolute path is never a live value — put one on an adjacent comment line
if you need it locally.

| file | purpose |
|---|---|
| `configs/mt/data/render.yaml` | bitext → instruction chat rows |
| `configs/mt/train/lora.yaml` | the 8k LoRA finetune (provisional) |
| `configs/mt/infer/offline_vllm.yaml` | offline batch inference defaults |
| `configs/mt/accelerate/single_node.yaml` | single-node DDP launcher |
| `configs/mt/accelerate/multinode.yaml` | multi-node DDP launcher (c10d rendezvous) |

All loaders **reject unknown keys**. A typo is an error, not a silently ignored setting.

---

## `configs/mt/data/render.yaml`

Consumed by `python -m bodhan_genai.mt.data.render`. Full narrative in
[data_pipeline.md](data_pipeline.md).

### top level

| key | type | default | notes |
|---|---|---|---|
| `seed` | int | 42 | everything derived from it is stable across runs and machines |
| `template_variant` | str | `target_only` | `target_only` \| `with_source`. `target_only` is the released contract |
| `dedup` | bool | true | drops duplicate `(direction, source, target)` triples across **all** sources |
| `extra_languages` | map | `{}` | languages outside the served 25, keyed by FLORES-style code |

### `output:`

| key | type | default | notes |
|---|---|---|---|
| `train` | str | — | **required** |
| `dev` | str | null | required in practice: the trainer selects on `eval_loss` |
| `dev_fraction` | float | 0.01 | must be in `[0, 1)` |
| `dev_max_rows` | int | 2000 | hard cap; eval runs often, so a huge dev set only costs wall-clock |

### `sources:` (list)

| key | type | default | notes |
|---|---|---|---|
| `path` | str | — | bitext JSONL |
| `src_field` / `tgt_field` | str | — | the JSONL columns holding the text |
| `src_lang` / `tgt_lang` | str | — | FLORES-style codes |
| `name` | str | filename stem | appears in the row's `corpus` field and the stats table |
| `reverse_fraction` | float | 0.0 | fraction also emitted reversed; must be in `[0, 1]` |
| `limit` | int | null | cap input rows for a smoke run |

---

## `configs/mt/train/lora.yaml`

Consumed by `python -m bodhan_genai.mt.training.train`, normally via
`scripts/mt/train_lora.sh`. **Provisional recipe** — see [training.md](training.md).

### `model:`

| key | type | default | notes |
|---|---|---|---|
| `model_path` | str | — | loaded with `AutoModelForCausalLM` (text-only view of the multimodal checkpoint) |
| `tokenizer_path` | str | `""` | empty = use `model_path` |
| `max_seq_length` | int | 8192 | the shipped recipe is the 8k one |
| `torch_dtype` | str | `bfloat16` | |
| `attn_implementation` | str | `sdpa` | no flash-attn in the MT env |
| `gradient_checkpointing` | bool | true | needed at 8k; trades compute for memory |
| `adapter_path` | str | null | seed weights from an existing adapter. **Requires `resume: false`** |

### `lora:`

| key | type | default | notes |
|---|---|---|---|
| `r` | int | 32 | must be > 0 |
| `lora_alpha` | int | 64 | conventional 2 × r; effective scale = alpha/r |
| `lora_dropout` | float | 0.05 | must be in `[0, 1)` |
| `bias` | str | `none` | `none` \| `all` \| `lora_only` |
| `task_type` | str | `CAUSAL_LM` | |
| `target_modules` | str \| list | `all-linear` | **keep `all-linear`**: an explicit list misses Gemma 4's `per_layer_input_gate` / `per_layer_projection` |
| `exclude_modules` | list | vision/audio/`lm_head` globs | translation never touches the towers |
| `modules_to_save` | list | `[]` | |
| `use_rslora` | bool | false | alpha/√r instead of alpha/r |
| `use_dora` | bool | false | |

### `data:`

| key | type | default | notes |
|---|---|---|---|
| `train_file` / `dev_file` | str | — | output of the render stage |
| `cache_dir` | str | — | tokenized-dataset cache; must be visible to every rank |
| `num_proc` | int | 16 | tokenization parallelism |
| `shuffle_seed` | int | 42 | |

### top level

| key | type | default | notes |
|---|---|---|---|
| `resume` | bool | true | continue from the last checkpoint in `output_dir`. Mutually exclusive with `model.adapter_path` |
| `eval_fraction` | float | 0.1 | eval/save cadence as a fraction of one epoch, resolved from the real dataset size. `training.eval_steps` overrides |

### `training:` → `trl.SFTConfig`

Passed through as `**kwargs`, so any `SFTConfig` field works. Shipped values:

| key | value | notes |
|---|---|---|
| `output_dir` | `training_output/mt-lora-8k` | re-running here resumes |
| `num_train_epochs` | 3 | |
| `learning_rate` | 1e-4 | suits LoRA |
| `lr_scheduler_type` | `cosine_with_min_lr` | |
| `lr_scheduler_kwargs.min_lr` | 1e-5 | floors the decay instead of reaching 0 |
| `warmup_steps` | 0.03 | float < 1 = **fraction** of total steps (transformers 5; `warmup_ratio` is deprecated) |
| `max_grad_norm` | 5.0 | set explicitly — frameworks disagree on the default |
| `per_device_train_batch_size` | 1 | memory-safe at 8192 on 80 GB |
| `gradient_accumulation_steps` | 4 | effective batch = `1 × 4 × world_size` |
| `bf16` | true | |
| `logging_steps` | 10 | |
| `save_total_limit` | 3 | |
| `load_best_model_at_end` | true | on `eval_loss` |
| `metric_for_best_model` | `eval_loss` | `greater_is_better: false` |
| `early_stopping_patience` | 5 | **not an SFTConfig field** — popped by the recipe, drives `EarlyStoppingCallback` |
| `report_to` | `wandb` | `none` to disable |

**Owned by the recipe, ignored here** (setting them logs a warning): `max_length`, `packing`,
`assistant_only_loss`.

### `logging:`

| key | default | notes |
|---|---|---|
| `wandb_project` | `bodhan-genai-mt` | |
| `wandb_run_name` | null | defaults to the trainer's own naming |
| `report_to` | `wandb` | |

---

## `configs/mt/infer/offline_vllm.yaml`

Installed as **argparse defaults**; explicit CLI flags always win. Nested sections
(`engine`, `sampling`, `io`) are flattened one level onto the flag names.

| key | default | notes |
|---|---|---|
| `model` | published hub id | local path or hub id |
| `engine.dtype` | `bfloat16` | |
| `engine.tensor_parallel_size` | 1 | |
| `engine.max_model_len` | 32768 | architecture allows 131072; only 32768 is validated. Drop to 8192 for sentences |
| `engine.gpu_memory_utilization` | 0.90 | |
| `engine.enforce_eager` | true | the validated serving configuration |
| `sampling.temperature` | 0.0 | greedy — recommended, and the only reproducible setting |
| `sampling.top_p` | 1.0 | only used when temperature > 0 |
| `sampling.repetition_penalty` | 1.0 | |
| `sampling.max_new_tokens` | 512 | 512 sentence / 2048 paragraph / 8192 document |
| `sampling.seed` | 0 | only used when temperature > 0 |

---

## `configs/mt/accelerate/*.yaml`

**DDP + bf16, not FSDP.** Only the LoRA adapter is trainable, so the optimizer state is small and
sharding would add communication for nothing. Copying the TTS FSDP2 config here would be a mistake
— TTS trains all 3B parameters. A test asserts `distributed_type: MULTI_GPU` and no `fsdp_config`.

`single_node.yaml` pins `machine_rank: 0` and `num_processes: 8`; `scripts/mt/train_lora.sh`
overrides the process count from the visible GPU count.

`multinode.yaml` differs only in the rendezvous — `rdzv_backend: c10d`, so ranks negotiate at
startup and `machine_rank` is not baked in. `num_machines`, `num_processes` and `main_process_ip`
come from the launch command, because the head node changes every allocation.
