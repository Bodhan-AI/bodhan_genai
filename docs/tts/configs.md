# Configuration reference

Field-by-field docs for every file under `configs/`.

**Path convention:** model/tokenizer/data references are written as **HF hub ids live** with the
**cluster-local absolute path on a comment line next to them**. On offline nodes, swap the comment
in and set `HF_HUB_OFFLINE=1`. Hardcoded cluster paths never appear as live values in shipped
configs.

---

## `configs/tts/data/tokenize.yaml` — stage 1 (python -m bodhan_genai.tts.data.tokenize)

| field | meaning |
|---|---|
| `ray.address` | `null` = start a local Ray; `"auto"` = join an existing cluster |
| `models.snac_model_path` | SNAC codec, e.g. `hubertsiuzdak/snac_24khz` (local path in comment) |
| `models.tokenizer_path` | the extended Llama-3 audio tokenizer (required; see docs/tts/token_layout.md) |
| `processing.workers_per_gpu` | Ray SNAC actors per GPU |
| `processing.encode_batch_size` | waveforms padded + encoded per GPU forward pass |
| `datasets[]` | list of dataset entries, each with `source`, `columns`, `output` |
| `datasets[].source.jsonl` | JSONL manifest path (one of `jsonl` / `hf_dataset`) |
| `datasets[].source.hf_dataset` / `.subset` / `.split` | HF Hub dataset coordinates |
| `datasets[].columns.{audio,text,language,speaker,style,accent,audio_caption}` | input-column → schema mapping (see docs/tts/data_pipeline.md) |
| `datasets[].output.dir` | output parquet directory (manifest written alongside) |
| `datasets[].output.rows_per_shard` | rows per parquet shard |

## `configs/tts/data/compile.yaml` — stage 2 (python -m bodhan_genai.tts.data.compile)

| field | meaning |
|---|---|
| `models.tokenizer_path` | same extended tokenizer as stage 1 (ids must match) |
| `input.datasets[].input_dir` | stage-1 parquet directory |
| `input.datasets[].output_dir` | compiled parquet output (input_ids/labels/length, length-desc sorted) |

---

## `configs/tts/train/pretrain.yaml` — full-FT pretraining

| field | meaning |
|---|---|
| `training_stage` | tag used in auto-generated wandb run names (`pt`) |
| `model.model_path` | base checkpoint, HF id live / local path commented |
| `model.tokenizer_path` | extended audio tokenizer; embedding auto-resized to `len(tokenizer)` |
| `model.max_seq_len` | packing bin size; every batch is `[1, max_seq_len]` |
| `model.attn_implementation` | `flash_attention_2` (required for packed cross-seq isolation) |
| `model.torch_dtype` | `bfloat16` |
| `model.activation_checkpointing` | keep `false` — FSDP owns it via the accelerate config; setting both causes a CheckpointError |
| `model.compile` / `model.compile_mode` | `torch.compile` toggle + mode; first step costs 10–20 min, set `false` for smoke runs |
| `data.train.datasets[]` | `{path, ratio, name}` entries; ratios are normalized sampling weights |
| `data.val.datasets[]` / `data.log.datasets[]` | validation / sample-logging parquet dirs |
| `training.*` | passed straight to HF `TrainingArguments` (`output_dir`, `num_train_epochs`, `learning_rate`, `warmup_steps`, `lr_scheduler_type`, `optim`, `max_grad_norm`, `bf16`, `logging_steps`, `save_strategy`/`save_steps`, `eval_strategy`/`eval_steps`, `seed`, dataloader knobs) |
| `training.gradient_accumulation_steps` | effective batch = bins × world size × this |
| `training.save_total_limit` | **must stay `null`** — retention is owned by the checkpoint keeper callback (top-K by metric ∪ last-N by step); a non-null value makes HF's `_rotate_checkpoints` delete the best checkpoints out from under it |
| `checkpoint_retention.{last_n,best_k,metric}` | the actual retention policy |
| `logging.{wandb_project,wandb_entity,wandb_run_name,peak_tflops_per_gpu}` | wandb + MFU logging; `WANDB_MODE=offline` for air-gapped runs |

Note: `per_device_train_batch_size` is forced to 1 — packing handles batching.

## `configs/tts/train/sft.yaml` — full-FT SFT

Same schema as `pretrain.yaml` with `training_stage: "sft"`, SFT-compiled datasets under
`data.train.datasets`, and typically a shorter schedule (lower LR, fewer epochs). No separate code
path — same trainer entry point.

## `configs/tts/train/lora.yaml` — LoRA adapter run

Everything from the full-FT schema, plus:

| field | meaning |
|---|---|
| `lora.r` / `lora.lora_alpha` | rank / scaling (defaults 32 / 64) |
| `lora.lora_dropout` | adapter dropout (0.05) |
| `lora.bias` / `lora.task_type` | `"none"` / `"CAUSAL_LM"` |
| `lora.target_modules` | attention + MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) |
| `lora.modules_to_save` | extra fully-tuned modules; `[]` for pure LoRA |
| `model.compile` | keep `false` — the LoRA entry point force-disables compile anyway (PEFT injection under FSDP2 causes Inductor recompile storms) |
| `training.learning_rate` | ~10× the full-FT LR (LoRA tolerates it), e.g. `1e-4`–`2e-4` |

Checkpoints contain only `adapter_model.safetensors` + `adapter_config.json` (adapter-only save
callback), not a full FSDP state dump.

---

## `configs/tts/accelerate/single_node_fsdp.yaml` — launcher config

| field | meaning |
|---|---|
| `distributed_type` | `FSDP` |
| `fsdp_config.fsdp_version` | `2` (FSDP2; requires torch >= 2.6) |
| `fsdp_config.fsdp_activation_checkpointing` | activation checkpointing lives HERE, not in train.yaml |
| `fsdp_config.fsdp_auto_wrap_policy` / `fsdp_transformer_layer_cls_to_wrap` | `TRANSFORMER_BASED_WRAP` on `LlamaDecoderLayer` |
| `fsdp_config.fsdp_reshard_after_forward` / `fsdp_forward_prefetch` | memory/throughput trade knobs |
| `fsdp_config.fsdp_state_dict_type` | `FULL_STATE_DICT` |
| `mixed_precision` | `bf16` |
| `num_processes` | reference GPU count — `scripts/tts/train.sh` overrides it with `NUM_GPUS` at launch |
| `num_machines` / `machine_rank` | `1` / `0` — single-node by design |

---

## `configs/tts/infer/offline_vllm.yaml` — python -m bodhan_genai.tts.inference.cli

| field | meaning |
|---|---|
| `model.checkpoint_path` | trained checkpoint — `bodhan-ai/indic-speak` by default (public), or a local path; its own tokenizer is used — see the vocab-mismatch trap in the README |
| `model.snac_model_path` | SNAC codec for phase-2 decode |
| `vocos` | decoder for phase-2 decode: `true` = fine-tuned Vocos decoder from the Hub (default), a path = local `vocos` `.pt`, `false` = SNAC's own decoder — see [Inference](inference.md#decoder-selection-vocos) |
| `engine.gpu_memory_utilization` | vLLM VRAM fraction; leave headroom for the SNAC decode phase |
| `engine.max_num_seqs` / `engine.max_model_len` | vLLM batch/context limits |
| `sampling.{temperature,top_p,max_new_tokens}` | generation params (0.7 / 0.8 defaults) |
| `io.jsonl_path` | prompt manifest (overridable by `--jsonl-path`) |
| `io.output_dir` | WAV output dir (overridable by `--output_dir`) |
| `decode.batch_size` | SNAC windows per phase-2 decode batch |

Two-phase flow: phase 1 generates audio token ids for all prompts with vLLM; phase 2 batch-decodes
the extracted SNAC windows to 24 kHz WAVs. Keeping the phases separate lets vLLM use the whole GPU
during generation.
