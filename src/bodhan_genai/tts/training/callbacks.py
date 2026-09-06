"""
Training callbacks.

TrainingMetricsCallback: Perplexity, GPU memory, and MFU (Model FLOPs Utilization).
EpochSamplerCallback: Calls sampler.set_epoch() at the start of each epoch.
BestAndLastCheckpointKeeper: Keeps top-N checkpoints by val loss + last-N by step
                             (replacement for HF's save_total_limit).
PeftAdapterSaveCallback: Saves LoRA adapter weights at every save_steps boundary
                         (used only by training/train_lora.py).
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _to_local_cpu(t):
    """Materialize sharded / distributed tensors into a regular CPU tensor for
    serialization. Handles:
      - DTensor (FSDP-2): ``.full_tensor()`` gathers across the mesh.
      - Older sharded tensors: ``.to_local()`` returns the shard owned by this rank.
      - Regular tensors: ``.detach().cpu()``.
    Anything not tensor-like is returned unchanged.
    """
    if not torch.is_tensor(t) and not hasattr(t, "to_local") and not hasattr(t, "full_tensor"):
        return t
    full_fn = getattr(t, "full_tensor", None)
    if callable(full_fn):
        try:
            t = full_fn()
        except (RuntimeError, NotImplementedError) as e:
            # Catching RuntimeError/NotImplementedError covers the realistic
            # FSDP/DTensor cases (uninitialized mesh, mixed-dim sharding, etc.).
            # A blanket `except Exception` previously masked genuine corruption
            # and let `to_local()` produce a partial tensor without anyone knowing.
            logging.getLogger(__name__).warning(
                f"_to_local_cpu: full_tensor() failed ({type(e).__name__}: {e}); "
                "falling back to to_local()"
            )
    to_local_fn = getattr(t, "to_local", None)
    if callable(to_local_fn):
        try:
            local = to_local_fn()
            if torch.is_tensor(local):
                t = local
        except (RuntimeError, NotImplementedError) as e:
            logging.getLogger(__name__).warning(
                f"_to_local_cpu: to_local() failed ({type(e).__name__}: {e}); "
                "returning tensor as-is — downstream save may write a partial shard"
            )
    return t.detach().cpu() if torch.is_tensor(t) else t


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU peak TFLOPS lookup (BF16 Tensor Core throughput)
# ---------------------------------------------------------------------------

_GPU_TFLOPS: dict[str, float] = {
    "A100": 312.0,
    "H100": 989.0,
    "H200": 1979.0,
    "A6000": 154.0,
    "RTX 4090": 165.2,
    "RTX 3090": 71.0,
    "V100": 125.0,
}


def _detect_peak_tflops() -> float | None:
    """Return peak BF16 TFLOPS for the current GPU, or None if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for key, tflops in _GPU_TFLOPS.items():
        if key in name:
            return tflops
    logger.debug(f"Unknown GPU model '{name}' — peak TFLOPS not available for MFU")
    return None


# ---------------------------------------------------------------------------
# Training metrics (perplexity + GPU memory + MFU)
# ---------------------------------------------------------------------------


class TrainingMetricsCallback(TrainerCallback):
    """
    Computes and logs per-step training metrics:

      perplexity                — from CE loss (WandB rewrites to train/perplexity)
      eval_perplexity           — from eval_loss (WandB rewrites to eval/perplexity)
      system/gpu_mem_alloc_gb   — peak allocated VRAM
      system/gpu_mem_reserved_gb
      mfu                       — Model FLOPs Utilization (rewritten to train/mfu)
      tflops_achieved           — Achieved model TFLOPS per GPU
      tflops_aggregate          — Achieved model TFLOPS across all ranks

    MFU formula:
      FLOPs_per_step = 6 * num_params * seq_len * grad_accum_steps
      time_per_step  = wall-clock seconds for logging_steps steps
      MFU            = per_gpu_TFLOPS / peak_tflops
    """

    def __init__(
        self,
        num_params: int,
        max_seq_len: int,
        gradient_accumulation_steps: int = 1,
        peak_tflops_per_gpu: float | None = None,
    ) -> None:
        self._num_params = num_params
        self._max_seq_len = max_seq_len
        self._grad_accum = gradient_accumulation_steps
        self._peak_tflops = peak_tflops_per_gpu or _detect_peak_tflops()
        if self._peak_tflops:
            logger.info(f"MFU tracking enabled: peak_tflops={self._peak_tflops:.1f}")
        else:
            logger.info("MFU tracking disabled: GPU peak TFLOPS unknown")

        self._last_log_time: float | None = None
        self._last_step: int = 0
        self._low_packing_streak: int = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return

        # Perplexity
        if "loss" in logs:
            try:
                logs["perplexity"] = math.exp(min(float(logs["loss"]), 100.0))
            except (OverflowError, ValueError):
                logs["perplexity"] = float("inf")

        if "eval_loss" in logs:
            try:
                logs["eval_perplexity"] = math.exp(min(float(logs["eval_loss"]), 100.0))
            except (OverflowError, ValueError):
                logs["eval_perplexity"] = float("inf")

        # GPU memory — console-only (not pushed to wandb).
        if state.is_world_process_zero and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
            gpu_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            logger.info(f"gpu_mem: alloc={gpu_alloc_gb:.2f}GB reserved={gpu_reserved_gb:.2f}GB")
            torch.cuda.reset_peak_memory_stats()

        # MFU
        now = time.monotonic()
        if "loss" in logs and self._peak_tflops and self._last_log_time is not None:
            steps_elapsed = state.global_step - self._last_step
            elapsed_secs = now - self._last_log_time
            if steps_elapsed > 0 and elapsed_secs > 0:
                flops_per_step = 6 * self._num_params * self._max_seq_len * self._grad_accum
                total_flops = flops_per_step * steps_elapsed
                achieved_tflops_per_gpu = total_flops / elapsed_secs / 1e12
                mfu = achieved_tflops_per_gpu / self._peak_tflops
                logs["mfu"] = round(mfu, 4)
                logs["tflops_achieved"] = round(achieved_tflops_per_gpu, 2)
                logs["tflops_aggregate"] = round(
                    achieved_tflops_per_gpu * max(1, args.world_size),
                    2,
                )

        if "loss" in logs:
            self._last_log_time = now
            self._last_step = state.global_step

        # Packing efficiency (attached by PackingCollator). Console-only; we
        # strip the key from `logs` so wandb never sees it but we still surface
        # the value in stdout for live monitoring.
        if "packing_efficiency" in logs:
            eff = float(logs.pop("packing_efficiency"))
            logger.info(f"packing_efficiency: {eff:.4f}")
            if eff < 0.85:
                self._low_packing_streak += 1
                if self._low_packing_streak >= 3:
                    logger.warning(
                        f"Packing efficiency {eff:.2%} < 85% for {self._low_packing_streak} "
                        "consecutive log steps — check sequence length distribution."
                    )
            else:
                self._low_packing_streak = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Keep train TFLOPS/MFU focused on training time, not inline eval time.
        if self._last_log_time is not None:
            self._last_log_time = time.monotonic()

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Checkpoint writes can be long on shared filesystems; exclude them from
        # the next train throughput interval without changing the step count.
        if self._last_log_time is not None:
            self._last_log_time = time.monotonic()


# ---------------------------------------------------------------------------
# Epoch sampler update
# ---------------------------------------------------------------------------


class EpochSamplerCallback(TrainerCallback):
    """
    Drives per-epoch shuffling on the *active* sampler. Resolves the sampler
    via the trainer each call — HF's Trainer re-creates the dataloader/sampler
    during train(), so a captured reference would silently orphan (pre-fix bug:
    set_epoch landed on a dead sampler while the live one kept epoch=0 forever).
    """

    def __init__(self, trainer) -> None:
        self._trainer = trainer

    def _active_sampler(self):
        getter = getattr(self._trainer, "get_train_sampler", None)
        return getter() if getter is not None else None

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        sampler = self._active_sampler()
        if sampler is None:
            return
        epoch = int(state.epoch) if state.epoch is not None else 0
        sampler.set_epoch(epoch)
        logger.debug(f"Sampler epoch set to {epoch}")


# ---------------------------------------------------------------------------
# Best-N + last-N checkpoint retention
# ---------------------------------------------------------------------------


class BestAndLastCheckpointKeeper(TrainerCallback):
    """Retains the union of:
      - top ``best_k`` checkpoints by ``metric`` (default: lowest ``eval_loss``)
      - last ``last_n`` checkpoints by step

    Anything else under ``args.output_dir/checkpoint-*`` is deleted on each save.

    **Important**: ``save_total_limit`` must be unset (``null``) in the training
    config — HF's ``_rotate_checkpoints`` runs **before** ``on_save`` fires, so
    if it's set, the oldest dirs (which may still be the lowest-loss ones) are
    gone before this callback can protect them.

    The eval-loss lookup walks ``state.log_history`` looking for entries that
    carry both ``step`` and ``metric``. If a saved step has no recorded metric
    (e.g. eval and save misaligned), that checkpoint is excluded from the
    "best" set but is still eligible for "last".
    """

    def __init__(
        self,
        last_n: int = 3,
        best_k: int = 3,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        if last_n < 0 or best_k < 0:
            raise ValueError(
                f"last_n and best_k must be >= 0, got last_n={last_n}, best_k={best_k}"
            )
        if last_n == 0 and best_k == 0:
            raise ValueError("at least one of last_n / best_k must be >= 1")
        self.last_n = last_n
        self.best_k = best_k
        self.metric = metric
        self.greater_is_better = greater_is_better

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # HF's TrainingArguments.save_total_limit triggers _rotate_checkpoints
        # AFTER each save; if it's set, HF will delete checkpoints before this
        # callback's union retention (best-K + last-N) gets a chance to weigh
        # in. The "best" set silently shrinks to whatever fits in the most-
        # recent window. Misconfig is easy (we hit it today on llama3 train.yaml),
        # so refuse to run with both retention systems live at the same time.
        if args.save_total_limit is not None:
            raise ValueError(
                f"BestAndLastCheckpointKeeper requires training.save_total_limit=null "
                f"in the YAML, got save_total_limit={args.save_total_limit}. "
                "Set it to null so HF's own _rotate_checkpoints does not preempt "
                "the union (best/last) retention this callback enforces."
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return

        output_dir = Path(args.output_dir)
        if not output_dir.is_dir():
            return

        # Discover all checkpoint dirs and tag with their step.
        ckpts: list[tuple[int, Path]] = []
        for p in output_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            try:
                step = int(p.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            ckpts.append((step, p))

        # Below max(last_n, best_k) checkpoints, neither set can be pruned —
        # whichever set is larger trivially contains all dirs.
        if len(ckpts) <= max(self.last_n, self.best_k):
            return

        ckpts.sort(key=lambda x: x[0])

        # step → metric value (most recent log entry wins on duplicates)
        metric_by_step: dict[int, float] = {}
        for entry in state.log_history:
            try:
                s = entry.get("step")
                v = entry.get(self.metric)
            except AttributeError:
                continue
            if s is None or v is None:
                continue
            try:
                metric_by_step[int(s)] = float(v)
            except (TypeError, ValueError):
                continue

        # Last-N by step
        last_steps: set[int] = (
            {step for step, _ in ckpts[-self.last_n :]} if self.last_n > 0 else set()
        )

        # Best-K by metric (only checkpoints with a recorded metric value)
        if self.best_k > 0:
            scored = [(step, metric_by_step[step]) for step, _ in ckpts if step in metric_by_step]
            scored.sort(key=lambda x: x[1], reverse=self.greater_is_better)
            best_steps = {step for step, _ in scored[: self.best_k]}
        else:
            best_steps = set()

        keep = last_steps | best_steps
        for step, path in ckpts:
            if step in keep:
                continue
            try:
                shutil.rmtree(path)
                logger.info(
                    "BestAndLastCheckpointKeeper: pruned %s "
                    "(not in best %d by %s or last %d by step)",
                    path.name,
                    self.best_k,
                    self.metric,
                    self.last_n,
                )
            except OSError as e:
                logger.warning(
                    "BestAndLastCheckpointKeeper: failed to delete %s: %s",
                    path,
                    e,
                )


# ---------------------------------------------------------------------------
# LoRA adapter save (used only by training/train_lora.py)
# ---------------------------------------------------------------------------


class PeftAdapterSaveCallback(TrainerCallback):
    """Saves the PEFT adapter (only) at every save_steps boundary.

    HF Trainer's standard FSDP save would dump the full base model + adapter
    via FULL_STATE_DICT — that's ~80 GB per save for a 3B model and defeats
    the point of LoRA. This callback hooks ``on_save`` and writes ONLY the
    adapter weights (``adapter_config.json`` + ``adapter_model.safetensors``)
    to ``{output_dir}/checkpoint-{step}/adapter/`` via ``PeftModel.save_pretrained``.

    Optimizer/scheduler/trainer state are still persisted by HF Trainer through
    its normal FSDP path, so resume continues to work.

    For inference, load the base model and then attach the adapter:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(<base_path>, ...)
        model = PeftModel.from_pretrained(base, <adapter_dir>)
    """

    def __init__(self, trainer) -> None:
        self._trainer = trainer

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # NOTE: the gather below (accelerator.get_state_dict) is a COLLECTIVE
        # operation under FSDP — every rank must call it. Don't early-return
        # on non-rank-0 before the gather, or FSDP will hang.
        wrapped = self._trainer.model
        accelerator = getattr(self._trainer, "accelerator", None)

        # Gather the full (unsharded) state dict on rank 0; other ranks see {}.
        # Without this, PEFT.save_pretrained tries to read DTensor storage and
        # safetensors blows up with "Attempted to access the data pointer on
        # an invalid python storage."
        try:
            full_state = (
                accelerator.get_state_dict(wrapped)
                if accelerator is not None
                else wrapped.state_dict()
            )
        except Exception as e:
            if state.is_world_process_zero:
                logger.exception(
                    "PeftAdapterSaveCallback: get_state_dict failed at step %d: %s",
                    state.global_step,
                    e,
                )
            return

        if not state.is_world_process_zero:
            return

        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = ckpt_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap accelerate / DDP / FSDP wrappers so we can reach the PeftModel.
        model = wrapped
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            try:
                model = accelerator.unwrap_model(wrapped)
            except Exception:
                model = wrapped
        for attr in ("module", "_orig_mod"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "save_pretrained"):
                model = inner

        if not hasattr(model, "save_pretrained"):
            logger.warning(
                "PeftAdapterSaveCallback: model has no save_pretrained method "
                "(type=%s); skipping adapter save at step %d",
                type(model).__name__,
                state.global_step,
            )
            return

        # We do NOT call ``model.save_pretrained`` because PEFT's implementation
        # runs an ``id_tensor_storage`` dedup pass over every tensor in the state
        # dict — that calls ``tensor.storage().data_ptr()``, which fails on
        # DTensors (the form FSDP-2 returns from accelerator.get_state_dict).
        # Instead: filter to LoRA params, fully-materialize any DTensors, then
        # torch.save the bin manually + emit adapter_config.json from the LoraConfig.
        try:
            from peft import get_peft_model_state_dict
        except ImportError:
            logger.warning("PeftAdapterSaveCallback: peft not importable; skipping adapter save.")
            return

        try:
            peft_state = get_peft_model_state_dict(model, state_dict=full_state)
            peft_state = {k: _to_local_cpu(v) for k, v in peft_state.items()}
            torch.save(peft_state, str(adapter_dir / "adapter_model.bin"))

            # adapter_config.json — write directly from the LoraConfig, not via
            # PeftModel.save_pretrained (which re-walks the state dict).
            peft_cfg_map = getattr(model, "peft_config", None)
            if peft_cfg_map:
                active = getattr(model, "active_adapter", None)
                peft_cfg = (peft_cfg_map.get(active) if active else None) or next(
                    iter(peft_cfg_map.values())
                )
                peft_cfg.save_pretrained(str(adapter_dir))
            else:
                logger.warning(
                    "PeftAdapterSaveCallback: model has no peft_config; "
                    "wrote weights but skipped adapter_config.json at step %d",
                    state.global_step,
                )

            logger.info(
                "Saved LoRA adapter (%d tensors) to %s",
                len(peft_state),
                adapter_dir,
            )
        except Exception as e:
            logger.exception(
                "PeftAdapterSaveCallback: failed to save adapter at step %d: %s",
                state.global_step,
                e,
            )
