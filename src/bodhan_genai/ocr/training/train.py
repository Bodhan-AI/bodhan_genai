"""Train IndicDocLayout: ``python -m bodhan_genai.ocr.training.train --config ...``.

Detection and reading order are learned jointly in one pass — see
:mod:`bodhan_genai.ocr.training.modeling` for why the order head is warm-started rather
than trained from scratch.

Two details that are easy to get wrong and expensive to discover late:

*   **The backbone gets its own, lower learning rate.** It arrives document-pretrained;
    driving it at the head's rate erases that in the first few hundred steps and the run
    never recovers to the warm-start's quality.
*   **Gradients are clipped hard (0.1).** RT-DETR's Hungarian matching makes the loss
    discontinuous — a step that flips an assignment produces a very large gradient — and
    without tight clipping a single such batch can wreck the run.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from bodhan_genai.ocr.training.config import LayoutTrainConfig

logger = logging.getLogger(__name__)


def _resolve_source_weights(cache, weights_by_name: dict[str, float]) -> dict[int, float]:
    """Map the config's source *names* onto the cache's integer source ids."""
    names = list(cache.source_names)
    if not names:
        raise ValueError(
            "this cache carries no source names, so per-source weights cannot be applied; "
            "repack it with a current bodhan_genai.ocr.data.blob"
        )
    unknown = set(weights_by_name) - set(names)
    if unknown:
        raise ValueError(f"source_weights names {sorted(unknown)} are not in the cache: {names}")
    if not weights_by_name:  # uniform over whatever the cache holds
        return {i: 1.0 for i, _ in enumerate(names)}
    return {i: weights_by_name.get(name, 0.0) for i, name in enumerate(names)}


def _build_optimizer(model, config: LayoutTrainConfig):
    import torch

    backbone, rest = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "backbone" in name else rest).append(parameter)
    logger.info("optimizer: %d backbone tensors, %d head tensors", len(backbone), len(rest))
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.backbone_learning_rate},
            {"params": rest, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def _lr_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(config: LayoutTrainConfig) -> Path:
    """Run training. Returns the output directory."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import RTDetrImageProcessorFast

    from bodhan_genai.ocr.training.dataset import (
        BlobLayoutDataset,
        MixedSourceSampler,
        make_collate,
    )
    from bodhan_genai.ocr.training.ema import ModelEma
    from bodhan_genai.ocr.training.modeling import build_model

    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accum,
        mixed_precision="bf16" if config.bf16 else "no",
        log_with="wandb" if config.wandb_project else None,
    )
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(config.as_dict(), indent=2), encoding="utf-8"
        )
    if config.wandb_project:
        accelerator.init_trackers(
            config.wandb_project,
            config=config.as_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

    dataset = BlobLayoutDataset(config.cache_prefix, image_size=config.image_size, train=True)
    sampler = MixedSourceSampler(
        dataset.source_ids,
        _resolve_source_weights(dataset.cache, config.source_weights),
        batch_size=config.batch_size,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )
    processor = RTDetrImageProcessorFast(
        size={"height": config.image_size, "width": config.image_size}
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate(processor, config.image_size),
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    model = build_model(config.checkpoint, lambda_order=config.lambda_order)
    optimizer = _build_optimizer(model, config)

    steps_per_epoch = max(1, len(sampler) // config.grad_accum)
    total_steps = config.max_steps or steps_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, warmup=config.warmup_steps, total=total_steps),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    ema = None
    if config.use_ema:
        ema = ModelEma(model, decay=config.ema_decay, warmup=config.ema_warmup)

    logger.info(
        "training: %d pages, %d batches/epoch, %d optimizer steps, %d process(es)",
        len(dataset),
        len(sampler),
        total_steps,
        accelerator.num_processes,
    )

    step, started = 0, time.perf_counter()
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if ema is not None:
                    ema.update(model)
                if step % config.log_every == 0:
                    rate = step / (time.perf_counter() - started)
                    logger.info(
                        "epoch %d step %d/%d loss %.4f lr %.2e %.2f steps/s",
                        epoch,
                        step,
                        total_steps,
                        outputs.loss.item(),
                        scheduler.get_last_lr()[-1],
                        rate,
                    )
                    accelerator.log({"loss": outputs.loss.item(), "step": step}, step=step)
                if config.max_steps and step >= config.max_steps:
                    break
        if config.max_steps and step >= config.max_steps:
            break

        if accelerator.is_main_process and (epoch + 1) % config.save_every_epochs == 0:
            _save(accelerator, model, ema, processor, output_dir / f"epoch{epoch + 1}")

    if accelerator.is_main_process:
        _save(accelerator, model, ema, processor, output_dir / "final")
    accelerator.end_training()
    return output_dir


def _save(accelerator, model, ema, processor, destination: Path) -> None:
    """Write the live weights, and the EMA shadow beside them.

    Both, deliberately: EMA is usually the better model but not always, and re-running a
    multi-day job to find out is not an option.
    """
    destination.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    if ema is not None:
        import copy

        shadow_model = copy.deepcopy(unwrapped)
        ema.copy_to(shadow_model)
        shadow_model.save_pretrained(destination / "ema", safe_serialization=True)
        processor.save_pretrained(destination / "ema")
    logger.info("saved %s", destination)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.training.train",
        description="Train IndicDocLayout (detection + reading order).",
    )
    parser.add_argument("--config", required=True, help="see configs/ocr/train/layout.yaml")
    parser.add_argument("--cache-prefix", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke-test a recipe")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = LayoutTrainConfig.from_yaml(
        args.config,
        cache_prefix=args.cache_prefix,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    train(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
