"""
PackingTrainer: custom HuggingFace Trainer that uses SequencePackingSampler.

Overrides get_train_dataloader and get_eval_dataloader to inject the packing
sampler and collator. All other Trainer functionality (checkpointing, FSDP,
gradient accumulation, logging, etc.) is inherited unchanged.

"""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader
from transformers import Trainer

from bodhan_genai.tts.training.collator import PackingCollator
from bodhan_genai.tts.training.config import PackingConfig
from bodhan_genai.tts.training.dataset import MixedDataset, ParquetTokenDataset
from bodhan_genai.tts.training.sampler import (
    SequencePackingSampler,
    build_train_sampler,
)

logger = logging.getLogger(__name__)


class PackingTrainer(Trainer):
    """
    Trainer subclass that replaces the default DataLoader with a packing-aware one.

    Extra constructor args (passed as kwargs, not positional):
      max_seq_len: Fixed sequence length for the collator.
      pad_token_id: Token ID for padding input_ids.
      train_mixed_dataset: Dataset to use for training (overrides train_dataset).
      eval_mixed_dataset: Dataset to use for evaluation (overrides eval_dataset).

    The per_device_train_batch_size is forced to 1; effective batching is handled
    by the sampler which packs multiple sequences into each max_seq_len block.
    """

    def __init__(
        self,
        *args,
        max_seq_len: int,
        pad_token_id: int,
        dataloader_prefetch_factor: int = 4,
        train_mixed_dataset: MixedDataset | ParquetTokenDataset | None = None,
        eval_mixed_dataset: MixedDataset | ParquetTokenDataset | None = None,
        packing_config: PackingConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_seq_len = max_seq_len
        self._pad_token_id = pad_token_id
        self._dataloader_prefetch_factor = max(1, int(dataloader_prefetch_factor))
        self._train_mixed = train_mixed_dataset
        self._eval_mixed = eval_mixed_dataset
        self._packing_config = packing_config or PackingConfig()
        self._train_sampler: SequencePackingSampler | None = None
        # Running mean of PackingCollator's per-batch packing_efficiency between
        # consecutive log() calls. Drained into `logs["packing_efficiency"]` by
        # log() before callbacks run, so they can inspect or rewrite it before
        # WandbCallback applies HF's train/eval key namespacing.
        self._pack_eff_sum: float = 0.0
        self._pack_eff_count: int = 0
        # HF Trainer registers WandbCallback (via report_to) BEFORE user-supplied
        # callbacks. CallbackHandler.on_log iterates callbacks in order and
        # WandbCallback.on_log immediately calls wandb.log() — so without
        # reordering, mutations to `logs` by TrainingMetricsCallback happen
        # after wandb has already pushed and never reach the server. Move
        # WandbCallback to the tail so it consumes the fully mutated dict.
        self._reorder_wandb_callback_last()
        # Memoized eval DataLoader. HF Trainer + accelerate end up calling
        # get_eval_dataloader() once per save event even with eval_strategy='no',
        # which (without caching) builds a fresh SequencePackingSampler + 8
        # workers each time. The per-call setup created uneven per-rank pack
        # times (rank 26: 6.4s vs others' 0.4s on save #13 of train-1439.err)
        # and was the most plausible trigger for post-save NCCL stalls.
        self._cached_eval_dataloader: DataLoader | None = None

    # ------------------------------------------------------------------
    # Accessors for EpochSamplerCallback
    # ------------------------------------------------------------------

    def get_train_sampler(self) -> SequencePackingSampler | None:
        return self._train_sampler

    # ------------------------------------------------------------------
    # Wandb plumbing
    # ------------------------------------------------------------------

    def _reorder_wandb_callback_last(self) -> None:
        try:
            from transformers.integrations import WandbCallback
        except ImportError:
            return
        cbs = self.callback_handler.callbacks
        wandb_cbs = [c for c in cbs if isinstance(c, WandbCallback)]
        for wc in wandb_cbs:
            cbs.remove(wc)
            cbs.append(wc)

    def add_callback(self, callback):
        # Trainer.__init__ runs _reorder_wandb_callback_last once, but user
        # callbacks (EpochSamplerCallback, BestAndLastCheckpointKeeper) are
        # added via add_callback AFTER init —
        # they would otherwise land *after* WandbCallback in the list, so their
        # mutations of `logs` wouldn't reach wandb. Re-pin WandbCallback to the
        # tail on every add_callback so the order is always correct at train()
        # / evaluate() time. The reorder is idempotent.
        super().add_callback(callback)
        self._reorder_wandb_callback_last()

    def training_step(self, model, inputs, *args, **kwargs):
        # Strip the collator's metadata before forward. Modern HF model forward
        # signatures silently absorb unknown kwargs via **kwargs, but stripping
        # also lets us capture the value to surface in wandb.
        pe = inputs.pop("packing_efficiency", None) if isinstance(inputs, dict) else None
        if pe is not None:
            self._pack_eff_sum += float(pe)
            self._pack_eff_count += 1
        return super().training_step(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        # Drain accumulated packing efficiency into logs BEFORE super().log
        # triggers callback_handler.on_log, giving callbacks a chance to mutate
        # the raw key before WandbCallback rewrites training metrics under
        # `train/...`.
        if self._pack_eff_count > 0:
            logs["packing_efficiency"] = self._pack_eff_sum / self._pack_eff_count
            self._pack_eff_sum = 0.0
            self._pack_eff_count = 0
        super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # DataLoader overrides
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
        dataset = self._train_mixed or self.train_dataset
        if dataset is None:
            raise ValueError("No training dataset provided.")

        sampler = build_train_sampler(
            dataset=dataset,
            max_seq_len=self._max_seq_len,
            shuffle=True,
            seed=self.args.seed,
            rank=self.args.process_index,
            world_size=self.args.world_size,
            pack_backend=self._packing_config.backend,
            rank_local=self._packing_config.rank_local,
            equalize_rank_bins=self._packing_config.equalize_rank_bins,
        )
        self._train_sampler = sampler

        collator = PackingCollator(
            max_seq_len=self._max_seq_len,
            pad_token_id=self._pad_token_id,
        )

        num_workers = self.args.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=(self._dataloader_prefetch_factor if num_workers > 0 else None),
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        # Cache the default (no explicit eval_dataset) DataLoader so that the
        # repeated calls HF Trainer makes around save events don't each spawn
        # a fresh SequencePackingSampler + worker pool. An explicit eval_dataset
        # arg bypasses the cache (different dataset = different DataLoader).
        if eval_dataset is None and self._cached_eval_dataloader is not None:
            return self._cached_eval_dataloader

        dataset = eval_dataset or self._eval_mixed or self.eval_dataset
        if dataset is None:
            return super().get_eval_dataloader(eval_dataset)

        sampler = SequencePackingSampler(
            dataset=dataset,
            max_seq_len=self._max_seq_len,
            shuffle=False,
            seed=self.args.seed,
            rank=self.args.process_index,
            world_size=self.args.world_size,
        )

        collator = PackingCollator(
            max_seq_len=self._max_seq_len,
            pad_token_id=self._pad_token_id,
        )

        num_workers = self.args.dataloader_num_workers
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=(self._dataloader_prefetch_factor if num_workers > 0 else None),
        )
        if eval_dataset is None:
            self._cached_eval_dataloader = dataloader
        return dataloader
