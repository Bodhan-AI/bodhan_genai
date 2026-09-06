"""Datasets and the mixed-source batch sampler.

Two dataset shapes, deliberately:

*   :class:`BlobLayoutDataset` for training, over the packed cache. Meta arrays are
    loaded once in the parent and inherited by workers through fork.
*   :class:`DiskLayoutDataset` for val/test, straight from ``images/`` + ``jsons/``.
    Evaluation runs rarely and reads each page once, so the cache is not worth building
    for it — and reading the real files is a check that the cache did not drift.

Both parse labels through :func:`labels_from_doc`, so train and val cannot disagree
about what the annotation means.

:class:`MixedSourceSampler` composes **every** batch to the configured per-source ratio,
rather than shuffling a weighted pool. The difference matters under DDP: a weighted pool
gives the right ratio in expectation over an epoch, but any individual step can be almost
all one source, and with gradient accumulation across ranks that is what the optimizer
actually sees.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bodhan_genai.ocr.data.taxonomy import labels_from_doc

logger = logging.getLogger(__name__)


def _to_tensors(boxes, classes, order):
    import torch

    return {
        "class_labels": torch.as_tensor(list(classes), dtype=torch.long),
        "boxes": torch.as_tensor(list(boxes), dtype=torch.float32).reshape(-1, 4).clamp(0, 1),
        "reading_order": torch.as_tensor(list(order), dtype=torch.float32),
    }


def _empty_target():
    import torch

    return {
        "class_labels": torch.zeros(0, dtype=torch.long),
        "boxes": torch.zeros(0, 4, dtype=torch.float32),
        "reading_order": torch.zeros(0, dtype=torch.float32),
    }


def build_transform(image_size: int, *, train: bool):
    """Augmentation pipeline, or ``None`` when albumentations is not installed.

    Augmentation is a soft dependency: it improves the trained model but is not needed
    to run, test or evaluate the pipeline, and albumentations pulls in OpenCV. Declared
    in the ``ocr-train`` extra; absent, training still runs and says so.
    """
    if not train:
        return None
    try:
        import albumentations as alb
    except ImportError:
        logger.warning(
            "albumentations is not installed, so training runs without augmentation. "
            "Install the ocr-train extra to match the reference recipe."
        )
        return None

    return alb.Compose(
        [
            alb.LongestMaxSize(max_size=image_size),
            alb.PadIfNeeded(image_size, image_size, border_mode=0, value=(255, 255, 255)),
            alb.RandomBrightnessContrast(p=0.3),
            alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            alb.GaussNoise(p=0.15),
        ],
        bbox_params=alb.BboxParams(
            format="yolo",  # cxcywh normalized, which is what the taxonomy emits
            label_fields=["class_labels", "reading_order"],
            min_visibility=0.3,
        ),
    )


class BlobLayoutDataset:
    """Training pages from the packed cache."""

    def __init__(self, cache_prefix: str | Path, *, image_size: int = 1024, train: bool = True):
        from bodhan_genai.ocr.data.blob import BlobCache

        self.cache = BlobCache(cache_prefix)
        self.image_size = image_size
        self.transform = build_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def source_ids(self):
        return self.cache.source

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        image = np.asarray(self.cache.image(index))
        boxes, classes, order = self.cache.labels(index)
        boxes, classes, order = boxes.tolist(), classes.tolist(), order.tolist()

        if self.transform is not None and boxes:
            augmented = self.transform(
                image=image, bboxes=boxes, class_labels=classes, reading_order=order
            )
            image = augmented["image"]
            boxes = augmented["bboxes"]
            classes = augmented["class_labels"]
            order = augmented["reading_order"]

        target = _to_tensors(boxes, classes, order) if boxes else _empty_target()
        return {"image": image, "target": target, "stem": self.cache.stems[index]}


class DiskLayoutDataset:
    """Val/test pages straight from disk."""

    def __init__(self, manifest_path: str | Path, *, image_size: int = 1024):
        self.pages = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["pages"]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        page = self.pages[index]
        root = Path(page["src"])
        with Image.open(root / "images" / page["image"]) as handle:
            image = np.asarray(handle.convert("RGB"))
        doc = json.loads((root / "jsons" / f"{page['stem']}.json").read_text(encoding="utf-8"))
        boxes, classes, order = labels_from_doc(doc)
        target = _to_tensors(boxes, classes, order) if boxes else _empty_target()
        return {"image": image, "target": target, "stem": page["stem"]}


def make_collate(processor=None, image_size: int = 1024):
    """Collate into ``pixel_values`` + a list of per-image label dicts.

    The RT-DETR loss wants normalized ``cxcywh`` boxes and one dict per image, so the
    image processor is used only for the pixels; labels are passed through untouched
    rather than round-tripped through COCO format and back.
    """

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images = [item["image"] for item in batch]
        if processor is not None:
            encoded = processor(images=images, return_tensors="pt")
            pixel_values = encoded["pixel_values"]
        else:
            import numpy as np

            stacked = np.stack(
                [np.asarray(im, dtype="float32").transpose(2, 0, 1) / 255.0 for im in images]
            )
            pixel_values = torch.from_numpy(stacked)
        return {
            "pixel_values": pixel_values,
            "labels": [item["target"] for item in batch],
            "stems": [item["stem"] for item in batch],
        }

    return collate


class MixedSourceSampler:
    """Every batch holds the configured number of pages from each source.

    ``counts`` is derived from the per-source weights once, then held fixed, so the
    composition is identical on every rank and every step. Sources whose weight rounds
    to zero are dropped with a warning rather than silently contributing nothing.
    """

    def __init__(
        self,
        source_ids,
        weights: dict[int, float],
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        import numpy as np

        self.source_ids = np.asarray(source_ids)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.pools = {
            source: np.nonzero(self.source_ids == source)[0]
            for source in sorted(set(int(s) for s in self.source_ids))
        }
        self.counts = self._resolve_counts(weights)
        empty = [s for s, n in self.counts.items() if n == 0]
        if empty:
            logger.warning(
                "source(s) %s round to 0 pages per batch of %d and will never be sampled; "
                "raise the batch size or their weight.",
                empty,
                batch_size,
            )
        self.global_batch = sum(self.counts.values()) * num_replicas

    def _resolve_counts(self, weights: dict[int, float]) -> dict[int, int]:
        """Largest-remainder apportionment, so the counts sum to exactly the batch size."""
        total = sum(weights.get(s, 0.0) for s in self.pools) or 1.0
        exact = {s: self.batch_size * weights.get(s, 0.0) / total for s in self.pools}
        counts = {s: int(v) for s, v in exact.items()}
        remainder = self.batch_size - sum(counts.values())
        for source in sorted(exact, key=lambda s: exact[s] - counts[s], reverse=True)[:remainder]:
            counts[source] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        limiting = [
            len(self.pools[s]) // (n * self.num_replicas)
            for s, n in self.counts.items()
            if n > 0 and len(self.pools[s])
        ]
        return min(limiting) if limiting else 0

    def __iter__(self):
        import numpy as np

        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled = {s: rng.permutation(idx) for s, idx in self.pools.items()}
        cursor = dict.fromkeys(self.pools, 0)

        for _ in range(len(self)):
            batch: list[int] = []
            for source, per_batch in self.counts.items():
                take = per_batch * self.num_replicas
                if not take:
                    continue
                pool, start = shuffled[source], cursor[source]
                if start + take > len(pool):  # exhausted this epoch: reshuffle and continue
                    shuffled[source] = pool = rng.permutation(self.pools[source])
                    cursor[source] = start = 0
                batch.extend(int(i) for i in pool[start : start + take])
                cursor[source] = start + take
            # deal the global batch across ranks so each rank keeps the same ratio
            yield batch[self.rank :: self.num_replicas]
