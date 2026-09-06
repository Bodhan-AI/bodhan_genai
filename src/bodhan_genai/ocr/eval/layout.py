"""Score a layout checkpoint: detection mAP and reading order.

    python -m bodhan_genai.ocr.eval.layout --ckpt runs/layout/final --manifest layout_test.json

This drives ``IndicDocLayout`` — the same backend inference and serving use — rather than
calling the model directly. An eval that runs its own forward pass measures a pipeline
nobody ships: it silently skips the confidence threshold, the dedup rules and the order
decode, which is where several of this stack's sharp edges live.

Reading order is scored only over predictions matched to ground truth at IoU >= 0.5.
Ranking boxes the model invented, or missing ones it never found, is a detection failure
and is already counted as such; folding it into the order number would double-count it
and make the two metrics move together for no reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bodhan_genai.ocr.data.taxonomy import ID2LABEL, LABEL2ID, labels_from_doc
from bodhan_genai.ocr.eval.metrics import (
    DetectionAccumulator,
    dense_ranks,
    iou_matrix,
    kendall_tau,
    normalized_edit_distance,
    pairwise_accuracy,
    raster_order,
    xyxy_from_cxcywh,
)

logger = logging.getLogger(__name__)

# A page whose ground-truth order already equals the raster order tells you nothing about
# the order head: raster gets it right for free. The "hard" slice is everything else.
HARD_TAU_CEILING = 0.999


@dataclass
class OrderAccumulator:
    """Reading-order metrics, over all pages and over the non-raster slice."""

    pages: int = 0
    hard_pages: int = 0
    totals: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ("tau_model", "tau_raster", "ned_model", "ned_raster", "pairwise",
         "tau_model_hard", "tau_raster_hard", "pairwise_hard"), 0.0,
    ))  # fmt: skip

    def add(self, predicted_ranks, raster_ranks, true_ranks) -> None:
        if len(true_ranks) < 2:
            return
        truth = dense_ranks(true_ranks)
        model = dense_ranks(predicted_ranks)
        raster = dense_ranks(raster_ranks)

        tau_model = kendall_tau(model, truth)
        tau_raster = kendall_tau(raster, truth)
        self.pages += 1
        self.totals["tau_model"] += tau_model
        self.totals["tau_raster"] += tau_raster
        self.totals["ned_model"] += normalized_edit_distance(model, truth)
        self.totals["ned_raster"] += normalized_edit_distance(raster, truth)
        self.totals["pairwise"] += pairwise_accuracy(model, truth)

        if tau_raster < HARD_TAU_CEILING:
            self.hard_pages += 1
            self.totals["tau_model_hard"] += tau_model
            self.totals["tau_raster_hard"] += tau_raster
            self.totals["pairwise_hard"] += pairwise_accuracy(model, truth)

    def summary(self) -> dict[str, float | int]:
        pages = max(self.pages, 1)
        hard = max(self.hard_pages, 1)
        out: dict[str, float | int] = {"order_pages": self.pages, "hard_pages": self.hard_pages}
        for key, total in self.totals.items():
            out[key] = total / (hard if key.endswith("_hard") else pages)
        return out


def _predicted_blocks(backend, image_path: Path) -> list[Any]:
    blocks = backend.detect(str(image_path))
    return list(blocks) if blocks is not None else []


def evaluate(
    checkpoint: str,
    manifest_path: str | Path,
    *,
    confidence: float = 0.5,
    image_size: int = 1024,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    """Run the layout backend over a manifest and score it."""
    from bodhan_genai.ocr import IndicDocLayout, LayoutConfig

    pages = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["pages"]
    if limit:
        pages = pages[:limit]

    detection = DetectionAccumulator()
    order = OrderAccumulator()
    skipped = 0

    layout = IndicDocLayout(
        checkpoint, config=LayoutConfig(conf=confidence, img_size=image_size, device=device)
    )
    try:
        for page in pages:
            root = Path(page["src"])
            try:
                doc = json.loads(
                    (root / "jsons" / f"{page['stem']}.json").read_text(encoding="utf-8")
                )
                blocks = _predicted_blocks(layout, root / "images" / page["image"])
            except Exception as exc:
                skipped += 1
                logger.warning("skip %s: %s", page["stem"], exc)
                continue

            gt_boxes_cxcywh, gt_classes, gt_order = labels_from_doc(doc)
            if not gt_boxes_cxcywh:
                continue

            pred_boxes = [b.bbox_xyxy for b in blocks]
            pred_classes = [LABEL2ID.get(b.label, -1) for b in blocks]
            pred_scores = [getattr(b, "conf", 1.0) for b in blocks]
            gt_boxes = xyxy_from_cxcywh(gt_boxes_cxcywh)

            detection.add(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)

            if len(blocks) >= 2:
                iou = iou_matrix(pred_boxes, gt_boxes)
                pairs = [
                    (p, int(iou[p].argmax()))
                    for p in range(len(blocks))
                    if iou.shape[1] and iou[p].max() >= 0.5
                ]
                # one prediction per ground-truth box, best IoU wins
                best: dict[int, int] = {}
                for p, g in sorted(pairs, key=lambda pg: -iou[pg[0], pg[1]]):
                    best.setdefault(g, p)
                if len(best) >= 2:
                    gt_indices = sorted(best)
                    order.add(
                        [blocks[best[g]].order for g in gt_indices],
                        [raster_order(gt_boxes_cxcywh)[g] for g in gt_indices],
                        [gt_order[g] for g in gt_indices],
                    )
    finally:
        close = getattr(layout, "close", None)
        if callable(close):
            close()

    result = {"pages": len(pages), "skipped": skipped}
    result.update(detection.summary(ID2LABEL))
    result.update(order.summary())
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.eval.layout",
        description="Detection mAP and reading-order metrics for a layout checkpoint.",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="write metrics.json here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = evaluate(
        args.ckpt,
        args.manifest,
        confidence=args.conf,
        image_size=args.img_size,
        device=args.device,
        limit=args.limit,
    )
    per_class = metrics.pop("per_class", {})
    print(json.dumps(metrics, indent=2))
    if args.out:
        metrics["per_class"] = per_class
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
