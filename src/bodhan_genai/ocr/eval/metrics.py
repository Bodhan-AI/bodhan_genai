"""Detection and reading-order metrics.

Detection is scored COCO-style: greedy one-to-one matching by descending confidence at
each of ten IoU thresholds from 0.50 to 0.95, then 101-point interpolated AP per class,
averaged over classes present in the ground truth.

AP is implemented here rather than pulled from ``ultralytics`` or ``pycocotools``. Both
would be a heavyweight dependency (OpenCV, a C extension) for one function, and both
differ from each other in the details — absent classes, the interpolation grid, how ties
are broken. A local implementation is one that can be unit-tested against hand-computed
cases, which is what makes the number comparable across runs.

Reading order gets three numbers because they fail differently:

*   **Kendall-tau** — global correlation. Insensitive to a single block moved far.
*   **Normalized edit distance** — sensitive to exactly that.
*   **Pairwise accuracy** — the fraction of block pairs ordered correctly, which is what
    the order head is actually trained on.

Each is also reported against a **raster baseline** (top-to-bottom, left-to-right).
Raster is very strong on single-column pages, so a model that has learned nothing still
scores well in aggregate; the gap over raster, and the score on the non-raster slice, are
the numbers that mean something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


def xyxy_from_cxcywh(boxes):
    import numpy as np

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def iou_matrix(a, b):
    """``[n, m]`` IoU between two sets of xyxy boxes."""
    import numpy as np

    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(bottom_right - top_left, 0, None).prod(-1)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(recall, precision) -> float:
    """101-point interpolated AP over a precision/recall curve.

    The closing ``precision = 0`` sentinel sits just *past* the achieved recall rather
    than at 1.0. Placing it exactly at 1.0 puts a discontinuity on the last grid point,
    and ``np.interp`` takes the right-hand value there — so a detector that finds
    everything at full precision scores 100/101 instead of 1.0.
    """
    import numpy as np

    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    beyond = (float(recall[-1]) if len(recall) else 0.0) + 1e-3
    recall = np.concatenate(([0.0], recall, [beyond]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    # make precision monotonically decreasing, so a later spike cannot inflate AP
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    return float(np.interp(grid, recall, precision).mean())


@dataclass
class DetectionAccumulator:
    """Collects per-page matches, then reduces to AP once at the end.

    AP is not an average of per-page APs — a page with one object would count as much as
    a page with fifty. Matches are pooled across the whole set and the curve is computed
    once.
    """

    records: list = field(default_factory=list)  # (confidence, pred_class, matched[10])
    gt_counts: dict = field(default_factory=dict)

    def add(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes) -> None:
        import numpy as np

        for class_id in np.asarray(gt_classes, dtype=int):
            self.gt_counts[int(class_id)] = self.gt_counts.get(int(class_id), 0) + 1

        pred_classes = np.asarray(pred_classes, dtype=int)
        pred_scores = np.asarray(pred_scores, dtype=float)
        gt_classes = np.asarray(gt_classes, dtype=int)
        if not len(pred_classes):
            return

        iou = iou_matrix(pred_boxes, gt_boxes) if len(gt_classes) else None
        by_confidence = np.argsort(-pred_scores)
        matched = np.zeros((len(pred_scores), len(IOU_THRESHOLDS)), dtype=bool)

        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            if iou is None:
                break  # no ground truth on this page: every prediction is a false positive
            taken: set[int] = set()
            # descending confidence, so the strongest prediction claims a box first
            for pred_index in by_confidence:
                candidates = [
                    gt_index
                    for gt_index in range(len(gt_classes))
                    if gt_index not in taken
                    and gt_classes[gt_index] == pred_classes[pred_index]
                    and iou[pred_index, gt_index] >= threshold
                ]
                if candidates:
                    best = max(candidates, key=lambda g, p=pred_index: iou[p, g])
                    taken.add(best)
                    matched[pred_index, threshold_index] = True

        for pred_index in range(len(pred_scores)):
            self.records.append(
                (
                    float(pred_scores[pred_index]),
                    int(pred_classes[pred_index]),
                    matched[pred_index].tolist(),
                )
            )

    def summary(self, id2label: dict[int, str] | None = None) -> dict:
        import numpy as np

        result = {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0, "per_class": {}}
        if not self.records or not self.gt_counts:
            return result

        by_class: dict[int, list] = {}
        for score, class_id, matched in self.records:
            by_class.setdefault(class_id, []).append((score, matched))

        aps, precisions, recalls = [], [], []
        for class_id, total in sorted(self.gt_counts.items()):
            entries = sorted(by_class.get(class_id, []), key=lambda e: -e[0])
            if not entries:
                aps.append([0.0] * len(IOU_THRESHOLDS))
                continue
            flags = np.array([e[1] for e in entries], dtype=bool)  # [n_pred, n_thresholds]
            per_threshold = []
            for threshold_index in range(len(IOU_THRESHOLDS)):
                true_positive = np.cumsum(flags[:, threshold_index])
                false_positive = np.cumsum(~flags[:, threshold_index])
                recall = true_positive / max(total, 1)
                precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
                per_threshold.append(average_precision(recall, precision))
                if threshold_index == 0:
                    precisions.append(float(precision[-1]))
                    recalls.append(float(recall[-1]))
            aps.append(per_threshold)
            if id2label is not None:
                result["per_class"][id2label.get(class_id, str(class_id))] = {
                    "AP50": per_threshold[0],
                    "AP50-95": float(np.mean(per_threshold)),
                    "n": total,
                }

        ap_array = np.asarray(aps, dtype=float)
        result["mAP50"] = float(ap_array[:, 0].mean())
        result["mAP50-95"] = float(ap_array.mean())
        result["precision"] = float(np.mean(precisions)) if precisions else 0.0
        result["recall"] = float(np.mean(recalls)) if recalls else 0.0
        return result


def kendall_tau(predicted_ranks, true_ranks) -> float:
    """Kendall's tau-b. Returns 0.0 for sequences too short or fully tied."""
    from scipy.stats import kendalltau

    if len(predicted_ranks) < 2:
        return 0.0
    tau = kendalltau(predicted_ranks, true_ranks).correlation
    return 0.0 if tau != tau else float(tau)  # NaN when one side is constant


def normalized_edit_distance(a, b) -> float:
    """Levenshtein between two sequences, divided by the longer length."""
    a, b = list(a), list(b)
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item_a != item_b))
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def pairwise_accuracy(predicted_ranks, true_ranks) -> float:
    """Fraction of block pairs placed in the correct relative order."""
    n = len(true_ranks)
    if n < 2:
        return 1.0
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true_ranks[i] == true_ranks[j]:
                continue
            total += 1
            same = (predicted_ranks[i] < predicted_ranks[j]) == (true_ranks[i] < true_ranks[j])
            correct += int(same)
    return correct / total if total else 1.0


def raster_order(boxes, *, row_tolerance: float = 0.02) -> list[int]:
    """Top-to-bottom, left-to-right ranks — the baseline every model must beat.

    ``row_tolerance`` groups boxes whose centres are within that fraction of the page
    height into the same visual row, so a slightly-tilted scan does not read as a
    zig-zag.
    """
    import numpy as np

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return []
    centre_y, centre_x = boxes[:, 1], boxes[:, 0]
    order = sorted(range(len(boxes)), key=lambda i: (centre_y[i], centre_x[i]))
    rows, current, anchor = [], [order[0]], centre_y[order[0]]
    for index in order[1:]:
        if abs(centre_y[index] - anchor) <= row_tolerance:
            current.append(index)
        else:
            rows.append(current)
            current, anchor = [index], centre_y[index]
    rows.append(current)

    ranks = [0] * len(boxes)
    position = 0
    for row in rows:
        for index in sorted(row, key=lambda i: centre_x[i]):
            ranks[index] = position
            position += 1
    return ranks


def dense_ranks(order) -> list[int]:
    """Arbitrary rank values to 0..n-1, so gaps and duplicates do not distort a metric."""
    import numpy as np

    order = np.asarray(order)
    return np.argsort(np.argsort(order)).tolist()
