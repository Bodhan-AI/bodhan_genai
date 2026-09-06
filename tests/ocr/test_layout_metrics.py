"""Detection and reading-order metrics.

These are scored numbers that get published, so the tests are mostly hand-computable
cases: a metric that is subtly wrong still produces plausible output, and there is
nothing downstream that would catch it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from bodhan_genai.ocr.eval.metrics import (
    DetectionAccumulator,
    average_precision,
    dense_ranks,
    iou_matrix,
    kendall_tau,
    normalized_edit_distance,
    pairwise_accuracy,
    raster_order,
    xyxy_from_cxcywh,
)

GT_BOXES = [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
GT_CLASSES = [0, 1]


def test_iou_identical_and_disjoint():
    assert iou_matrix(GT_BOXES, GT_BOXES)[0, 0] == pytest.approx(1.0)
    assert iou_matrix([[0, 0, 1, 1]], [[5, 5, 6, 6]])[0, 0] == 0.0


def test_iou_half_overlap():
    # [0,0,2,1] vs [1,0,3,1]: intersection 1, union 3
    assert iou_matrix([[0, 0, 2, 1]], [[1, 0, 3, 1]])[0, 0] == pytest.approx(1 / 3)


def test_xyxy_conversion():
    assert xyxy_from_cxcywh([[0.5, 0.5, 0.2, 0.4]])[0].tolist() == pytest.approx(
        [0.4, 0.3, 0.6, 0.7]
    )


def test_a_perfect_detector_scores_exactly_one():
    """Regression: the closing sentinel used to sit on the 101-point grid and cap AP at
    100/101 = 0.990, so a flawless run silently looked like a 1% miss."""
    accumulator = DetectionAccumulator()
    accumulator.add(GT_BOXES, GT_CLASSES, [0.9, 0.8], GT_BOXES, GT_CLASSES)
    summary = accumulator.summary()
    assert summary["mAP50"] == pytest.approx(1.0)
    assert summary["mAP50-95"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)


def test_average_precision_of_a_perfect_curve_is_one():
    assert average_precision([0.5, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_missing_one_of_two_classes_halves_map():
    accumulator = DetectionAccumulator()
    accumulator.add([GT_BOXES[0]], [0], [0.9], GT_BOXES, GT_CLASSES)
    assert accumulator.summary()["mAP50"] == pytest.approx(0.5)


def test_a_false_positive_costs_precision():
    accumulator = DetectionAccumulator()
    accumulator.add(
        [*GT_BOXES, [9.0, 9.0, 10.0, 10.0]],
        [*GT_CLASSES, 0],
        [0.9, 0.8, 0.95],
        GT_BOXES,
        GT_CLASSES,
    )
    summary = accumulator.summary()
    assert summary["precision"] < 1.0
    assert summary["mAP50"] < 1.0


def test_no_predictions_scores_zero_not_an_error():
    accumulator = DetectionAccumulator()
    accumulator.add([], [], [], GT_BOXES, GT_CLASSES)
    assert accumulator.summary()["mAP50"] == 0.0


def test_predictions_with_no_ground_truth_are_all_false_positives():
    accumulator = DetectionAccumulator()
    accumulator.add(GT_BOXES, GT_CLASSES, [0.9, 0.8], [], [])
    assert accumulator.summary()["mAP50"] == 0.0


def test_right_box_wrong_class_does_not_match():
    accumulator = DetectionAccumulator()
    accumulator.add(GT_BOXES, [1, 0], [0.9, 0.8], GT_BOXES, GT_CLASSES)
    assert accumulator.summary()["mAP50"] == 0.0


def test_loose_boxes_pass_at_iou50_but_not_at_higher_thresholds():
    # shifted so IoU is comfortably above 0.5 but well below 0.95
    loose = [[0.05, 0.05, 1.05, 1.05], [2.05, 2.05, 3.05, 3.05]]
    accumulator = DetectionAccumulator()
    accumulator.add(loose, GT_CLASSES, [0.9, 0.8], GT_BOXES, GT_CLASSES)
    summary = accumulator.summary()
    assert summary["mAP50"] == pytest.approx(1.0)
    assert summary["mAP50-95"] < summary["mAP50"]


def test_per_class_breakdown_is_reported():
    accumulator = DetectionAccumulator()
    accumulator.add(GT_BOXES, GT_CLASSES, [0.9, 0.8], GT_BOXES, GT_CLASSES)
    per_class = accumulator.summary({0: "Question", 1: "Paragraph"})["per_class"]
    assert set(per_class) == {"Question", "Paragraph"}
    assert per_class["Question"]["n"] == 1


def test_two_predictions_cannot_claim_the_same_box():
    """One-to-one matching: a duplicate detection is a false positive, not a second hit."""
    accumulator = DetectionAccumulator()
    accumulator.add([GT_BOXES[0], GT_BOXES[0]], [0, 0], [0.9, 0.8], [GT_BOXES[0]], [0])
    assert accumulator.summary()["precision"] < 1.0


# --------------------------------------------------------------------------- #
# Reading order
# --------------------------------------------------------------------------- #


def test_kendall_tau_perfect_and_reversed():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau([3, 2, 1, 0], [0, 1, 2, 3]) == pytest.approx(-1.0)


def test_kendall_tau_degenerate_inputs_return_zero_not_nan():
    assert kendall_tau([1], [1]) == 0.0
    assert kendall_tau([1, 1, 1], [0, 1, 2]) == 0.0


def test_normalized_edit_distance_bounds():
    assert normalized_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert normalized_edit_distance([1, 2, 3], [4, 5, 6]) == pytest.approx(1.0)
    assert normalized_edit_distance([], []) == 0.0


def test_pairwise_accuracy_bounds():
    assert pairwise_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert pairwise_accuracy([2, 1, 0], [0, 1, 2]) == 0.0
    assert pairwise_accuracy([0], [0]) == 1.0


def test_pairwise_accuracy_ignores_tied_ground_truth():
    """Tied blocks have no correct relative order, so they must not be scored."""
    assert pairwise_accuracy([0, 1], [5, 5]) == 1.0


def test_raster_reads_rows_left_to_right():
    # two rows of two columns, given out of order
    boxes = [
        [0.75, 0.8, 0.4, 0.1],  # bottom-right
        [0.25, 0.2, 0.4, 0.1],  # top-left
        [0.75, 0.2, 0.4, 0.1],  # top-right
        [0.25, 0.8, 0.4, 0.1],  # bottom-left
    ]
    assert raster_order(boxes) == [3, 0, 1, 2]


def test_raster_tolerates_a_slightly_tilted_row():
    boxes = [[0.25, 0.200, 0.4, 0.1], [0.75, 0.205, 0.4, 0.1]]
    assert raster_order(boxes) == [0, 1], "a 0.005 skew must not read as two rows"


def test_raster_of_nothing_is_empty():
    assert raster_order([]) == []


def test_dense_ranks_normalizes_gaps_and_order():
    assert dense_ranks([10, 5, 30]) == [1, 0, 2]
    assert dense_ranks([3, 1, 2]) == [2, 0, 1]
