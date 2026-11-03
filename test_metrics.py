"""Unit tests for metrics and experiment partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from demo import (
    COMPARISON_INCLUDES,
    compare_result_dirs,
    generate_dataset,
    make_calibrated_classifier,
    numerical_equal,
    stratified_disjoint_partitions,
)
from sklearn.naive_bayes import GaussianNB
from metrics import (
    UNAVAILABLE,
    accuracy,
    brier_score,
    confidence,
    log_loss,
    selective_prediction,
)


def test_brier_score_hand_calculated_example() -> None:
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.2, 0.6, 0.4])
    # ((0.8-1)^2 + (0.2-0)^2 + (0.6-1)^2 + (0.4-0)^2) / 4 = 0.1
    assert brier_score(labels, probabilities) == pytest.approx(0.1)


def test_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        brier_score([], [])
    with pytest.raises(ValueError, match="non-empty"):
        log_loss(np.array([]), np.array([]))


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        accuracy([0, 1], [0.1])


def test_rejects_out_of_range_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0, 1], [-0.1, 0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([0, 1], [0.2, 1.01])


def test_rejects_nonfinite_probabilities() -> None:
    with pytest.raises(ValueError, match="finite"):
        log_loss([0, 1], [0.2, np.nan])
    with pytest.raises(ValueError, match="finite"):
        accuracy([0, 1], [0.2, np.inf])


def test_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        brier_score([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        brier_score([0, np.nan], [0.1, 0.2])


def test_confidence_is_max_p_or_one_minus_p() -> None:
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(confidence(p), [0.9, 0.5, 0.9])


def test_threshold_boundary_includes_equality() -> None:
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.2, 0.7, 0.3])
    # confidences: 0.8, 0.8, 0.7, 0.7
    at_point = selective_prediction(labels, probabilities, 0.70)
    assert at_point["accepted_count"] == 4
    assert at_point["coverage"] == pytest.approx(1.0)
    just_above = selective_prediction(labels, probabilities, 0.80)
    assert just_above["accepted_count"] == 2
    assert just_above["coverage"] == pytest.approx(0.5)


def test_threshold_rejects_values_outside_unit_confidence_range() -> None:
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    with pytest.raises(ValueError, match=r"\[0.5, 1.0\]"):
        selective_prediction(labels, probabilities, 0.49)
    with pytest.raises(ValueError, match=r"\[0.5, 1.0\]"):
        selective_prediction(labels, probabilities, 1.01)


def test_zero_coverage_reports_unavailable_metrics() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.5, 0.5, 0.5, 0.5])
    result = selective_prediction(labels, probabilities, 0.60)
    assert result["accepted_count"] == 0
    assert result["coverage"] == pytest.approx(0.0)
    assert result["accepted_accuracy"] == UNAVAILABLE
    assert result["accepted_error_rate"] == UNAVAILABLE


def test_threshold_one_accepts_only_hard_certainties() -> None:
    labels = np.array([1, 0, 1])
    probabilities = np.array([1.0, 0.0, 0.99])
    result = selective_prediction(labels, probabilities, 1.0)
    assert result["accepted_count"] == 2
    assert result["accepted_accuracy"] == pytest.approx(1.0)


def test_partitions_are_mutually_disjoint_and_complete() -> None:
    X, y = generate_dataset(42)
    parts = stratified_disjoint_partitions(X, y, 42)
    train = set(parts["train_idx"].tolist())
    calib = set(parts["calib_idx"].tolist())
    test = set(parts["test_idx"].tolist())
    assert train.isdisjoint(calib)
    assert train.isdisjoint(test)
    assert calib.isdisjoint(test)
    assert train | calib | test == set(range(X.shape[0]))
    assert parts["counts"]["n_train"] == 6000
    assert parts["counts"]["n_calib"] == 2000
    assert parts["counts"]["n_test"] == 2000


def test_partitions_are_reproducible() -> None:
    X, y = generate_dataset(44)
    a = stratified_disjoint_partitions(X, y, 44)
    b = stratified_disjoint_partitions(X, y, 44)
    np.testing.assert_array_equal(a["train_idx"], b["train_idx"])
    np.testing.assert_array_equal(a["calib_idx"], b["calib_idx"])
    np.testing.assert_array_equal(a["test_idx"], b["test_idx"])


def test_dataset_generation_is_reproducible() -> None:
    X1, y1 = generate_dataset(46)
    X2, y2 = generate_dataset(46)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_calibrator_uses_the_same_fitted_base_model() -> None:
    X, y = generate_dataset(42)
    parts = stratified_disjoint_partitions(X, y, 42)
    base = GaussianNB()
    base.fit(parts["X_train"], parts["y_train"])
    theta_before = base.theta_.copy()
    calibrated = make_calibrated_classifier(base)
    assert calibrated.estimator.estimator is base
    calibrated.fit(parts["X_calib"], parts["y_calib"])
    np.testing.assert_array_equal(base.theta_, theta_before)
    np.testing.assert_allclose(
        base.predict_proba(parts["X_test"]),
        calibrated.estimator.predict_proba(parts["X_test"]),
    )


def test_numerical_equal_detects_selective_metric_changes() -> None:
    original = {
        "methods": {
            "raw": {
                "selective": [
                    {"threshold": 0.9, "coverage": 0.5562, "accepted_error_rate": 0.0899}
                ]
            }
        }
    }
    changed_coverage = {
        "methods": {
            "raw": {
                "selective": [
                    {"threshold": 0.9, "coverage": 0.3351, "accepted_error_rate": 0.0899}
                ]
            }
        }
    }
    assert numerical_equal(original, original)
    assert not numerical_equal(original, changed_coverage)


def test_compare_result_dirs_covers_threshold_and_coverage_tables() -> None:
    results = Path("results")
    report = compare_result_dirs(results, results)
    assert report["equal"] is True
    joined = " ".join(report["files_compared"])
    assert "all_seeds.json" in joined
    assert "per_seed_selective.csv" in joined
    assert "summary_selective.csv" in joined
    assert "seed_42/metrics.json" in joined
    assert "seed_42/selective_prediction.csv" in joined
    include_text = " ".join(report["includes"])
    assert "coverage" in include_text
    assert "accepted_error_rate" in include_text
    assert any("threshold" in item for item in COMPARISON_INCLUDES)
