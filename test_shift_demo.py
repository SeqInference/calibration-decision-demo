"""Tests for synthetic feature-corruption utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from demo import generate_dataset, make_calibrated_classifier, stratified_disjoint_partitions
from metrics import matched_coverage_selection
from shift_demo import (
    NOISE_SCALES,
    corrupt_features,
    corruption_generator,
    snapshot_gaussiannb,
    snapshots_equal,
    standard_normal_noise,
    training_feature_stds,
    validate_noise_scale,
)


def test_zero_noise_reproduces_clean_test_features() -> None:
    X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    stds = np.array([10.0, 20.0])
    noise = np.ones_like(X_test)
    corrupted = corrupt_features(X_test, stds, noise, 0.0)
    np.testing.assert_array_equal(corrupted, X_test)
    assert corrupted is not X_test


def test_source_features_and_labels_remain_unchanged() -> None:
    X, y = generate_dataset(42)
    X_before = X.copy()
    y_before = y.copy()
    parts = stratified_disjoint_partitions(X, y, 42)
    stds = training_feature_stds(parts["X_train"])
    noise = np.ones_like(parts["X_test"])
    _ = corrupt_features(parts["X_test"], stds, noise, 1.0)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    np.testing.assert_array_equal(parts["X_test"], X[parts["test_idx"]])
    np.testing.assert_array_equal(parts["y_test"], y[parts["test_idx"]])


def test_training_data_alone_determines_corruption_scales() -> None:
    X_train = np.array([[0.0, 0.0], [2.0, 8.0], [4.0, 16.0]], dtype=float)
    X_test = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    X_test_before = X_test.copy()
    stds = training_feature_stds(X_train)
    np.testing.assert_allclose(stds, np.array([np.std([0.0, 2.0, 4.0]), np.std([0.0, 8.0, 16.0])]))
    noise = np.ones_like(X_test)
    corrupted = corrupt_features(X_test, stds, noise, 1.0)
    np.testing.assert_allclose(corrupted, X_test + stds)
    np.testing.assert_array_equal(X_test, X_test_before)
    wrong_stds = np.std(X_test, axis=0)
    assert not np.allclose(stds, wrong_stds)


def test_corruption_rng_is_reproducible_and_independent_per_seed() -> None:
    a = standard_normal_noise(corruption_generator(42), (5, 3))
    b = standard_normal_noise(corruption_generator(42), (5, 3))
    c = standard_normal_noise(corruption_generator(43), (5, 3))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_same_noise_array_is_reused_across_severity_by_scaling() -> None:
    X_test = np.arange(12, dtype=float).reshape(4, 3)
    stds = np.array([0.5, 1.0, 2.0])
    noise = standard_normal_noise(corruption_generator(42), X_test.shape)
    low = corrupt_features(X_test, stds, noise, 0.25)
    high = corrupt_features(X_test, stds, noise, 1.00)
    np.testing.assert_allclose(high - X_test, 4.0 * (low - X_test))


def test_invalid_noise_scales_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_noise_scale(np.nan)
    with pytest.raises(ValueError, match="finite"):
        validate_noise_scale(np.inf)
    with pytest.raises(ValueError, match=">= 0"):
        validate_noise_scale(-0.01)
    X_test = np.zeros((2, 2))
    stds = np.ones(2)
    noise = np.zeros((2, 2))
    with pytest.raises(ValueError, match=">= 0"):
        corrupt_features(X_test, stds, noise, -1.0)


def test_model_parameters_remain_unchanged_across_conditions() -> None:
    X, y = generate_dataset(42)
    parts = stratified_disjoint_partitions(X, y, 42)
    base = GaussianNB()
    base.fit(parts["X_train"], parts["y_train"])
    calibrated = make_calibrated_classifier(base)
    calibrated.fit(parts["X_calib"], parts["y_calib"])
    before = snapshot_gaussiannb(base)
    stds = training_feature_stds(parts["X_train"])
    noise = standard_normal_noise(corruption_generator(42), parts["X_test"].shape)
    for scale in NOISE_SCALES:
        X_corrupted = corrupt_features(parts["X_test"], stds, noise, scale)
        _ = base.predict_proba(X_corrupted)
        _ = calibrated.predict_proba(X_corrupted)
        assert snapshots_equal(before, snapshot_gaussiannb(base))


def test_matched_coverage_counts() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    probabilities = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.40, 0.20, 0.05])
    indices = np.arange(8)
    half = matched_coverage_selection(labels, probabilities, 0.50, indices)
    assert half["accepted_count"] == 4
    assert half["realized_coverage"] == pytest.approx(0.5)
    full = matched_coverage_selection(labels, probabilities, 1.00, indices)
    assert full["accepted_count"] == 8
    assert full["accepted_error_rate"] == pytest.approx(0.375)


def test_matched_coverage_breaks_ties_by_original_row_index() -> None:
    labels = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.9, 0.6])
    indices = np.array([10, 2, 5])
    one = matched_coverage_selection(labels, probabilities, 1.0 / 3.0, indices)
    assert one["accepted_count"] == 1
    # Both first two have confidence 0.9; smaller original index (2) wins.
    assert one["selected_row_indices"] == [2]
    two = matched_coverage_selection(labels, probabilities, 2.0 / 3.0, indices)
    assert two["accepted_count"] == 2
    assert two["selected_row_indices"] == [2, 10]
    assert two["cutoff_tie"] is True


def test_constant_baseline_is_entirely_tied() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.3, 0.3, 0.3, 0.3])
    indices = np.array([3, 1, 2, 0])
    result = matched_coverage_selection(labels, probabilities, 0.50, indices)
    assert result["n_unique_confidences"] == 1
    assert result["n_tied_examples"] == 4
    assert result["max_tie_group_size"] == 4
    assert result["cutoff_tie"] is True
    assert result["accepted_count"] == 2
    # Ranking is original index order; p=0.3 predicts class 0 for every row.
    assert result["selected_row_indices"] == [0, 1]
    assert result["accepted_accuracy"] == pytest.approx(0.0)


def test_matched_coverage_rejects_invalid_targets() -> None:
    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    indices = np.array([0, 1])
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        matched_coverage_selection(labels, probabilities, 0.0, indices)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        matched_coverage_selection(labels, probabilities, 1.1, indices)
