"""Probability, accuracy, and selective-prediction metrics.

Brier score and log loss are proper scoring rules. They penalize both
miscalibration and lack of sharpness (overly uncertain but well-calibrated
predictions can still score poorly relative to a more informative model).
They are therefore not pure calibration diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

UNAVAILABLE = "unavailable"


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.ndim != 1:
        arr = np.ravel(arr)
        if arr.size != np.asarray(values).size:
            raise ValueError(f"{name} must be one-dimensional")
    return arr


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    probs = _as_1d_float(probabilities, "probabilities")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    return probs


def _validate_labels(labels: Any) -> np.ndarray:
    y = _as_1d_float(labels, "labels")
    if not np.all(np.isfinite(y)):
        raise ValueError("labels must be finite")
    unique = np.unique(y)
    if not np.all(np.isin(unique, np.array([0.0, 1.0]))):
        raise ValueError("labels must be binary in {0, 1}")
    return y.astype(int)


def _align_inputs(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = _validate_labels(labels)
    p = _validate_probabilities(probabilities)
    if y.shape[0] != p.shape[0]:
        raise ValueError("labels and probabilities must have the same length")
    return y, p


def brier_score(labels: Any, probabilities: Any) -> float:
    """Mean squared error between binary labels and predicted probabilities."""
    y, p = _align_inputs(labels, probabilities)
    return float(np.mean((p - y) ** 2))


def log_loss(labels: Any, probabilities: Any, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of binary labels given probabilities."""
    y, p = _align_inputs(labels, probabilities)
    clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def accuracy(labels: Any, probabilities: Any, threshold: float = 0.5) -> float:
    """Hard-label accuracy using a probability threshold (default 0.5)."""
    y, p = _align_inputs(labels, probabilities)
    preds = (p >= threshold).astype(int)
    return float(np.mean(preds == y))


def confidence(probabilities: Any) -> np.ndarray:
    """Binary confidence: max(p, 1 - p)."""
    p = _validate_probabilities(probabilities)
    return np.maximum(p, 1.0 - p)


def reliability_diagram_data(
    labels: Any,
    probabilities: Any,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Equal-width reliability bins with counts, mean predicted, and observed."""
    y, p = _align_inputs(labels, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)
    mean_predicted = np.full(n_bins, np.nan)
    fraction_positive = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_ids == i
        counts[i] = int(np.sum(mask))
        if counts[i] > 0:
            mean_predicted[i] = float(np.mean(p[mask]))
            fraction_positive[i] = float(np.mean(y[mask]))
    return {
        "bin_edges": edges,
        "counts": counts,
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
    }


def selective_prediction(
    labels: Any,
    probabilities: Any,
    threshold: float,
) -> dict[str, float | int | str]:
    """Accept examples whose confidence is at least `threshold`.

    Confidence is max(p, 1-p). When no examples are accepted, coverage is 0
    and accuracy/error metrics are reported as unavailable.
    """
    y, p = _align_inputs(labels, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if threshold < 0.5 or threshold > 1.0:
        raise ValueError("threshold must lie in [0.5, 1.0]")
    conf = confidence(p)
    accepted = conf >= threshold
    accepted_count = int(np.sum(accepted))
    n = int(y.shape[0])
    coverage = float(accepted_count / n)
    if accepted_count == 0:
        return {
            "threshold": float(threshold),
            "coverage": coverage,
            "accepted_count": accepted_count,
            "accepted_accuracy": UNAVAILABLE,
            "accepted_error_rate": UNAVAILABLE,
        }
    y_acc = y[accepted]
    p_acc = p[accepted]
    acc = accuracy(y_acc, p_acc)
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "accepted_count": accepted_count,
        "accepted_accuracy": acc,
        "accepted_error_rate": float(1.0 - acc),
    }


def confidence_tie_summary(probabilities: Any) -> dict[str, int]:
    """Count duplicate confidence values. Constant predictors produce one large tie."""
    conf = np.round(confidence(probabilities), decimals=12)
    _, counts = np.unique(conf, return_counts=True)
    tied = counts[counts > 1]
    return {
        "n_unique_confidences": int(counts.shape[0]),
        "n_tied_examples": int(np.sum(tied)) if tied.size else 0,
        "max_tie_group_size": int(np.max(counts)),
        "n_tie_groups": int(tied.shape[0]),
    }


def matched_coverage_selection(
    labels: Any,
    probabilities: Any,
    coverage: float,
    row_indices: Any,
) -> dict[str, float | int | str | bool]:
    """Accept one method's own top `coverage` fraction by confidence, without labels.

    This is a retrospective diagnostic: the accepted set is chosen from a
    ranking of already-scored examples for that method. It does not select
    the same rows across methods, and it is not a deployment-ready threshold
    that can be applied to a future stream without a ranking of that stream.

    Ties in confidence are broken by original row index, ascending, so the
    outcome is deterministic. For a constant baseline every example has the
    same confidence, so the subset is an arbitrary index-selected reference
    set rather than a confidence-based selection.
    """
    y, p = _align_inputs(labels, probabilities)
    indices = np.asarray(row_indices)
    if indices.shape[0] != y.shape[0]:
        raise ValueError("row_indices must have the same length as labels")
    if not np.isfinite(coverage):
        raise ValueError("coverage must be finite")
    if coverage <= 0.0 or coverage > 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    n = int(y.shape[0])
    k = int(round(coverage * n))
    if k < 1:
        raise ValueError("coverage selected zero examples")
    conf = confidence(p)
    order = np.lexsort((indices, -conf))
    selected = order[:k]
    cutoff_conf = float(conf[selected[-1]])
    n_at_cutoff = int(np.sum(np.isclose(conf, cutoff_conf, rtol=0.0, atol=1e-12)))
    acc = accuracy(y[selected], p[selected])
    ties = confidence_tie_summary(p)
    return {
        "target_coverage": float(coverage),
        "accepted_count": int(k),
        "realized_coverage": float(k / n),
        "accepted_accuracy": acc,
        "accepted_error_rate": float(1.0 - acc),
        "cutoff_confidence": cutoff_conf,
        "n_at_cutoff_confidence": n_at_cutoff,
        "cutoff_tie": bool(n_at_cutoff > 1),
        "selected_row_indices": indices[selected].tolist(),
        **ties,
    }


def evaluate_probability_metrics(labels: Any, probabilities: Any) -> dict[str, float]:
    y, p = _align_inputs(labels, probabilities)
    return {
        "brier_score": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "accuracy": accuracy(y, p),
    }
