"""Standalone probability-calibration and selective-prediction demo.

Synthetic binary classification only. Thresholds are predefined and are not
tuned on the test partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from metrics import (
    UNAVAILABLE,
    evaluate_probability_metrics,
    reliability_diagram_data,
    selective_prediction,
)

SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 10000
N_FEATURES = 20
N_INFORMATIVE = 10
N_REDUNDANT = 5
WEIGHTS = [0.7, 0.3]
CLASS_SEP = 0.8
FLIP_Y = 0.05
TRAIN_SIZE = 0.60
CALIB_SIZE = 0.20
TEST_SIZE = 0.20
CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
N_RELIABILITY_BINS = 10
PACKAGES = ["numpy", "pandas", "scikit-learn", "matplotlib", "pytest"]
METHODS = ("raw", "calibrated", "baseline")


def calibration_api_name() -> str:
    try:
        from sklearn.frozen import FrozenEstimator  # noqa: F401

        return "CalibratedClassifierCV(FrozenEstimator(...), method='sigmoid')"
    except ImportError:
        return "CalibratedClassifierCV(..., method='sigmoid', cv='prefit')"


def make_calibrated_classifier(fitted_estimator: GaussianNB) -> CalibratedClassifierCV:
    """Sigmoid calibration for an already fitted estimator.

    Uses FrozenEstimator when available (scikit-learn >= 1.6). Falls back to
    cv='prefit' on older releases.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            FrozenEstimator(fitted_estimator),
            method="sigmoid",
        )
    except ImportError:
        return CalibratedClassifierCV(
            fitted_estimator,
            method="sigmoid",
            cv="prefit",
        )


def generate_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_classes=2,
        weights=WEIGHTS,
        class_sep=CLASS_SEP,
        flip_y=FLIP_Y,
        random_state=seed,
    )
    return X, y


def stratified_disjoint_partitions(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Create 60/20/20 stratified partitions with mutually disjoint indices."""
    if abs(TRAIN_SIZE + CALIB_SIZE + TEST_SIZE - 1.0) > 1e-12:
        raise ValueError("partition fractions must sum to 1")
    n = X.shape[0]
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(
        indices,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=seed,
    )
    calib_frac_of_rest = CALIB_SIZE / (CALIB_SIZE + TEST_SIZE)
    calib_idx, test_idx = train_test_split(
        rest_idx,
        train_size=calib_frac_of_rest,
        stratify=y[rest_idx],
        random_state=seed,
    )
    train_idx = np.asarray(train_idx)
    calib_idx = np.asarray(calib_idx)
    test_idx = np.asarray(test_idx)
    if len(set(train_idx) & set(calib_idx)) or len(set(train_idx) & set(test_idx)) or len(
        set(calib_idx) & set(test_idx)
    ):
        raise RuntimeError("partitions are not mutually disjoint")
    if len(train_idx) + len(calib_idx) + len(test_idx) != n:
        raise RuntimeError("partitions do not cover the full dataset")
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_calib": X[calib_idx],
        "y_calib": y[calib_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "calib_idx": calib_idx,
        "test_idx": test_idx,
        "counts": {
            "n_total": int(n),
            "n_train": int(len(train_idx)),
            "n_calib": int(len(calib_idx)),
            "n_test": int(len(test_idx)),
            "train_positive": int(np.sum(y[train_idx] == 1)),
            "calib_positive": int(np.sum(y[calib_idx] == 1)),
            "test_positive": int(np.sum(y[test_idx] == 1)),
        },
    }


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " ")}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def experiment_config() -> dict[str, Any]:
    return {
        "dataset": "sklearn.datasets.make_classification (synthetic)",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_informative": N_INFORMATIVE,
        "n_redundant": N_REDUNDANT,
        "weights": WEIGHTS,
        "class_sep": CLASS_SEP,
        "flip_y": FLIP_Y,
        "seeds": SEEDS,
        "train_size": TRAIN_SIZE,
        "calib_size": CALIB_SIZE,
        "test_size": TEST_SIZE,
        "base_estimator": "GaussianNB",
        "calibration": "sigmoid (Platt scaling) on a frozen/prefit GaussianNB",
        "baseline": "constant probability equal to training positive-class frequency",
        "evaluation_split": "untouched test partition only",
        "confidence": "max(p, 1-p)",
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "threshold_selection": "predefined; not tuned on test results",
        "reliability_bins": N_RELIABILITY_BINS,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is UNAVAILABLE:
        return UNAVAILABLE
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def plot_reliability(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str,
    path: Path,
) -> dict[str, Any]:
    data = reliability_diagram_data(labels, probabilities, n_bins=N_RELIABILITY_BINS)
    edges = data["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    valid = data["counts"] > 0
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(
        data["mean_predicted"][valid],
        data["fraction_positive"][valid],
        marker="o",
        color="C0",
        label="empirical",
    )
    ax.bar(
        centers,
        np.nan_to_num(data["fraction_positive"], nan=0.0),
        width=np.diff(edges),
        align="center",
        alpha=0.25,
        color="C0",
        edgecolor="none",
    )
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    axes[1].bar(
        centers,
        data["counts"],
        width=np.diff(edges),
        align="center",
        color="C1",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Bin count")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {
        "bin_edges": data["bin_edges"].tolist(),
        "counts": data["counts"].tolist(),
        "mean_predicted": [None if np.isnan(v) else float(v) for v in data["mean_predicted"]],
        "fraction_positive": [None if np.isnan(v) else float(v) for v in data["fraction_positive"]],
    }


def plot_probability_histogram(
    probabilities: dict[str, np.ndarray],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name, p in probabilities.items():
        ax.hist(p, bins=20, range=(0, 1), density=True, histtype="step", label=name, linewidth=1.6)
    ax.set_xlabel("Predicted P(y=1)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_vs_coverage(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]
        xs: list[float] = []
        ys: list[float] = []
        for r in method_rows:
            if r["accepted_error_rate"] == UNAVAILABLE:
                continue
            xs.append(float(r["coverage"]))
            ys.append(float(r["accepted_error_rate"]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=method)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accepted error rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_method(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_probability_metrics(labels, probabilities)
    selective = [selective_prediction(labels, probabilities, t) for t in CONFIDENCE_THRESHOLDS]
    return {
        "method": name,
        "metrics": metrics,
        "selective": selective,
        "probability_summary": {
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
        },
    }


def run_seed(seed: int, output_dir: Path) -> dict[str, Any]:
    X, y = generate_dataset(seed)
    parts = stratified_disjoint_partitions(X, y, seed)
    base = GaussianNB()
    base.fit(parts["X_train"], parts["y_train"])
    calibrated = make_calibrated_classifier(base)
    calibrated.fit(parts["X_calib"], parts["y_calib"])

    raw_p = base.predict_proba(parts["X_test"])[:, 1]
    cal_p = calibrated.predict_proba(parts["X_test"])[:, 1]
    train_positive_rate = float(np.mean(parts["y_train"] == 1))
    baseline_p = np.full(parts["y_test"].shape[0], train_positive_rate, dtype=float)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    method_probs = {
        "raw": raw_p,
        "calibrated": cal_p,
        "baseline": baseline_p,
    }
    method_results = {}
    reliability = {}
    selective_rows = []
    for name, probs in method_probs.items():
        method_results[name] = evaluate_method(name, parts["y_test"], probs)
        reliability[name] = plot_reliability(
            parts["y_test"],
            probs,
            title=f"Reliability diagram — {name} (seed {seed})",
            path=seed_dir / f"reliability_{name}.png",
        )
        for item in method_results[name]["selective"]:
            selective_rows.append(
                {
                    "seed": seed,
                    "method": name,
                    **item,
                }
            )

    plot_probability_histogram(
        method_probs,
        seed_dir / "probability_histogram.png",
        title=f"Predicted probability histograms (seed {seed})",
    )
    plot_error_vs_coverage(
        selective_rows,
        seed_dir / "accepted_error_vs_coverage.png",
        title=f"Accepted error rate vs coverage (seed {seed})",
    )

    record = {
        "seed": seed,
        "partition_counts": parts["counts"],
        "train_positive_rate": train_positive_rate,
        "methods": method_results,
        "reliability": reliability,
        "calibration_api": calibration_api_name(),
    }

    write_json(seed_dir / "metrics.json", record)
    pd.DataFrame(selective_rows).to_csv(seed_dir / "selective_prediction.csv", index=False)
    return record


def _flatten_seed_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for method, payload in rec["methods"].items():
            rows.append(
                {
                    "seed": rec["seed"],
                    "method": method,
                    "brier_score": payload["metrics"]["brier_score"],
                    "log_loss": payload["metrics"]["log_loss"],
                    "accuracy": payload["metrics"]["accuracy"],
                    "train_positive_rate": rec["train_positive_rate"],
                    "n_train": rec["partition_counts"]["n_train"],
                    "n_calib": rec["partition_counts"]["n_calib"],
                    "n_test": rec["partition_counts"]["n_test"],
                }
            )
    return pd.DataFrame(rows)


def _flatten_selective(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for method, payload in rec["methods"].items():
            for item in payload["selective"]:
                rows.append({"seed": rec["seed"], "method": method, **item})
    return pd.DataFrame(rows)


def summarize(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    metric_df = _flatten_seed_metrics(records)
    selective_df = _flatten_selective(records)
    metric_summary = (
        metric_df.groupby("method")[["brier_score", "log_loss", "accuracy"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    metric_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in metric_summary.columns
    ]

    selective_numeric = selective_df.copy()
    for col in ("accepted_accuracy", "accepted_error_rate"):
        selective_numeric[col] = pd.to_numeric(selective_numeric[col], errors="coerce")
    selective_summary = (
        selective_numeric.groupby(["method", "threshold"])[
            ["coverage", "accepted_count", "accepted_accuracy", "accepted_error_rate"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    selective_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in selective_summary.columns
    ]

    metric_df.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    selective_df.to_csv(output_dir / "per_seed_selective.csv", index=False)
    metric_summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    selective_summary.to_csv(output_dir / "summary_selective.csv", index=False)

    all_selective_rows = selective_df.to_dict(orient="records")
    plot_error_vs_coverage(
        all_selective_rows,
        output_dir / "accepted_error_vs_coverage_all_seeds.png",
        title="Accepted error rate vs coverage (all seeds)",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, col in zip(axes, ["brier_score", "log_loss", "accuracy"]):
        metric_df.boxplot(column=col, by="method", ax=ax)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("")
    fig.suptitle("Test-set scores across seeds (synthetic data)")
    fig.tight_layout()
    fig.savefig(output_dir / "metric_spread_across_seeds.png", dpi=120)
    plt.close(fig)

    summary = {
        "n_seeds": len(records),
        "seeds": [r["seed"] for r in records],
        "metrics_mean_std": metric_summary.to_dict(orient="records"),
        "selective_mean_std": selective_summary.to_dict(orient="records"),
        "notes": [
            "Brier score and log loss are proper scoring rules; they reflect calibration and sharpness together, not calibration alone.",
            "Thresholds were predefined and were not selected using test-set results.",
            "accepted_accuracy and accepted_error_rate are omitted from means when a seed has zero accepted examples.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", experiment_config())
    write_json(output_dir / "versions.json", package_versions())
    records = [run_seed(seed, output_dir) for seed in SEEDS]
    write_json(output_dir / "all_seeds.json", records)
    summary = summarize(records, output_dir)
    return {"records": records, "summary": summary}


NUMERICAL_ARTIFACTS = (
    "config.json",
    "all_seeds.json",
    "summary.json",
    "per_seed_metrics.csv",
    "per_seed_selective.csv",
    "summary_metrics.csv",
    "summary_selective.csv",
)

PER_SEED_NUMERICAL_ARTIFACTS = (
    "metrics.json",
    "selective_prediction.csv",
)

COMPARISON_INCLUDES = (
    "per-seed Brier score, log loss, and accuracy",
    "per-seed selective metrics: threshold, coverage, accepted_count, accepted_accuracy, accepted_error_rate",
    "reliability-diagram bin edges, counts, mean predicted probability, and observed fraction",
    "partition counts and training positive-class frequency",
    "descriptive mean/std summary tables",
    "experiment configuration",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[str(key)] = None
            else:
                cleaned[str(key)] = value
        records.append(cleaned)
    return records


def load_numerical_results(path: Path) -> dict[str, Any]:
    """Load every published numerical artifact under an experiment output directory."""
    payload: dict[str, Any] = {}
    for name in NUMERICAL_ARTIFACTS:
        file_path = path / name
        if not file_path.is_file():
            raise FileNotFoundError(f"missing numerical artifact: {file_path}")
        if name.endswith(".json"):
            payload[name] = _read_json(file_path)
        else:
            payload[name] = _read_csv_records(file_path)
    seed_dirs = sorted(p for p in path.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        raise FileNotFoundError(f"no seed_* directories under {path}")
    for seed_dir in seed_dirs:
        for name in PER_SEED_NUMERICAL_ARTIFACTS:
            file_path = seed_dir / name
            if not file_path.is_file():
                raise FileNotFoundError(f"missing numerical artifact: {file_path}")
            key = f"{seed_dir.name}/{name}"
            if name.endswith(".json"):
                payload[key] = _read_json(file_path)
            else:
                payload[key] = _read_csv_records(file_path)
    return payload


def comparable_payload(path: Path) -> dict[str, Any]:
    """Backward-compatible alias: full numerical payload, not a summary subset."""
    return load_numerical_results(path)


def compare_result_dirs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    payload_a = load_numerical_results(dir_a)
    payload_b = load_numerical_results(dir_b)
    files = sorted(set(payload_a) | set(payload_b))
    missing_in_a = sorted(set(payload_b) - set(payload_a))
    missing_in_b = sorted(set(payload_a) - set(payload_b))
    mismatches = [name for name in sorted(set(payload_a) & set(payload_b)) if not numerical_equal(payload_a[name], payload_b[name])]
    equal = not missing_in_a and not missing_in_b and not mismatches
    return {
        "equal": equal,
        "files_compared": files,
        "includes": list(COMPARISON_INCLUDES),
        "excluded": [
            "PNG plots (raster rendering, not numerical tables)",
            "versions.json (environment metadata, not experiment metrics)",
        ],
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "mismatches": mismatches,
    }


def numerical_equal(a: Any, b: Any, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        if a is None or b is None:
            return a == b
        if isinstance(a, str) or isinstance(b, str):
            return a == b
    if isinstance(a, dict):
        return set(a) == set(b) and all(numerical_equal(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(numerical_equal(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a is None or b is None:
            return a == b
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True))
    return a == b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics, tables, and plots.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="Compare all published numerical artifacts in two output directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        dir_a = Path(args.compare[0]).resolve()
        dir_b = Path(args.compare[1]).resolve()
        report = compare_result_dirs(dir_a, dir_b)
        print(json.dumps(_json_ready(report), indent=2))
        if not report["equal"]:
            raise SystemExit(1)
        return
    run_experiment(args.output_dir.resolve())
    print(f"Wrote experiment outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
