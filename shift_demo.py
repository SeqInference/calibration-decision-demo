"""Synthetic feature-corruption stress test of a frozen sigmoid calibrator.

Labels stay fixed. Additive Gaussian noise on test features models measurement
corruption, not a general distribution shift, financial process, or LLM task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from demo import (
    METHODS,
    SEEDS,
    _json_ready,
    calibration_api_name,
    evaluate_method,
    generate_dataset,
    make_calibrated_classifier,
    numerical_equal,
    package_versions,
    plot_error_vs_coverage,
    plot_reliability,
    stratified_disjoint_partitions,
    write_json,
)
from metrics import matched_coverage_selection

NOISE_SCALES = [0.00, 0.25, 0.50, 1.00]
MATCHED_COVERAGES = [0.25, 0.50, 0.75, 1.00]
# Independent of sklearn's make_classification / train_test_split RNGs.
# ASCII "SHIF" used as a documented stream separator.
CORRUPTION_STREAM_KEY = 0x53484946
STD_DDOF = 0


def shift_config() -> dict[str, Any]:
    from demo import experiment_config

    config = experiment_config()
    config.update(
        {
            "experiment": "synthetic feature-corruption stress test",
            "label_handling": "test labels are unchanged (measurement corruption)",
            "noise_scales": NOISE_SCALES,
            "noise_model": "X_test + scale * train_feature_std * Z, Z ~ N(0,1) iid",
            "train_std_ddof": STD_DDOF,
            "corruption_rng": {
                "type": "numpy.random.Generator",
                "bit_generator": "PCG64",
                "seed_sequence": "numpy.random.SeedSequence([seed, CORRUPTION_STREAM_KEY])",
                "stream_key": CORRUPTION_STREAM_KEY,
                "stream_key_meaning": "ASCII SHIF; independent of sklearn RNGs",
            },
            "noise_reuse": "one standard-normal array per seed, reused across scales",
            "model_policy": "fit once on train/calib; freeze across corruption conditions",
            "matched_coverages": MATCHED_COVERAGES,
            "matched_coverage_note": (
                "Retrospective diagnostic: rank already-scored test rows by "
                "confidence, breaking ties by original dataset row index. "
                "Not a deployment-ready threshold."
            ),
        }
    )
    return config


def corruption_generator(seed: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), CORRUPTION_STREAM_KEY])
    return np.random.default_rng(sequence)


def training_feature_stds(X_train: np.ndarray) -> np.ndarray:
    return np.std(np.asarray(X_train, dtype=float), axis=0, ddof=STD_DDOF)


def validate_noise_scale(scale: float) -> float:
    value = float(scale)
    if not np.isfinite(value):
        raise ValueError("noise scale must be finite")
    if value < 0.0:
        raise ValueError("noise scale must be >= 0")
    return value


def standard_normal_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> np.ndarray:
    return rng.standard_normal(size=shape)


def corrupt_features(
    X_test: np.ndarray,
    train_stds: np.ndarray,
    noise: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Return a new array; never modify X_test, train_stds, or noise."""
    scale = validate_noise_scale(scale)
    X = np.asarray(X_test, dtype=float)
    stds = np.asarray(train_stds, dtype=float)
    Z = np.asarray(noise, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_test must be 2-dimensional")
    if stds.shape != (X.shape[1],):
        raise ValueError("train_stds must have one entry per feature")
    if Z.shape != X.shape:
        raise ValueError("noise must match X_test shape")
    return X + scale * stds * Z


def snapshot_gaussiannb(estimator: GaussianNB) -> dict[str, np.ndarray]:
    return {
        "theta_": np.array(estimator.theta_, copy=True),
        "var_": np.array(estimator.var_, copy=True),
        "class_prior_": np.array(estimator.class_prior_, copy=True),
    }


def snapshots_equal(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(before[key], after[key]) for key in before)


def evaluate_condition(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, Any]:
    payload = evaluate_method(name, labels, probabilities)
    payload["matched_coverage"] = []
    for coverage in MATCHED_COVERAGES:
        item = matched_coverage_selection(labels, probabilities, coverage, row_indices)
        item.pop("selected_row_indices", None)
        payload["matched_coverage"].append(item)
    return payload


def run_seed(seed: int, output_dir: Path) -> dict[str, Any]:
    X, y = generate_dataset(seed)
    X_source = np.array(X, copy=True)
    y_source = np.array(y, copy=True)
    parts = stratified_disjoint_partitions(X, y, seed)
    X_train = parts["X_train"]
    y_train = parts["y_train"]
    X_test = np.array(parts["X_test"], copy=True)
    y_test = np.array(parts["y_test"], copy=True)
    test_row_indices = np.array(parts["test_idx"], copy=True)

    base = GaussianNB()
    base.fit(X_train, y_train)
    calibrated = make_calibrated_classifier(base)
    calibrated.fit(parts["X_calib"], parts["y_calib"])
    base_snapshot = snapshot_gaussiannb(base)
    train_positive_rate = float(np.mean(y_train == 1))
    train_stds = training_feature_stds(X_train)

    rng = corruption_generator(seed)
    noise = standard_normal_noise(rng, X_test.shape)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, Any] = {}
    selective_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for scale in NOISE_SCALES:
        X_corrupted = corrupt_features(X_test, train_stds, noise, scale)
        if not snapshots_equal(base_snapshot, snapshot_gaussiannb(base)):
            raise RuntimeError("base GaussianNB parameters changed during evaluation")
        raw_p = base.predict_proba(X_corrupted)[:, 1]
        cal_p = calibrated.predict_proba(X_corrupted)[:, 1]
        baseline_p = np.full(y_test.shape[0], train_positive_rate, dtype=float)
        if not snapshots_equal(base_snapshot, snapshot_gaussiannb(base)):
            raise RuntimeError("base GaussianNB parameters changed after predict")

        method_probs = {"raw": raw_p, "calibrated": cal_p, "baseline": baseline_p}
        scale_key = f"{scale:.2f}"
        scale_dir = seed_dir / f"scale_{scale_key}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        method_results: dict[str, Any] = {}
        reliability: dict[str, Any] = {}
        condition_selective: list[dict[str, Any]] = []
        for name, probs in method_probs.items():
            method_results[name] = evaluate_condition(name, y_test, probs, test_row_indices)
            reliability[name] = plot_reliability(
                y_test,
                probs,
                title=f"Reliability — {name} (seed {seed}, scale {scale_key})",
                path=scale_dir / f"reliability_{name}.png",
            )
            for item in method_results[name]["selective"]:
                row = {"seed": seed, "noise_scale": scale, "method": name, **item}
                selective_rows.append(row)
                condition_selective.append(row)
            for item in method_results[name]["matched_coverage"]:
                matched_rows.append({"seed": seed, "noise_scale": scale, "method": name, **item})
            metric_rows.append(
                {
                    "seed": seed,
                    "noise_scale": scale,
                    "method": name,
                    **method_results[name]["metrics"],
                }
            )
        plot_error_vs_coverage(
            condition_selective,
            scale_dir / "accepted_error_vs_coverage.png",
            title=f"Accepted error vs coverage (seed {seed}, scale {scale_key})",
        )
        conditions[scale_key] = {
            "noise_scale": scale,
            "methods": method_results,
            "reliability": reliability,
        }

    if not np.array_equal(X, X_source) or not np.array_equal(y, y_source):
        raise RuntimeError("source dataset arrays were modified")
    if not np.array_equal(X_test, parts["X_test"]) or not np.array_equal(y_test, parts["y_test"]):
        raise RuntimeError("copied test arrays diverged from partition views")
    if not snapshots_equal(base_snapshot, snapshot_gaussiannb(base)):
        raise RuntimeError("base GaussianNB parameters changed across conditions")

    record = {
        "seed": seed,
        "partition_counts": parts["counts"],
        "train_positive_rate": train_positive_rate,
        "train_feature_stds": train_stds.tolist(),
        "corruption_rng": {
            "seed": seed,
            "stream_key": CORRUPTION_STREAM_KEY,
            "seed_sequence": [seed, CORRUPTION_STREAM_KEY],
        },
        "calibration_api": calibration_api_name(),
        "base_parameters_frozen": True,
        "conditions": conditions,
    }
    write_json(seed_dir / "metrics.json", record)
    pd.DataFrame(selective_rows).to_csv(seed_dir / "selective_prediction.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(seed_dir / "matched_coverage.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(seed_dir / "metrics_by_scale.csv", index=False)
    return record


def _flatten_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for scale_key, condition in rec["conditions"].items():
            scale = condition["noise_scale"]
            for method, payload in condition["methods"].items():
                rows.append(
                    {
                        "seed": rec["seed"],
                        "noise_scale": scale,
                        "method": method,
                        **payload["metrics"],
                        "train_positive_rate": rec["train_positive_rate"],
                    }
                )
    return pd.DataFrame(rows)


def _flatten_selective(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for condition in rec["conditions"].values():
            scale = condition["noise_scale"]
            for method, payload in condition["methods"].items():
                for item in payload["selective"]:
                    rows.append(
                        {"seed": rec["seed"], "noise_scale": scale, "method": method, **item}
                    )
    return pd.DataFrame(rows)


def _flatten_matched(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for condition in rec["conditions"].values():
            scale = condition["noise_scale"]
            for method, payload in condition["methods"].items():
                for item in payload["matched_coverage"]:
                    rows.append(
                        {"seed": rec["seed"], "noise_scale": scale, "method": method, **item}
                    )
    return pd.DataFrame(rows)


def _add_clean_deltas(metric_df: pd.DataFrame) -> pd.DataFrame:
    clean = metric_df.loc[metric_df["noise_scale"] == 0.0, ["seed", "method", "brier_score", "log_loss", "accuracy"]]
    clean = clean.rename(
        columns={
            "brier_score": "brier_score_clean",
            "log_loss": "log_loss_clean",
            "accuracy": "accuracy_clean",
        }
    )
    merged = metric_df.merge(clean, on=["seed", "method"], how="left")
    merged["brier_score_delta"] = merged["brier_score"] - merged["brier_score_clean"]
    merged["log_loss_delta"] = merged["log_loss"] - merged["log_loss_clean"]
    merged["accuracy_delta"] = merged["accuracy"] - merged["accuracy_clean"]
    return merged


def _agg_mean_std(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for col in value_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    summary = numeric.groupby(group_cols)[value_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def plot_scores_vs_corruption(metric_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, col in zip(axes, ["brier_score", "log_loss", "accuracy"]):
        for method in METHODS:
            subset = metric_df[metric_df["method"] == method]
            grouped = subset.groupby("noise_scale")[col].mean()
            ax.plot(grouped.index, grouped.values, marker="o", label=method)
        ax.set_xlabel("Noise scale (× training std)")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(col.replace("_", " "))
    axes[0].legend()
    fig.suptitle("Test scores vs feature-corruption severity (mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_selective_vs_coverage(selective_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, scale in zip(axes, NOISE_SCALES):
        for method in METHODS:
            subset = selective_df[
                (selective_df["method"] == method) & (selective_df["noise_scale"] == scale)
            ].copy()
            subset["accepted_error_rate"] = pd.to_numeric(
                subset["accepted_error_rate"], errors="coerce"
            )
            means = subset.groupby("coverage", dropna=True)["accepted_error_rate"].mean()
            means = means.dropna()
            if not means.empty:
                ax.plot(means.index, means.values, marker="o", label=method)
        ax.set_title(f"scale {scale:.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Accepted error rate")
    axes[0].legend()
    fig.suptitle(
        "Accepted error vs coverage by corruption scale (mean over seeds; same-threshold points)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_matched_error_vs_coverage(matched_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, scale in zip(axes, NOISE_SCALES):
        for method in METHODS:
            subset = matched_df[
                (matched_df["method"] == method) & (matched_df["noise_scale"] == scale)
            ]
            means = subset.groupby("target_coverage")["accepted_error_rate"].mean()
            ax.plot(means.index, means.values, marker="o", label=method)
        ax.set_title(f"scale {scale:.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Matched coverage")
        ax.set_ylabel("Accepted error rate")
    axes[0].legend()
    fig.suptitle(
        "Matched-coverage accepted error (retrospective ranking; mean over seeds)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    metric_df = _add_clean_deltas(_flatten_metrics(records))
    selective_df = _flatten_selective(records)
    matched_df = _flatten_matched(records)

    metric_summary = _agg_mean_std(
        metric_df,
        ["noise_scale", "method"],
        ["brier_score", "log_loss", "accuracy", "brier_score_delta", "log_loss_delta", "accuracy_delta"],
    )
    selective_summary = _agg_mean_std(
        selective_df,
        ["noise_scale", "method", "threshold"],
        ["coverage", "accepted_count", "accepted_accuracy", "accepted_error_rate"],
    )
    matched_summary = _agg_mean_std(
        matched_df,
        ["noise_scale", "method", "target_coverage"],
        [
            "accepted_count",
            "realized_coverage",
            "accepted_accuracy",
            "accepted_error_rate",
            "n_tied_examples",
            "max_tie_group_size",
            "n_unique_confidences",
        ],
    )

    metric_df.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    selective_df.to_csv(output_dir / "per_seed_selective.csv", index=False)
    matched_df.to_csv(output_dir / "per_seed_matched_coverage.csv", index=False)
    metric_summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    selective_summary.to_csv(output_dir / "summary_selective.csv", index=False)
    matched_summary.to_csv(output_dir / "summary_matched_coverage.csv", index=False)

    plot_scores_vs_corruption(metric_df, output_dir / "scores_vs_corruption.png")
    plot_selective_vs_coverage(
        selective_df, output_dir / "accepted_error_vs_coverage_by_scale.png"
    )
    plot_matched_error_vs_coverage(
        matched_df, output_dir / "matched_coverage_error_by_scale.png"
    )

    summary = {
        "n_seeds": len(records),
        "seeds": [r["seed"] for r in records],
        "noise_scales": NOISE_SCALES,
        "metrics_mean_std": metric_summary.to_dict(orient="records"),
        "selective_mean_std": selective_summary.to_dict(orient="records"),
        "matched_coverage_mean_std": matched_summary.to_dict(orient="records"),
        "notes": [
            "Synthetic feature corruption only; labels are held fixed.",
            "This is measurement-noise stress testing, not a claim of general shift robustness.",
            "Matched coverage is a retrospective ranking diagnostic, not a deployment threshold.",
            "Mean/std values are descriptive summaries of the five seeds, not confidence intervals.",
            "Brier score and log loss mix calibration and sharpness.",
            "The constant baseline is independent of features, so its scores do not change with noise.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", shift_config())
    write_json(output_dir / "versions.json", package_versions())
    records = [run_seed(seed, output_dir) for seed in SEEDS]
    write_json(output_dir / "all_seeds.json", records)
    summary = summarize(records, output_dir)
    return {"records": records, "summary": summary}


def load_numerical_shift_results(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".json", ".csv"}:
            continue
        if file_path.name == "versions.json":
            continue
        rel = file_path.relative_to(path).as_posix()
        if file_path.suffix == ".json":
            payload[rel] = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            frame = pd.read_csv(file_path)
            records = []
            for row in frame.to_dict(orient="records"):
                cleaned = {}
                for key, value in row.items():
                    if isinstance(value, float) and not np.isfinite(value):
                        cleaned[str(key)] = None
                    else:
                        cleaned[str(key)] = value
                records.append(cleaned)
            payload[rel] = records
    if not payload:
        raise FileNotFoundError(f"no numerical artifacts under {path}")
    return payload


def compare_shift_dirs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    payload_a = load_numerical_shift_results(dir_a)
    payload_b = load_numerical_shift_results(dir_b)
    files = sorted(set(payload_a) | set(payload_b))
    missing_in_a = sorted(set(payload_b) - set(payload_a))
    missing_in_b = sorted(set(payload_a) - set(payload_b))
    mismatches = [
        name
        for name in sorted(set(payload_a) & set(payload_b))
        if not numerical_equal(payload_a[name], payload_b[name])
    ]
    return {
        "equal": not missing_in_a and not missing_in_b and not mismatches,
        "files_compared": files,
        "includes": [
            "per-seed Brier, log loss, accuracy at each noise scale",
            "thresholded selective metrics including coverage and accepted error",
            "matched-coverage ranking metrics and tie summaries",
            "reliability-bin statistics",
            "deltas versus the clean (scale 0) condition",
            "configuration",
        ],
        "excluded": [
            "PNG plots (raster rendering, not numerical tables)",
            "versions.json (environment metadata, not experiment metrics)",
        ],
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_shift"),
        help="Directory for shift-test metrics and plots. Must not be results/.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="Compare all numerical artifacts in two shift-test output directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        dir_a = Path(args.compare[0]).resolve()
        dir_b = Path(args.compare[1]).resolve()
        report = compare_shift_dirs(dir_a, dir_b)
        print(json.dumps(_json_ready(report), indent=2))
        if not report["equal"]:
            raise SystemExit(1)
        return
    output_dir = args.output_dir.resolve()
    original_results = (Path.cwd() / "results").resolve()
    if output_dir == original_results:
        raise SystemExit("Refusing to write shift outputs into results/; use results_shift/")
    run_experiment(output_dir)
    print(f"Wrote shift experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
