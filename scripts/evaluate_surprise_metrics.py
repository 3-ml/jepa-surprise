from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROLE_ALIASES = {
    "possible": "possible",
    "plausible": "possible",
    "positive": "possible",
    "v+": "possible",
    "impossible": "impossible",
    "implausible": "impossible",
    "negative": "impossible",
    "violation": "impossible",
    "v-": "impossible",
}


@dataclass(frozen=True)
class SurpriseRow:
    pair_id: str
    condition: str
    role: str
    surprise: float
    temporal: tuple[float, ...]
    video_id: str
    source: dict[str, str]


@dataclass(frozen=True)
class PairMetric:
    pair_id: str
    condition: str
    possible_surprise: float
    impossible_surprise: float
    margin: float
    correct: bool


def parse_temporal(value: str) -> tuple[float, ...]:
    value = value.strip()
    if not value:
        return ()
    if value.startswith("["):
        loaded = json.loads(value)
        return tuple(float(x) for x in loaded)
    return tuple(float(x) for x in value.replace(";", ",").split(",") if x.strip())


def aggregate_temporal(values: tuple[float, ...], mode: str) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(arr.mean())
    if mode == "max":
        return float(arr.max())
    if mode == "last":
        return float(arr[-1])
    raise ValueError(f"Unsupported aggregate mode: {mode}")


def normalize_role(role: str) -> str:
    normalized = ROLE_ALIASES.get(role.strip().lower())
    if normalized is None:
        raise ValueError(f"Unknown role '{role}'. Use possible/impossible.")
    return normalized


def read_surprise_csv(path: Path, aggregate: str) -> list[SurpriseRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"pair_id", "condition", "role"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    parsed = []
    for i, row in enumerate(rows, start=2):
        temporal = parse_temporal(
            row.get("temporal_surprise", "")
            or row.get("surprise_t", "")
            or row.get("S_t", "")
            or row.get("s_t", "")
        )
        surprise_text = row.get("surprise", "") or row.get("S", "") or row.get("score", "")
        if surprise_text:
            surprise = float(surprise_text)
        else:
            surprise = aggregate_temporal(temporal, aggregate)
        if not math.isfinite(surprise):
            raise ValueError(f"Row {i} has no scalar surprise and no usable temporal surprise.")

        parsed.append(
            SurpriseRow(
                pair_id=row["pair_id"].strip(),
                condition=row["condition"].strip(),
                role=normalize_role(row["role"]),
                surprise=surprise,
                temporal=temporal,
                video_id=(row.get("video_id") or row.get("path") or row.get("video_path") or "").strip(),
                source=row,
            )
        )
    return parsed


def compute_pair_metrics(rows: list[SurpriseRow]) -> list[PairMetric]:
    grouped: dict[tuple[str, str], dict[str, SurpriseRow]] = {}
    for row in rows:
        key = (row.condition, row.pair_id)
        if row.role in grouped.setdefault(key, {}):
            raise ValueError(f"Duplicate {row.role} row for condition={row.condition} pair_id={row.pair_id}")
        grouped[key][row.role] = row

    metrics = []
    for (condition, pair_id), members in sorted(grouped.items()):
        if set(members) != {"possible", "impossible"}:
            raise ValueError(
                f"Need one possible and one impossible row for condition={condition} pair_id={pair_id}; got {sorted(members)}"
            )
        possible = members["possible"].surprise
        impossible = members["impossible"].surprise
        margin = impossible - possible
        metrics.append(
            PairMetric(
                pair_id=pair_id,
                condition=condition,
                possible_surprise=possible,
                impossible_surprise=impossible,
                margin=margin,
                correct=margin > 0.0,
            )
        )
    return metrics


def bootstrap_ci(values: list[float], reducer: Callable[[np.ndarray], float], seed: int, n_boot: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        estimates[i] = reducer(sample)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def condition_summaries(metrics: list[PairMetric], seed: int, n_boot: int) -> dict[str, dict[str, float | int | list[float]]]:
    out = {}
    for condition in sorted({m.condition for m in metrics}):
        selected = [m for m in metrics if m.condition == condition]
        correct = [1.0 if m.correct else 0.0 for m in selected]
        margins = [m.margin for m in selected]
        acc_ci = bootstrap_ci(correct, lambda x: float(x.mean()), seed, n_boot)
        mean_margin_ci = bootstrap_ci(margins, lambda x: float(x.mean()), seed, n_boot)
        median_margin_ci = bootstrap_ci(margins, lambda x: float(np.median(x)), seed, n_boot)
        out[condition] = {
            "n_pairs": len(selected),
            "pairwise_accuracy": float(np.mean(correct)),
            "pairwise_accuracy_ci95": [acc_ci[0], acc_ci[1]],
            "mean_margin": float(np.mean(margins)),
            "mean_margin_ci95": [mean_margin_ci[0], mean_margin_ci[1]],
            "median_margin": float(median(margins)),
            "median_margin_ci95": [median_margin_ci[0], median_margin_ci[1]],
            "mean_possible_surprise": float(np.mean([m.possible_surprise for m in selected])),
            "mean_impossible_surprise": float(np.mean([m.impossible_surprise for m in selected])),
        }
    return out


def compare_to_clean(summaries: dict[str, dict[str, float | int | list[float]]], clean_condition: str) -> list[dict[str, float | str]]:
    if clean_condition not in summaries:
        raise ValueError(f"Clean condition '{clean_condition}' not found. Available: {sorted(summaries)}")
    clean = summaries[clean_condition]
    comparisons = []
    for condition, summary in summaries.items():
        if condition == clean_condition:
            continue
        comparisons.append(
            {
                "condition": condition,
                "clean_condition": clean_condition,
                "accuracy": float(summary["pairwise_accuracy"]),
                "clean_accuracy": float(clean["pairwise_accuracy"]),
                "accuracy_drop": float(clean["pairwise_accuracy"]) - float(summary["pairwise_accuracy"]),
                "mean_margin": float(summary["mean_margin"]),
                "clean_mean_margin": float(clean["mean_margin"]),
                "mean_margin_delta": float(summary["mean_margin"]) - float(clean["mean_margin"]),
                "median_margin": float(summary["median_margin"]),
                "clean_median_margin": float(clean["median_margin"]),
                "median_margin_delta": float(summary["median_margin"]) - float(clean["median_margin"]),
            }
        )
    return comparisons


def write_pair_metrics(path: Path, metrics: list[PairMetric]) -> None:
    fields = ["condition", "pair_id", "possible_surprise", "impossible_surprise", "margin", "correct"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "condition": metric.condition,
                    "pair_id": metric.pair_id,
                    "possible_surprise": metric.possible_surprise,
                    "impossible_surprise": metric.impossible_surprise,
                    "margin": metric.margin,
                    "correct": int(metric.correct),
                }
            )


def write_comparisons(path: Path, comparisons: list[dict[str, float | str]]) -> None:
    fields = [
        "condition",
        "clean_condition",
        "accuracy",
        "clean_accuracy",
        "accuracy_drop",
        "mean_margin",
        "clean_mean_margin",
        "mean_margin_delta",
        "median_margin",
        "clean_median_margin",
        "median_margin_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)


def plot_temporal(rows: list[SurpriseRow], out_dir: Path) -> list[str]:
    temporal_rows = [row for row in rows if row.temporal]
    if not temporal_rows:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for condition in sorted({row.condition for row in temporal_rows}):
        plt.figure(figsize=(8, 4.5))
        for role in ["possible", "impossible"]:
            series = [np.asarray(row.temporal, dtype=np.float64) for row in temporal_rows if row.condition == condition and row.role == role]
            if not series:
                continue
            min_len = min(len(x) for x in series)
            arr = np.stack([x[:min_len] for x in series])
            mean = arr.mean(axis=0)
            low = np.quantile(arr, 0.025, axis=0)
            high = np.quantile(arr, 0.975, axis=0)
            x = np.arange(min_len)
            plt.plot(x, mean, label=role)
            plt.fill_between(x, low, high, alpha=0.2)
        plt.title(f"Temporal Surprise: {condition}")
        plt.xlabel("time index")
        plt.ylabel("surprise")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"temporal_surprise_{condition}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired physical-surprise metrics.")
    parser.add_argument("--input", required=True, help="CSV with pair_id, condition, role, and surprise or temporal_surprise.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-condition", default="clean")
    parser.add_argument("--aggregate", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-temporal", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_surprise_csv(input_path, args.aggregate)
    metrics = compute_pair_metrics(rows)
    summaries = condition_summaries(metrics, args.seed, args.bootstrap)
    comparisons = compare_to_clean(summaries, args.clean_condition)

    pair_metrics_path = out_dir / "pair_metrics.csv"
    comparisons_path = out_dir / "condition_comparisons.csv"
    summary_path = out_dir / "summary.json"
    write_pair_metrics(pair_metrics_path, metrics)
    write_comparisons(comparisons_path, comparisons)
    plot_paths = plot_temporal(rows, out_dir / "plots") if args.plot_temporal else []

    summary = {
        "input": str(input_path),
        "clean_condition": args.clean_condition,
        "aggregate": args.aggregate,
        "n_rows": len(rows),
        "n_pair_condition_metrics": len(metrics),
        "conditions": summaries,
        "comparisons_to_clean": comparisons,
        "outputs": {
            "pair_metrics": str(pair_metrics_path),
            "condition_comparisons": str(comparisons_path),
            "temporal_plots": plot_paths,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
