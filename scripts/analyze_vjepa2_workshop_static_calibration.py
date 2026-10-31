#!/usr/bin/env python3
"""Analyze V-JEPA2 IntPhys2 static-control scores for the workshop protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


MOTION_FIELDS = (
    "motion_global_consecutive_l1_01",
    "motion_patch_p95_consecutive_l1_01",
    "motion_first_last_l1_01",
)

SCORE_SPECS = {
    "raw_original": "original",
    "full_start_residual": "delta_original_minus_full_start",
    "full_boundary_residual": "delta_original_minus_full_boundary",
    "full_end_residual": "delta_original_minus_full_end",
    "multi_full_freeze_residual": "multi_full_freeze_residual",
    "target_hold_residual": "delta_original_minus_target_hold",
    "target_shuffle_residual": "delta_original_minus_target_shuffle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_score_frame(windows: pd.DataFrame, score_name: str, score_field: str) -> pd.DataFrame:
    video_means = (
        windows.groupby(["pair_id", "pair_label", "name"], as_index=False)
        .agg(
            score=(score_field, "mean"),
            condition=("condition", first_unique),
            difficulty=("difficulty", first_unique),
            camera=("camera", first_unique),
            windows=("window_id", "count"),
        )
    )
    paired = video_means.pivot(index="pair_id", columns="pair_label", values="score").dropna()
    meta = (
        video_means.groupby("pair_id", as_index=True)
        .agg(
            condition=("condition", first_unique),
            difficulty=("difficulty", first_unique),
            camera=("camera", first_unique),
            possible_windows=("windows", "min"),
            impossible_windows=("windows", "max"),
        )
        .loc[paired.index]
    )
    out = meta.reset_index()
    out["score_name"] = score_name
    out["possible_score"] = paired["possible"].to_numpy(dtype=np.float64)
    out["impossible_score"] = paired["impossible"].to_numpy(dtype=np.float64)
    out["margin"] = out["impossible_score"] - out["possible_score"]
    out["correct"] = out["margin"] > 0
    return out


def first_unique(values: pd.Series) -> object:
    unique = [value for value in values.dropna().unique()]
    if not unique:
        return ""
    if len(unique) > 1:
        return "|".join(map(str, sorted(unique)))
    return unique[0]


def score_metrics(pair_rows: pd.DataFrame, raw_rows: pd.DataFrame, samples: int, seed: int) -> dict[str, object]:
    margins = pair_rows["margin"].to_numpy(dtype=np.float64)
    correct = pair_rows["correct"].to_numpy(dtype=np.float64)
    raw_correct = raw_rows.set_index("pair_id").loc[pair_rows["pair_id"], "correct"].to_numpy(dtype=bool)
    now_correct = pair_rows["correct"].to_numpy(dtype=bool)
    delta_correct = now_correct.astype(np.int8) - raw_correct.astype(np.int8)
    return {
        "pairs": int(len(pair_rows)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": bootstrap_ci(correct, samples, seed),
        "mean_margin": float(margins.mean()),
        "mean_margin_ci95": bootstrap_ci(margins, samples, seed + 101),
        "median_margin": float(np.median(margins)),
        "pairs_corrected_vs_raw": int((delta_correct > 0).sum()),
        "pairs_harmed_vs_raw": int((delta_correct < 0).sum()),
        "pairs_unchanged_vs_raw": int((delta_correct == 0).sum()),
        "accuracy_delta_vs_raw": float(delta_correct.mean()),
        "accuracy_delta_vs_raw_ci95": bootstrap_ci(delta_correct.astype(np.float64), samples, seed + 202),
    }


def high_score_low_motion(windows: pd.DataFrame, score_name: str, score_field: str) -> dict[str, object]:
    top_count = max(1, int(np.ceil(0.1 * len(windows))))
    top = windows.sort_values([score_field, "window_id"], ascending=[False, True]).head(top_count)
    low = windows[windows["lowest_visual_change_decile"].astype(bool)]
    overlap = set(top["window_id"]) & set(low["window_id"])
    return {
        "score_name": score_name,
        "windows": int(len(windows)),
        "top_decile_windows": int(top_count),
        "lowest_visual_change_windows": int(len(low)),
        "top_score_and_low_visual_change": int(len(overlap)),
        "top_score_low_visual_change_fraction": float(len(overlap) / top_count),
    }


def motion_correlations(windows: pd.DataFrame, score_name: str, score_field: str) -> list[dict[str, object]]:
    rows = []
    for field in MOTION_FIELDS:
        rows.append(
            {
                "score_name": score_name,
                "motion_field": field,
                "spearman_rho": float(windows[[score_field, field]].corr(method="spearman").iloc[0, 1]),
            }
        )
    return rows


def condition_breakdown(pair_rows: pd.DataFrame, score_name: str) -> list[dict[str, object]]:
    rows = []
    for condition, group in sorted(pair_rows.groupby("condition"), key=lambda item: str(item[0])):
        margins = group["margin"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "score_name": score_name,
                "condition": condition,
                "pairs": int(len(group)),
                "accuracy": float((margins > 0).mean()),
                "mean_margin": float(margins.mean()),
                "median_margin": float(np.median(margins)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(args.input_result_dir / "analysis_windows.csv")
    required = {"window_id", "pair_id", "pair_label", "name", "condition", "lowest_visual_change_decile", *MOTION_FIELDS}
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    windows["multi_full_freeze_score"] = windows[["full_start", "full_boundary", "full_end"]].mean(axis=1)
    windows["multi_full_freeze_residual"] = windows["original"] - windows["multi_full_freeze_score"]

    raw_pairs = paired_score_frame(windows, "raw_original", "original")
    all_pair_rows = []
    metric_rows = []
    high_motion_rows = []
    correlation_rows = []
    breakdown_rows = []
    summary_scores = {}
    for offset, (score_name, score_field) in enumerate(SCORE_SPECS.items()):
        pair_rows = paired_score_frame(windows, score_name, score_field)
        all_pair_rows.extend(pair_rows.to_dict("records"))
        metrics = score_metrics(pair_rows, raw_pairs, args.bootstrap_samples, args.seed + offset * 1009)
        metric_rows.append({"score_name": score_name, **metrics})
        summary_scores[score_name] = metrics
        high_motion_rows.append(high_score_low_motion(windows, score_name, score_field))
        correlation_rows.extend(motion_correlations(windows, score_name, score_field))
        breakdown_rows.extend(condition_breakdown(pair_rows, score_name))

    write_csv(args.output_dir / "score_metrics.csv", metric_rows)
    write_csv(args.output_dir / "pair_outcomes.csv", all_pair_rows)
    write_csv(args.output_dir / "high_score_low_motion.csv", high_motion_rows)
    write_csv(args.output_dir / "motion_correlations.csv", correlation_rows)
    write_csv(args.output_dir / "condition_breakdown.csv", breakdown_rows)

    summary = {
        "status": "complete",
        "input_result_dir": str(args.input_result_dir),
        "windows": int(len(windows)),
        "videos": int(windows["name"].nunique()),
        "pairs": int(windows["pair_id"].nunique()),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "score_definitions": SCORE_SPECS,
        "scores": summary_scores,
        "interpretation": [
            "Pairwise VoE uses video-mean scores and pair bootstrap confidence intervals.",
            "The low-motion false-positive proxy is score-blind: lowest visual-change decile versus top score decile.",
            "Static residuals are control-specific diagnostics; target hold and target shuffle are not treated as universal calibrations.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
