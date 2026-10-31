#!/usr/bin/env python3
"""Produce descriptive construct-validity diagnostics from locked control scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROLS = (
    "full_start",
    "full_boundary",
    "full_end",
    "target_hold",
    "target_shuffle",
)
MOTION_FIELDS = (
    "motion_global_consecutive_l1_01",
    "motion_patch_p95_consecutive_l1_01",
    "motion_first_last_l1_01",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def pair_metrics(frame: pd.DataFrame, value: str) -> dict:
    videos = (
        frame.groupby(["pair_id", "pair_label", "name"], as_index=False)[value]
        .mean()
    )
    paired = videos.pivot(index="pair_id", columns="pair_label", values=value).dropna()
    margins = paired["impossible"] - paired["possible"]
    return {
        "pairs": int(len(margins)),
        "correct": int((margins > 0).sum()),
        "ties": int((margins == 0).sum()),
        "accuracy": float((margins > 0).mean()),
        "mean_margin": float(margins.mean()),
        "median_margin": float(margins.median()),
    }


def main():
    args = parse_args()
    scores = pd.read_csv(args.result_dir / "scores.csv")
    expected = {"original", *CONTROLS}
    if set(scores["control"]) != expected:
        raise ValueError(f"Unexpected controls: {sorted(set(scores['control']))}")
    pivot = scores.pivot(index="window_id", columns="control", values="scalar_l1")
    metadata = (
        scores[scores["control"] == "original"]
        .set_index("window_id")[
            [
                "name",
                "pair_id",
                "pair_label",
                "condition",
                "difficulty",
                "camera",
                *MOTION_FIELDS,
                "lowest_visual_change_decile",
            ]
        ]
    )
    analysis = metadata.join(pivot)
    for control in CONTROLS:
        analysis[f"delta_original_minus_{control}"] = (
            analysis["original"] - analysis[control]
        )
    analysis.reset_index().to_csv(
        args.result_dir / "analysis_windows.csv", index=False, lineterminator="\n"
    )

    delta_fields = [f"delta_original_minus_{control}" for control in CONTROLS]
    correlations = analysis[["original", *MOTION_FIELDS, *delta_fields]].corr(
        method="spearman"
    )
    correlations.to_csv(
        args.result_dir / "spearman_correlations.csv", lineterminator="\n"
    )
    delta_agreement = analysis[delta_fields].corr(method="spearman")
    delta_agreement.to_csv(
        args.result_dir / "control_delta_rank_agreement.csv", lineterminator="\n"
    )

    low = analysis[analysis["lowest_visual_change_decile"].astype(bool)]
    top_count = max(1, int(np.ceil(0.1 * len(analysis))))
    top_raw = analysis.sort_values(["original"], ascending=False).head(top_count)
    control_rows = []
    for control, delta in zip(CONTROLS, delta_fields):
        control_rows.append(
            {
                "control": control,
                "all_window_mean_delta": float(analysis[delta].mean()),
                "all_window_positive_delta_fraction": float((analysis[delta] > 0).mean()),
                "lowest_visual_change_mean_delta": float(low[delta].mean()),
                "lowest_visual_change_positive_delta_fraction": float((low[delta] > 0).mean()),
                "raw_delta_spearman": float(analysis[["original", delta]].corr(method="spearman").iloc[0, 1]),
            }
        )
    pd.DataFrame(control_rows).to_csv(
        args.result_dir / "control_descriptives.csv", index=False, lineterminator="\n"
    )

    pairwise = {"raw_original": pair_metrics(analysis.reset_index(), "original")}
    for control, delta in zip(CONTROLS, delta_fields):
        pairwise[f"control_score_{control}"] = pair_metrics(
            analysis.reset_index(), control
        )
        pairwise[f"delta_original_minus_{control}"] = pair_metrics(
            analysis.reset_index(), delta
        )

    sign_agreement = {}
    for left_index, left in enumerate(delta_fields):
        for right in delta_fields[left_index + 1 :]:
            sign_agreement[f"{left}__{right}"] = float(
                (np.sign(analysis[left]) == np.sign(analysis[right])).mean()
            )
    summary = {
        "status": "descriptive_only_no_bootstrap_or_event_labels",
        "windows": int(len(analysis)),
        "videos": int(analysis["name"].nunique()),
        "pairs": int(analysis["pair_id"].nunique()),
        "lowest_visual_change_windows": int(len(low)),
        "top_raw_decile_windows": int(len(top_raw)),
        "lowest_visual_change_among_top_raw_decile": int(
            top_raw["lowest_visual_change_decile"].astype(bool).sum()
        ),
        "raw_motion_spearman": {
            field: float(analysis[["original", field]].corr(method="spearman").iloc[0, 1])
            for field in MOTION_FIELDS
        },
        "pairwise_video_mean": pairwise,
        "control_delta_sign_agreement": sign_agreement,
        "limitations": [
            "No event labels are used; low visual change is not a non-event label.",
            "No confidence intervals or inferential gate decisions are computed here.",
            "Control deltas remain control-specific diagnostics, not calibrated event surprise.",
        ],
    }
    (args.result_dir / "descriptive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
