#!/usr/bin/env python3
"""Analyze paired V-JEPA2 appearance-by-temporal-control pilot scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_scene_interval(
    values: pd.DataFrame,
    value_column: str,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    by_scene = values.groupby("scene")[value_column].agg(["sum", "count"])
    draws = rng.integers(
        0, len(by_scene), size=(resamples, len(by_scene)), endpoint=False
    )
    sampled_sum = by_scene["sum"].to_numpy()[draws].sum(axis=1)
    sampled_count = by_scene["count"].to_numpy()[draws].sum(axis=1)
    means = sampled_sum / sampled_count
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores)
    expected_rows = (
        20 * 2 * len(config["conditions"]) * len(config["controls"])
    )
    if len(scores) != expected_rows:
        raise ValueError(f"Expected {expected_rows} score rows, found {len(scores)}")
    cell_counts = scores.groupby("sample_id")["control"].nunique()
    if not cell_counts.eq(len(config["controls"])).all():
        raise ValueError("At least one sample lacks a temporal control")
    if not np.isfinite(scores["scalar_surprise"]).all():
        raise ValueError("Non-finite scalar score")

    pivot = scores.pivot_table(
        index=[
            "split",
            "condition",
            "control",
            "base_pair_id",
            "scene",
        ],
        columns="possible",
        values="scalar_surprise",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={False: "impossible_score", True: "possible_score"})
    pivot["margin_impossible_minus_possible"] = (
        pivot["impossible_score"] - pivot["possible_score"]
    )
    pivot["correct_strict"] = (
        pivot["margin_impossible_minus_possible"] > 0
    ).astype(float)
    pivot["tie"] = (pivot["margin_impossible_minus_possible"] == 0).astype(float)
    pivot["correct_half_ties"] = pivot["correct_strict"] + 0.5 * pivot["tie"]
    pivot.to_csv(args.output_root / "pair_margins.csv", index=False)

    rng = np.random.default_rng(config["bootstrap_seed"])
    summary_rows = []
    for (split, condition, control), group in pivot.groupby(
        ["split", "condition", "control"], sort=True
    ):
        accuracy_low, accuracy_high = bootstrap_scene_interval(
            group,
            "correct_half_ties",
            int(config["bootstrap_resamples"]),
            rng,
        )
        margin_low, margin_high = bootstrap_scene_interval(
            group,
            "margin_impossible_minus_possible",
            int(config["bootstrap_resamples"]),
            rng,
        )
        video_rows = scores[
            scores["split"].eq(split)
            & scores["condition"].eq(condition)
            & scores["control"].eq(control)
        ]
        summary_rows.append(
            {
                "split": split,
                "condition": condition,
                "control": control,
                "mean_scalar_surprise": video_rows["scalar_surprise"].mean(),
                "mean_pixel_motion": video_rows[
                    "mean_consecutive_rgb_difference"
                ].mean(),
                "pair_accuracy_half_ties": group["correct_half_ties"].mean(),
                "pair_accuracy_ci_low": accuracy_low,
                "pair_accuracy_ci_high": accuracy_high,
                "tie_fraction": group["tie"].mean(),
                "mean_margin": group["margin_impossible_minus_possible"].mean(),
                "mean_margin_ci_low": margin_low,
                "mean_margin_ci_high": margin_high,
                "n_pairs": len(group),
            }
        )
    condition_control = pd.DataFrame(summary_rows)

    originals = scores[scores["control"].eq("original")][
        ["sample_id", "scalar_surprise"]
    ].rename(columns={"scalar_surprise": "original_score"})
    deltas = scores.merge(originals, on="sample_id", validate="many_to_one")
    deltas["delta_original_minus_control"] = (
        deltas["original_score"] - deltas["scalar_surprise"]
    )
    control_rows = []
    for (split, condition, control), group in deltas.groupby(
        ["split", "condition", "control"], sort=True
    ):
        delta_low, delta_high = bootstrap_scene_interval(
            group,
            "delta_original_minus_control",
            int(config["bootstrap_resamples"]),
            rng,
        )
        correlation = (
            1.0
            if control == "original"
            else float(spearmanr(group["original_score"], group["scalar_surprise"]).statistic)
        )
        control_rows.append(
            {
                "split": split,
                "condition": condition,
                "control": control,
                "mean_delta_original_minus_control": group[
                    "delta_original_minus_control"
                ].mean(),
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "spearman_raw_vs_control": correlation,
                "n_videos": len(group),
            }
        )
    control_metrics = pd.DataFrame(control_rows)

    original_pairs = pivot[pivot["control"].eq("original")][
        ["split", "condition", "base_pair_id", "correct_strict", "tie"]
    ].rename(
        columns={
            "correct_strict": "original_correct",
            "tie": "original_tie",
        }
    )
    comparison = pivot.merge(
        original_pairs,
        on=["split", "condition", "base_pair_id"],
        validate="many_to_one",
    )
    comparison["corrected"] = (
        comparison["original_correct"].eq(0)
        & comparison["original_tie"].eq(0)
        & comparison["correct_strict"].eq(1)
    )
    comparison["harmed"] = (
        comparison["original_correct"].eq(1)
        & comparison["correct_strict"].eq(0)
        & comparison["tie"].eq(0)
    )
    correction_counts = (
        comparison.groupby(["split", "condition", "control"])
        .agg(corrected_pairs=("corrected", "sum"), harmed_pairs=("harmed", "sum"))
        .reset_index()
    )
    condition_control = condition_control.merge(
        correction_counts,
        on=["split", "condition", "control"],
        validate="one_to_one",
    )

    nonclean = deltas[~deltas["condition"].eq("clean")]
    clean_delta = deltas[deltas["condition"].eq("clean")][
        [
            "split",
            "base_video_id",
            "control",
            "delta_original_minus_control",
        ]
    ].rename(columns={"delta_original_minus_control": "clean_delta"})
    interactions = nonclean.merge(
        clean_delta,
        on=["split", "base_video_id", "control"],
        validate="many_to_one",
    )
    interactions["appearance_by_control_interaction"] = (
        interactions["delta_original_minus_control"] - interactions["clean_delta"]
    )
    interaction_rows = []
    for (split, condition, control), group in interactions.groupby(
        ["split", "condition", "control"], sort=True
    ):
        low, high = bootstrap_scene_interval(
            group,
            "appearance_by_control_interaction",
            int(config["bootstrap_resamples"]),
            rng,
        )
        interaction_rows.append(
            {
                "split": split,
                "condition": condition,
                "control": control,
                "mean_interaction": group[
                    "appearance_by_control_interaction"
                ].mean(),
                "interaction_ci_low": low,
                "interaction_ci_high": high,
                "n_videos": len(group),
            }
        )
    interaction_frame = pd.DataFrame(interaction_rows)

    condition_control.to_csv(
        args.output_root / "condition_control_metrics.csv", index=False
    )
    control_metrics.to_csv(args.output_root / "control_deltas.csv", index=False)
    interaction_frame.to_csv(
        args.output_root / "appearance_control_interactions.csv", index=False
    )

    test_metrics = condition_control[condition_control["split"].eq("test")]
    test_deltas = control_metrics[control_metrics["split"].eq("test")]
    clean_test = test_metrics[test_metrics["condition"].eq("clean")]
    clean_original = clean_test[clean_test["control"].eq("original")].iloc[0]
    transformed_original = test_metrics[
        ~test_metrics["condition"].eq("clean")
        & test_metrics["control"].eq("original")
    ]
    full_freezes = test_deltas[
        test_deltas["control"].isin(["freeze_first", "freeze_middle", "freeze_last"])
    ]
    gate = {
        "all_cells_complete": True,
        "clean_raw_voe_headroom": bool(
            clean_original["pair_accuracy_half_ties"] >= 0.6
        ),
        "two_controls_agree_on_event_signal": False,
        "static_false_positive_reduction_demonstrated": False,
        "event_discrimination_preserved": False,
        "heldout_voe_preserved_or_improved": False,
        "qualifies_as_event_sensitive": False,
    }
    summary = {
        "schema_version": 1,
        "scope": config["scope_note"],
        "score_rows": len(scores),
        "pair_rows": len(pivot),
        "test_clean_raw_pair_accuracy_half_ties": float(
            clean_original["pair_accuracy_half_ties"]
        ),
        "test_clean_raw_pair_accuracy_ci": [
            float(clean_original["pair_accuracy_ci_low"]),
            float(clean_original["pair_accuracy_ci_high"]),
        ],
        "test_transformed_raw_pair_accuracy_range": [
            float(transformed_original["pair_accuracy_half_ties"].min()),
            float(transformed_original["pair_accuracy_half_ties"].max()),
        ],
        "test_clean_raw_mean_margin": float(clean_original["mean_margin"]),
        "test_full_freeze_mean_delta_range": [
            float(full_freezes["mean_delta_original_minus_control"].min()),
            float(full_freezes["mean_delta_original_minus_control"].max()),
        ],
        "gate": gate,
        "interpretation": (
            "Full-freeze scores often exceed original-dynamics scores, consistent with a "
            "static-content difficulty component. The 16-frame raw protocol has "
            "insufficient clean VoE headroom, so none of the control deltas is calibrated "
            "physical/event surprise."
        ),
        "limitations": [
            "Original IntPhys O1 feasibility data, not IntPhys2 Main.",
            "Only four test scenes and eight test pairs.",
            "Sixteen frames uniformly sampled over each 100-frame video.",
            "No released event timestamps were used, so event discrimination and static false-positive reduction cannot be validated.",
            "The natural-low-motion window is a different source interval and is a baseline, not an exact counterfactual of the uniform clip.",
            "No pixel-predictive comparator was run.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "scores_sha256": sha256_file(args.scores),
            "config_sha256": sha256_file(args.config),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (args.output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
