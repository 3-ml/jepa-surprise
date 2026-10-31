#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACTION_SCALES = (0.25, 0.5, 0.75, 1.0)
HUE_SCALE_MATCHES = {30.0: 0.25, 60.0: 0.75, 120.0: 1.0, 180.0: 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dino-result-dir", type=Path, required=True)
    return parser.parse_args()


def scale_token(value: float) -> str:
    return str(value).replace(".", "p")


def close(actual: float, expected: float, name: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise RuntimeError(f"{name}: {actual} != {expected}")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = 10_000) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(n, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.result_dir / "scores.csv")
    condition_summary = pd.read_csv(args.result_dir / "condition_summary.csv")
    action_summary = pd.read_csv(args.result_dir / "same_target_contrasts.csv")
    construct_summary = pd.read_csv(args.result_dir / "construct_comparisons.csv")
    selected = pd.read_csv(args.result_dir / "selection_manifest.csv")
    provenance = json.loads((args.result_dir / "provenance.json").read_text())
    dino_scores = pd.read_csv(args.dino_result_dir / "scores.csv")
    dino_selected = pd.read_csv(args.dino_result_dir / "selection_manifest.csv")

    if len(scores) != 256 * 18 or scores.base_id.nunique() != 256 or scores.condition.nunique() != 18:
        raise RuntimeError("Unexpected score matrix dimensions")
    if scores.duplicated(["base_id", "condition"]).any():
        raise RuntimeError("Duplicate base-condition rows")
    if not scores.groupby("base_id").size().eq(18).all():
        raise RuntimeError("Incomplete per-episode condition support")
    if not np.isfinite(scores.prediction_mse).all():
        raise RuntimeError("Nonfinite scores")
    identity_columns = ["base_id", "dataset_index", "episode", "start_step"]
    if not selected[identity_columns].equals(dino_selected[identity_columns]):
        raise RuntimeError("JEPA-WM and DINO-WM selection identities differ")

    target_join = scores.merge(
        dino_scores[["base_id", "condition", "target_sha256"]],
        on=["base_id", "condition"],
        suffixes=("_jepa", "_dino"),
        validate="one_to_one",
    )
    if len(target_join) != len(scores) or not target_join.target_sha256_jepa.eq(target_join.target_sha256_dino).all():
        raise RuntimeError("JEPA-WM and DINO-WM target bytes differ")

    pivot = scores.pivot(index="base_id", columns="condition", values="prediction_mse")
    rng = np.random.default_rng(20260903)
    for row in condition_summary.itertuples(index=False):
        values = pivot[row.condition].to_numpy()
        close(float(values.mean()), float(row.score_mean), f"condition mean {row.condition}")
        low, high = bootstrap_mean(values, rng)
        close(low, float(row.score_ci95_low), f"condition CI low {row.condition}")
        close(high, float(row.score_ci95_high), f"condition CI high {row.condition}")
        if len(values) != row.n_episodes:
            raise RuntimeError(f"Condition count mismatch for {row.condition}")

    action_means = {}
    for scale in ACTION_SCALES:
        token = scale_token(scale)
        lawful = pivot[f"lawful_scale_{token}"]
        for mode in ("zero", "opposite"):
            values = pivot[f"{mode}_scale_{token}"] - lawful
            row = action_summary[
                (action_summary.action_mode == mode)
                & np.isclose(action_summary.intended_displacement_scale, scale)
            ].iloc[0]
            close(float(values.mean()), float(row.inconsistent_minus_lawful_mean), f"{mode} {scale}")
            low, high = bootstrap_mean(values.to_numpy(), rng)
            close(low, float(row.ci95_low), f"{mode} CI low {scale}")
            close(high, float(row.ci95_high), f"{mode} CI high {scale}")
            action_means[(mode, scale)] = float(values.mean())

    construct_means = {}
    static = pivot["lawful_scale_0p0"]
    for degrees, scale in HUE_SCALE_MATCHES.items():
        hue = pivot[f"appearance_hue_{int(degrees):03d}"]
        row = construct_summary[np.isclose(construct_summary.hue_degrees, degrees)].iloc[0]
        appearance_delta = hue - static
        close(float(appearance_delta.mean()), float(row.appearance_minus_static_mean), f"hue-static {degrees}")
        low, high = bootstrap_mean(appearance_delta.to_numpy(), rng)
        close(low, float(row.appearance_ci95_low), f"hue-static CI low {degrees}")
        close(high, float(row.appearance_ci95_high), f"hue-static CI high {degrees}")
        for mode in ("zero", "opposite"):
            values = hue - pivot[f"{mode}_scale_{scale_token(scale)}"]
            expected = float(row[f"appearance_minus_{mode}_mean"])
            close(float(values.mean()), expected, f"hue-{mode} {degrees}")
            low, high = bootstrap_mean(values.to_numpy(), rng)
            close(low, float(row[f"appearance_minus_{mode}_ci95_low"]), f"hue-{mode} CI low {degrees}")
            close(high, float(row[f"appearance_minus_{mode}_ci95_high"]), f"hue-{mode} CI high {degrees}")
            construct_means[(degrees, mode)] = float(values.mean())

    zero_ordered = all(action_means[("zero", left)] < action_means[("zero", right)] for left, right in zip(ACTION_SCALES, ACTION_SCALES[1:]))
    opposite_ordered = all(
        action_means[("opposite", left)] < action_means[("opposite", right)]
        for left, right in zip(ACTION_SCALES, ACTION_SCALES[1:])
    )
    opposite_above_zero = all(
        action_means[("opposite", scale)] > action_means[("zero", scale)] for scale in ACTION_SCALES
    )
    all_action_intervals_positive = bool((action_summary.ci95_low > 0).all())
    reported_hue_intervals_negative = bool(
        (
            construct_summary[construct_summary.hue_degrees.isin([30.0, 60.0])][
                "appearance_minus_opposite_ci95_high"
            ]
            < 0
        ).all()
    )

    validation = {
        "status": "passed",
        "rows": len(scores),
        "episodes": int(scores.base_id.nunique()),
        "conditions": int(scores.condition.nunique()),
        "selection_identity_with_dino_wm": True,
        "target_byte_identity_with_dino_wm": True,
        "all_bootstrap_intervals_recomputed": True,
        "target_hash_mismatches": int(provenance["target_hash_mismatches"]),
        "maximum_start_state_error": provenance["maximum_start_state_error"],
        "maximum_official_endpoint_error": provenance["maximum_official_endpoint_error"],
        "maximum_same_context_visual_embedding_difference": provenance[
            "maximum_same_context_visual_embedding_difference"
        ],
        "maximum_same_target_visual_embedding_difference": provenance[
            "maximum_same_target_visual_embedding_difference"
        ],
        "maximum_repeated_score_difference": provenance["maximum_repeated_score_difference"],
        "all_action_intervals_positive": all_action_intervals_positive,
        "zero_action_contrasts_strictly_increase": zero_ordered,
        "opposite_action_contrasts_strictly_increase": opposite_ordered,
        "opposite_above_zero_at_every_scale": opposite_above_zero,
        "reported_hue_minus_opposite_intervals_negative": reported_hue_intervals_negative,
        "paper_values": {
            "zero_minus_lawful_0.25": action_means[("zero", 0.25)],
            "zero_minus_lawful_1.0": action_means[("zero", 1.0)],
            "opposite_minus_lawful_0.25": action_means[("opposite", 0.25)],
            "opposite_minus_lawful_1.0": action_means[("opposite", 1.0)],
            "hue_minus_opposite_30deg_0.25": construct_means[(30.0, "opposite")],
            "hue_minus_opposite_60deg_0.75": construct_means[(60.0, "opposite")],
        },
    }
    if not all(
        [
            all_action_intervals_positive,
            zero_ordered,
            opposite_ordered,
            opposite_above_zero,
            reported_hue_intervals_negative,
        ]
    ):
        raise RuntimeError(json.dumps(validation, indent=2))
    output = args.result_dir / "validation.json"
    output.write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
