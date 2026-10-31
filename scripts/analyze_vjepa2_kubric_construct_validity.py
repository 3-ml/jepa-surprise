#!/usr/bin/env python3
"""Analyze the exploratory V-JEPA2-G Kubric construct-validity pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INTERVENTION_CONDITIONS = (
    "hue-030",
    "hue-060",
    "hue-120",
    "hue-180",
    "physical-z-0p25",
    "physical-z-0p5",
    "physical-z-1p0",
    "physical-z-1p5",
)
CONTEXT = 12
CONTROL_NAMES = ("target_hold", "target_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--render-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=2308)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    key: str,
    replicates: int,
) -> tuple[float, float]:
    material = f"{seed}:{key}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = np.random.default_rng(derived_seed)
    sampled = values[rng.integers(0, len(values), size=(replicates, len(values)))]
    means = sampled.mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def spearman(x, y) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.scores)
    render = pd.read_csv(args.render_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted(scores["scene_id"].unique())
    if len(scenes) != 8 or len(scores) != 88:
        raise ValueError(f"Expected 8 scenes and 88 scores, found {len(scenes)} and {len(scores)}")
    counts = scores.groupby("scene_id")["condition"].agg(list)
    expected_conditions = ("clean", *INTERVENTION_CONDITIONS, *CONTROL_NAMES)
    expected_set = set(expected_conditions)
    if any(set(values) != expected_set for values in counts):
        raise ValueError("Each scene must contain all 11 conditions exactly once")
    if not np.isfinite(scores["scalar_l1"]).all():
        raise ValueError("Nonfinite scores")
    if scores["predictor_max_abs_difference_from_clean"].max() > 1e-4:
        raise ValueError("Same-context predictor invariant failed")

    clean = scores[scores["condition"] == "clean"].set_index("scene_id")
    scores["score_delta_from_clean"] = scores.apply(
        lambda row: row["scalar_l1"] - clean.loc[row["scene_id"], "scalar_l1"], axis=1
    )
    clean_temporal = {
        scene_id: np.asarray(json.loads(row["temporal_tubelet_l1"]), dtype=np.float64)
        for scene_id, row in clean.iterrows()
    }
    scores["temporal_delta"] = scores.apply(
        lambda row: np.asarray(json.loads(row["temporal_tubelet_l1"]), dtype=np.float64)
        - clean_temporal[row["scene_id"]],
        axis=1,
    )
    merged = scores.merge(
        render,
        on=["scene_id", "seed", "condition", "family", "dose", "dose_unit", "target_object"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_render"),
    )
    summary_rows = []
    for condition, group in scores.groupby("condition", sort=False):
        values = group["score_delta_from_clean"].to_numpy(dtype=np.float64)
        low, high = bootstrap_mean(
            values,
            seed=args.bootstrap_seed,
            key=f"condition:{condition}:scalar_delta",
            replicates=args.bootstrap_replicates,
        )
        temporal = np.stack(group["temporal_delta"].to_numpy())
        summary_rows.append(
            {
                "condition": condition,
                "family": group["family"].iloc[0],
                "dose": float(group["dose"].iloc[0]),
                "mean_scalar_l1": float(group["scalar_l1"].mean()),
                "mean_delta_from_clean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_scenes": int(np.count_nonzero(values > 0)),
                "scenes": len(values),
                "mean_first_target_tubelet_delta": float(temporal[:, 0].mean()),
                "mean_target_shift_l1_layernorm": float(
                    group["target_shift_l1_layernorm"].mean()
                ),
            }
        )
    write_csv(args.output_dir / "condition_summary.csv", summary_rows)
    summary_by_condition = {row["condition"]: row for row in summary_rows}

    per_scene_rows = []
    for scene_id in scenes:
        scene = scores[scores["scene_id"] == scene_id].set_index("condition")
        row = {"scene_id": scene_id}
        for condition in expected_conditions:
            row[f"delta_{condition}"] = float(
                scene.loc[condition, "score_delta_from_clean"]
            )
        for family, conditions, doses in (
            ("appearance", ["clean", "hue-030", "hue-060", "hue-120", "hue-180"], [0, 30, 60, 120, 180]),
            ("physical", ["clean", "physical-z-0p25", "physical-z-0p5", "physical-z-1p0", "physical-z-1p5"], [0, 0.25, 0.5, 1.0, 1.5]),
        ):
            row[f"{family}_dose_spearman"] = spearman(
                doses, [scene.loc[condition, "scalar_l1"] for condition in conditions]
            )
        interventions = merged[
            (merged["scene_id"] == scene_id)
            & merged["condition"].isin(INTERVENTION_CONDITIONS)
        ]
        row["score_delta_vs_rgb_l1_spearman"] = spearman(
            interventions["score_delta_from_clean"], interventions["target_rgb_l1_01"]
        )
        row["score_delta_vs_target_shift_spearman"] = spearman(
            interventions["score_delta_from_clean"],
            interventions["target_shift_l1_layernorm"],
        )
        per_scene_rows.append(row)
    write_csv(args.output_dir / "per_scene_deltas.csv", per_scene_rows)

    render_means = (
        render[render["condition"].isin(INTERVENTION_CONDITIONS)]
        .groupby("condition")["target_rgb_l1_01"]
        .mean()
    )
    appearance = ["hue-030", "hue-060", "hue-120", "hue-180"]
    physical = ["physical-z-0p25", "physical-z-0p5", "physical-z-1p0", "physical-z-1p5"]
    matched_rows = []
    for appearance_condition in appearance:
        physical_condition = min(
            physical,
            key=lambda condition: abs(
                render_means[condition] - render_means[appearance_condition]
            ),
        )
        rendered_relative_gap = abs(
            render_means[physical_condition] - render_means[appearance_condition]
        ) / np.mean([render_means[physical_condition], render_means[appearance_condition]])
        if rendered_relative_gap > 0.20:
            continue
        a = scores[scores["condition"] == appearance_condition].set_index("scene_id")
        p = scores[scores["condition"] == physical_condition].set_index("scene_id")
        values = (
            p["score_delta_from_clean"] - a["score_delta_from_clean"]
        ).to_numpy()
        low, high = bootstrap_mean(
            values,
            seed=args.bootstrap_seed,
            key=f"matched:{appearance_condition}:{physical_condition}:scalar",
            replicates=args.bootstrap_replicates,
        )
        first_tubelet_values = np.asarray(
            [
                p.loc[scene_id, "temporal_delta"][0]
                - a.loc[scene_id, "temporal_delta"][0]
                for scene_id in scenes
            ],
            dtype=np.float64,
        )
        first_low, first_high = bootstrap_mean(
            first_tubelet_values,
            seed=args.bootstrap_seed,
            key=f"matched:{appearance_condition}:{physical_condition}:first_tubelet",
            replicates=args.bootstrap_replicates,
        )
        matched_rows.append(
            {
                "appearance_condition": appearance_condition,
                "physical_condition": physical_condition,
                "appearance_mean_rgb_l1": float(render_means[appearance_condition]),
                "physical_mean_rgb_l1": float(render_means[physical_condition]),
                "rendered_relative_gap": float(rendered_relative_gap),
                "physical_minus_appearance_score_delta": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_scenes": int(np.count_nonzero(values > 0)),
                "first_tubelet_physical_minus_appearance": float(
                    first_tubelet_values.mean()
                ),
                "first_tubelet_ci95_low": first_low,
                "first_tubelet_ci95_high": first_high,
                "scenes": len(values),
                "physical_minus_appearance_target_shift_l1": float(
                    p["target_shift_l1_layernorm"].mean()
                    - a["target_shift_l1_layernorm"].mean()
                ),
            }
        )
    write_csv(args.output_dir / "matched_comparisons.csv", matched_rows)

    temporal_rows = []
    for condition in ["clean", *INTERVENTION_CONDITIONS, *CONTROL_NAMES]:
        group = scores[scores["condition"] == condition]
        values = np.stack(group["temporal_delta"].to_numpy())
        for tubelet in range(18):
            low, high = bootstrap_mean(
                values[:, tubelet],
                seed=args.bootstrap_seed,
                key=f"temporal:{condition}:{tubelet}",
                replicates=args.bootstrap_replicates,
            )
            temporal_rows.append(
                {
                    "condition": condition,
                    "target_tubelet_index": tubelet,
                    "source_frame_0based_a": CONTEXT + 2 * tubelet,
                    "source_frame_0based_b": CONTEXT + 2 * tubelet + 1,
                    "mean_delta_from_clean": float(values[:, tubelet].mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(args.output_dir / "temporal_summary.csv", temporal_rows)

    shuffle = scores[scores["condition"] == "target_shuffle"][
        "score_delta_from_clean"
    ].to_numpy()
    shuffle_low = summary_by_condition["target_shuffle"]["ci95_low"]
    shuffle_high = summary_by_condition["target_shuffle"]["ci95_high"]
    hold = scores[scores["condition"] == "target_hold"][
        "score_delta_from_clean"
    ].to_numpy()
    hold_low = summary_by_condition["target_hold"]["ci95_low"]
    hold_high = summary_by_condition["target_hold"]["ci95_high"]
    intervention_means = (
        merged[merged["condition"].isin(INTERVENTION_CONDITIONS)]
        .groupby("condition")
        .agg(
            score_delta=("score_delta_from_clean", "mean"),
            rgb_l1=("target_rgb_l1_01", "mean"),
            target_shift=("target_shift_l1_layernorm", "mean"),
        )
    )
    high_dose_influence = {}
    for condition in ("physical-z-1p0", "physical-z-1p5"):
        values = scores[scores["condition"] == condition].set_index("scene_id")[
            "score_delta_from_clean"
        ]
        leave_one_out = {
            scene_id: float(values.drop(scene_id).mean()) for scene_id in values.index
        }
        high_dose_influence[condition] = {
            "full_mean": float(values.mean()),
            "leave_one_out_min": min(leave_one_out.values()),
            "leave_one_out_max": max(leave_one_out.values()),
            "largest_value_scene": str(values.idxmax()),
            "largest_value": float(values.max()),
        }
    analysis = {
        "status": "complete_exploratory_pilot",
        "scenes": 8,
        "scores": len(scores),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_streams": "independent SHA-256-derived NumPy streams per estimand key",
        "same_context_predictor_max_abs_difference": float(
            scores["predictor_max_abs_difference_from_clean"].max()
        ),
        "clean_score_mean": float(clean["scalar_l1"].mean()),
        "clean_score_between_scene_sd": float(clean["scalar_l1"].std(ddof=1)),
        "clean_predictive_sensitivity_gate": {
            "passed": bool(np.count_nonzero(shuffle > 0) >= 6 and shuffle_low > 0),
            "target_shuffle_mean_delta": float(shuffle.mean()),
            "target_shuffle_ci95": [shuffle_low, shuffle_high],
            "target_shuffle_positive_scenes": int(np.count_nonzero(shuffle > 0)),
            "target_hold_mean_delta": float(hold.mean()),
            "target_hold_ci95": [hold_low, hold_high],
            "target_hold_positive_scenes": int(np.count_nonzero(hold > 0)),
        },
        "aggregate_condition_mean_correlations": {
            "score_delta_vs_rgb_l1_spearman": spearman(
                intervention_means["score_delta"], intervention_means["rgb_l1"]
            ),
            "score_delta_vs_target_shift_spearman": spearman(
                intervention_means["score_delta"], intervention_means["target_shift"]
            ),
        },
        "median_per_scene_correlations": {
            "score_delta_vs_rgb_l1_spearman": float(
                np.median([row["score_delta_vs_rgb_l1_spearman"] for row in per_scene_rows])
            ),
            "score_delta_vs_target_shift_spearman": float(
                np.median([row["score_delta_vs_target_shift_spearman"] for row in per_scene_rows])
            ),
            "appearance_dose_spearman": float(
                np.median([row["appearance_dose_spearman"] for row in per_scene_rows])
            ),
            "physical_dose_spearman": float(
                np.median([row["physical_dose_spearman"] for row in per_scene_rows])
            ),
        },
        "high_physical_dose_scene_influence": high_dose_influence,
        "matched_comparisons": matched_rows,
        "scores_sha256": sha256_file(args.scores),
        "render_metrics_sha256": sha256_file(args.render_metrics),
        "interpretation_boundary": "Scores establish representation-space prediction-difficulty responses for direct state displacement and appearance events; they do not establish calibrated physical surprise.",
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
