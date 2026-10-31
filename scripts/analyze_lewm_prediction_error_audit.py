#!/usr/bin/env python3
"""Analyze paired LeWM prediction-error audit outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser.parse_args()


def interval(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Bootstrap input must be finite and nonempty")
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    args = parse_args()
    scores_path = args.result_dir / "scores.csv"
    manifest_path = args.result_dir / "selection_manifest.csv"
    scores = pd.read_csv(scores_path)
    manifest = pd.read_csv(manifest_path)
    if not np.isfinite(scores["prediction_mse"]).all():
        raise ValueError("Scores contain nonfinite prediction MSE")
    if scores.duplicated(["base_id", "condition"]).any():
        raise ValueError("Duplicate base-condition rows")
    counts = scores.groupby("base_id")["condition"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Base episodes do not share one complete condition matrix")
    if set(scores.base_id) != set(manifest.base_id):
        raise ValueError("Score and manifest base IDs differ")

    task = scores.task.unique()
    if len(task) != 1:
        raise ValueError("Expected one task")
    task = task[0]
    provenance_path = args.result_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    phase = provenance.get("phase", "exploratory")
    seed_material = hashlib.sha256(
        f"{args.result_dir.resolve()}:{args.bootstrap}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(seed_material[:8], "big"))
    clean = scores[scores.condition == "clean"].set_index("base_id").prediction_mse
    if len(clean) != len(manifest):
        raise ValueError("Exactly one clean row per base episode is required")

    merged = scores.merge(manifest, on=["base_id", "episode", "start_step"], validate="many_to_one")
    condition_rows = []
    contrast_rows = []
    for condition, group in merged.groupby("condition", sort=False):
        group = group.set_index("base_id").loc[clean.index]
        values = group.prediction_mse.to_numpy()
        score_low, score_high = interval(values, rng, args.bootstrap)
        deltas = values - clean.to_numpy()
        delta_low, delta_high = interval(deltas, rng, args.bootstrap)
        positive = int((deltas > 0).sum())
        negative = int((deltas < 0).sum())
        ties = len(deltas) - positive - negative
        sign_n = positive + negative
        sign_p = float(binomtest(positive, sign_n, 0.5).pvalue) if sign_n else 1.0
        agent_rho, agent_p = safe_spearman(
            pd.Series(deltas), group.agent_displacement_diameters.reset_index(drop=True)
        )
        block_rho, block_p = safe_spearman(
            pd.Series(deltas), group.block_displacement_state.reset_index(drop=True)
        )
        first = group.iloc[0]
        condition_rows.append(
            {
                "task": task,
                "condition": condition,
                "family": first.family,
                "scope": first.scope,
                "value": first.value,
                "n_episodes": len(values),
                "score_mean": values.mean(),
                "score_ci95_low": score_low,
                "score_ci95_high": score_high,
                "score_median": np.median(values),
                "score_std": values.std(ddof=1),
                "delta_from_clean_mean": deltas.mean(),
                "delta_ci95_low": delta_low,
                "delta_ci95_high": delta_high,
                "delta_from_clean_median": np.median(deltas),
                "positive": positive,
                "negative": negative,
                "ties": ties,
                "two_sided_sign_p": sign_p,
                "delta_agent_displacement_spearman": agent_rho,
                "delta_agent_displacement_p": agent_p,
                "delta_block_displacement_spearman": block_rho,
                "delta_block_displacement_p": block_p,
                "target_rgb_l1_from_clean_01_mean": group.target_rgb_l1_from_clean_01.mean(),
                "target_changed_pixel_fraction_mean": group.get(
                    "target_changed_pixel_fraction", pd.Series(np.nan, index=group.index)
                ).mean(),
            }
        )
        for base_id, delta in zip(clean.index, deltas, strict=True):
            contrast_rows.append(
                {
                    "task": task,
                    "base_id": base_id,
                    "condition": condition,
                    "delta_from_clean": delta,
                }
            )

    summary = pd.DataFrame(condition_rows)
    contrasts = pd.DataFrame(contrast_rows)
    write_csv(args.result_dir / "condition_summary.csv", summary)
    write_csv(args.result_dir / "paired_contrasts.csv", contrasts)

    score_pivot = scores.pivot(index="base_id", columns="condition", values="prediction_mse")
    scope_rows = []
    for family in ("hue", "brightness"):
        family_summary = summary[summary.family == family]
        for value in sorted(family_summary.value.unique()):
            target_name = family_summary[
                (family_summary.value == value) & (family_summary.scope == "target")
            ].condition.item()
            persistent_name = family_summary[
                (family_summary.value == value) & (family_summary.scope == "persistent")
            ].condition.item()
            difference = (
                score_pivot[target_name] - score_pivot[persistent_name]
            ).to_numpy()
            low, high = interval(difference, rng, args.bootstrap)
            scope_rows.append(
                {
                    "task": task,
                    "family": family,
                    "value": value,
                    "target_condition": target_name,
                    "persistent_condition": persistent_name,
                    "n_episodes": len(difference),
                    "target_minus_persistent_mean": difference.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": np.median(difference),
                    "positive": int((difference > 0).sum()),
                }
            )
    scope_summary = pd.DataFrame(scope_rows)
    write_csv(args.result_dir / "target_minus_persistent_summary.csv", scope_summary)

    descriptive_rows = []
    for value in (120.0, 180.0):
        hue_name = summary[
            (summary.family == "hue")
            & (summary.scope == "target")
            & (summary.value == value)
        ].condition.item()
        difference = (
            score_pivot[hue_name] - score_pivot["action_opposite_final"]
        ).to_numpy()
        low, high = interval(difference, rng, args.bootstrap)
        descriptive_rows.append(
            {
                "task": task,
                "hue_degrees": value,
                "comparison": f"{hue_name}_minus_action_opposite_final",
                "n_episodes": len(difference),
                "score_difference_mean": difference.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "positive": int((difference > 0).sum()),
            }
        )
    descriptive = pd.DataFrame(descriptive_rows)
    write_csv(args.result_dir / "cross_construct_descriptive.csv", descriptive)

    action = summary[summary.family == "action"].copy()
    target_hue = summary[(summary.family == "hue") & (summary.scope == "target")]
    persistent_hue = summary[(summary.family == "hue") & (summary.scope == "persistent")]
    strongest_target_hue = target_hue.loc[target_hue.delta_from_clean_mean.idxmax()]
    strongest_action = action.loc[action.delta_from_clean_mean.idxmax()]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ordering = summary.sort_values("delta_from_clean_mean")
    axes[0, 0].barh(ordering.condition, ordering.delta_from_clean_mean)
    axes[0, 0].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xlabel("Prediction MSE change from clean")
    axes[0, 0].set_title(f"{task}: paired intervention effects")
    axes[0, 0].tick_params(axis="y", labelsize=7)

    pivot = contrasts.pivot(index="base_id", columns="condition", values="delta_from_clean")
    manifest_indexed = manifest.set_index("base_id")
    for condition, marker in [("action_zero_final", "o"), ("action_opposite_final", "^")]:
        axes[0, 1].scatter(
            manifest_indexed.loc[pivot.index, "agent_displacement_diameters"],
            pivot[condition],
            alpha=0.7,
            s=24,
            marker=marker,
            label=condition,
        )
    axes[0, 1].set_xlabel("Actual agent displacement (diameters)")
    axes[0, 1].set_ylabel("Prediction MSE change from clean")
    axes[0, 1].set_title("Same-target action inconsistency")
    axes[0, 1].legend(fontsize=8)

    for scope, marker in [("target", "o"), ("persistent", "s")]:
        group = summary[(summary.family == "hue") & (summary.scope == scope)].sort_values("value")
        axes[1, 0].errorbar(
            group.value,
            group.delta_from_clean_mean,
            yerr=[
                group.delta_from_clean_mean - group.delta_ci95_low,
                group.delta_ci95_high - group.delta_from_clean_mean,
            ],
            marker=marker,
            capsize=3,
            label=scope,
        )
    axes[1, 0].set_xlabel("Hue rotation (degrees)")
    axes[1, 0].set_ylabel("Prediction MSE change from clean")
    axes[1, 0].set_title("Appearance dose response: hue")
    axes[1, 0].legend()

    for scope, marker in [("target", "o"), ("persistent", "s")]:
        group = summary[
            (summary.family == "brightness") & (summary.scope == scope)
        ].sort_values("value")
        axes[1, 1].errorbar(
            np.log2(group.value),
            group.delta_from_clean_mean,
            yerr=[
                group.delta_from_clean_mean - group.delta_ci95_low,
                group.delta_ci95_high - group.delta_from_clean_mean,
            ],
            marker=marker,
            capsize=3,
            label=scope,
        )
    axes[1, 1].set_xlabel("Exposure change (stops, log2 factor)")
    axes[1, 1].set_ylabel("Prediction MSE change from clean")
    axes[1, 1].set_title("Appearance dose response: brightness")
    axes[1, 1].legend()
    figure.savefig(args.result_dir / "prediction_error_audit.png", dpi=180)
    plt.close(figure)

    action_names = summary[summary.family == "action"].condition.tolist()
    action_factors = summary.set_index("condition").loc[action_names, "value"].to_numpy()
    mismatch = np.abs(1.0 - action_factors)
    action_episode_rhos = []
    action_episode_monotonic = []
    order = np.argsort(mismatch)
    for base_id in score_pivot.index:
        deltas = score_pivot.loc[base_id, action_names].to_numpy() - score_pivot.loc[
            base_id, "clean"
        ]
        action_episode_rhos.append(float(spearmanr(mismatch, deltas).statistic))
        action_episode_monotonic.append(bool(np.all(np.diff(deltas[order]) >= 0)))

    analysis = {
        "status": f"complete_{phase}_analysis",
        "phase": phase,
        "task": task,
        "n_independent_episodes": int(len(clean)),
        "n_conditions": int(len(summary)),
        "bootstrap_replicates": args.bootstrap,
        "clean_score_mean": float(clean.mean()),
        "clean_score_median": float(clean.median()),
        "strongest_action_condition": strongest_action.condition,
        "strongest_action_delta_mean": float(strongest_action.delta_from_clean_mean),
        "strongest_target_hue_condition": strongest_target_hue.condition,
        "strongest_target_hue_delta_mean": float(
            strongest_target_hue.delta_from_clean_mean
        ),
        "action_dose_episode_median_spearman": float(np.median(action_episode_rhos)),
        "action_dose_perfectly_monotonic_episodes": int(sum(action_episode_monotonic)),
        "target_minus_persistent": scope_summary.to_dict(orient="records"),
        "target_hue_minus_opposite_action_descriptive": descriptive.to_dict(
            orient="records"
        ),
        "interpretation_boundary": (
            "These scores establish representation-space prediction difficulty. "
            "Action conditions isolate action-outcome inconsistency with identical target bytes; "
            "they are not a reconstruction of the paper's random teleport intervention."
        ),
    }
    (args.result_dir / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )

    selected = summary[
        summary.condition.isin(
            [
                "clean",
                "action_zero_final",
                "action_opposite_final",
                "hue_target_060",
                "hue_target_120",
                "hue_persistent_120",
                "brightness_target_0p85",
                "brightness_persistent_0p85",
            ]
        )
    ]
    lines = [
        f"# Exploratory LeWM prediction-error audit: {task}",
        "",
        (
            "Each row below summarizes a paired intervention over independent official-dataset "
            "episodes; intervals are episode-bootstrap 95% intervals for the mean change from clean."
        ),
        "",
        "| Condition | Mean score | Mean Δ from clean | 95% interval | Positive pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples():
        lines.append(
            f"| {row.condition} | {row.score_mean:.6f} | {row.delta_from_clean_mean:.6f} "
            f"| [{row.delta_ci95_low:.6f}, {row.delta_ci95_high:.6f}] "
            f"| {row.positive}/{row.n_episodes} |"
        )
    lines.extend(
        [
            "",
            f"The {phase} experiment measures latent prediction difficulty; physical, novelty, and behavioral interpretations require their own controls.",
            "",
        ]
    )
    (args.result_dir / "analysis.md").write_text("\n".join(lines))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
