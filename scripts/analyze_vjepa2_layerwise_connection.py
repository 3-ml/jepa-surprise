#!/usr/bin/env python3
"""Connect V-JEPA2 layerwise probes, representation distances, and static-control metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRANSFORM_ORDER = ["grayscale", "brightness_contrast", "hue_shift"]
STATIC_CORRELATION_METRICS = [
    "raw_original_mean",
    "delta_original_minus_multifreeze_mean",
    "abs_delta_original_minus_multifreeze_mean",
    "delta_original_minus_target_hold_mean",
    "delta_original_minus_target_shuffle_mean",
    "motion_global_consecutive_l1_01_mean",
    "motion_patch_p95_consecutive_l1_01_mean",
    "motion_first_last_l1_01_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--analysis-windows", type=Path, required=True)
    parser.add_argument("--pair-outcomes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def metadata_from_payload(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    columns = [
        "sample_id",
        "name",
        "file_name",
        "pair_id",
        "pair_label",
        "pair_label_value",
        "appearance_condition",
        "condition",
        "difficulty",
        "camera",
        "window_count",
    ]
    return pd.DataFrame({column: payload[column] for column in columns})


def cosine_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def off_diagonal_median_cosine_distance(z: np.ndarray) -> float:
    if len(z) < 2:
        return 1.0
    distances = 1.0 - z @ z.T
    mask = ~np.eye(len(z), dtype=bool)
    value = float(np.median(distances[mask]))
    return value if value > 1e-12 else 1.0


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def aggregate_static_metrics(analysis_windows: pd.DataFrame) -> pd.DataFrame:
    windows = analysis_windows.copy()
    full_controls = ["full_start", "full_boundary", "full_end"]
    windows["multifreeze"] = windows[full_controls].mean(axis=1)
    windows["delta_original_minus_multifreeze"] = (
        windows["original"] - windows["multifreeze"]
    )
    windows["abs_delta_original_minus_multifreeze"] = windows[
        "delta_original_minus_multifreeze"
    ].abs()
    agg_spec = {
        "pair_id": "first",
        "pair_label": "first",
        "condition": "first",
        "difficulty": "first",
        "camera": "first",
        "window_id": "count",
        "original": "mean",
        "multifreeze": "mean",
        "delta_original_minus_multifreeze": "mean",
        "abs_delta_original_minus_multifreeze": "mean",
        "delta_original_minus_target_hold": "mean",
        "delta_original_minus_target_shuffle": "mean",
        "motion_global_consecutive_l1_01": "mean",
        "motion_patch_p95_consecutive_l1_01": "mean",
        "motion_first_last_l1_01": "mean",
        "lowest_visual_change_decile": "mean",
    }
    video = windows.groupby("name", as_index=False).agg(agg_spec)
    video = video.rename(
        columns={
            "window_id": "window_count_static",
            "original": "raw_original_mean",
            "multifreeze": "multifreeze_mean",
            "delta_original_minus_multifreeze": "delta_original_minus_multifreeze_mean",
            "abs_delta_original_minus_multifreeze": "abs_delta_original_minus_multifreeze_mean",
            "delta_original_minus_target_hold": "delta_original_minus_target_hold_mean",
            "delta_original_minus_target_shuffle": "delta_original_minus_target_shuffle_mean",
            "motion_global_consecutive_l1_01": "motion_global_consecutive_l1_01_mean",
            "motion_patch_p95_consecutive_l1_01": "motion_patch_p95_consecutive_l1_01_mean",
            "motion_first_last_l1_01": "motion_first_last_l1_01_mean",
            "lowest_visual_change_decile": "lowest_visual_change_fraction",
        }
    )
    return video


def compute_layer_geometry(
    payload: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = payload["X"]
    layer_names = [str(value) for value in payload["layer_names"]]
    original_mask = metadata["appearance_condition"].eq("original").to_numpy()
    original_indices = np.flatnonzero(original_mask)
    original_names = metadata.iloc[original_indices]["name"].to_numpy()
    original_index_by_name = {
        name: index for name, index in zip(original_names.tolist(), original_indices.tolist())
    }

    layer_rows: list[dict] = []
    video_rows: list[dict] = []
    physical_rows: list[dict] = []
    for layer_index, layer_name in enumerate(layer_names):
        z = cosine_normalize(x[:, layer_index, :])
        z_original = z[original_indices]
        normalization = off_diagonal_median_cosine_distance(z_original)
        retrieval_hits: list[float] = []
        all_transform_distances: list[float] = []
        condition_distances: dict[str, list[float]] = {
            condition: [] for condition in TRANSFORM_ORDER
        }

        for name, group in metadata.groupby("name", sort=True):
            if name not in original_index_by_name:
                continue
            base_index = original_index_by_name[name]
            row = {
                "name": name,
                "pair_id": str(group["pair_id"].iloc[0]),
                "pair_label": str(group["pair_label"].iloc[0]),
                "condition": str(group["condition"].iloc[0]),
                "difficulty": str(group["difficulty"].iloc[0]),
                "camera": str(group["camera"].iloc[0]),
                "window_count": int(group["window_count"].iloc[0]),
                "layer_index": layer_index,
                "layer_name": layer_name,
                "normalization_original_pairwise_median": normalization,
            }
            per_video_distances: list[float] = []
            for condition in TRANSFORM_ORDER:
                condition_index = group.index[
                    group["appearance_condition"].eq(condition)
                ]
                if len(condition_index) != 1:
                    row[f"appearance_distance_{condition}"] = np.nan
                    continue
                distance = float(
                    (1.0 - z[base_index] @ z[int(condition_index[0])]) / normalization
                )
                row[f"appearance_distance_{condition}"] = distance
                per_video_distances.append(distance)
                all_transform_distances.append(distance)
                condition_distances[condition].append(distance)

                nearest = original_names[int(np.argmin(1.0 - z_original @ z[int(condition_index[0])] ))]
                retrieval_hits.append(float(nearest == name))
            row["appearance_distance_mean"] = float(np.mean(per_video_distances))
            row["appearance_distance_median"] = float(np.median(per_video_distances))
            video_rows.append(row)

        physical_distances: list[float] = []
        physical_original_distances: list[float] = []
        for (pair_id, appearance), group in metadata.groupby(
            ["pair_id", "appearance_condition"], sort=True
        ):
            if len(group) != 2 or group["pair_label"].nunique() != 2:
                continue
            possible = group[group["pair_label"].eq("possible")]
            impossible = group[group["pair_label"].eq("impossible")]
            if len(possible) != 1 or len(impossible) != 1:
                continue
            distance = float(
                (
                    1.0
                    - z[int(possible.index[0])] @ z[int(impossible.index[0])]
                )
                / normalization
            )
            physical_rows.append(
                {
                    "pair_id": str(pair_id),
                    "appearance_condition": str(appearance),
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "normalized_matched_physical_distance": distance,
                }
            )
            physical_distances.append(distance)
            if appearance == "original":
                physical_original_distances.append(distance)

        appearance_median = float(np.median(all_transform_distances))
        physical_median = float(np.median(physical_distances))
        layer_row = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "normalization_original_pairwise_median": normalization,
            "appearance_distance_mean": float(np.mean(all_transform_distances)),
            "appearance_distance_median": appearance_median,
            "appearance_retrieval_top1": float(np.mean(retrieval_hits)),
            "physical_distance_mean": float(np.mean(physical_distances)),
            "physical_distance_median": physical_median,
            "physical_original_distance_median": float(
                np.median(physical_original_distances)
            ),
            "appearance_to_physical_distance_ratio": appearance_median
            / max(physical_median, 1e-12),
            "n_appearance_distances": len(all_transform_distances),
            "n_physical_distances": len(physical_distances),
        }
        for condition, values in condition_distances.items():
            layer_row[f"appearance_distance_{condition}_median"] = float(
                np.median(values)
            )
        layer_rows.append(layer_row)

    return pd.DataFrame(layer_rows), pd.DataFrame(video_rows), pd.DataFrame(physical_rows)


def load_probe_curves(probe_summary: pd.DataFrame) -> pd.DataFrame:
    test = probe_summary[probe_summary["split"].eq("test")].copy()
    rows = []
    for layer_index, group in test.groupby("layer_index", sort=True):
        row = {
            "layer_index": int(layer_index),
            "layer_name": str(group["layer_name"].iloc[0]),
        }
        for task in ("appearance", "physics_all", "physics_original_only"):
            task_row = group[group["task"].eq(task)]
            if task_row.empty:
                continue
            row[f"{task}_accuracy_mean"] = float(task_row["accuracy_mean"].iloc[0])
            row[f"{task}_accuracy_std"] = float(task_row["accuracy_std"].iloc[0])
            row[f"{task}_baseline_mean"] = float(
                task_row["majority_baseline_mean"].iloc[0]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def static_correlations_by_layer(video_distances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer_index, group in video_distances.groupby("layer_index", sort=True):
        for metric in STATIC_CORRELATION_METRICS:
            rows.append(
                {
                    "layer_index": int(layer_index),
                    "layer_name": str(group["layer_name"].iloc[0]),
                    "static_metric": metric,
                    "spearman_r": spearman(group["appearance_distance_mean"], group[metric]),
                    "n_videos": int(group[["appearance_distance_mean", metric]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def cross_layer_correlations(layer_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "appearance_accuracy_mean",
        "physics_all_accuracy_mean",
        "physics_original_only_accuracy_mean",
        "appearance_distance_median",
        "physical_distance_median",
        "appearance_to_physical_distance_ratio",
        "appearance_retrieval_top1",
    ]
    rows = []
    for left in columns:
        if left not in layer_metrics:
            continue
        for right in columns + ["layer_index"]:
            if right not in layer_metrics or left >= right:
                continue
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_r": spearman(layer_metrics[left], layer_metrics[right]),
                    "n_layers": int(layer_metrics[[left, right]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def attach_pair_outcomes(
    physical_distances: pd.DataFrame,
    pair_outcomes: pd.DataFrame | None,
) -> pd.DataFrame:
    if pair_outcomes is None:
        return physical_distances
    raw = pair_outcomes[pair_outcomes["score_name"].eq("raw_original")][
        ["pair_id", "margin", "correct"]
    ].rename(columns={"margin": "raw_original_margin", "correct": "raw_original_correct"})
    return physical_distances.merge(raw, on="pair_id", how="left")


def draw_connection_plot(layer_metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    for column, label, color in [
        ("appearance_accuracy_mean", "Appearance probe", "#0072B2"),
        ("physics_all_accuracy_mean", "Physics all", "#D55E00"),
        ("physics_original_only_accuracy_mean", "Physics original", "#009E73"),
    ]:
        if column in layer_metrics:
            ax.plot(
                layer_metrics["layer_index"],
                layer_metrics[column],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
                color=color,
            )
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for column, label, color in [
        ("appearance_distance_median", "Same-video appearance displacement", "#0072B2"),
        ("physical_distance_median", "Matched physical displacement", "#D55E00"),
    ]:
        ax.plot(
            layer_metrics["layer_index"],
            layer_metrics[column],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=label,
            color=color,
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Normalized cosine distance")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")

    fig.suptitle("V-JEPA2-G layerwise probes and representation sensitivity")
    fig.tight_layout()
    fig.savefig(output, dpi=200)


def draw_probe_distance_scatter(layer_metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        layer_metrics["appearance_distance_median"],
        layer_metrics["appearance_accuracy_mean"],
        c=layer_metrics["layer_index"],
        cmap="viridis",
        s=48,
    )
    for _, row in layer_metrics.iterrows():
        if int(row["layer_index"]) in {0, 10, 18, 25, 27, 40}:
            ax.annotate(
                str(int(row["layer_index"])),
                (row["appearance_distance_median"], row["appearance_accuracy_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Median normalized appearance displacement")
    ax.set_ylabel("Appearance probe test accuracy")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)


def draw_static_correlation_plot(static_correlations: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = {
        "delta_original_minus_multifreeze_mean": "multi-freeze delta",
        "abs_delta_original_minus_multifreeze_mean": "abs multi-freeze delta",
        "motion_global_consecutive_l1_01_mean": "global motion",
        "motion_first_last_l1_01_mean": "first-last motion",
    }
    for metric, label in labels.items():
        rows = static_correlations[static_correlations["static_metric"].eq(metric)]
        ax.plot(
            rows["layer_index"],
            rows["spearman_r"],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=label,
        )
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Spearman r with per-video appearance displacement")
    ax.set_ylim(-1.02, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_npz(args.representations)
    metadata = metadata_from_payload(payload)
    analysis_windows = pd.read_csv(args.analysis_windows)
    probe_summary = pd.read_csv(args.probe_summary)
    pair_outcomes = pd.read_csv(args.pair_outcomes) if args.pair_outcomes else None

    layer_geometry, video_distances, physical_distances = compute_layer_geometry(
        payload, metadata
    )
    static_metrics = aggregate_static_metrics(analysis_windows)
    video_distances = video_distances.merge(
        static_metrics, on=["name", "pair_id", "pair_label", "condition", "difficulty", "camera"], how="left"
    )
    physical_distances = attach_pair_outcomes(physical_distances, pair_outcomes)
    static_correlations = static_correlations_by_layer(video_distances)
    probe_curves = load_probe_curves(probe_summary)
    layer_metrics = layer_geometry.merge(
        probe_curves, on=["layer_index", "layer_name"], how="left"
    )
    cross_layer = cross_layer_correlations(layer_metrics)

    layer_metrics.to_csv(args.output_dir / "layerwise_connection_metrics.csv", index=False)
    video_distances.to_csv(args.output_dir / "video_layer_appearance_static_metrics.csv", index=False)
    physical_distances.to_csv(args.output_dir / "physical_layer_distances.csv", index=False)
    static_correlations.to_csv(args.output_dir / "static_sensitivity_correlations_by_layer.csv", index=False)
    cross_layer.to_csv(args.output_dir / "cross_layer_correlations.csv", index=False)

    draw_connection_plot(layer_metrics, args.output_dir / "layerwise_connection_curves.png")
    draw_probe_distance_scatter(layer_metrics, args.output_dir / "probe_vs_appearance_distance.png")
    draw_static_correlation_plot(static_correlations, args.output_dir / "static_sensitivity_correlation_by_layer.png")

    key_pairs = {
        "appearance_accuracy_vs_appearance_distance": (
            "appearance_accuracy_mean",
            "appearance_distance_median",
        ),
        "appearance_accuracy_vs_depth": ("appearance_accuracy_mean", "layer_index"),
        "appearance_distance_vs_depth": ("appearance_distance_median", "layer_index"),
        "physics_all_accuracy_vs_depth": ("physics_all_accuracy_mean", "layer_index"),
        "physical_distance_vs_depth": ("physical_distance_median", "layer_index"),
    }
    summary = {
        "representations": str(args.representations),
        "representations_sha256": sha256_file(args.representations),
        "probe_summary": str(args.probe_summary),
        "probe_summary_sha256": sha256_file(args.probe_summary),
        "analysis_windows": str(args.analysis_windows),
        "analysis_windows_sha256": sha256_file(args.analysis_windows),
        "pair_outcomes": None if args.pair_outcomes is None else str(args.pair_outcomes),
        "n_samples": int(len(metadata)),
        "n_layers": int(len(layer_metrics)),
        "n_videos": int(metadata["name"].nunique()),
        "n_pairs": int(metadata["pair_id"].nunique()),
        "appearance_conditions": sorted(metadata["appearance_condition"].unique().tolist()),
        "key_cross_layer_spearman": {
            name: spearman(layer_metrics[left], layer_metrics[right])
            for name, (left, right) in key_pairs.items()
        },
        "best_layers": {
            "appearance_accuracy": int(
                layer_metrics.sort_values(
                    ["appearance_accuracy_mean", "layer_index"],
                    ascending=[False, True],
                ).iloc[0]["layer_index"]
            ),
            "appearance_distance": int(
                layer_metrics.sort_values(
                    ["appearance_distance_median", "layer_index"],
                    ascending=[False, True],
                ).iloc[0]["layer_index"]
            ),
            "physics_all_accuracy": int(
                layer_metrics.sort_values(
                    ["physics_all_accuracy_mean", "layer_index"],
                    ascending=[False, True],
                ).iloc[0]["layer_index"]
            ),
        },
        "limitations": [
            "Descriptive correlations on the inspected locked 24-pair workshop subset.",
            "Static sensitivity comes from final V-JEPA2 predictor control scores, not layerwise predictor outputs.",
            "No pixel-predictive comparator or fresh confirmatory split is included in this analysis.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
