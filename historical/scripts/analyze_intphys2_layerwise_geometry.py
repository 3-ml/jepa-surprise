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


SOURCE_APPEARANCE_NAMES = ("original", "clean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="model")
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
    columns = [str(column) for column in payload["metadata_columns"].tolist()]
    metadata = pd.DataFrame(
        {column: payload[column] for column in columns if column in payload}
    )
    if "pair_label" not in metadata and "physical_label" in metadata:
        metadata["pair_label"] = metadata["physical_label"]
    return metadata


def source_appearance_name(metadata: pd.DataFrame) -> str:
    for name in SOURCE_APPEARANCE_NAMES:
        if metadata["appearance_condition"].eq(name).any():
            return name
    raise ValueError(
        "Could not find a source appearance condition; expected one of "
        f"{SOURCE_APPEARANCE_NAMES}"
    )


def transform_order(metadata: pd.DataFrame, source_name: str) -> list[str]:
    return sorted(
        str(value)
        for value in metadata["appearance_condition"].dropna().unique().tolist()
        if str(value) != source_name
    )


def cosine_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
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


def compute_layer_geometry(
    payload: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = payload["X"]
    layer_names = [str(value) for value in payload["layer_names"]]
    source_name = source_appearance_name(metadata)
    transforms = transform_order(metadata, source_name)
    source_mask = metadata["appearance_condition"].eq(source_name).to_numpy()
    source_indices = np.flatnonzero(source_mask)
    source_names = metadata.iloc[source_indices]["name"].to_numpy()
    source_index_by_name = {
        name: index
        for name, index in zip(source_names.tolist(), source_indices.tolist())
    }

    layer_rows: list[dict] = []
    video_rows: list[dict] = []
    physical_rows: list[dict] = []
    for layer_index, layer_name in enumerate(layer_names):
        z = cosine_normalize(x[:, layer_index, :])
        z_source = z[source_indices]
        normalization = off_diagonal_median_cosine_distance(z_source)
        retrieval_hits: list[float] = []
        all_transform_distances: list[float] = []
        condition_distances: dict[str, list[float]] = {
            condition: [] for condition in transforms
        }

        for name, group in metadata.groupby("name", sort=True):
            if name not in source_index_by_name:
                continue
            base_index = source_index_by_name[name]
            row = {
                "name": name,
                "pair_id": str(group["pair_id"].iloc[0]),
                "pair_label": str(group["pair_label"].iloc[0]),
                "condition": str(group["condition"].iloc[0]),
                "difficulty": str(group["difficulty"].iloc[0]),
                "camera": str(group["camera"].iloc[0]),
                "layer_index": layer_index,
                "layer_name": layer_name,
                "normalization_original_pairwise_median": normalization,
            }
            if "partition" in group:
                row["partition"] = str(group["partition"].iloc[0])
            per_video_distances: list[float] = []
            for condition in transforms:
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

                nearest = source_names[
                    int(np.argmin(1.0 - z_source @ z[int(condition_index[0])]))
                ]
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
                (1.0 - z[int(possible.index[0])] @ z[int(impossible.index[0])])
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
            if appearance == source_name:
                physical_original_distances.append(distance)

        appearance_median = float(np.median(all_transform_distances))
        physical_median = float(np.median(physical_distances))
        layer_row = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "normalization_original_pairwise_median": normalization,
            "source_appearance_condition": source_name,
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

    return (
        pd.DataFrame(layer_rows),
        pd.DataFrame(video_rows),
        pd.DataFrame(physical_rows),
    )


def load_probe_curves(probe_summary: pd.DataFrame) -> pd.DataFrame:
    test = probe_summary[probe_summary["split"].eq("test")].copy()
    rows = []
    for layer_index, group in test.groupby("layer_index", sort=True):
        row = {
            "layer_index": int(layer_index),
            "layer_name": str(group["layer_name"].iloc[0]),
        }
        for task in (
            "appearance",
            "physics_all",
            "physics_original_only",
            "physics_clean_only",
        ):
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


def cross_layer_correlations(layer_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "appearance_accuracy_mean",
        "physics_all_accuracy_mean",
        "physics_original_only_accuracy_mean",
        "physics_clean_only_accuracy_mean",
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


def draw_layer_plot(
    layer_metrics: pd.DataFrame, output: Path, model_label: str
) -> None:
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
    ax.set_ylabel("Held-out probe accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    for column, label, color in [
        ("appearance_distance_median", "Appearance displacement", "#0072B2"),
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

    fig.suptitle(f"{model_label}: layerwise probes and geometry")
    fig.tight_layout()
    fig.savefig(output, dpi=200)


def summarize(
    layer_metrics: pd.DataFrame, args: argparse.Namespace, metadata: pd.DataFrame
) -> dict:
    key_pairs = {
        "appearance_accuracy_vs_appearance_distance": (
            "appearance_accuracy_mean",
            "appearance_distance_median",
        ),
        "appearance_accuracy_vs_depth": ("appearance_accuracy_mean", "layer_index"),
        "appearance_distance_vs_depth": ("appearance_distance_median", "layer_index"),
        "physics_all_accuracy_vs_depth": ("physics_all_accuracy_mean", "layer_index"),
        "physical_distance_vs_depth": ("physical_distance_median", "layer_index"),
        "retrieval_vs_depth": ("appearance_retrieval_top1", "layer_index"),
    }
    best_layers: dict[str, int] = {}
    for label, column in [
        ("appearance_accuracy", "appearance_accuracy_mean"),
        ("physics_all_accuracy", "physics_all_accuracy_mean"),
        ("physics_original_only_accuracy", "physics_original_only_accuracy_mean"),
        ("physics_clean_only_accuracy", "physics_clean_only_accuracy_mean"),
        ("appearance_distance", "appearance_distance_median"),
        ("physical_distance", "physical_distance_median"),
        ("retrieval", "appearance_retrieval_top1"),
    ]:
        if column not in layer_metrics:
            continue
        ascending = [False, True]
        best = layer_metrics.sort_values(
            [column, "layer_index"], ascending=ascending
        ).iloc[0]
        best_layers[label] = int(best["layer_index"])

    endpoints = {}
    for _, row in layer_metrics[
        layer_metrics["layer_index"].isin(
            [
                int(layer_metrics["layer_index"].min()),
                int(layer_metrics["layer_index"].max()),
            ]
        )
    ].iterrows():
        endpoints[str(row["layer_name"])] = {
            "layer_index": int(row["layer_index"]),
            "appearance_distance_median": float(row["appearance_distance_median"]),
            "physical_distance_median": float(row["physical_distance_median"]),
            "appearance_retrieval_top1": float(row["appearance_retrieval_top1"]),
            "appearance_accuracy_mean": None
            if "appearance_accuracy_mean" not in row
            or pd.isna(row["appearance_accuracy_mean"])
            else float(row["appearance_accuracy_mean"]),
            "physics_all_accuracy_mean": None
            if "physics_all_accuracy_mean" not in row
            or pd.isna(row["physics_all_accuracy_mean"])
            else float(row["physics_all_accuracy_mean"]),
            "physics_original_only_accuracy_mean": None
            if "physics_original_only_accuracy_mean" not in row
            or pd.isna(row["physics_original_only_accuracy_mean"])
            else float(row["physics_original_only_accuracy_mean"]),
            "physics_clean_only_accuracy_mean": None
            if "physics_clean_only_accuracy_mean" not in row
            or pd.isna(row["physics_clean_only_accuracy_mean"])
            else float(row["physics_clean_only_accuracy_mean"]),
        }

    return {
        "model_label": args.model_label,
        "representations": str(args.representations),
        "representations_sha256": sha256_file(args.representations),
        "probe_summary": str(args.probe_summary),
        "probe_summary_sha256": sha256_file(args.probe_summary),
        "n_samples": int(len(metadata)),
        "n_layers": int(len(layer_metrics)),
        "n_videos": int(metadata["name"].nunique()),
        "n_pairs": int(metadata["pair_id"].nunique()),
        "appearance_conditions": sorted(
            metadata["appearance_condition"].unique().tolist()
        ),
        "partitions": sorted(metadata["partition"].unique().tolist())
        if "partition" in metadata
        else [],
        "key_cross_layer_spearman": {
            name: spearman(layer_metrics[left], layer_metrics[right])
            for name, (left, right) in key_pairs.items()
            if left in layer_metrics and right in layer_metrics
        },
        "best_layers": best_layers,
        "endpoints": endpoints,
    }


def main() -> None:
    args = parse_args()
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_npz(args.representations)
    metadata = metadata_from_payload(payload)
    probe_summary = pd.read_csv(args.probe_summary)
    layer_geometry, video_distances, physical_distances = compute_layer_geometry(
        payload, metadata
    )
    probe_curves = load_probe_curves(probe_summary)
    layer_metrics = layer_geometry.merge(
        probe_curves, on=["layer_index", "layer_name"], how="left"
    )
    cross_layer = cross_layer_correlations(layer_metrics)

    layer_metrics.to_csv(
        args.output_dir / "layerwise_geometry_metrics.csv", index=False
    )
    video_distances.to_csv(
        args.output_dir / "video_layer_appearance_metrics.csv", index=False
    )
    physical_distances.to_csv(
        args.output_dir / "physical_layer_distances.csv", index=False
    )
    cross_layer.to_csv(args.output_dir / "cross_layer_correlations.csv", index=False)
    draw_layer_plot(
        layer_metrics,
        args.output_dir / "layerwise_probe_geometry.png",
        args.model_label,
    )
    summary = summarize(layer_metrics, args, metadata)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
