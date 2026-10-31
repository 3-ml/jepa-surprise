from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vjepa-predictions", type=Path, required=True)
    parser.add_argument("--videomae-predictions", type=Path, required=True)
    parser.add_argument("--pixel-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vjepa-label", default="V-JEPA2-G")
    parser.add_argument("--videomae-label", default="VideoMAE2-G")
    parser.add_argument("--pixel-label", default="PixelSummary")
    parser.add_argument("--vjepa-physics-layer", default="block_30")
    parser.add_argument("--videomae-physics-layer", default="block_00")
    parser.add_argument("--vjepa-clean-layer", default="block_30")
    parser.add_argument("--videomae-clean-layer", default="block_30")
    parser.add_argument("--vjepa-appearance-layer", default="block_20")
    parser.add_argument("--videomae-appearance-layer", default="block_30")
    parser.add_argument("--pixel-layer", default="pixel_summary")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-mode", choices=("within_split", "global_scene"), default="within_split")
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ci(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def load_prediction_set(
    *,
    label: str,
    path: Path,
    layers: dict[str, str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scene_id" not in frame.columns:
        raise ValueError(f"{path} does not contain scene_id")
    rows = []
    for task, layer_name in layers.items():
        selected = frame[
            frame["task"].eq(task)
            & frame["layer_name"].eq(layer_name)
            & frame["split"].eq("test")
        ].copy()
        if selected.empty:
            raise ValueError(f"No predictions for {label} task={task} layer={layer_name}")
        selected["model"] = label
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def pair_level(frame: pd.DataFrame) -> pd.DataFrame:
    fields = ["model", "task", "layer_name", "seed", "scene_id", "pair_id"]
    return (
        frame.groupby(fields, as_index=False)
        .agg(
            correct=("correct", "mean"),
            n_samples=("correct", "size"),
            condition=("condition", "first"),
            difficulty=("difficulty", "first"),
            camera=("camera", "first"),
        )
    )


def validate_crossfit_predictions(
    frame: pd.DataFrame, *, expected_scenes: int, expected_pairs: int, expected_folds: int
) -> None:
    required = {
        "split_mode", "split_index", "task", "seed", "scene_id", "pair_id",
        "sample_id", "y_true", "y_true_label", "appearance_condition",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Crossfit predictions are missing columns: {missing}")
    if set(frame["split_mode"].astype(str).unique().tolist()) != {"crossfit"}:
        raise ValueError("Global scene bootstrap requires crossfit predictions")
    folds = sorted(frame["seed"].astype(int).unique().tolist())
    if folds != list(range(expected_folds)):
        raise ValueError(f"Expected folds 0..{expected_folds - 1}, found {folds}")
    if not np.array_equal(frame["seed"].to_numpy(), frame["split_index"].to_numpy()):
        raise ValueError("seed and split_index columns disagree")
    if frame.duplicated(["task", "seed", "scene_id", "pair_id", "sample_id"]).any():
        raise ValueError("Crossfit predictions contain duplicate sample support")
    for task, task_rows in frame.groupby("task", sort=True):
        if task_rows["scene_id"].nunique() != expected_scenes:
            raise ValueError(f"{task} does not cover {expected_scenes} scenes")
        if task_rows["pair_id"].nunique() != expected_pairs:
            raise ValueError(f"{task} does not cover {expected_pairs} pairs")
        if not task_rows.groupby("scene_id")["seed"].nunique().eq(1).all():
            raise ValueError(f"{task} does not test every scene exactly once")
        if not task_rows.groupby("sample_id")["seed"].nunique().eq(1).all():
            raise ValueError(f"{task} does not predict every sample exactly once")
        expected_samples = 2 if task == "physics_clean_only" else 8
        if not task_rows.groupby("pair_id")["sample_id"].nunique().eq(expected_samples).all():
            raise ValueError(f"{task} has incomplete pair sample support")
        if not task_rows.groupby("scene_id")["pair_id"].nunique().eq(2).all():
            raise ValueError(f"{task} does not carry both pairs per scene")


def validate_cross_model_support(frames: list[pd.DataFrame]) -> None:
    columns = [
        "task", "seed", "split_index", "scene_id", "pair_id", "sample_id",
        "y_true", "y_true_label", "appearance_condition",
    ]
    reference = frames[0][columns].sort_values(columns).reset_index(drop=True)
    for frame in frames[1:]:
        candidate = frame[columns].sort_values(columns).reset_index(drop=True)
        if not reference.equals(candidate):
            raise ValueError("Cross-model prediction support or true labels differ")


def draw_cluster_means(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    draws_by_seed = []
    for _, seed_rows in frame.groupby("seed", sort=True):
        scenes = np.asarray(sorted(seed_rows["scene_id"].astype(str).unique().tolist()), dtype=object)
        scene_arrays = {
            scene: seed_rows[seed_rows["scene_id"].astype(str).eq(scene)]["correct"].to_numpy(dtype=float)
            for scene in scenes.tolist()
        }
        sampled = rng.integers(0, len(scenes), size=(n_boot, len(scenes)))
        seed_draws = np.empty(n_boot, dtype=float)
        for draw_index, draw in enumerate(sampled):
            values = [scene_arrays[str(scenes[index])] for index in draw.tolist()]
            seed_draws[draw_index] = float(np.mean(np.concatenate(values)))
        draws_by_seed.append(seed_draws)
    return np.vstack(draws_by_seed).mean(axis=0)


def global_scene_values(frame: pd.DataFrame) -> np.ndarray:
    scene_split_counts = frame.groupby("scene_id")["seed"].nunique()
    if not scene_split_counts.eq(1).all():
        repeated = scene_split_counts[scene_split_counts.ne(1)].to_dict()
        raise ValueError(f"Global scene bootstrap requires one test fold per scene: {repeated}")
    return frame.groupby("scene_id", sort=True)["correct"].mean().to_numpy(dtype=float)


def draw_global_scene_means(
    frame: pd.DataFrame, *, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    values = global_scene_values(frame)
    sampled = rng.integers(0, len(values), size=(n_boot, len(values)))
    return np.mean(values[sampled], axis=1)


def bootstrap_accuracy(
    pair_frame: pd.DataFrame, *, n_boot: int, rng: np.random.Generator, bootstrap_mode: str
) -> dict[str, float]:
    if bootstrap_mode == "global_scene":
        stats = ci(draw_global_scene_means(pair_frame, n_boot=n_boot, rng=rng))
        stats["mean"] = float(np.mean(global_scene_values(pair_frame)))
        return stats
    return ci(draw_cluster_means(pair_frame, n_boot=n_boot, rng=rng))


def bootstrap_delta(
    left: pd.DataFrame, right: pd.DataFrame, *, n_boot: int,
    rng: np.random.Generator, bootstrap_mode: str
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["seed", "scene_id", "pair_id"],
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired model predictions are not aligned by fold, scene, and pair")
    if not merged["n_samples_left"].eq(merged["n_samples_right"]).all():
        raise ValueError("Paired model predictions have different within-pair sample counts")
    merged["correct"] = merged["correct_left"].astype(float) - merged["correct_right"].astype(float)
    if bootstrap_mode == "global_scene":
        stats = ci(draw_global_scene_means(merged, n_boot=n_boot, rng=rng))
        stats["mean"] = float(np.mean(global_scene_values(merged)))
        return stats
    return ci(draw_cluster_means(merged, n_boot=n_boot, rng=rng))


def observed_seed_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_rows.groupby(["model", "task", "layer_name", "seed"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n_scenes=("scene_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
            n_pair_rows=("correct", "size"),
        )
        .sort_values(["model", "task", "layer_name", "seed"])
    )


def split_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, seed), group in predictions.groupby(["model", "task", "seed"], sort=True):
        scene_pairs = (
            group.groupby("scene_id", as_index=False)
            .agg(n_pairs=("pair_id", "nunique"), n_samples=("sample_id", "nunique"))
        )
        rows.append(
            {
                "model": model,
                "task": task,
                "seed": int(seed),
                "test_scenes": int(scene_pairs["scene_id"].nunique()),
                "test_pairs": int(group["pair_id"].nunique()),
                "test_samples": int(group["sample_id"].nunique()),
                "min_pairs_per_scene": int(scene_pairs["n_pairs"].min()),
                "max_pairs_per_scene": int(scene_pairs["n_pairs"].max()),
            }
        )
    return pd.DataFrame(rows)


def coverage_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task), group in predictions.groupby(["model", "task"], sort=True):
        scene_rows = group.groupby("scene_id", as_index=False).agg(
            test_folds=("seed", "nunique"), n_pairs=("pair_id", "nunique")
        )
        rows.append(
            {
                "model": model,
                "task": task,
                "unique_test_scenes": int(scene_rows["scene_id"].nunique()),
                "unique_test_pairs": int(group["pair_id"].nunique()),
                "test_folds_per_scene_min": int(scene_rows["test_folds"].min()),
                "test_folds_per_scene_max": int(scene_rows["test_folds"].max()),
                "pairs_per_scene_min": int(scene_rows["n_pairs"].min()),
                "pairs_per_scene_max": int(scene_rows["n_pairs"].max()),
            }
        )
    return pd.DataFrame(rows)


def accuracy_tables(
    predictions: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    pixel_model: str | None,
    n_boot: int,
    rng: np.random.Generator,
    bootstrap_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows = pair_level(predictions)
    accuracy_rows = []
    for (model, task, layer_name), group in pair_rows.groupby(["model", "task", "layer_name"], sort=True):
        accuracy_rows.append(
            {
                "model": model,
                "task": task,
                "layer_name": layer_name,
                "n_seed_rows": int(group["seed"].nunique()),
                "n_scenes_min": int(group.groupby("seed")["scene_id"].nunique().min()),
                "n_scenes_max": int(group.groupby("seed")["scene_id"].nunique().max()),
                "n_pairs_min": int(group.groupby("seed")["pair_id"].nunique().min()),
                "n_pairs_max": int(group.groupby("seed")["pair_id"].nunique().max()),
                "n_scenes_union": int(group["scene_id"].nunique()),
                **bootstrap_accuracy(group, n_boot=n_boot, rng=rng, bootstrap_mode=bootstrap_mode),
            }
        )

    delta_rows = []
    right_models = [right_model]
    if pixel_model is not None:
        right_models.append(pixel_model)
    for task in sorted(pair_rows["task"].unique().tolist()):
        left = pair_rows[pair_rows["model"].eq(left_model) & pair_rows["task"].eq(task)]
        for comparison_model in right_models:
            right = pair_rows[pair_rows["model"].eq(comparison_model) & pair_rows["task"].eq(task)]
            if left.empty or right.empty:
                continue
            delta_rows.append(
                {
                    "left_model": left_model,
                    "right_model": comparison_model,
                    "task": task,
                    "metric": "scene_clustered_pair_accuracy_delta",
                    **bootstrap_delta(
                        left, right, n_boot=n_boot, rng=rng, bootstrap_mode=bootstrap_mode
                    ),
                }
            )
    return pd.DataFrame(accuracy_rows), pd.DataFrame(delta_rows), observed_seed_rows(pair_rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        load_prediction_set(
            label=args.vjepa_label,
            path=args.vjepa_predictions,
            layers={
                "appearance": args.vjepa_appearance_layer,
                "physics_all": args.vjepa_physics_layer,
                "physics_clean_only": args.vjepa_clean_layer,
            },
        ),
        load_prediction_set(
            label=args.videomae_label,
            path=args.videomae_predictions,
            layers={
                "appearance": args.videomae_appearance_layer,
                "physics_all": args.videomae_physics_layer,
                "physics_clean_only": args.videomae_clean_layer,
            },
        ),
    ]
    run_records: list[dict[str, Any]] = [
        {
            "label": args.vjepa_label,
            "predictions": str(args.vjepa_predictions),
            "prediction_sha256": sha256_file(args.vjepa_predictions),
        },
        {
            "label": args.videomae_label,
            "predictions": str(args.videomae_predictions),
            "prediction_sha256": sha256_file(args.videomae_predictions),
        },
    ]
    pixel_model = None
    if args.pixel_predictions is not None:
        pixel_model = args.pixel_label
        frames.append(
            load_prediction_set(
                label=args.pixel_label,
                path=args.pixel_predictions,
                layers={
                    "appearance": args.pixel_layer,
                    "physics_all": args.pixel_layer,
                    "physics_clean_only": args.pixel_layer,
                },
            )
        )
        run_records.append(
            {
                "label": args.pixel_label,
                "predictions": str(args.pixel_predictions),
                "prediction_sha256": sha256_file(args.pixel_predictions),
            }
        )

    if args.bootstrap_mode == "global_scene":
        if args.expected_scenes is None or args.expected_pairs is None:
            raise ValueError("Global scene bootstrap requires expected scene and pair counts")
        for frame in frames:
            validate_crossfit_predictions(
                frame,
                expected_scenes=args.expected_scenes,
                expected_pairs=args.expected_pairs,
                expected_folds=args.expected_folds,
            )
        validate_cross_model_support(frames)

    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(args.output_dir / "selected_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    rng = np.random.default_rng(args.seed)
    accuracy, delta, per_seed = accuracy_tables(
        predictions,
        left_model=args.vjepa_label,
        right_model=args.videomae_label,
        pixel_model=pixel_model,
        n_boot=args.bootstrap,
        rng=rng,
        bootstrap_mode=args.bootstrap_mode,
    )
    audit = split_audit(predictions)
    coverage = coverage_audit(predictions)

    accuracy.to_csv(args.output_dir / "scene_clustered_accuracy_ci.csv", index=False)
    delta.to_csv(args.output_dir / "scene_clustered_model_delta_ci.csv", index=False)
    per_seed.to_csv(args.output_dir / "per_seed_accuracy.csv", index=False)
    audit.to_csv(args.output_dir / "split_audit.csv", index=False)
    coverage.to_csv(args.output_dir / "coverage_audit.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": "python -m scripts.analyze_intphys2_grouped_probe_statistics " + " ".join(__import__("sys").argv[1:]),
                "bootstrap": args.bootstrap,
                "seed": args.seed,
                "cluster": "scene_id",
                "unit_before_cluster": "pair_id mean over selected samples",
                "bootstrap_mode": args.bootstrap_mode,
                "expected_scenes": args.expected_scenes,
                "expected_pairs": args.expected_pairs,
                "expected_folds": args.expected_folds,
                "aggregation": (
                    "all unique out-of-fold scenes are resampled globally"
                    if args.bootstrap_mode == "global_scene"
                    else "bootstrap means are computed per split and averaged across splits"
                ),
                "runs": run_records,
                "accuracy_ci": accuracy.to_dict(orient="records"),
                "model_delta_ci": delta.to_dict(orient="records"),
                "split_audit": audit.to_dict(orient="records"),
                "coverage_audit": coverage.to_dict(orient="records"),
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
