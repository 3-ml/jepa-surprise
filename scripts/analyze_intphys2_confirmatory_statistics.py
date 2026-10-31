#!/usr/bin/env python3
"""Pair-bootstrap statistics and stratified robustness for IntPhys2 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from train_vjepa2_layerwise_probes import (
        deterministic_group_split,
        encode_labels,
        split_masks,
    )
except ModuleNotFoundError:
    from scripts.train_vjepa2_layerwise_probes import (
        deterministic_group_split,
        encode_labels,
        split_masks,
    )


TASK_LABELS = {
    "appearance": "appearance_condition",
    "physics_all": "pair_label",
    "physics_original_only": "pair_label",
    "physics_clean_only": "pair_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa-run", type=Path, required=True)
    parser.add_argument("--videomae-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-run", type=Path)
    parser.add_argument("--vjepa-label", default="V-JEPA2-G")
    parser.add_argument("--videomae-label", default="VideoMAE2-G")
    parser.add_argument("--pixel-label", default="PixelSummary")
    parser.add_argument("--vjepa-physics-layer", default="block_34")
    parser.add_argument("--videomae-physics-layer", default="block_03")
    parser.add_argument("--vjepa-original-layer", default="block_31")
    parser.add_argument("--videomae-original-layer", default="block_32")
    parser.add_argument("--vjepa-appearance-layer", default="block_15")
    parser.add_argument("--videomae-appearance-layer", default="block_10")
    parser.add_argument("--pixel-layer", default="pixel_summary")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def metadata_from_payload(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    columns = [str(column) for column in payload["metadata_columns"].tolist()]
    return pd.DataFrame({column: payload[column] for column in columns if column in payload})


def layer_index(payload: dict[str, np.ndarray], layer_name: str) -> int:
    names = [str(value) for value in payload["layer_names"].tolist()]
    if layer_name not in names:
        raise ValueError(f"Layer {layer_name!r} not in {names}")
    return names.index(layer_name)


def task_mask_and_labels(metadata: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "physics_original_only":
        mask = metadata["appearance_condition"].eq("original").to_numpy()
    elif task == "physics_clean_only":
        mask = metadata["appearance_condition"].eq("clean").to_numpy()
    else:
        mask = np.ones(len(metadata), dtype=bool)
    labels = metadata[TASK_LABELS[task]].to_numpy()
    return mask, labels


def predict_task_layer(
    *,
    label: str,
    run_dir: Path,
    task: str,
    layer_name: str,
    seeds: list[int],
    train_fraction: float,
    val_fraction: float,
    max_iter: int,
    c_value: float,
) -> pd.DataFrame:
    payload = load_npz(run_dir / "representations.npz")
    metadata = metadata_from_payload(payload)
    index = layer_index(payload, layer_name)
    task_mask, label_values = task_mask_and_labels(metadata, task)
    x = payload["X"][task_mask, index, :]
    task_metadata = metadata[task_mask].reset_index(drop=True)
    y, classes = encode_labels(label_values[task_mask])
    pair_ids = payload["pair_id"][task_mask]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split = deterministic_group_split(
            pair_ids,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        masks = split_masks(pair_ids, split)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=max_iter,
                random_state=seed,
                class_weight="balanced",
            ),
        )
        model.fit(x[masks["train"]], y[masks["train"]])
        predictions = model.predict(x[masks["test"]])
        test_indices = np.flatnonzero(masks["test"])
        for local_index, pred in zip(test_indices, predictions.tolist()):
            meta = task_metadata.iloc[int(local_index)]
            true = int(y[int(local_index)])
            rows.append(
                {
                    "model": label,
                    "run_dir": str(run_dir),
                    "task": task,
                    "layer_name": layer_name,
                    "layer_index": index,
                    "seed": seed,
                    "sample_id": str(meta["sample_id"]),
                    "name": str(meta["name"]),
                    "pair_id": str(meta["pair_id"]),
                    "pair_label": str(meta["pair_label"]),
                    "appearance_condition": str(meta["appearance_condition"]),
                    "condition": str(meta["condition"]),
                    "difficulty": str(meta["difficulty"]),
                    "camera": str(meta["camera"]),
                    "y_true": true,
                    "y_pred": int(pred),
                    "y_true_label": classes[true],
                    "y_pred_label": classes[int(pred)],
                    "correct": float(pred == true),
                }
            )
    return pd.DataFrame(rows)


def ci(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def pair_level(frame: pd.DataFrame, group_fields: list[str] | None = None) -> pd.DataFrame:
    fields = ["model", "task", "layer_name", "seed", "pair_id"]
    if group_fields:
        fields.extend(group_fields)
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


def bootstrap_accuracy(pair_frame: pd.DataFrame, *, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    seed_values = sorted(pair_frame["seed"].unique().tolist())
    per_seed_draws = []
    for seed in seed_values:
        seed_rows = pair_frame[pair_frame["seed"].eq(seed)]
        values = seed_rows["correct"].to_numpy(dtype=float)
        sampled = rng.integers(0, len(values), size=(n_boot, len(values)))
        per_seed_draws.append(np.mean(values[sampled], axis=1))
    return ci(np.mean(np.stack(per_seed_draws, axis=0), axis=0))


def bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["seed", "pair_id"],
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    seed_values = sorted(merged["seed"].unique().tolist())
    per_seed_draws = []
    for seed in seed_values:
        seed_rows = merged[merged["seed"].eq(seed)]
        values = (
            seed_rows["correct_left"].to_numpy(dtype=float)
            - seed_rows["correct_right"].to_numpy(dtype=float)
        )
        sampled = rng.integers(0, len(values), size=(n_boot, len(values)))
        per_seed_draws.append(np.mean(values[sampled], axis=1))
    return ci(np.mean(np.stack(per_seed_draws, axis=0), axis=0))


def summarize_predictions(predictions: pd.DataFrame, *, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_rows = pair_level(predictions)
    for (model, task, layer_name), group in pair_rows.groupby(["model", "task", "layer_name"], sort=True):
        stats = bootstrap_accuracy(group, n_boot=n_boot, rng=rng)
        rows.append(
            {
                "model": model,
                "task": task,
                "layer_name": layer_name,
                "n_seed_pair_rows": int(len(group)),
                "n_pairs_union": int(group["pair_id"].nunique()),
                **stats,
            }
        )

    stratum_rows = []
    for field in ("appearance_condition", "condition", "difficulty", "camera"):
        grouped = pair_level(predictions, [field])
        for (model, task, layer_name, value), group in grouped.groupby(
            ["model", "task", "layer_name", field],
            sort=True,
        ):
            if group["pair_id"].nunique() < 2:
                continue
            stats = bootstrap_accuracy(group, n_boot=n_boot, rng=rng)
            stratum_rows.append(
                {
                    "stratum_field": field,
                    "stratum_value": value,
                    "model": model,
                    "task": task,
                    "layer_name": layer_name,
                    "n_seed_pair_rows": int(len(group)),
                    "n_pairs_union": int(group["pair_id"].nunique()),
                    **stats,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(stratum_rows)


def summarize_deltas(
    predictions: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = pair_level(predictions)
    rows = []
    for task in sorted(pair_rows["task"].unique().tolist()):
        left = pair_rows[pair_rows["model"].eq(left_model) & pair_rows["task"].eq(task)]
        right = pair_rows[pair_rows["model"].eq(right_model) & pair_rows["task"].eq(task)]
        if left.empty or right.empty:
            continue
        stats = bootstrap_delta(left, right, n_boot=n_boot, rng=rng)
        rows.append(
            {
                "left_model": left_model,
                "right_model": right_model,
                "task": task,
                "metric": "pair_accuracy_delta",
                **stats,
            }
        )

    stratum_rows = []
    for field in ("appearance_condition", "condition", "difficulty", "camera"):
        grouped = pair_level(predictions, [field])
        for task in sorted(grouped["task"].unique().tolist()):
            for value in sorted(grouped[field].dropna().unique().tolist()):
                left = grouped[
                    grouped["model"].eq(left_model)
                    & grouped["task"].eq(task)
                    & grouped[field].eq(value)
                ]
                right = grouped[
                    grouped["model"].eq(right_model)
                    & grouped["task"].eq(task)
                    & grouped[field].eq(value)
                ]
                if left.empty or right.empty:
                    continue
                stats = bootstrap_delta(left, right, n_boot=n_boot, rng=rng)
                stratum_rows.append(
                    {
                        "stratum_field": field,
                        "stratum_value": value,
                        "left_model": left_model,
                        "right_model": right_model,
                        "task": task,
                        "metric": "pair_accuracy_delta",
                        **stats,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(stratum_rows)


def pair_layer_matrix(frame: pd.DataFrame, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    layers = np.asarray(sorted(frame["layer_index"].unique().tolist()), dtype=np.int64)
    pivot = (
        frame.groupby(["pair_id", "layer_index"], as_index=False)
        .agg(distance=("normalized_matched_physical_distance", "median"))
        .pivot(index="pair_id", columns="layer_index", values="distance")
        .reindex(index=pairs.tolist(), columns=layers.tolist())
    )
    values = pivot.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Physical-distance matrix contains missing pair/layer values")
    return layers, values


def physical_trend(layers: np.ndarray, pair_layer_values: np.ndarray, sample_indices: np.ndarray) -> float:
    layer_medians = np.median(pair_layer_values[sample_indices], axis=0)
    value = spearmanr(layers, layer_medians).statistic
    return float(value)


def summarize_physical_trend(
    left_run: Path,
    right_run: Path,
    *,
    left_model: str,
    right_model: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    left = pd.read_csv(left_run / "geometry" / "physical_layer_distances.csv")
    right = pd.read_csv(right_run / "geometry" / "physical_layer_distances.csv")
    pairs = np.asarray(
        sorted(set(left["pair_id"].astype(str)).intersection(set(right["pair_id"].astype(str)))),
        dtype=object,
    )
    left["pair_id"] = left["pair_id"].astype(str)
    right["pair_id"] = right["pair_id"].astype(str)
    left_layers, left_values = pair_layer_matrix(left, pairs)
    right_layers, right_values = pair_layer_matrix(right, pairs)
    if not np.array_equal(left_layers, right_layers):
        raise ValueError("Left/right physical-distance layers differ")
    left_draws = []
    right_draws = []
    delta_draws = []
    sampled_indices = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    for sampled in sampled_indices:
        left_value = physical_trend(left_layers, left_values, sampled)
        right_value = physical_trend(right_layers, right_values, sampled)
        left_draws.append(left_value)
        right_draws.append(right_value)
        delta_draws.append(left_value - right_value)
    return {
        "metric": "physical_distance_vs_depth_spearman",
        "n_pairs": int(len(pairs)),
        left_model: ci(np.asarray(left_draws)),
        right_model: ci(np.asarray(right_draws)),
        f"{left_model}_minus_{right_model}": ci(np.asarray(delta_draws)),
    }


def write_csv(path: Path, rows: pd.DataFrame) -> None:
    rows.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    model_specs = [
        (args.vjepa_label, args.vjepa_run, {
            "appearance": args.vjepa_appearance_layer,
            "physics_all": args.vjepa_physics_layer,
            "physics_clean_only": args.vjepa_original_layer,
        }),
        (args.videomae_label, args.videomae_run, {
            "appearance": args.videomae_appearance_layer,
            "physics_all": args.videomae_physics_layer,
            "physics_clean_only": args.videomae_original_layer,
        }),
    ]
    if args.pixel_run is not None:
        model_specs.append(
            (
                args.pixel_label,
                args.pixel_run,
                {
                    "appearance": args.pixel_layer,
                    "physics_all": args.pixel_layer,
                    "physics_clean_only": args.pixel_layer,
                },
            )
        )

    prediction_frames = []
    for label, run_dir, task_layers in model_specs:
        for task, layer_name in task_layers.items():
            prediction_frames.append(
                predict_task_layer(
                    label=label,
                    run_dir=run_dir,
                    task=task,
                    layer_name=layer_name,
                    seeds=seeds,
                    train_fraction=args.train_fraction,
                    val_fraction=args.val_fraction,
                    max_iter=args.max_iter,
                    c_value=args.c,
                )
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_csv(args.output_dir / "heldout_predictions.csv", predictions)

    rng = np.random.default_rng(args.seed)
    accuracy, stratified_accuracy = summarize_predictions(
        predictions,
        n_boot=args.bootstrap,
        rng=rng,
    )
    delta, stratified_delta = summarize_deltas(
        predictions,
        left_model=args.vjepa_label,
        right_model=args.videomae_label,
        n_boot=args.bootstrap,
        rng=rng,
    )
    trend = summarize_physical_trend(
        args.vjepa_run,
        args.videomae_run,
        left_model=args.vjepa_label,
        right_model=args.videomae_label,
        n_boot=args.bootstrap,
        rng=rng,
    )

    write_csv(args.output_dir / "accuracy_ci.csv", accuracy)
    write_csv(args.output_dir / "stratified_accuracy_ci.csv", stratified_accuracy)
    write_csv(args.output_dir / "model_delta_ci.csv", delta)
    write_csv(args.output_dir / "stratified_model_delta_ci.csv", stratified_delta)
    (args.output_dir / "physical_distance_trend_ci.json").write_text(
        json.dumps(trend, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "command": "python " + " ".join(__import__("sys").argv),
        "seeds": seeds,
        "bootstrap": args.bootstrap,
        "runs": [
            {
                "label": label,
                "run_dir": str(run_dir),
                "representations_sha256": sha256_file(run_dir / "representations.npz"),
                "task_layers": task_layers,
            }
            for label, run_dir, task_layers in model_specs
        ],
        "accuracy_ci": accuracy.to_dict(orient="records"),
        "model_delta_ci": delta.to_dict(orient="records"),
        "physical_distance_trend_ci": trend,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
