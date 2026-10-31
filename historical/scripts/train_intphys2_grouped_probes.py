from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="scene_id")
    parser.add_argument("--split-mode", choices=("repeated_holdout", "crossfit"), default="repeated_holdout")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--crossfit-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def deterministic_group_split(
    groups: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, set[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    unique_groups = sorted({str(group) for group in groups})
    if len(unique_groups) < 3:
        raise ValueError("At least three groups are required for train/val/test splits")
    ordered = sorted(unique_groups, key=lambda group: stable_key(group, seed))
    n_groups = len(ordered)
    n_train = max(1, int(round(n_groups * train_fraction)))
    n_val = max(1, int(round(n_groups * val_fraction))) if val_fraction > 0 else 0
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train : n_train + n_val]),
        "test": set(ordered[n_train + n_val :]),
    }


def deterministic_crossfit_splits(
    groups: np.ndarray, *, folds: int, seed: int
) -> list[dict[str, set[str]]]:
    unique_groups = sorted({str(group) for group in groups})
    if folds < 3:
        raise ValueError("Cross-fitting requires at least three folds")
    if len(unique_groups) < folds:
        raise ValueError("Cross-fitting requires at least one group per fold")
    ordered = sorted(unique_groups, key=lambda group: stable_key(group, seed))
    fold_groups = [set(ordered[index::folds]) for index in range(folds)]
    splits = []
    for test_index in range(folds):
        val_index = (test_index + 1) % folds
        train = set().union(
            *(fold for index, fold in enumerate(fold_groups) if index not in {test_index, val_index})
        )
        splits.append({"train": train, "val": fold_groups[val_index], "test": fold_groups[test_index]})
    return splits


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def metadata_from_payload(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    columns = [str(column) for column in payload["metadata_columns"].tolist()]
    return pd.DataFrame({column: payload[column] for column in columns if column in payload})


def scene_id_from_pair_id(pair_id: str) -> str:
    value = str(pair_id)
    if ":" in value:
        value = value.split(":", 1)[1]
    return value.split("/", 1)[0]

def task_specs(metadata: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    appearance = metadata["appearance_condition"].astype(str).to_numpy()
    specs = [("physics_all", np.ones(len(metadata), dtype=bool), "pair_label")]
    for source_name in ("original", "clean"):
        source_mask = appearance == source_name
        if source_mask.any():
            specs.append((f"physics_{source_name}_only", source_mask, "pair_label"))
    if len(set(appearance.tolist())) > 1:
        specs.append(("appearance", np.ones(len(metadata), dtype=bool), "appearance_condition"))
    return specs


def split_masks(groups: np.ndarray, split: dict[str, set[str]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([str(group) in members for group in groups], dtype=bool)
        for name, members in split.items()
    }


def encode_labels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted({str(value) for value in values})
    if len(classes) < 2:
        raise ValueError("Probe labels need at least two classes")
    mapping = {label: index for index, label in enumerate(classes)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64), classes


def majority_accuracy(train_labels: np.ndarray, eval_labels: np.ndarray) -> float:
    majority_label = Counter(train_labels.tolist()).most_common(1)[0][0]
    return float(np.mean(eval_labels == majority_label))


def validate_pair_group_containment(metadata: pd.DataFrame, group_column: str) -> None:
    leaking_pairs = []
    for pair_id, group in metadata.groupby("pair_id", sort=True):
        groups = sorted({str(value) for value in group[group_column].tolist()})
        if len(groups) != 1:
            leaking_pairs.append({"pair_id": str(pair_id), "groups": groups})
    if leaking_pairs:
        preview = leaking_pairs[:5]
        raise ValueError(f"Pair IDs are not contained in single groups: {preview}")


def class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(str(value) for value in labels.tolist())
    return dict(sorted(counts.items()))


def fit_predict_layer(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    seed: int,
    max_iter: int,
    c_value: float,
) -> tuple[dict[str, tuple[float, float, int]], dict[str, np.ndarray]]:
    train_mask = masks["train"]
    if len(set(y[train_mask].tolist())) < 2:
        raise ValueError("Training split contains fewer than two classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            random_state=seed,
            class_weight="balanced",
        ),
    )
    model.fit(x[train_mask], y[train_mask])
    metrics: dict[str, tuple[float, float, int]] = {}
    predictions: dict[str, np.ndarray] = {}
    for split_name, mask in masks.items():
        split_predictions = model.predict(x[mask])
        predictions[split_name] = split_predictions
        accuracy = float(np.mean(split_predictions == y[mask]))
        baseline = majority_accuracy(y[train_mask], y[mask])
        metrics[split_name] = (accuracy, baseline, int(mask.sum()))
    return metrics, predictions


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and (args.output_dir / "probe_results.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_npz(args.representations)
    metadata = metadata_from_payload(payload)
    if args.group_column == "scene_id" and "scene_id" not in metadata.columns:
        if "pair_id" not in metadata.columns:
            raise ValueError("Cannot derive scene_id without pair_id metadata")
        metadata["scene_id"] = metadata["pair_id"].map(scene_id_from_pair_id)
    if args.group_column not in metadata.columns:
        raise ValueError(f"Missing group column {args.group_column!r}")
    validate_pair_group_containment(metadata, args.group_column)

    x = payload["X"]
    layer_names = [str(name) for name in payload["layer_names"].tolist()]
    all_groups = metadata[args.group_column].astype(str).to_numpy()
    all_pair_ids = metadata["pair_id"].astype(str).to_numpy()
    if args.split_mode == "crossfit":
        split_specs = list(enumerate(deterministic_crossfit_splits(all_groups, folds=args.folds, seed=args.crossfit_seed)))
    else:
        split_specs = [
            (
                seed,
                deterministic_group_split(
                    all_groups,
                    seed=seed,
                    train_fraction=args.train_fraction,
                    val_fraction=args.val_fraction,
                ),
            )
            for seed in parse_seeds(args.seeds)
        ]

    rows: list[dict] = []
    prediction_rows: list[dict] = []
    split_records: dict[str, dict[str, dict[str, list[str]]]] = {}
    skipped: list[str] = []

    for split_index, split in split_specs:
        split_records[str(split_index)] = {}
        for split_name, members in split.items():
            mask = np.asarray([str(group) in members for group in all_groups], dtype=bool)
            split_records[str(split_index)][split_name] = {
                "groups": sorted(members),
                "pair_ids": sorted(set(all_pair_ids[mask].tolist())),
            }

        for task_name, task_mask, label_column in task_specs(metadata):
            task_groups = all_groups[task_mask]
            task_pair_ids = all_pair_ids[task_mask]
            task_x = x[task_mask]
            task_meta = metadata[task_mask].reset_index(drop=True)
            task_y_values = task_meta[label_column].astype(str).to_numpy()
            try:
                task_y, classes = encode_labels(task_y_values)
            except ValueError as error:
                skipped.append(f"{task_name}: {error}")
                continue
            masks = split_masks(task_groups, split)
            if not all(mask.any() for mask in masks.values()):
                skipped.append(f"{task_name}: empty split for index {split_index}")
                continue
            if len(set(task_y[masks["train"]].tolist())) < 2:
                skipped.append(f"{task_name}: one-class train split for index {split_index}")
                continue

            split_class_counts = {
                split_name: class_counts(task_y_values[mask])
                for split_name, mask in masks.items()
            }
            for layer_index, layer_name in enumerate(layer_names):
                metrics, predictions = fit_predict_layer(
                    task_x[:, layer_index, :],
                    task_y,
                    masks,
                    seed=split_index,
                    max_iter=args.max_iter,
                    c_value=args.c,
                )
                for split_name, (accuracy, baseline, count) in metrics.items():
                    split_mask = masks[split_name]
                    rows.append(
                        {
                            "task": task_name,
                            "seed": split_index,
                            "split_mode": args.split_mode,
                            "split_index": split_index,
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "split": split_name,
                            "n": count,
                            "classes": "|".join(classes),
                            "accuracy": accuracy,
                            "majority_baseline_accuracy": baseline,
                            "group_column": args.group_column,
                            "train_groups": len(split["train"]),
                            "val_groups": len(split["val"]),
                            "test_groups": len(split["test"]),
                            "train_pairs": int(len(set(task_pair_ids[masks["train"]].tolist()))),
                            "val_pairs": int(len(set(task_pair_ids[masks["val"]].tolist()))),
                            "test_pairs": int(len(set(task_pair_ids[masks["test"]].tolist()))),
                            "train_class_counts_json": json.dumps(split_class_counts["train"], sort_keys=True),
                            "val_class_counts_json": json.dumps(split_class_counts["val"], sort_keys=True),
                            "test_class_counts_json": json.dumps(split_class_counts["test"], sort_keys=True),
                        }
                    )
                    split_indices = np.flatnonzero(split_mask)
                    for local_index, pred in zip(split_indices.tolist(), predictions[split_name].tolist()):
                        meta = task_meta.iloc[local_index]
                        true = int(task_y[local_index])
                        prediction_rows.append(
                            {
                                "task": task_name,
                                "seed": split_index,
                                "split_mode": args.split_mode,
                                "split_index": split_index,
                                "layer_index": layer_index,
                                "layer_name": layer_name,
                                "split": split_name,
                                "sample_id": str(meta["sample_id"]),
                                "name": str(meta["name"]),
                                "pair_id": str(meta["pair_id"]),
                                "scene_id": str(meta["scene_id"]),
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

    if not rows:
        raise ValueError(f"No probe rows were produced; skipped={skipped}")
    if args.split_mode == "crossfit" and skipped:
        raise ValueError(f"Crossfit cannot skip any task or fold: {skipped}")

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["task", "layer_index", "layer_name", "split"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            majority_baseline_mean=("majority_baseline_accuracy", "mean"),
            n_mean=("n", "mean"),
            seeds=("seed", "nunique"),
            groups_mean=("test_groups", "mean"),
            pairs_mean=("test_pairs", "mean"),
        )
        .sort_values(["task", "split", "accuracy_mean", "layer_index"], ascending=[True, True, False, True])
    )
    test_summary = summary[summary["split"].eq("test")]
    best_rows = []
    for task, group in test_summary.groupby("task", sort=True):
        best = group.sort_values(["accuracy_mean", "layer_index"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "task": task,
                "best_test_layer_index": int(best["layer_index"]),
                "best_test_layer_name": str(best["layer_name"]),
                "best_test_accuracy_mean": float(best["accuracy_mean"]),
                "best_test_accuracy_std": None
                if pd.isna(best["accuracy_std"])
                else float(best["accuracy_std"]),
                "best_test_majority_baseline_mean": float(best["majority_baseline_mean"]),
            }
        )

    results.to_csv(args.output_dir / "probe_results.csv", index=False)
    summary.to_csv(args.output_dir / "probe_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "probe_predictions.csv", index=False)
    (args.output_dir / "split_assignments.json").write_text(
        json.dumps(split_records, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_test_layers": best_rows,
                "rows": int(len(results)),
                "prediction_rows": int(len(prediction_rows)),
                "tasks": sorted(results["task"].unique().tolist()),
                "split_mode": args.split_mode,
                "split_indices": [int(index) for index, _ in split_specs],
                "crossfit_folds": args.folds if args.split_mode == "crossfit" else None,
                "crossfit_seed": args.crossfit_seed if args.split_mode == "crossfit" else None,
                "representations": str(args.representations),
                "representations_shape": [int(value) for value in payload["X"].shape],
                "group_column": args.group_column,
                "unique_groups": int(metadata[args.group_column].nunique()),
                "unique_pairs": int(metadata["pair_id"].nunique()),
                "skipped": skipped,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
                "c": args.c,
                "max_iter": args.max_iter,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
