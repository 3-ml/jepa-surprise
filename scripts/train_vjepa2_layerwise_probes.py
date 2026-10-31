#!/usr/bin/env python3
"""Train pair-held-out probes on cached V-JEPA2 layerwise representations."""

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
    parser.add_argument("--seeds", default="0,1,2,3,4")
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
    pair_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> dict[str, set[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")
    unique_pairs = sorted({str(pair_id) for pair_id in pair_ids})
    if len(unique_pairs) < 3:
        raise ValueError("At least three pair IDs are required for train/val/test splits")
    ordered = sorted(unique_pairs, key=lambda pair_id: stable_key(pair_id, seed))
    n_pairs = len(ordered)
    n_train = max(1, int(round(n_pairs * train_fraction)))
    n_val = max(1, int(round(n_pairs * val_fraction))) if val_fraction > 0 else 0
    if n_train + n_val >= n_pairs:
        n_train = max(1, n_pairs - 2)
        n_val = 1
    train = set(ordered[:n_train])
    val = set(ordered[n_train : n_train + n_val])
    test = set(ordered[n_train + n_val :])
    if not test:
        raise ValueError("Split left no held-out test pairs")
    return {"train": train, "val": val, "test": test}


def split_masks(pair_ids: np.ndarray, split: dict[str, set[str]]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray([str(pair_id) in members for pair_id in pair_ids], dtype=bool)
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


def fit_layer_probe(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    seed: int,
    max_iter: int,
    c_value: float,
) -> dict[str, tuple[float, float, int]]:
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
    for split_name, mask in masks.items():
        predictions = model.predict(x[mask])
        accuracy = float(np.mean(predictions == y[mask]))
        baseline = majority_accuracy(y[train_mask], y[mask])
        metrics[split_name] = (accuracy, baseline, int(mask.sum()))
    return metrics


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def probe_specs(payload: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    pair_label = payload["pair_label"]
    appearance = payload["appearance_condition"]
    specs = [
        ("physics_all", np.ones(len(pair_label), dtype=bool), pair_label),
    ]
    for source_name in ("original", "clean"):
        source_mask = appearance == source_name
        if source_mask.any():
            specs.append((f"physics_{source_name}_only", source_mask, pair_label))
    if len(set(appearance.tolist())) > 1:
        specs.append(("appearance", np.ones(len(appearance), dtype=bool), appearance))
    return specs


def run_probe_tasks(
    payload: dict[str, np.ndarray],
    *,
    seeds: list[int],
    train_fraction: float,
    val_fraction: float,
    max_iter: int,
    c_value: float,
) -> tuple[list[dict], dict[str, dict[str, list[str]]], list[str]]:
    x = payload["X"]
    layer_names = [str(name) for name in payload["layer_names"].tolist()]
    all_pair_ids = payload["pair_id"]
    rows: list[dict] = []
    split_records: dict[str, dict[str, list[str]]] = {}
    skipped: list[str] = []
    for seed in seeds:
        split = deterministic_group_split(
            all_pair_ids,
            seed=seed,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        split_records[str(seed)] = {
            name: sorted(members) for name, members in split.items()
        }
        for task_name, task_mask, labels in probe_specs(payload):
            task_pair_ids = all_pair_ids[task_mask]
            task_x = x[task_mask]
            task_y_values = labels[task_mask]
            try:
                task_y, classes = encode_labels(task_y_values)
            except ValueError as error:
                skipped.append(f"{task_name}: {error}")
                continue
            masks = split_masks(task_pair_ids, split)
            if not all(mask.any() for mask in masks.values()):
                skipped.append(f"{task_name}: empty split for seed {seed}")
                continue
            if len(set(task_y[masks["train"]].tolist())) < 2:
                skipped.append(f"{task_name}: one-class train split for seed {seed}")
                continue
            for layer_index, layer_name in enumerate(layer_names):
                metrics = fit_layer_probe(
                    task_x[:, layer_index, :],
                    task_y,
                    masks,
                    seed=seed,
                    max_iter=max_iter,
                    c_value=c_value,
                )
                for split_name, (accuracy, baseline, count) in metrics.items():
                    rows.append(
                        {
                            "task": task_name,
                            "seed": seed,
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "split": split_name,
                            "n": count,
                            "classes": "|".join(classes),
                            "accuracy": accuracy,
                            "majority_baseline_accuracy": baseline,
                            "train_pairs": len(split["train"]),
                            "val_pairs": len(split["val"]),
                            "test_pairs": len(split["test"]),
                        }
                    )
    if not rows:
        raise ValueError(f"No probe rows were produced; skipped={skipped}")
    return rows, split_records, skipped


def summarize(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["task", "layer_index", "layer_name", "split"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            majority_baseline_mean=("majority_baseline_accuracy", "mean"),
            n_mean=("n", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["task", "split", "accuracy_mean", "layer_index"], ascending=[True, True, False, True])
    )
    best_rows = []
    test_summary = summary[summary["split"] == "test"]
    for task, group in test_summary.groupby("task"):
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
    payload = {
        "best_test_layers": best_rows,
        "rows": int(len(frame)),
        "tasks": sorted(frame["task"].unique().tolist()),
        "seeds": sorted(int(seed) for seed in frame["seed"].unique().tolist()),
    }
    return summary, payload


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        if (args.output_dir / "probe_results.csv").exists():
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_npz(args.representations)
    rows, split_records, skipped = run_probe_tasks(
        payload,
        seeds=parse_seeds(args.seeds),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        max_iter=args.max_iter,
        c_value=args.c,
    )
    results = pd.DataFrame(rows)
    summary, summary_payload = summarize(rows)
    results.to_csv(args.output_dir / "probe_results.csv", index=False)
    summary.to_csv(args.output_dir / "probe_summary.csv", index=False)
    (args.output_dir / "split_assignments.json").write_text(
        json.dumps(split_records, indent=2) + "\n"
    )
    summary_payload.update(
        {
            "representations": str(args.representations),
            "representations_shape": [int(value) for value in payload["X"].shape],
            "skipped": skipped,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "c": args.c,
            "max_iter": args.max_iter,
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
