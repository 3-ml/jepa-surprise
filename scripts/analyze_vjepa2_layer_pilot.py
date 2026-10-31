#!/usr/bin/env python3
"""Run locked grouped probes and normalized geometry for the V-JEPA2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def classifier(c_value: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def select_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> tuple[float, dict[str, float]]:
    splitter = GroupKFold(n_splits=3)
    scores = {}
    for c_value in c_grid:
        fold_scores = []
        for train, validation in splitter.split(x, y, groups):
            model = classifier(c_value, seed)
            model.fit(x[train], y[train])
            prediction = model.predict(x[validation])
            fold_scores.append(balanced_accuracy_score(y[validation], prediction))
        scores[str(c_value)] = float(np.mean(fold_scores))
    best_score = max(scores.values())
    selected = min(float(c) for c, score in scores.items() if score == best_score)
    return selected, scores


def grouped_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    unique_groups = np.unique(groups)
    label_index = {label: index for index, label in enumerate(labels)}
    group_confusions = np.zeros(
        (len(unique_groups), len(labels), len(labels)), dtype=np.int64
    )
    for group_index, group in enumerate(unique_groups):
        for truth, prediction in zip(y_true[groups == group], y_pred[groups == group]):
            group_confusions[
                group_index, label_index[truth], label_index[prediction]
            ] += 1
    draws = rng.integers(
        0, len(unique_groups), size=(resamples, len(unique_groups))
    )
    confusion = group_confusions[draws].sum(axis=1)
    true_support = confusion.sum(axis=2)
    predicted_support = confusion.sum(axis=1)
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=true_support != 0,
    )
    balanced_accuracy = recall.mean(axis=1)
    f1 = np.divide(
        2 * true_positive,
        true_support + predicted_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=(true_support + predicted_support) != 0,
    ).mean(axis=1)
    ba_low, ba_high = np.percentile(balanced_accuracy, [2.5, 97.5])
    f1_low, f1_high = np.percentile(f1, [2.5, 97.5])
    return float(ba_low), float(ba_high), float(f1_low), float(f1_high)


def cosine_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def geometry_rows(
    x: np.ndarray,
    metadata: pd.DataFrame,
    view: str,
    layer: int,
) -> list[dict]:
    test = metadata["split"].eq("test").to_numpy()
    clean = metadata["condition"].eq("clean").to_numpy() & test
    changed = ~metadata["condition"].eq("clean").to_numpy() & test
    z = cosine_rows(x)
    clean_indices = np.flatnonzero(clean)
    changed_indices = np.flatnonzero(changed)
    clean_distances = pdist(z[clean_indices], metric="cosine")
    normalization = float(np.median(clean_distances))
    clean_ids = metadata.iloc[clean_indices]["base_video_id"].to_numpy()

    retrieval_hits = []
    same_state_distances = []
    by_condition: dict[str, list[float]] = {}
    for query_index in changed_indices:
        distances = 1.0 - z[clean_indices] @ z[query_index]
        nearest = clean_ids[int(np.argmin(distances))]
        expected = metadata.iloc[query_index]["base_video_id"]
        retrieval_hits.append(float(nearest == expected))
        exact_index = clean_indices[np.flatnonzero(clean_ids == expected)[0]]
        normalized_distance = float((1.0 - z[query_index] @ z[exact_index]) / normalization)
        same_state_distances.append(normalized_distance)
        condition = metadata.iloc[query_index]["condition"]
        by_condition.setdefault(condition, []).append(normalized_distance)

    physical_distances = []
    test_metadata = metadata.loc[test]
    for (_pair_id, _condition), group in test_metadata.groupby(
        ["base_pair_id", "condition"], sort=True
    ):
        group_indices = group.index.to_numpy()
        if len(group_indices) != 2 or group["possible"].nunique() != 2:
            continue
        physical_distances.append(
            float((1.0 - z[group_indices[0]] @ z[group_indices[1]]) / normalization)
        )
    rows = [
        {
            "view": view,
            "layer": layer,
            "metric": "same_state_cross_appearance_retrieval_top1",
            "condition": "all_transforms",
            "value": float(np.mean(retrieval_hits)),
            "n": len(retrieval_hits),
        },
        {
            "view": view,
            "layer": layer,
            "metric": "normalized_same_state_appearance_displacement",
            "condition": "all_transforms",
            "value": float(np.median(same_state_distances)),
            "n": len(same_state_distances),
        },
        {
            "view": view,
            "layer": layer,
            "metric": "normalized_matched_physical_displacement",
            "condition": "all_conditions",
            "value": float(np.median(physical_distances)),
            "n": len(physical_distances),
        },
    ]
    for condition, values in sorted(by_condition.items()):
        rows.append(
            {
                "view": view,
                "layer": layer,
                "metric": "normalized_same_state_appearance_displacement",
                "condition": condition,
                "value": float(np.median(values)),
                "n": len(values),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    features_path = args.features_root / "layer_features.npz"
    samples_path = args.features_root / "samples.json"
    provenance_path = args.features_root / "provenance.json"
    data = np.load(features_path)
    metadata = pd.DataFrame(json.loads(samples_path.read_text(encoding="utf-8")))
    layers = data["layers"].astype(int).tolist()
    if len(metadata) != data["pooled"].shape[0]:
        raise ValueError("Feature and metadata row counts differ")
    if metadata.groupby("base_video_id")["split"].nunique().max() != 1:
        raise ValueError("Base-video split leakage detected")
    if metadata.groupby("base_pair_id")["split"].nunique().max() != 1:
        raise ValueError("Base-pair split leakage detected")

    train_mask = metadata["split"].eq("development").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    clean_mask = metadata["condition"].eq("clean").to_numpy()
    c_grid = [float(value) for value in config["probe"]["c_grid"]]
    seeds = [int(value) for value in config["probe"]["seeds"]]
    resamples = int(config["probe"]["bootstrap_resamples"])
    rng = np.random.default_rng(config["probe"]["bootstrap_seed"])
    probe_rows: list[dict] = []
    prediction_rows: list[dict] = []
    geometry: list[dict] = []

    views = {"pooled": data["pooled"], "token_mean_std": data["token_mean_std"]}
    for view, features in views.items():
        for layer_index, layer in enumerate(layers):
            x = features[:, layer_index, :]
            geometry.extend(geometry_rows(x, metadata, view, layer))
            y_appearance = metadata["condition"].to_numpy()
            appearance_groups = metadata["scene"].to_numpy()
            for seed in seeds:
                selected_c, cv_scores = select_c(
                    x[train_mask],
                    y_appearance[train_mask],
                    appearance_groups[train_mask],
                    c_grid,
                    seed,
                )
                model = classifier(selected_c, seed)
                model.fit(x[train_mask], y_appearance[train_mask])
                prediction = model.predict(x[test_mask])
                y_test = y_appearance[test_mask]
                groups_test = appearance_groups[test_mask]
                appearance_labels = np.unique(y_appearance)
                ba_low, ba_high, f1_low, f1_high = grouped_bootstrap_intervals(
                    y_test,
                    prediction,
                    groups_test,
                    appearance_labels,
                    resamples,
                    rng,
                )
                f1_metric = lambda truth, pred: f1_score(
                    truth, pred, average="macro", labels=appearance_labels
                )
                probe_rows.append(
                    {
                        "view": view,
                        "layer": layer,
                        "task": "appearance_family",
                        "train_condition": "all",
                        "test_condition": "all",
                        "seed": seed,
                        "selected_c": selected_c,
                        "cv_scores": json.dumps(cv_scores, sort_keys=True),
                        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                        "balanced_accuracy_ci_low": ba_low,
                        "balanced_accuracy_ci_high": ba_high,
                        "macro_f1": f1_metric(y_test, prediction),
                        "macro_f1_ci_low": f1_low,
                        "macro_f1_ci_high": f1_high,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                    }
                )
                for row_index, truth, pred in zip(
                    metadata.index[test_mask], y_test, prediction
                ):
                    prediction_rows.append(
                        {
                            "view": view,
                            "layer": layer,
                            "task": "appearance_family",
                            "train_condition": "all",
                            "test_condition": "all",
                            "seed": seed,
                            "row_index": int(row_index),
                            "truth": truth,
                            "prediction": pred,
                        }
                    )

            y_physics = metadata["possible"].astype(int).to_numpy()
            physics_groups = metadata["scene"].to_numpy()
            clean_train = train_mask & clean_mask
            selected_by_seed = {}
            for seed in seeds:
                selected_c, cv_scores = select_c(
                    x[clean_train],
                    y_physics[clean_train],
                    physics_groups[clean_train],
                    c_grid,
                    seed,
                )
                selected_by_seed[seed] = (selected_c, cv_scores)
                model = classifier(selected_c, seed)
                model.fit(x[clean_train], y_physics[clean_train])
                for condition in metadata["condition"].unique():
                    condition_test = test_mask & metadata["condition"].eq(condition).to_numpy()
                    prediction = model.predict(x[condition_test])
                    truth = y_physics[condition_test]
                    groups_test = physics_groups[condition_test]
                    ba_low, ba_high, f1_low, f1_high = grouped_bootstrap_intervals(
                        truth,
                        prediction,
                        groups_test,
                        np.asarray([0, 1]),
                        resamples,
                        rng,
                    )
                    f1_metric = lambda target, pred: f1_score(
                        target, pred, average="macro", labels=[0, 1]
                    )
                    probe_rows.append(
                        {
                            "view": view,
                            "layer": layer,
                            "task": "physics_clean_trained",
                            "train_condition": "clean",
                            "test_condition": condition,
                            "seed": seed,
                            "selected_c": selected_c,
                            "cv_scores": json.dumps(cv_scores, sort_keys=True),
                            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                            "balanced_accuracy_ci_low": ba_low,
                            "balanced_accuracy_ci_high": ba_high,
                            "macro_f1": f1_metric(truth, prediction),
                            "macro_f1_ci_low": f1_low,
                            "macro_f1_ci_high": f1_high,
                            "n_train": int(clean_train.sum()),
                            "n_test": int(condition_test.sum()),
                        }
                    )
                    for row_index, target, pred in zip(
                        metadata.index[condition_test], truth, prediction
                    ):
                        prediction_rows.append(
                            {
                                "view": view,
                                "layer": layer,
                                "task": "physics_clean_trained",
                                "train_condition": "clean",
                                "test_condition": condition,
                                "seed": seed,
                                "row_index": int(row_index),
                                "truth": int(target),
                                "prediction": int(pred),
                            }
                        )

                for condition in metadata["condition"].unique():
                    oracle_train = train_mask & metadata["condition"].eq(condition).to_numpy()
                    oracle_test = test_mask & metadata["condition"].eq(condition).to_numpy()
                    oracle_c, oracle_cv = select_c(
                        x[oracle_train],
                        y_physics[oracle_train],
                        physics_groups[oracle_train],
                        c_grid,
                        seed,
                    )
                    oracle = classifier(oracle_c, seed)
                    oracle.fit(x[oracle_train], y_physics[oracle_train])
                    prediction = oracle.predict(x[oracle_test])
                    truth = y_physics[oracle_test]
                    groups_test = physics_groups[oracle_test]
                    ba_low, ba_high, f1_low, f1_high = grouped_bootstrap_intervals(
                        truth,
                        prediction,
                        groups_test,
                        np.asarray([0, 1]),
                        resamples,
                        rng,
                    )
                    f1_metric = lambda target, pred: f1_score(
                        target, pred, average="macro", labels=[0, 1]
                    )
                    probe_rows.append(
                        {
                            "view": view,
                            "layer": layer,
                            "task": "physics_condition_oracle",
                            "train_condition": condition,
                            "test_condition": condition,
                            "seed": seed,
                            "selected_c": oracle_c,
                            "cv_scores": json.dumps(oracle_cv, sort_keys=True),
                            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                            "balanced_accuracy_ci_low": ba_low,
                            "balanced_accuracy_ci_high": ba_high,
                            "macro_f1": f1_metric(truth, prediction),
                            "macro_f1_ci_low": f1_low,
                            "macro_f1_ci_high": f1_high,
                            "n_train": int(oracle_train.sum()),
                            "n_test": int(oracle_test.sum()),
                        }
                    )

    probes = pd.DataFrame(probe_rows)
    predictions = pd.DataFrame(prediction_rows)
    geometry_frame = pd.DataFrame(geometry)
    probes.to_csv(args.output_root / "probe_metrics.csv", index=False)
    predictions.to_csv(args.output_root / "test_predictions.csv", index=False)
    geometry_frame.to_csv(args.output_root / "geometry_metrics.csv", index=False)

    appearance = probes[probes["task"].eq("appearance_family")]
    physics_clean = probes[
        probes["task"].eq("physics_clean_trained")
        & probes["test_condition"].eq("clean")
    ]
    view_appearance_max = appearance.groupby("view")["balanced_accuracy"].max()
    gate = {
        "extraction_alignment": True,
        "appearance_above_floor_any_layer": bool(
            (view_appearance_max >= config["gate"]["appearance_balanced_accuracy_floor"]).all()
        ),
        "physics_clean_headroom": bool(
            physics_clean["balanced_accuracy"].max()
            >= config["gate"]["physics_clean_balanced_accuracy_floor"]
        ),
        "both_pooling_views_present": len(view_appearance_max)
        == config["gate"]["required_pooling_views"],
        "split_leakage": False,
    }
    gate["continue_to_confirmatory"] = bool(
        gate["extraction_alignment"]
        and gate["appearance_above_floor_any_layer"]
        and gate["physics_clean_headroom"]
        and gate["both_pooling_views_present"]
        and not gate["split_leakage"]
    )
    summary = {
        "schema_version": 1,
        "scope": config["scope_note"],
        "n_samples": len(metadata),
        "development_scenes": config["development_scenes"],
        "test_scenes": config["test_scenes"],
        "layers": layers,
        "seeds": seeds,
        "appearance_chance_balanced_accuracy": 0.25,
        "appearance_best": appearance.loc[
            appearance["balanced_accuracy"].idxmax()
        ].to_dict(),
        "physics_clean_best": physics_clean.loc[
            physics_clean["balanced_accuracy"].idxmax()
        ].to_dict(),
        "gate": gate,
        "limitations": [
            "Original IntPhys O1 feasibility data, not IntPhys2 Main.",
            "Only four untouched test scenes; bootstrap intervals are descriptive.",
            "Sixteen uniformly sampled frames were used to fit 4 GB VRAM.",
            "No pixel-predictive comparator was run, so no latent-prediction-specific claim is permitted.",
            "Five sampled depths cannot establish a discrete transition zone.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": {
            "features_sha256": sha256_file(features_path),
            "samples_sha256": sha256_file(samples_path),
            "extraction_provenance_sha256": sha256_file(provenance_path),
            "config_sha256": sha256_file(args.config),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (args.output_root / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
