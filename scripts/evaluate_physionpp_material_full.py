from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed),
    )


def bootstrap_pair_ci(records: list[dict[str, object]], seed: int, n_boot: int = 5000) -> tuple[float, float]:
    by_pair: dict[str, list[float]] = {}
    for row in records:
        pair_key = str(row.get("pair_key", ""))
        if not pair_key:
            continue
        by_pair.setdefault(pair_key, []).append(float(row["correct"]))
    pair_scores = np.asarray([np.mean(values) for values in by_pair.values()], dtype=np.float64)
    if pair_scores.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(pair_scores, size=pair_scores.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pair_accuracy(records: list[dict[str, object]]) -> dict[str, float]:
    by_pair: dict[str, list[float]] = {}
    for row in records:
        pair_key = str(row.get("pair_key", ""))
        if pair_key:
            by_pair.setdefault(pair_key, []).append(float(row["correct"]))
    return {key: float(np.mean(values)) for key, values in by_pair.items()}


def bootstrap_values_ci(values: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_delta_summary(
    baseline_records: list[dict[str, object]],
    comparison_records: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    base = pair_accuracy(baseline_records)
    comp = pair_accuracy(comparison_records)
    common = sorted(set(base) & set(comp))
    deltas = np.asarray([comp[key] - base[key] for key in common], dtype=np.float64)
    ci_low, ci_high = bootstrap_values_ci(deltas, seed=seed)
    video_deltas = [int(row.get("correctness_delta_vs_clean_source", 0)) for row in comparison_records]
    return {
        "n_pairs": len(common),
        "mean_pair_accuracy_delta": float(deltas.mean()) if deltas.size else float("nan"),
        "pair_delta_bootstrap_ci95": [ci_low, ci_high],
        "pairs_improved": int((deltas > 0).sum()),
        "pairs_harmed": int((deltas < 0).sum()),
        "pairs_unchanged": int((deltas == 0).sum()),
        "videos_corrected": int(sum(1 for value in video_deltas if value > 0)),
        "videos_harmed": int(sum(1 for value in video_deltas if value < 0)),
        "videos_unchanged": int(sum(1 for value in video_deltas if value == 0)),
    }


def summarize(records: list[dict[str, object]], seed: int) -> dict[str, object]:
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in records], dtype=np.int64)
    probabilities = np.asarray([row["positive_probability"] for row in records], dtype=np.float64)
    pair_keys = sorted({str(row["pair_key"]) for row in records if str(row.get("pair_key", ""))})
    ci_low, ci_high = bootstrap_pair_ci(records, seed=seed)
    by_pair = []
    for key in pair_keys:
        rows = [row for row in records if row.get("pair_key") == key]
        by_pair.append(float(np.mean([row["correct"] for row in rows])))
    return {
        "n_videos": len(records),
        "n_pairs": len(pair_keys),
        "accuracy": float(accuracy_score(labels, predictions)),
        "pair_mean_accuracy": float(np.mean(by_pair)) if by_pair else float("nan"),
        "pair_bootstrap_ci95": [ci_low, ci_high],
        "positive_rate": float(labels.mean()) if labels.size else float("nan"),
        "pred_positive_rate": float(predictions.mean()) if predictions.size else float("nan"),
        "mean_positive_probability": float(probabilities.mean()) if probabilities.size else float("nan"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in records:
        pair_key = str(row.get("pair_key", ""))
        if not pair_key:
            continue
        key = (str(row["protocol"]), str(row["eval_variant"]), pair_key)
        grouped.setdefault(key, []).append(row)

    clean_baseline: dict[str, tuple[float, float]] = {}
    for (protocol, eval_variant, pair_key), rows in grouped.items():
        if protocol == "clean_mass_readout" and eval_variant == "clean_source":
            clean_baseline[pair_key] = (
                float(np.mean([row["correct"] for row in rows])),
                float(np.mean([row["positive_probability"] for row in rows])),
            )

    out: list[dict[str, object]] = []
    for (protocol, eval_variant, pair_key), rows in sorted(grouped.items()):
        pair_accuracy = float(np.mean([row["correct"] for row in rows]))
        mean_probability = float(np.mean([row["positive_probability"] for row in rows]))
        record: dict[str, object] = {
            "protocol": protocol,
            "eval_variant": eval_variant,
            "pair_key": pair_key,
            "n_members": len(rows),
            "pair_accuracy": pair_accuracy,
            "mean_positive_probability": mean_probability,
            "labels": ";".join(str(row["label"]) for row in rows),
            "predictions": ";".join(str(row["prediction"]) for row in rows),
            "sources": ";".join(str(row["source_rgb_mp4"]) for row in rows),
        }
        if pair_key in clean_baseline and not (protocol == "clean_mass_readout" and eval_variant == "clean_source"):
            base_acc, base_prob = clean_baseline[pair_key]
            record["pair_accuracy_delta_vs_clean_mass_source"] = pair_accuracy - base_acc
            record["mean_probability_delta_vs_clean_mass_source"] = mean_probability - base_prob
        out.append(record)
    return out


def validate_feature_manifest(features: np.lib.npyio.NpzFile, rows: list[dict[str, str]], name: str) -> None:
    if len(rows) != len(features["X"]):
        raise RuntimeError(f"{name} manifest rows ({len(rows)}) do not match features ({len(features['X'])}).")
    feature_rgb = features["rgb_mp4"].astype(str)
    for i, row in enumerate(rows):
        if row["rgb_mp4"] != feature_rgb[i]:
            raise RuntimeError(f"{name} row {i} does not match manifest: {feature_rgb[i]} vs {row['rgb_mp4']}")


def records_from_predictions(
    *,
    protocol: str,
    train_variant: str,
    eval_variant: str,
    rows: list[dict[str, str]],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    source_probabilities: dict[str, float] | None = None,
    source_correct: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row, label, prob, pred in zip(rows, y_true, probabilities, predictions):
        source = row.get("source_rgb_mp4", row["rgb_mp4"])
        correct = int(pred == label)
        record: dict[str, object] = {
            "protocol": protocol,
            "train_variant": train_variant,
            "eval_variant": eval_variant,
            "split": row["split"],
            "scenario": row["scenario"],
            "pair_key": row.get("pair_key", ""),
            "rgb_mp4": row["rgb_mp4"],
            "source_rgb_mp4": source,
            "label": int(label),
            "prediction": int(pred),
            "positive_probability": float(prob),
            "correct": correct,
        }
        if source_probabilities is not None and source in source_probabilities:
            record["clean_source_positive_probability"] = float(source_probabilities[source])
            record["probability_shift_vs_clean_source"] = float(prob - source_probabilities[source])
        if source_correct is not None and source in source_correct:
            record["correctness_delta_vs_clean_source"] = int(correct - source_correct[source])
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clean-trained and appearance-oracle Physion++ Mass-Collision material readouts."
    )
    parser.add_argument("--clean-features", default="data/physionpp/mass_only/features_vjepa2_mass_full.npz")
    parser.add_argument("--shift-features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-shift-features", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument(
        "--clean-train-manifest",
        default=None,
        help=(
            "Optional material manifest whose unique readout source_rgb_mp4 rows define "
            "the clean-source videos used to train the clean baseline."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean = np.load(args.clean_features, allow_pickle=False)
    eval_shifted = np.load(args.shift_features, allow_pickle=False)
    eval_manifest_rows = read_manifest(Path(args.manifest))
    train_shifted = np.load(args.train_shift_features, allow_pickle=False) if args.train_shift_features else eval_shifted
    train_manifest_rows = read_manifest(Path(args.train_manifest)) if args.train_manifest else eval_manifest_rows

    validate_feature_manifest(eval_shifted, eval_manifest_rows, "eval")
    validate_feature_manifest(train_shifted, train_manifest_rows, "train")

    clean_X = clean["X"]
    clean_y = clean["y"]
    clean_split = clean["split"].astype(str)
    clean_scenario = clean["scenario"].astype(str)
    clean_rgb = clean["rgb_mp4"].astype(str)
    clean_by_rgb = {path: i for i, path in enumerate(clean_rgb)}

    eval_shifted_X = eval_shifted["X"]
    eval_shifted_y = eval_shifted["y"]
    train_shifted_X = train_shifted["X"]
    train_shifted_y = train_shifted["y"]
    train_variants = {row["material_variant"] for row in train_manifest_rows if row["split"] == "readout"}
    eval_variants = {row["material_variant"] for row in eval_manifest_rows if row["split"] == "test"}
    variants = sorted(train_variants & eval_variants)
    if not variants:
        raise RuntimeError("No material variants have both train/readout and eval/test rows.")

    source_test_rows_by_path: dict[str, dict[str, str]] = {}
    for row in eval_manifest_rows:
        if row["split"] == "test":
            source_test_rows_by_path.setdefault(row["source_rgb_mp4"], row)
    source_test_paths = sorted(source_test_rows_by_path)
    missing_sources = [path for path in source_test_paths if path not in clean_by_rgb]
    if missing_sources:
        raise RuntimeError(f"Clean feature file is missing {len(missing_sources)} source videos.")

    if args.clean_train_manifest:
        clean_train_manifest_rows = read_manifest(Path(args.clean_train_manifest))
        clean_train_sources = sorted(
            {
                row.get("source_rgb_mp4", row["rgb_mp4"])
                for row in clean_train_manifest_rows
                if row["split"] == "readout"
            }
        )
        missing_clean_train_sources = [path for path in clean_train_sources if path not in clean_by_rgb]
        if missing_clean_train_sources:
            raise RuntimeError(
                f"Clean feature file is missing {len(missing_clean_train_sources)} clean train source videos."
            )
        clean_mass_train = np.zeros_like(clean_split, dtype=bool)
        clean_mass_train[np.asarray([clean_by_rgb[path] for path in clean_train_sources], dtype=np.int64)] = True
        clean_train_variant = "clean_sources_from_clean_train_manifest"
    else:
        clean_train_sources = sorted(clean_rgb[clean_split == "readout"].tolist())
        clean_mass_train = clean_split == "readout"
        clean_train_variant = "clean_all_mass_readout"
    clean_collision_train = (clean_split == "readout") & (clean_scenario == "Mass-Collision")
    clean_test_idx = np.asarray([clean_by_rgb[path] for path in source_test_paths], dtype=np.int64)
    clean_test_rows = [source_test_rows_by_path[path] | {"rgb_mp4": path} for path in source_test_paths]

    clean_mass_clf = classifier(args.seed)
    clean_mass_clf.fit(clean_X[clean_mass_train], clean_y[clean_mass_train])
    clean_source_prob = clean_mass_clf.predict_proba(clean_X[clean_test_idx])[:, 1]
    clean_source_pred = (clean_source_prob >= 0.5).astype(np.int64)
    clean_source_records = records_from_predictions(
        protocol="clean_mass_readout",
        train_variant=clean_train_variant,
        eval_variant="clean_source",
        rows=clean_test_rows,
        y_true=clean_y[clean_test_idx],
        probabilities=clean_source_prob,
        predictions=clean_source_pred,
    )
    clean_prob_by_source = {
        row["source_rgb_mp4"]: float(prob) for row, prob in zip(clean_source_records, clean_source_prob)
    }
    clean_correct_by_source = {
        row["source_rgb_mp4"]: int(row["correct"]) for row in clean_source_records
    }

    clean_collision_clf = classifier(args.seed)
    clean_collision_clf.fit(clean_X[clean_collision_train], clean_y[clean_collision_train])
    collision_prob = clean_collision_clf.predict_proba(clean_X[clean_test_idx])[:, 1]
    collision_pred = (collision_prob >= 0.5).astype(np.int64)
    clean_collision_records = records_from_predictions(
        protocol="clean_collision_readout",
        train_variant="clean_collision_readout",
        eval_variant="clean_source",
        rows=clean_test_rows,
        y_true=clean_y[clean_test_idx],
        probabilities=collision_prob,
        predictions=collision_pred,
    )

    all_records: list[dict[str, object]] = clean_source_records + clean_collision_records
    summary: dict[str, object] = {
        "clean_features": args.clean_features,
        "shift_features": args.shift_features,
        "manifest": args.manifest,
        "train_shift_features": args.train_shift_features or args.shift_features,
        "train_manifest": args.train_manifest or args.manifest,
        "clean_train_manifest": args.clean_train_manifest,
        "seed": args.seed,
        "train_counts": {
            "clean_mass_readout": int(clean_mass_train.sum()),
            "clean_mass_readout_variant": clean_train_variant,
            "clean_collision_readout": int(clean_collision_train.sum()),
            "clean_train_source_scenarios": {
                scenario: int(
                    sum(
                        1
                        for path in clean_train_sources
                        if clean_scenario[clean_by_rgb[path]] == scenario
                    )
                )
                for scenario in sorted({clean_scenario[clean_by_rgb[path]] for path in clean_train_sources})
            },
        },
        "test_sources": {
            "n_videos": len(source_test_paths),
            "n_pairs": len({row.get("pair_key", "") for row in clean_test_rows}),
        },
        "clean_mass_readout_on_clean_source": summarize(clean_source_records, args.seed),
        "clean_collision_readout_on_clean_source": summarize(clean_collision_records, args.seed),
        "variants": {},
    }

    variant_summary_rows: list[dict[str, object]] = []
    for variant in variants:
        train_idx = np.asarray(
            [
                i
                for i, row in enumerate(train_manifest_rows)
                if row["material_variant"] == variant and row["split"] == "readout"
            ],
            dtype=np.int64,
        )
        test_idx = np.asarray(
            [
                i
                for i, row in enumerate(eval_manifest_rows)
                if row["material_variant"] == variant and row["split"] == "test"
            ],
            dtype=np.int64,
        )
        if train_idx.size == 0 or test_idx.size == 0:
            raise RuntimeError(f"Variant {variant} needs both readout and test rows for oracle evaluation.")

        test_rows = [row for row in eval_manifest_rows if row["material_variant"] == variant and row["split"] == "test"]
        clean_shift_prob = clean_mass_clf.predict_proba(eval_shifted_X[test_idx])[:, 1]
        clean_shift_pred = (clean_shift_prob >= 0.5).astype(np.int64)
        clean_shift_records = records_from_predictions(
            protocol="clean_mass_readout",
            train_variant=clean_train_variant,
            eval_variant=variant,
            rows=test_rows,
            y_true=eval_shifted_y[test_idx],
            probabilities=clean_shift_prob,
            predictions=clean_shift_pred,
            source_probabilities=clean_prob_by_source,
            source_correct=clean_correct_by_source,
        )
        all_records.extend(clean_shift_records)

        oracle = classifier(args.seed)
        oracle.fit(train_shifted_X[train_idx], train_shifted_y[train_idx])
        oracle_train_prob = oracle.predict_proba(train_shifted_X[train_idx])[:, 1]
        oracle_train_pred = (oracle_train_prob >= 0.5).astype(np.int64)
        oracle_test_prob = oracle.predict_proba(eval_shifted_X[test_idx])[:, 1]
        oracle_test_pred = (oracle_test_prob >= 0.5).astype(np.int64)
        oracle_records = records_from_predictions(
            protocol=f"oracle_{variant}_readout",
            train_variant=f"{variant}_shifted_readout",
            eval_variant=variant,
            rows=test_rows,
            y_true=eval_shifted_y[test_idx],
            probabilities=oracle_test_prob,
            predictions=oracle_test_pred,
            source_probabilities=clean_prob_by_source,
            source_correct=clean_correct_by_source,
        )
        all_records.extend(oracle_records)

        clean_shift_summary = summarize(clean_shift_records, args.seed)
        oracle_summary = summarize(oracle_records, args.seed)
        clean_shift_summary["accuracy_delta_vs_clean_mass_source"] = float(
            clean_shift_summary["accuracy"] - summary["clean_mass_readout_on_clean_source"]["accuracy"]  # type: ignore[index]
        )
        clean_shift_summary["mean_probability_shift_vs_clean_source"] = float(
            np.mean([row["probability_shift_vs_clean_source"] for row in clean_shift_records])
        )
        oracle_summary["accuracy_delta_vs_clean_mass_source"] = float(
            oracle_summary["accuracy"] - summary["clean_mass_readout_on_clean_source"]["accuracy"]  # type: ignore[index]
        )
        clean_shift_summary["paired_delta_vs_clean_mass_source"] = paired_delta_summary(
            clean_source_records, clean_shift_records, args.seed
        )
        oracle_summary["paired_delta_vs_clean_mass_source"] = paired_delta_summary(
            clean_source_records, oracle_records, args.seed
        )
        summary["variants"][variant] = {
            "shifted_rows": {
                "readout": int(train_idx.size),
                "test": int(test_idx.size),
            },
            "oracle_train_accuracy": float(accuracy_score(train_shifted_y[train_idx], oracle_train_pred)),
            "oracle_train_mean_positive_probability": float(oracle_train_prob.mean()),
            "clean_mass_readout_on_shifted_test": clean_shift_summary,
            "appearance_oracle_on_shifted_test": oracle_summary,
        }
        variant_summary_rows.extend(
            [
                {"variant": variant, "protocol": "clean_mass_readout", **clean_shift_summary},
                {"variant": variant, "protocol": f"oracle_{variant}_readout", **oracle_summary},
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_name(output.stem + "_predictions.csv")
    variant_summary_path = output.with_name(output.stem + "_variant_summary.csv")
    pair_summary_path = output.with_name(output.stem + "_pair_summary.csv")
    write_csv(prediction_path, all_records)
    write_csv(variant_summary_path, variant_summary_rows)
    write_csv(pair_summary_path, build_pair_summary(all_records))
    summary["predictions_csv"] = str(prediction_path)
    summary["variant_summary_csv"] = str(variant_summary_path)
    summary["pair_summary_csv"] = str(pair_summary_path)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
