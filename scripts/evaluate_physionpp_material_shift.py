from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
    probabilities = np.asarray([row["positive_probability"] for row in rows], dtype=np.float64)
    pair_scores = []
    for key in sorted({str(row["pair_key"]) for row in rows}):
        pair_rows = [row for row in rows if row["pair_key"] == key]
        pair_scores.append(float(np.mean([row["correct"] for row in pair_rows])))
    return {
        "n_videos": len(rows),
        "n_pairs": len(pair_scores),
        "accuracy": float(accuracy_score(labels, predictions)),
        "pair_mean_accuracy": float(np.mean(pair_scores)),
        "mean_positive_probability": float(probabilities.mean()),
        "pred_positive_rate": float(predictions.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a clean-trained Physion++ Mass readout on material-shifted videos."
    )
    parser.add_argument("--clean-features", default="data/physionpp/mass_only/features_vjepa2_mass_full.npz")
    parser.add_argument("--shift-features", default="data/physionpp/mass_material_ball_pilot/features_vjepa2_material_ball_pilot.npz")
    parser.add_argument("--manifest", default="data/physionpp/mass_material_ball_pilot/manifest_material_mass_collision.csv")
    parser.add_argument("--output", default="results/physionpp_material_shift/summary.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean = np.load(args.clean_features, allow_pickle=False)
    shifted = np.load(args.shift_features, allow_pickle=False)
    manifest_rows = read_manifest(Path(args.manifest))

    X = clean["X"]
    y = clean["y"]
    split = clean["split"].astype(str)
    clean_rgb = clean["rgb_mp4"].astype(str)
    clean_by_rgb = {path: i for i, path in enumerate(clean_rgb)}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=args.seed),
    )
    clf.fit(X[split == "readout"], y[split == "readout"])

    source_paths: list[str] = []
    source_meta: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        source = row["source_rgb_mp4"]
        if source not in source_paths:
            source_paths.append(source)
        source_meta.setdefault(source, row)

    source_idx = np.asarray([clean_by_rgb[path] for path in source_paths], dtype=np.int64)
    clean_prob = clf.predict_proba(X[source_idx])[:, 1]
    clean_pred = (clean_prob >= 0.5).astype(np.int64)
    clean_y = y[source_idx]

    clean_records: list[dict[str, object]] = []
    for source, probability, prediction, label in zip(source_paths, clean_prob, clean_pred, clean_y):
        meta = source_meta[source]
        clean_records.append(
            {
                "variant": "clean",
                "rgb_mp4": source,
                "source_rgb_mp4": source,
                "pair_key": meta["pair_key"],
                "label": int(label),
                "prediction": int(prediction),
                "positive_probability": float(probability),
                "correct": int(prediction == label),
            }
        )

    shift_prob = clf.predict_proba(shifted["X"])[:, 1]
    shift_pred = (shift_prob >= 0.5).astype(np.int64)
    shift_y = shifted["y"]

    clean_index_by_source = {source: i for i, source in enumerate(source_paths)}
    shifted_records: list[dict[str, object]] = []
    for i, row in enumerate(manifest_rows):
        source = row["source_rgb_mp4"]
        clean_i = clean_index_by_source[source]
        correct = int(shift_pred[i] == shift_y[i])
        clean_correct = int(clean_pred[clean_i] == clean_y[clean_i])
        shifted_records.append(
            {
                "variant": row["material_variant"],
                "rgb_mp4": row["rgb_mp4"],
                "source_rgb_mp4": source,
                "pair_key": row["pair_key"],
                "label": int(shift_y[i]),
                "prediction": int(shift_pred[i]),
                "positive_probability": float(shift_prob[i]),
                "correct": correct,
                "clean_prediction": int(clean_pred[clean_i]),
                "clean_positive_probability": float(clean_prob[clean_i]),
                "probability_shift_vs_clean": float(shift_prob[i] - clean_prob[clean_i]),
                "correctness_delta_vs_clean": correct - clean_correct,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_name(output.stem + "_predictions.csv")
    all_records = clean_records + shifted_records
    fields: list[str] = []
    for row in all_records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_records)

    summary: dict[str, object] = {
        "clean_train_features": args.clean_features,
        "shift_features": args.shift_features,
        "manifest": args.manifest,
        "train_readout_videos": int((split == "readout").sum()),
        "selected_clean_source": summarize(clean_records),
        "variants": {},
        "predictions": str(prediction_path),
        "seed": args.seed,
    }
    clean_accuracy = summary["selected_clean_source"]["accuracy"]  # type: ignore[index]
    for variant in sorted({str(row["variant"]) for row in shifted_records}):
        rows = [row for row in shifted_records if row["variant"] == variant]
        variant_summary = summarize(rows)
        variant_summary["mean_probability_shift_vs_clean"] = float(
            np.mean([row["probability_shift_vs_clean"] for row in rows])
        )
        variant_summary["accuracy_delta_vs_clean_source"] = float(variant_summary["accuracy"] - clean_accuracy)
        summary["variants"][variant] = variant_summary  # type: ignore[index]

    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
