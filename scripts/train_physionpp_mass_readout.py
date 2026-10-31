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


def bootstrap_pair_ci(correct_by_pair: list[float], seed: int, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(correct_by_pair, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a logistic OCP readout on Physion++ Mass features.")
    parser.add_argument("--features", default="data/physionpp/mass_only/features_vjepa2_debug20.npz")
    parser.add_argument("--output", default="results/physionpp_mass_debug20_readout.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    split = data["split"].astype(str)
    scenario = data["scenario"].astype(str)
    pair_key = data["pair_key"].astype(str)
    rgb_mp4 = data["rgb_mp4"].astype(str)

    train_mask = split == "readout"
    test_mask = split == "test"
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=args.seed),
    )
    clf.fit(X[train_mask], y[train_mask])

    pred = clf.predict(X[test_mask])
    prob = clf.predict_proba(X[test_mask])[:, 1]
    y_test = y[test_mask]
    pair_test = pair_key[test_mask]
    scenario_test = scenario[test_mask]
    rgb_test = rgb_mp4[test_mask]

    pair_scores = []
    for key in sorted(set(pair_test)):
        idx = np.where(pair_test == key)[0]
        pair_scores.append(float((pred[idx] == y_test[idx]).mean()))

    by_scenario = {}
    for name in sorted(set(scenario_test)):
        idx = scenario_test == name
        by_scenario[name] = {
            "n_test": int(idx.sum()),
            "accuracy": float(accuracy_score(y_test[idx], pred[idx])),
            "positive_rate": float(y_test[idx].mean()),
            "pred_positive_rate": float(pred[idx].mean()),
        }

    ci_low, ci_high = bootstrap_pair_ci(pair_scores, seed=args.seed)
    result = {
        "features": args.features,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_pairs": len(pair_scores),
        "train_accuracy": float(accuracy_score(y[train_mask], clf.predict(X[train_mask]))),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_pair_bootstrap_ci95": [ci_low, ci_high],
        "test_positive_rate": float(y_test.mean()),
        "pred_positive_rate": float(pred.mean()),
        "mean_positive_probability": float(prob.mean()),
        "by_scenario": by_scenario,
        "seed": args.seed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    prediction_path = output.with_name(output.stem + "_predictions.csv")
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rgb_mp4", "scenario", "pair_key", "label", "prediction", "positive_probability"],
        )
        writer.writeheader()
        for path, scen, key, label, prediction, probability in zip(
            rgb_test, scenario_test, pair_test, y_test, pred, prob
        ):
            writer.writerow(
                {
                    "rgb_mp4": path,
                    "scenario": scen,
                    "pair_key": key,
                    "label": int(label),
                    "prediction": int(prediction),
                    "positive_probability": float(probability),
                }
            )
    result["predictions"] = str(prediction_path)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
