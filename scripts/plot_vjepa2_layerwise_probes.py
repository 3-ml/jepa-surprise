import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TASK_LABELS = {
    "appearance": "Appearance",
    "physics_all": "Physics, all appearances",
    "physics_original_only": "Physics, original only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="V-JEPA2-G Layerwise Probe Accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.summary_csv)
    test = data[data["split"] == "test"].copy()
    if test.empty:
        raise ValueError(f"No test rows found in {args.summary_csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = {
        "appearance": "#0072B2",
        "physics_all": "#D55E00",
        "physics_original_only": "#009E73",
    }
    for task in ("appearance", "physics_all", "physics_original_only"):
        rows = test[test["task"] == task].sort_values("layer_index")
        if rows.empty:
            continue
        x = rows["layer_index"].to_numpy()
        y = rows["accuracy_mean"].to_numpy()
        std = rows["accuracy_std"].fillna(0.0).to_numpy()
        baseline = rows["majority_baseline_mean"].to_numpy()
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, color=colors[task], label=TASK_LABELS[task])
        ax.fill_between(x, y - std, y + std, color=colors[task], alpha=0.13, linewidth=0)
        ax.plot(x, baseline, linestyle="--", linewidth=1.0, color=colors[task], alpha=0.45)

    ax.set_title(args.title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Pair-held-out test accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(test["layer_index"].min(), test["layer_index"].max())
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
