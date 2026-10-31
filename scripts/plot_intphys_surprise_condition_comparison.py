from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    estimates = values[indices].mean(axis=1)
    return np.quantile(estimates, 0.025, axis=0), np.quantile(estimates, 0.975, axis=0)


def load_condition(path: Path, condition: str) -> dict[str, object]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["condition"] != condition:
                continue
            possible = np.asarray(json.loads(row["possible_windows"]), dtype=np.float64)
            impossible = np.asarray(json.loads(row["impossible_windows"]), dtype=np.float64)
            rows.append(
                {
                    "pair_id": row["pair_id"],
                    "possible": possible,
                    "impossible": impossible,
                    "margin": impossible - possible,
                    "average_margin": float(row["average_margin"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found for condition {condition!r} in {path}")
    possible = np.stack([row["possible"] for row in rows])
    impossible = np.stack([row["impossible"] for row in rows])
    margins = np.stack([row["margin"] for row in rows])
    return {
        "rows": rows,
        "possible": possible,
        "impossible": impossible,
        "margins": margins,
        "average_margins": np.asarray([row["average_margin"] for row in rows], dtype=np.float64),
    }


def line_with_band(ax, x: np.ndarray, values: np.ndarray, label: str, color: str, seed: int) -> None:
    mean = values.mean(axis=0)
    low, high = bootstrap_mean(values, seed, 5000)
    ax.plot(x, mean, marker="o", markersize=3.5, linewidth=1.9, label=label, color=color)
    ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)


def plot_curves(
    cond_a: dict[str, object],
    cond_b: dict[str, object],
    label_a: str,
    label_b: str,
    output: Path,
    title_prefix: str,
) -> None:
    possible_a = cond_a["possible"]
    possible_b = cond_b["possible"]
    impossible_a = cond_a["impossible"]
    impossible_b = cond_b["impossible"]
    margins_a = cond_a["margins"]
    margins_b = cond_b["margins"]
    assert isinstance(possible_a, np.ndarray)
    assert isinstance(possible_b, np.ndarray)
    assert isinstance(impossible_a, np.ndarray)
    assert isinstance(impossible_b, np.ndarray)
    assert isinstance(margins_a, np.ndarray)
    assert isinstance(margins_b, np.ndarray)

    x = np.arange(possible_a.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    line_with_band(axes[0, 0], x, possible_a, f"{label_a} V+", "#1f77b4", 0)
    line_with_band(axes[0, 0], x, possible_b, f"{label_b} V+", "#ff7f0e", 1)
    axes[0, 0].set_title("Possible video surprise")
    axes[0, 0].set_ylabel("S_t(V+)")
    axes[0, 0].legend()

    line_with_band(axes[0, 1], x, impossible_a, f"{label_a} V-", "#1f77b4", 2)
    line_with_band(axes[0, 1], x, impossible_b, f"{label_b} V-", "#ff7f0e", 3)
    axes[0, 1].set_title("Impossible video surprise")
    axes[0, 1].set_ylabel("S_t(V-)")
    axes[0, 1].legend()

    line_with_band(axes[1, 0], x, margins_a, f"{label_a} margin", "#1f77b4", 4)
    line_with_band(axes[1, 0], x, margins_b, f"{label_b} margin", "#ff7f0e", 5)
    axes[1, 0].axhline(0, color="0.25", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Surprise margin")
    axes[1, 0].set_xlabel("prediction-window index")
    axes[1, 0].set_ylabel("S_t(V-) - S_t(V+)")
    axes[1, 0].legend()

    margin_shift = margins_b - margins_a
    line_with_band(axes[1, 1], x, margin_shift, f"{label_b} - {label_a}", "#2ca02c", 6)
    axes[1, 1].axhline(0, color="0.25", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Hue-induced margin shift")
    axes[1, 1].set_xlabel("prediction-window index")
    axes[1, 1].set_ylabel("delta margin")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(alpha=0.2, linewidth=0.7)

    fig.suptitle(f"{title_prefix}: clean vs hue surprise curves", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter(
    cond_a: dict[str, object],
    cond_b: dict[str, object],
    label_a: str,
    label_b: str,
    output: Path,
    title_prefix: str,
) -> None:
    rows_a = cond_a["rows"]
    rows_b = cond_b["rows"]
    assert isinstance(rows_a, list)
    assert isinstance(rows_b, list)
    by_pair_b = {row["pair_id"]: row for row in rows_b}
    points = []
    for row_a in rows_a:
        row_b = by_pair_b.get(row_a["pair_id"])
        if row_b is None:
            continue
        clean = float(row_a["average_margin"])
        shifted = float(row_b["average_margin"])
        points.append((clean, shifted, clean > 0, shifted > 0))
    arr = np.asarray([[p[0], p[1]] for p in points], dtype=np.float64)
    colors = []
    for _, _, clean_ok, shifted_ok in points:
        if clean_ok and shifted_ok:
            colors.append("#2ca02c")
        elif clean_ok and not shifted_ok:
            colors.append("#d62728")
        elif not clean_ok and shifted_ok:
            colors.append("#9467bd")
        else:
            colors.append("#7f7f7f")

    lim = float(np.quantile(np.abs(arr), 0.98))
    lim = max(lim, 1e-4) * 1.15
    plt.figure(figsize=(6.7, 6.4))
    plt.scatter(arr[:, 0], arr[:, 1], c=colors, s=42, alpha=0.82, edgecolor="white", linewidth=0.5)
    plt.axhline(0, color="0.25", linestyle="--", linewidth=1)
    plt.axvline(0, color="0.25", linestyle="--", linewidth=1)
    plt.plot([-lim, lim], [-lim, lim], color="0.55", linestyle=":", linewidth=1)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel(f"{label_a} average margin")
    plt.ylabel(f"{label_b} average margin")
    plt.title(f"{title_prefix}: pair margins, {label_a} vs {label_b}")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IntPhys surprise curves for two conditions.")
    parser.add_argument("--input", required=True, help="pair_surprises.csv")
    parser.add_argument("--condition-a", default="clean")
    parser.add_argument("--condition-b", default="hue-mild")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--title-prefix", default="IntPhys")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    path = Path(args.input)
    label_a = args.label_a or args.condition_a
    label_b = args.label_b or args.condition_b
    cond_a = load_condition(path, args.condition_a)
    cond_b = load_condition(path, args.condition_b)

    prefix = Path(args.output_prefix)
    plot_curves(
        cond_a,
        cond_b,
        label_a,
        label_b,
        prefix.with_name(prefix.name + "_curves.png"),
        args.title_prefix,
    )
    plot_scatter(
        cond_a,
        cond_b,
        label_a,
        label_b,
        prefix.with_name(prefix.name + "_pair_margin_scatter.png"),
        args.title_prefix,
    )


if __name__ == "__main__":
    main()
