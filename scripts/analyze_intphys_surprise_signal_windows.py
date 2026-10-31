from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_CONDITIONS = [
    "clean",
    "hue-mild",
    "hue-medium",
    "hue-strong",
    "brightness-mild",
    "brightness-medium",
    "brightness-strong",
    "selective_recolor-mild",
    "selective_recolor-medium",
    "selective_recolor-strong",
]

SELECTED_PLOT_CONDITIONS = ["clean", "hue-mild", "brightness-strong"]


def bootstrap_mean(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def condition_order(conditions: set[str]) -> list[str]:
    return [c for c in PREFERRED_CONDITIONS if c in conditions] + sorted(conditions - set(PREFERRED_CONDITIONS))


def parse_input_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected BLOCK=PATH, got {spec}")
        block, path = spec.split("=", 1)
        parsed.append((block, Path(path)))
    return parsed


def longest_true_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def read_rows(input_specs: list[tuple[str, Path]], zero_tol: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block, path in input_specs:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                possible = np.asarray(json.loads(raw["possible_windows"]), dtype=np.float64)
                impossible = np.asarray(json.loads(raw["impossible_windows"]), dtype=np.float64)
                margins = impossible - possible
                denom = (possible + impossible) / 2.0
                relative = np.divide(margins, denom, out=np.full_like(margins, np.nan), where=denom != 0)
                positive = margins > 0
                nonzero = np.abs(margins) > zero_tol
                avg_margin = float(margins.mean())
                rows.append(
                    {
                        "block": block,
                        "condition": raw["condition"],
                        "pair_id": raw["pair_id"],
                        "possible_windows": possible,
                        "impossible_windows": impossible,
                        "window_margins": margins,
                        "window_relative_margins": relative,
                        "average_margin": avg_margin,
                        "average_relative_margin": float(np.nanmean(relative)),
                        "average_correct": int(avg_margin > 0),
                        "positive_window_fraction": float(positive.mean()),
                        "zero_window_fraction": float((~nonzero).mean()),
                        "nonzero_positive_window_fraction": float(positive[nonzero].mean())
                        if nonzero.any()
                        else float("nan"),
                        "majority_positive_windows": int(positive.mean() > 0.5),
                        "any_positive_window": int(positive.any()),
                        "all_positive_windows": int(positive.all()),
                        "longest_positive_run": longest_true_run(positive),
                        "peak_margin": float(margins.max()),
                        "peak_margin_window": int(margins.argmax()),
                        "trough_margin": float(margins.min()),
                        "trough_margin_window": int(margins.argmin()),
                        "peak_relative_margin": float(np.nanmax(relative)),
                        "peak_to_abs_average_margin": float(margins.max() / abs(avg_margin))
                        if abs(avg_margin) > 0
                        else float("nan"),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def serializable_pair_row(row: dict[str, object]) -> dict[str, object]:
    skip = {"possible_windows", "impossible_windows", "window_margins", "window_relative_margins"}
    return {key: value for key, value in row.items() if key not in skip}


def make_window_rows(
    rows: list[dict[str, object]], seed: int, n_boot: int, zero_tol: float
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for block in sorted({str(row["block"]) for row in rows}):
        block_rows = [row for row in rows if row["block"] == block]
        for condition in condition_order({str(row["condition"]) for row in block_rows}):
            selected = [row for row in block_rows if row["condition"] == condition]
            n_windows = min(len(row["window_margins"]) for row in selected)
            for window_idx in range(n_windows):
                possible = np.asarray([row["possible_windows"][window_idx] for row in selected], dtype=np.float64)
                impossible = np.asarray([row["impossible_windows"][window_idx] for row in selected], dtype=np.float64)
                margins = impossible - possible
                denom = (possible + impossible) / 2.0
                relative = np.divide(margins, denom, out=np.full_like(margins, np.nan), where=denom != 0)
                correct = (margins > 0).astype(np.float64)
                nonzero = np.abs(margins) > zero_tol
                acc_low, acc_high = bootstrap_mean(correct, seed + window_idx, n_boot)
                margin_low, margin_high = bootstrap_mean(margins, seed + 1000 + window_idx, n_boot)
                rel_low, rel_high = bootstrap_mean(relative, seed + 2000 + window_idx, n_boot)
                out.append(
                    {
                        "block": block,
                        "condition": condition,
                        "window_index": window_idx,
                        "n_pairs": len(selected),
                        "window_pair_accuracy": float(correct.mean()),
                        "window_pair_accuracy_ci95_low": acc_low,
                        "window_pair_accuracy_ci95_high": acc_high,
                        "mean_possible": float(possible.mean()),
                        "mean_impossible": float(impossible.mean()),
                        "mean_margin": float(margins.mean()),
                        "mean_margin_ci95_low": margin_low,
                        "mean_margin_ci95_high": margin_high,
                        "median_margin": float(np.median(margins)),
                        "mean_relative_margin": float(np.nanmean(relative)),
                        "mean_relative_margin_ci95_low": rel_low,
                        "mean_relative_margin_ci95_high": rel_high,
                        "median_relative_margin": float(np.nanmedian(relative)),
                        "fraction_positive_margin": float((margins > 0).mean()),
                        "fraction_zero_margin": float((~nonzero).mean()),
                        "nonzero_window_pair_accuracy": float((margins[nonzero] > 0).mean())
                        if nonzero.any()
                        else float("nan"),
                        "fraction_abs_relative_margin_lt_0p5pct": float((np.abs(relative) < 0.005).mean()),
                    }
                )
    return out


def make_condition_rows(
    pair_rows: list[dict[str, object]], window_rows: list[dict[str, object]], seed: int, n_boot: int
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    clean_acc: dict[str, float] = {}
    for block in sorted({str(row["block"]) for row in pair_rows}):
        block_pair_rows = [row for row in pair_rows if row["block"] == block]
        for condition in condition_order({str(row["condition"]) for row in block_pair_rows}):
            selected_pairs = [row for row in block_pair_rows if row["condition"] == condition]
            selected_windows = [row for row in window_rows if row["block"] == block and row["condition"] == condition]
            avg_correct = np.asarray([row["average_correct"] for row in selected_pairs], dtype=np.float64)
            positive_fractions = np.asarray(
                [row["positive_window_fraction"] for row in selected_pairs], dtype=np.float64
            )
            zero_fractions = np.asarray([row["zero_window_fraction"] for row in selected_pairs], dtype=np.float64)
            nonzero_positive_fractions = np.asarray(
                [row["nonzero_positive_window_fraction"] for row in selected_pairs], dtype=np.float64
            )
            majority_positive = np.asarray(
                [row["majority_positive_windows"] for row in selected_pairs], dtype=np.float64
            )
            any_positive = np.asarray([row["any_positive_window"] for row in selected_pairs], dtype=np.float64)
            all_positive = np.asarray([row["all_positive_windows"] for row in selected_pairs], dtype=np.float64)
            longest_runs = np.asarray([row["longest_positive_run"] for row in selected_pairs], dtype=np.float64)
            avg_margins = np.asarray([row["average_margin"] for row in selected_pairs], dtype=np.float64)
            avg_rel = np.asarray([row["average_relative_margin"] for row in selected_pairs], dtype=np.float64)
            peak_ratios = np.asarray([row["peak_to_abs_average_margin"] for row in selected_pairs], dtype=np.float64)
            window_acc = np.asarray([row["window_pair_accuracy"] for row in selected_windows], dtype=np.float64)
            window_margins = np.asarray([row["mean_margin"] for row in selected_windows], dtype=np.float64)
            best_window_idx = int(selected_windows[int(window_acc.argmax())]["window_index"])

            pair_acc = float(avg_correct.mean())
            if condition == "clean":
                clean_acc[block] = pair_acc
            out.append(
                {
                    "block": block,
                    "condition": condition,
                    "n_pairs": len(selected_pairs),
                    "average_pair_accuracy": pair_acc,
                    "average_pair_accuracy_ci95_low": bootstrap_mean(avg_correct, seed, n_boot)[0],
                    "average_pair_accuracy_ci95_high": bootstrap_mean(avg_correct, seed, n_boot)[1],
                    "accuracy_drop_from_clean": "",
                    "mean_average_margin": float(avg_margins.mean()),
                    "median_average_margin": float(np.median(avg_margins)),
                    "mean_average_relative_margin": float(np.nanmean(avg_rel)),
                    "median_average_relative_margin": float(np.nanmedian(avg_rel)),
                    "mean_positive_window_fraction": float(positive_fractions.mean()),
                    "median_positive_window_fraction": float(np.median(positive_fractions)),
                    "mean_zero_window_fraction": float(zero_fractions.mean()),
                    "median_zero_window_fraction": float(np.median(zero_fractions)),
                    "mean_nonzero_positive_window_fraction": float(np.nanmean(nonzero_positive_fractions)),
                    "median_nonzero_positive_window_fraction": float(np.nanmedian(nonzero_positive_fractions)),
                    "fraction_pairs_majority_positive_windows": float(majority_positive.mean()),
                    "fraction_pairs_any_positive_window": float(any_positive.mean()),
                    "fraction_pairs_all_positive_windows": float(all_positive.mean()),
                    "median_longest_positive_run_windows": float(np.median(longest_runs)),
                    "median_peak_to_abs_average_margin": float(np.nanmedian(peak_ratios)),
                    "mean_window_pair_accuracy": float(window_acc.mean()),
                    "min_window_pair_accuracy": float(window_acc.min()),
                    "max_window_pair_accuracy": float(window_acc.max()),
                    "best_window_index_by_accuracy": best_window_idx,
                    "n_windows_accuracy_gt_0p5": int((window_acc > 0.5).sum()),
                    "n_windows_accuracy_gt_0p67": int((window_acc > 2 / 3).sum()),
                    "n_windows_accuracy_gt_0p8": int((window_acc > 0.8).sum()),
                    "n_windows_mean_margin_positive": int((window_margins > 0).sum()),
                }
            )

    for row in out:
        clean = clean_acc.get(str(row["block"]))
        if clean is not None:
            row["accuracy_drop_from_clean"] = clean - float(row["average_pair_accuracy"])
    return out


def plot_window_accuracy(window_rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in sorted({str(row["block"]) for row in window_rows}):
        conditions = [
            condition
            for condition in ["clean", "hue-mild", "hue-medium", "hue-strong", "brightness-strong"]
            if any(row["block"] == block and row["condition"] == condition for row in window_rows)
        ]
        if not conditions:
            continue
        plt.figure(figsize=(10, 4.8))
        for condition in conditions:
            selected = [row for row in window_rows if row["block"] == block and row["condition"] == condition]
            x = np.asarray([row["window_index"] for row in selected], dtype=np.int64)
            y = np.asarray([row["window_pair_accuracy"] for row in selected], dtype=np.float64)
            low = np.asarray([row["window_pair_accuracy_ci95_low"] for row in selected], dtype=np.float64)
            high = np.asarray([row["window_pair_accuracy_ci95_high"] for row in selected], dtype=np.float64)
            plt.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=condition)
            if condition in SELECTED_PLOT_CONDITIONS:
                plt.fill_between(x, low, high, alpha=0.12)
        plt.axhline(0.5, color="0.25", linestyle="--", linewidth=1)
        plt.ylim(-0.02, 1.02)
        plt.xlabel("prediction-window index")
        plt.ylabel("pair accuracy at this window")
        plt.title(f"{block}: segment-level surprise ordering")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_window_pair_accuracy.png", dpi=170)
        plt.close()


def plot_margin_heatmaps(pair_rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in sorted({str(row["block"]) for row in pair_rows}):
        for condition in SELECTED_PLOT_CONDITIONS:
            selected = [row for row in pair_rows if row["block"] == block and row["condition"] == condition]
            if not selected:
                continue
            selected = sorted(selected, key=lambda row: float(row["average_margin"]))
            matrix = np.asarray([row["window_margins"] for row in selected], dtype=np.float64)
            vmax = float(np.quantile(np.abs(matrix), 0.98))
            if vmax == 0:
                vmax = 1.0
            plt.figure(figsize=(10, 6.5))
            plt.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
            plt.axvline(matrix.shape[1] / 3 - 0.5, color="0.2", linestyle=":", linewidth=0.8)
            plt.axvline(2 * matrix.shape[1] / 3 - 0.5, color="0.2", linestyle=":", linewidth=0.8)
            plt.colorbar(label="S_t(impossible) - S_t(possible)")
            plt.xlabel("prediction-window index")
            plt.ylabel("pairs sorted by average margin")
            plt.title(f"{block} {condition}: pair-by-window surprise margin")
            plt.tight_layout()
            plt.savefig(out_dir / f"{block.lower()}_{condition}_window_margin_heatmap.png", dpi=170)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IntPhys surprise signal segment by segment.")
    parser.add_argument("--input", action="append", required=True, help="Input as BLOCK=pair_surprises.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zero-tol", type=float, default=1e-12)
    args = parser.parse_args()

    pair_rows = read_rows(parse_input_specs(args.input), args.zero_tol)
    window_rows = make_window_rows(pair_rows, args.seed, args.bootstrap, args.zero_tol)
    condition_rows = make_condition_rows(pair_rows, window_rows, args.seed, args.bootstrap)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "pair_window_signal.csv",
        [serializable_pair_row(row) for row in pair_rows],
        [
            "block",
            "condition",
            "pair_id",
            "average_margin",
            "average_relative_margin",
            "average_correct",
            "positive_window_fraction",
            "zero_window_fraction",
            "nonzero_positive_window_fraction",
            "majority_positive_windows",
            "any_positive_window",
            "all_positive_windows",
            "longest_positive_run",
            "peak_margin",
            "peak_margin_window",
            "trough_margin",
            "trough_margin_window",
            "peak_relative_margin",
            "peak_to_abs_average_margin",
        ],
    )
    write_csv(out_dir / "window_signal_summary.csv", window_rows, list(window_rows[0]))
    write_csv(out_dir / "condition_signal_summary.csv", condition_rows, list(condition_rows[0]))

    plots_dir = out_dir / "plots"
    plot_window_accuracy(window_rows, plots_dir)
    plot_margin_heatmaps(pair_rows, plots_dir)

    summary = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap,
        "zero_tolerance": args.zero_tol,
        "note": "This is a segment/window-level audit of stored V-JEPA prediction-window surprises, not true frame-level loss.",
        "conditions": condition_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
