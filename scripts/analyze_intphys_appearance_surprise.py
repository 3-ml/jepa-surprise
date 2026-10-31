from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("average", "maximum")


def parse_windows(text: str) -> np.ndarray:
    values = json.loads(text)
    return np.asarray(values, dtype=np.float64)


def read_pair_surprises(path: Path, block: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        item: dict[str, object] = {
            "block": block,
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "possible_windows": parse_windows(row["possible_windows"]),
            "impossible_windows": parse_windows(row["impossible_windows"]),
        }
        for metric in METRICS:
            item[f"{metric}_margin"] = float(row[f"{metric}_margin"])
            item[f"{metric}_correct"] = int(row[f"{metric}_correct"])
            item[f"possible_{metric}"] = float(row[f"possible_{metric}"])
            item[f"impossible_{metric}"] = float(row[f"impossible_{metric}"])
        parsed.append(item)
    return parsed


def bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        estimates[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarize_condition(rows: list[dict[str, object]], seed: int, n_boot: int) -> dict[str, object]:
    out: dict[str, object] = {"n_pairs": len(rows)}
    for metric in METRICS:
        correct = np.asarray([row[f"{metric}_correct"] for row in rows], dtype=np.float64)
        margins = np.asarray([row[f"{metric}_margin"] for row in rows], dtype=np.float64)
        out[metric] = {
            "pairwise_accuracy": float(correct.mean()),
            "pairwise_accuracy_ci95": list(bootstrap(correct, seed, n_boot)),
            "mean_margin": float(margins.mean()),
            "mean_margin_ci95": list(bootstrap(margins, seed, n_boot)),
            "median_margin": float(np.median(margins)),
            "fraction_negative_or_zero_margin": float((margins <= 0).mean()),
        }
    return out


def paired_delta(
    clean_rows: list[dict[str, object]],
    shifted_rows: list[dict[str, object]],
    metric: str,
    seed: int,
    n_boot: int,
) -> dict[str, object]:
    clean_by_pair = {row["pair_id"]: row for row in clean_rows}
    shifted_by_pair = {row["pair_id"]: row for row in shifted_rows}
    pair_ids = sorted(set(clean_by_pair) & set(shifted_by_pair))
    if len(pair_ids) != len(clean_rows) or len(pair_ids) != len(shifted_rows):
        raise ValueError("Clean/shifted conditions do not have the same pair IDs.")

    clean_correct = np.asarray([clean_by_pair[p][f"{metric}_correct"] for p in pair_ids], dtype=np.float64)
    shifted_correct = np.asarray([shifted_by_pair[p][f"{metric}_correct"] for p in pair_ids], dtype=np.float64)
    clean_margin = np.asarray([clean_by_pair[p][f"{metric}_margin"] for p in pair_ids], dtype=np.float64)
    shifted_margin = np.asarray([shifted_by_pair[p][f"{metric}_margin"] for p in pair_ids], dtype=np.float64)
    acc_delta = shifted_correct - clean_correct
    margin_delta = shifted_margin - clean_margin

    lost = int(((clean_correct == 1) & (shifted_correct == 0)).sum())
    gained = int(((clean_correct == 0) & (shifted_correct == 1)).sum())
    unchanged = int((clean_correct == shifted_correct).sum())

    return {
        "n_pairs": len(pair_ids),
        "accuracy_change": float(acc_delta.mean()),
        "accuracy_change_ci95": list(bootstrap(acc_delta, seed, n_boot)),
        "margin_delta_mean": float(margin_delta.mean()),
        "margin_delta_mean_ci95": list(bootstrap(margin_delta, seed, n_boot)),
        "margin_delta_median": float(np.median(margin_delta)),
        "fraction_pairs_margin_decreased": float((margin_delta < 0).mean()),
        "lost_correct_pairs": lost,
        "gained_correct_pairs": gained,
        "unchanged_correctness_pairs": unchanged,
        "mean_possible_shift": float(
            np.mean([shifted_by_pair[p][f"possible_{metric}"] - clean_by_pair[p][f"possible_{metric}"] for p in pair_ids])
        ),
        "mean_impossible_shift": float(
            np.mean([shifted_by_pair[p][f"impossible_{metric}"] - clean_by_pair[p][f"impossible_{metric}"] for p in pair_ids])
        ),
    }


def condition_order(conditions: set[str]) -> list[str]:
    preferred = [
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
    return [c for c in preferred if c in conditions] + sorted(conditions - set(preferred))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for block, block_summary in summary["blocks"].items():
        conditions = condition_order(set(block_summary["conditions"]))
        x = np.arange(len(conditions))
        plt.figure(figsize=(11, 4.5))
        for metric, offset in [("average", -0.18), ("maximum", 0.18)]:
            y = [block_summary["conditions"][c][metric]["pairwise_accuracy"] for c in conditions]
            lows = [y_i - block_summary["conditions"][c][metric]["pairwise_accuracy_ci95"][0] for y_i, c in zip(y, conditions)]
            highs = [block_summary["conditions"][c][metric]["pairwise_accuracy_ci95"][1] - y_i for y_i, c in zip(y, conditions)]
            plt.errorbar(x + offset, y, yerr=[lows, highs], fmt="o", capsize=3, label=metric)
        plt.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
        plt.xticks(x, conditions, rotation=35, ha="right")
        plt.ylabel("pairwise accuracy")
        plt.ylim(0.45, 1.02)
        plt.title(f"{block} Appearance Surprise Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_accuracy_ci.png", dpi=160)
        plt.close()


def plot_temporal(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block_condition: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_block_condition.setdefault((str(row["block"]), str(row["condition"])), []).append(row)

    for (block, condition), selected in sorted(by_block_condition.items()):
        possible = [row["possible_windows"] for row in selected]
        impossible = [row["impossible_windows"] for row in selected]
        min_len = min(min(len(x) for x in possible), min(len(x) for x in impossible))
        possible_arr = np.stack([x[:min_len] for x in possible])
        impossible_arr = np.stack([x[:min_len] for x in impossible])
        margin_arr = impossible_arr - possible_arr
        x = np.arange(min_len)

        plt.figure(figsize=(9, 4.8))
        for label, arr in [("possible", possible_arr), ("impossible", impossible_arr)]:
            mean = arr.mean(axis=0)
            low = np.quantile(arr, 0.025, axis=0)
            high = np.quantile(arr, 0.975, axis=0)
            plt.plot(x, mean, label=label)
            plt.fill_between(x, low, high, alpha=0.2)
        plt.ylabel("window surprise")
        plt.xlabel("window index")
        plt.title(f"{block} {condition}: temporal surprise")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_{condition}_temporal_surprise.png", dpi=160)
        plt.close()

        plt.figure(figsize=(9, 4.0))
        mean = margin_arr.mean(axis=0)
        low = np.quantile(margin_arr, 0.025, axis=0)
        high = np.quantile(margin_arr, 0.975, axis=0)
        plt.plot(x, mean, label="impossible - possible")
        plt.fill_between(x, low, high, alpha=0.2)
        plt.axhline(0, color="0.5", linestyle="--", linewidth=1)
        plt.ylabel("window margin")
        plt.xlabel("window index")
        plt.title(f"{block} {condition}: temporal margin")
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_{condition}_temporal_margin.png", dpi=160)
        plt.close()


def analyze(inputs: list[tuple[str, Path]], out_dir: Path, seed: int, n_boot: int, make_plots: bool) -> dict[str, object]:
    all_rows = []
    for block, path in inputs:
        all_rows.extend(read_pair_surprises(path, block))

    summary: dict[str, object] = {"seed": seed, "bootstrap_samples": n_boot, "blocks": {}}
    condition_rows = []
    comparison_rows = []
    for block in sorted({str(row["block"]) for row in all_rows}):
        block_rows = [row for row in all_rows if row["block"] == block]
        by_condition = {
            condition: [row for row in block_rows if row["condition"] == condition]
            for condition in condition_order({str(row["condition"]) for row in block_rows})
        }
        block_summary = {"conditions": {}, "comparisons_to_clean": {}}
        for condition, rows in by_condition.items():
            condition_summary = summarize_condition(rows, seed, n_boot)
            block_summary["conditions"][condition] = condition_summary
            for metric in METRICS:
                condition_rows.append(
                    {
                        "block": block,
                        "condition": condition,
                        "metric": metric,
                        **condition_summary[metric],
                    }
                )
        clean_rows = by_condition["clean"]
        for condition, rows in by_condition.items():
            if condition == "clean":
                continue
            block_summary["comparisons_to_clean"][condition] = {}
            for metric in METRICS:
                delta = paired_delta(clean_rows, rows, metric, seed, n_boot)
                block_summary["comparisons_to_clean"][condition][metric] = delta
                comparison_rows.append({"block": block, "condition": condition, "metric": metric, **delta})
        summary["blocks"][block] = block_summary

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "condition_metrics.csv", condition_rows)
    write_csv(out_dir / "paired_condition_deltas.csv", comparison_rows)
    if make_plots:
        plot_accuracy(summary, out_dir / "plots")
        plot_temporal(all_rows, out_dir / "plots")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_input_specs(specs: list[str]) -> list[tuple[str, Path]]:
    inputs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Input must be BLOCK=PATH, got: {spec}")
        block, path = spec.split("=", 1)
        inputs.append((block, Path(path)))
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze IntPhys appearance pair-surprise CSVs.")
    parser.add_argument("--input", action="append", required=True, help="Input as BLOCK=path/to/pair_surprises.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    summary = analyze(parse_input_specs(args.input), Path(args.output_dir), args.seed, args.bootstrap, args.plots)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
