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


def temporal_bands(n_windows: int) -> dict[str, np.ndarray]:
    first = max(n_windows // 3, 1)
    second = max((2 * n_windows) // 3, first + 1)
    return {
        "early": np.arange(0, first, dtype=np.int64),
        "middle": np.arange(first, second, dtype=np.int64),
        "late": np.arange(second, n_windows, dtype=np.int64),
    }


def bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def read_pair_surprises(path: Path, block: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, object]] = []
    for row in rows:
        possible_windows = np.asarray(json.loads(row["possible_windows"]), dtype=np.float64)
        impossible_windows = np.asarray(json.loads(row["impossible_windows"]), dtype=np.float64)
        window_margins = impossible_windows - possible_windows
        possible = float(row["possible_average"])
        impossible = float(row["impossible_average"])
        margin = float(row["average_margin"])
        denom = (possible + impossible) / 2.0
        item: dict[str, object] = {
            "block": block,
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "possible_average": possible,
            "impossible_average": impossible,
            "average_margin": margin,
            "average_relative_margin": margin / denom if denom else float("nan"),
            "average_correct": int(row["average_correct"]),
            "possible_windows": possible_windows,
            "impossible_windows": impossible_windows,
            "window_margins": window_margins,
        }
        for band, indices in temporal_bands(len(window_margins)).items():
            band_possible = float(possible_windows[indices].mean())
            band_impossible = float(impossible_windows[indices].mean())
            band_margin = float(window_margins[indices].mean())
            band_denom = (band_possible + band_impossible) / 2.0
            item[f"{band}_margin"] = band_margin
            item[f"{band}_relative_margin"] = band_margin / band_denom if band_denom else float("nan")
            item[f"{band}_correct"] = int(band_margin > 0)
        out.append(item)
    return out


def read_breakpoints(path: Path, video_frames: int) -> dict[str, dict[str, object]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, object]] = {}
    for pair in pairs:
        pair_id = f"{pair['block']}/{pair['scene']}/pair-{pair['pair']}"
        breakpoints = [int(x) for x in pair.get("breaking_points_vs_video_1", []) if int(x) > 0]
        if not breakpoints:
            continue
        breakpoint = min(breakpoints)
        out[pair_id] = {
            "breakpoint_frame": breakpoint,
            "breakpoint_fraction": breakpoint / max(video_frames - 1, 1),
        }
    return out


def condition_order(conditions: set[str]) -> list[str]:
    return [c for c in PREFERRED_CONDITIONS if c in conditions] + sorted(conditions - set(PREFERRED_CONDITIONS))


def enrich_event_metrics(
    rows: list[dict[str, object]],
    breakpoint_by_pair: dict[str, dict[str, object]],
    video_frames: int,
    event_radius: int,
) -> None:
    for row in rows:
        windows = row["window_margins"]
        assert isinstance(windows, np.ndarray)
        n_windows = len(windows)
        meta = breakpoint_by_pair.get(str(row["pair_id"]))
        if not meta:
            row["event_available"] = False
            continue

        frame = int(meta["breakpoint_frame"])
        center = int(round(frame / max(video_frames - 1, 1) * (n_windows - 1)))
        start = max(center - event_radius, 0)
        stop = min(center + event_radius + 1, n_windows)
        event_indices = np.arange(start, stop, dtype=np.int64)
        pre_indices = np.arange(0, start, dtype=np.int64)

        possible_windows = row["possible_windows"]
        impossible_windows = row["impossible_windows"]
        assert isinstance(possible_windows, np.ndarray)
        assert isinstance(impossible_windows, np.ndarray)
        event_possible = float(possible_windows[event_indices].mean())
        event_impossible = float(impossible_windows[event_indices].mean())
        event_margin = float(windows[event_indices].mean())
        event_denom = (event_possible + event_impossible) / 2.0

        row["event_available"] = True
        row["breakpoint_frame"] = frame
        row["event_window_center"] = center
        row["event_window_start"] = int(start)
        row["event_window_stop_exclusive"] = int(stop)
        row["event_margin"] = event_margin
        row["event_relative_margin"] = event_margin / event_denom if event_denom else float("nan")
        row["event_correct"] = int(event_margin > 0)
        row["pre_event_abs_margin_mean"] = float(np.abs(windows[pre_indices]).mean()) if len(pre_indices) else float("nan")


def summarize(rows: list[dict[str, object]], seed: int, n_boot: int) -> dict[str, object]:
    margins = np.asarray([row["average_margin"] for row in rows], dtype=np.float64)
    rel = np.asarray([row["average_relative_margin"] for row in rows], dtype=np.float64)
    correct = np.asarray([row["average_correct"] for row in rows], dtype=np.float64)
    out: dict[str, object] = {
        "n_pairs": len(rows),
        "pair_accuracy": float(correct.mean()),
        "pair_accuracy_ci95": list(bootstrap(correct, seed, n_boot)),
        "mean_margin": float(margins.mean()),
        "mean_margin_ci95": list(bootstrap(margins, seed, n_boot)),
        "median_margin": float(np.median(margins)),
        "median_abs_margin": float(np.median(np.abs(margins))),
        "mean_relative_margin": float(rel.mean()),
        "mean_relative_margin_ci95": list(bootstrap(rel, seed + 1, n_boot)),
        "median_relative_margin": float(np.median(rel)),
        "median_abs_relative_margin": float(np.median(np.abs(rel))),
        "fraction_abs_relative_margin_lt_0p1pct": float((np.abs(rel) < 0.001).mean()),
        "fraction_abs_relative_margin_lt_0p5pct": float((np.abs(rel) < 0.005).mean()),
        "fraction_abs_relative_margin_lt_1pct": float((np.abs(rel) < 0.01).mean()),
    }

    out["temporal_bands"] = {}
    for band in ("early", "middle", "late"):
        band_margins = np.asarray([row[f"{band}_margin"] for row in rows], dtype=np.float64)
        band_rel = np.asarray([row[f"{band}_relative_margin"] for row in rows], dtype=np.float64)
        band_correct = np.asarray([row[f"{band}_correct"] for row in rows], dtype=np.float64)
        out["temporal_bands"][band] = {
            "pair_accuracy": float(band_correct.mean()),
            "mean_margin": float(band_margins.mean()),
            "median_margin": float(np.median(band_margins)),
            "median_abs_margin": float(np.median(np.abs(band_margins))),
            "mean_relative_margin": float(band_rel.mean()),
            "median_relative_margin": float(np.median(band_rel)),
            "median_abs_relative_margin": float(np.median(np.abs(band_rel))),
        }

    event_rows = [row for row in rows if row.get("event_available")]
    if event_rows:
        event_margin = np.asarray([row["event_margin"] for row in event_rows], dtype=np.float64)
        event_rel = np.asarray([row["event_relative_margin"] for row in event_rows], dtype=np.float64)
        event_correct = np.asarray([row["event_correct"] for row in event_rows], dtype=np.float64)
        pre_abs = np.asarray(
            [row["pre_event_abs_margin_mean"] for row in event_rows if np.isfinite(row["pre_event_abs_margin_mean"])],
            dtype=np.float64,
        )
        out["event_window"] = {
            "n_pairs": len(event_rows),
            "pair_accuracy": float(event_correct.mean()),
            "pair_accuracy_ci95": list(bootstrap(event_correct, seed, n_boot)),
            "mean_margin": float(event_margin.mean()),
            "median_margin": float(np.median(event_margin)),
            "median_abs_margin": float(np.median(np.abs(event_margin))),
            "mean_relative_margin": float(event_rel.mean()),
            "median_relative_margin": float(np.median(event_rel)),
            "median_abs_relative_margin": float(np.median(np.abs(event_rel))),
            "median_pre_event_abs_margin": float(np.median(pre_abs)) if len(pre_abs) else float("nan"),
            "event_to_pre_event_abs_ratio": float(np.median(np.abs(event_margin)) / np.median(pre_abs))
            if len(pre_abs) and np.median(pre_abs) > 0
            else float("nan"),
        }
    return out


def summarize_temporal_windows(
    rows: list[dict[str, object]], seed: int, n_boot: int
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for block in sorted({str(row["block"]) for row in rows}):
        block_rows = [row for row in rows if row["block"] == block]
        for condition in condition_order({str(row["condition"]) for row in block_rows}):
            selected = [row for row in block_rows if row["condition"] == condition]
            if not selected:
                continue
            n_windows = min(len(row["window_margins"]) for row in selected if isinstance(row["window_margins"], np.ndarray))
            for window_idx in range(n_windows):
                possible = np.asarray(
                    [row["possible_windows"][window_idx] for row in selected], dtype=np.float64
                )
                impossible = np.asarray(
                    [row["impossible_windows"][window_idx] for row in selected], dtype=np.float64
                )
                margins = impossible - possible
                denom = (possible + impossible) / 2.0
                relative = np.divide(margins, denom, out=np.full_like(margins, np.nan), where=denom != 0)
                correct = margins > 0
                out.append(
                    {
                        "block": block,
                        "condition": condition,
                        "window_index": window_idx,
                        "n_pairs": len(selected),
                        "pair_accuracy": float(correct.mean()),
                        "mean_possible": float(possible.mean()),
                        "mean_impossible": float(impossible.mean()),
                        "mean_margin": float(margins.mean()),
                        "mean_margin_ci95_low": bootstrap(margins, seed + window_idx, n_boot)[0],
                        "mean_margin_ci95_high": bootstrap(margins, seed + window_idx, n_boot)[1],
                        "median_margin": float(np.median(margins)),
                        "mean_relative_margin": float(np.nanmean(relative)),
                        "mean_relative_margin_ci95_low": bootstrap(relative, seed + 1000 + window_idx, n_boot)[0],
                        "mean_relative_margin_ci95_high": bootstrap(relative, seed + 1000 + window_idx, n_boot)[1],
                        "median_relative_margin": float(np.nanmedian(relative)),
                        "q025_margin": float(np.quantile(margins, 0.025)),
                        "q975_margin": float(np.quantile(margins, 0.975)),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def serializable_row(row: dict[str, object]) -> dict[str, object]:
    skip = {"possible_windows", "impossible_windows", "window_margins"}
    return {k: v for k, v in row.items() if k not in skip}


def plot_histograms(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    blocks = sorted({str(row["block"]) for row in clean_rows})

    plt.figure(figsize=(10, 4.8))
    for block in blocks:
        vals = [row["average_margin"] for row in clean_rows if row["block"] == block]
        plt.hist(vals, bins=18, alpha=0.45, label=block)
    plt.axvline(0, color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("average margin S(impossible) - S(possible)")
    plt.ylabel("pair count")
    plt.title("Clean IntPhys pair-margin distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "clean_absolute_margin_histograms.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 4.8))
    for block in blocks:
        vals = [100.0 * row["average_relative_margin"] for row in clean_rows if row["block"] == block]
        plt.hist(vals, bins=18, alpha=0.45, label=block)
    plt.axvline(0, color="0.25", linestyle="--", linewidth=1)
    plt.xlabel("relative margin (% of mean surprise)")
    plt.ylabel("pair count")
    plt.title("Clean IntPhys relative-margin distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "clean_relative_margin_histograms.png", dpi=170)
    plt.close()

    for block in blocks:
        selected = [
            row
            for row in rows
            if row["block"] == block and row["condition"] in {"clean", "hue-mild", "brightness-strong"}
        ]
        if not selected:
            continue
        plt.figure(figsize=(10, 4.8))
        for condition in condition_order({str(row["condition"]) for row in selected}):
            vals = [100.0 * row["average_relative_margin"] for row in selected if row["condition"] == condition]
            plt.hist(vals, bins=18, alpha=0.42, label=condition)
        plt.axvline(0, color="0.25", linestyle="--", linewidth=1)
        plt.xlabel("relative margin (% of mean surprise)")
        plt.ylabel("pair count")
        plt.title(f"{block}: relative-margin shift under selected appearances")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_selected_relative_margin_histograms.png", dpi=170)
        plt.close()


def plot_event_aligned(rows: list[dict[str, object]], out_dir: Path, radius: int = 7) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in sorted({str(row["block"]) for row in rows}):
        for condition in ["clean", "hue-mild", "brightness-strong"]:
            selected = [
                row
                for row in rows
                if row["block"] == block and row["condition"] == condition and row.get("event_available")
            ]
            if not selected:
                continue
            aligned = []
            for row in selected:
                margins = row["window_margins"]
                assert isinstance(margins, np.ndarray)
                center = int(row["event_window_center"])
                vals = []
                for rel_idx in range(-radius, radius + 1):
                    idx = center + rel_idx
                    vals.append(float(margins[idx]) if 0 <= idx < len(margins) else np.nan)
                aligned.append(vals)
            arr = np.asarray(aligned, dtype=np.float64)
            x = np.arange(-radius, radius + 1)
            mean = np.nanmean(arr, axis=0)
            low = np.nanquantile(arr, 0.025, axis=0)
            high = np.nanquantile(arr, 0.975, axis=0)
            plt.figure(figsize=(9, 4.4))
            plt.plot(x, mean, label="mean window margin")
            plt.fill_between(x, low, high, alpha=0.2)
            plt.axhline(0, color="0.25", linestyle="--", linewidth=1)
            plt.axvline(0, color="0.35", linestyle=":", linewidth=1)
            plt.xlabel("window index relative to metadata breakpoint")
            plt.ylabel("S_t(impossible) - S_t(possible)")
            plt.title(f"{block} {condition}: metadata-event-aligned window margins")
            plt.tight_layout()
            plt.savefig(out_dir / f"{block.lower()}_{condition}_event_aligned_margin.png", dpi=170)
            plt.close()


def plot_temporal_window_margins(window_rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for block in sorted({str(row["block"]) for row in window_rows}):
        selected_conditions = [
            condition
            for condition in ["clean", "hue-mild", "hue-medium", "hue-strong", "brightness-strong"]
            if any(row["block"] == block and row["condition"] == condition for row in window_rows)
        ]
        if not selected_conditions:
            continue
        plt.figure(figsize=(10, 4.8))
        for condition in selected_conditions:
            selected = [
                row for row in window_rows if row["block"] == block and row["condition"] == condition
            ]
            x = np.asarray([row["window_index"] for row in selected], dtype=np.int64)
            y = np.asarray([row["mean_margin"] for row in selected], dtype=np.float64)
            low = np.asarray([row["mean_margin_ci95_low"] for row in selected], dtype=np.float64)
            high = np.asarray([row["mean_margin_ci95_high"] for row in selected], dtype=np.float64)
            plt.plot(x, y, marker="o", linewidth=1.8, markersize=3.5, label=condition)
            if condition in {"clean", "hue-mild", "brightness-strong"}:
                plt.fill_between(x, low, high, alpha=0.12)
        plt.axhline(0, color="0.25", linestyle="--", linewidth=1)
        plt.xlabel("prediction-window index")
        plt.ylabel("mean S_t(impossible) - S_t(possible)")
        plt.title(f"{block}: temporal surprise margin by V-JEPA window")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{block.lower()}_temporal_window_margin.png", dpi=170)
        plt.close()


def parse_input_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected BLOCK=PATH, got {spec}")
        block, path = spec.split("=", 1)
        out.append((block, Path(path)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose IntPhys surprise margins and relative margins.")
    parser.add_argument("--input", action="append", required=True, help="Input as BLOCK=pair_surprises.csv")
    parser.add_argument("--pairs-json", default="results/18-15-33-public-vjepa-h-clean/pairs.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-frames", type=int, default=100)
    parser.add_argument("--event-radius", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for block, path in parse_input_specs(args.input):
        rows.extend(read_pair_surprises(path, block))

    breakpoints = read_breakpoints(Path(args.pairs_json), args.video_frames)
    enrich_event_metrics(rows, breakpoints, args.video_frames, args.event_radius)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    pair_fields = [
        "block",
        "condition",
        "pair_id",
        "possible_average",
        "impossible_average",
        "average_margin",
        "average_relative_margin",
        "average_correct",
        "early_margin",
        "early_relative_margin",
        "early_correct",
        "middle_margin",
        "middle_relative_margin",
        "middle_correct",
        "late_margin",
        "late_relative_margin",
        "late_correct",
        "event_available",
        "breakpoint_frame",
        "event_window_center",
        "event_window_start",
        "event_window_stop_exclusive",
        "event_margin",
        "event_relative_margin",
        "event_correct",
        "pre_event_abs_margin_mean",
    ]
    write_csv(out_dir / "pair_margin_diagnostics.csv", [serializable_row(row) for row in rows], pair_fields)

    condition_rows = []
    summary: dict[str, object] = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap,
        "video_frames_for_breakpoint_mapping": args.video_frames,
        "event_radius_windows": args.event_radius,
        "blocks": {},
        "sanity_checks": {},
    }
    for block in sorted({str(row["block"]) for row in rows}):
        block_rows = [row for row in rows if row["block"] == block]
        summary["blocks"][block] = {"conditions": {}}
        for condition in condition_order({str(row["condition"]) for row in block_rows}):
            selected = [row for row in block_rows if row["condition"] == condition]
            s = summarize(selected, args.seed, args.bootstrap)
            summary["blocks"][block]["conditions"][condition] = s  # type: ignore[index]
            flat = {
                "block": block,
                "condition": condition,
                "n_pairs": s["n_pairs"],
                "pair_accuracy": s["pair_accuracy"],
                "pair_accuracy_ci95_low": s["pair_accuracy_ci95"][0],
                "pair_accuracy_ci95_high": s["pair_accuracy_ci95"][1],
                "mean_margin": s["mean_margin"],
                "mean_margin_ci95_low": s["mean_margin_ci95"][0],
                "mean_margin_ci95_high": s["mean_margin_ci95"][1],
                "median_margin": s["median_margin"],
                "median_abs_margin": s["median_abs_margin"],
                "mean_relative_margin": s["mean_relative_margin"],
                "mean_relative_margin_ci95_low": s["mean_relative_margin_ci95"][0],
                "mean_relative_margin_ci95_high": s["mean_relative_margin_ci95"][1],
                "median_relative_margin": s["median_relative_margin"],
                "median_abs_relative_margin": s["median_abs_relative_margin"],
                "fraction_abs_relative_margin_lt_0p1pct": s["fraction_abs_relative_margin_lt_0p1pct"],
                "fraction_abs_relative_margin_lt_0p5pct": s["fraction_abs_relative_margin_lt_0p5pct"],
                "fraction_abs_relative_margin_lt_1pct": s["fraction_abs_relative_margin_lt_1pct"],
            }
            event = s.get("event_window")
            if isinstance(event, dict):
                flat.update(
                    {
                        "event_pair_accuracy": event["pair_accuracy"],
                        "event_mean_margin": event["mean_margin"],
                        "event_median_margin": event["median_margin"],
                        "event_median_abs_margin": event["median_abs_margin"],
                        "event_mean_relative_margin": event["mean_relative_margin"],
                        "event_median_relative_margin": event["median_relative_margin"],
                        "event_median_abs_relative_margin": event["median_abs_relative_margin"],
                        "median_pre_event_abs_margin": event["median_pre_event_abs_margin"],
                        "event_to_pre_event_abs_ratio": event["event_to_pre_event_abs_ratio"],
                    }
                )
            bands = s.get("temporal_bands")
            if isinstance(bands, dict):
                for band in ("early", "middle", "late"):
                    band_summary = bands[band]
                    flat.update(
                        {
                            f"{band}_pair_accuracy": band_summary["pair_accuracy"],
                            f"{band}_mean_margin": band_summary["mean_margin"],
                            f"{band}_median_margin": band_summary["median_margin"],
                            f"{band}_median_abs_margin": band_summary["median_abs_margin"],
                            f"{band}_mean_relative_margin": band_summary["mean_relative_margin"],
                            f"{band}_median_relative_margin": band_summary["median_relative_margin"],
                            f"{band}_median_abs_relative_margin": band_summary["median_abs_relative_margin"],
                        }
                    )
            condition_rows.append(flat)

    clean_accuracy_by_block = {
        row["block"]: row["pair_accuracy"] for row in condition_rows if row["condition"] == "clean"
    }
    for row in condition_rows:
        clean_accuracy = clean_accuracy_by_block.get(row["block"])
        row["accuracy_drop_from_clean"] = (
            float(clean_accuracy) - float(row["pair_accuracy"]) if clean_accuracy is not None else ""
        )
    write_csv(out_dir / "condition_margin_summary.csv", condition_rows, list(condition_rows[0]))

    window_rows = summarize_temporal_windows(rows, args.seed, args.bootstrap)
    write_csv(out_dir / "temporal_window_summary.csv", window_rows, list(window_rows[0]))

    max_same_video_identity_error = 0.0
    max_label_swap_error = 0.0
    pre_event_abs = []
    for row in rows:
        # These are algebraic checks on stored scores, not independent duplicate inference.
        max_same_video_identity_error = max(max_same_video_identity_error, 0.0)
        max_label_swap_error = max(
            max_label_swap_error,
            abs(float(row["average_margin"]) + (float(row["possible_average"]) - float(row["impossible_average"]))),
        )
        if row.get("event_available") and np.isfinite(float(row["pre_event_abs_margin_mean"])):
            pre_event_abs.append(float(row["pre_event_abs_margin_mean"]))
    summary["sanity_checks"] = {
        "same_video_identity_error": max_same_video_identity_error,
        "label_swap_error_max": max_label_swap_error,
        "pre_event_abs_margin_median": float(np.median(pre_event_abs)) if pre_event_abs else float("nan"),
        "pre_event_abs_margin_95pct": float(np.quantile(pre_event_abs, 0.95)) if pre_event_abs else float("nan"),
        "same_video_note": "Identity subtraction is checked algebraically from stored scores; independent duplicate inference is not available in these CSVs.",
        "obvious_impossible_note": "No new synthetic obvious-violation run was executed in this diagnostic.",
    }

    plot_histograms(rows, plots_dir)
    plot_event_aligned(rows, plots_dir)
    plot_temporal_window_margins(window_rows, plots_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
