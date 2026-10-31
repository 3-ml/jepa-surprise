#!/usr/bin/env python3
"""Run the locked paired-statistics portion of the IntPhys 2 endpoint gate.

This script intentionally consumes only the committed pair-level handoff.  It
does not make a temporal-localization claim: that requires the retained
per-window arrays from the R2 archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ENDPOINTS = (
    "hue-mild",
    "hue-strong",
    "brightness-mild",
    "brightness-strong",
    "selective-recolor-mild",
    "selective-recolor-strong",
)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {
        "condition", "pair", "margin", "correct", "clean_correct",
        "condition_name", "difficulty", "camera",
    }
    if not raw_rows or set(raw_rows[0]) < required:
        raise ValueError(f"{path} is missing required columns")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        rows.append(
            {
                **raw,
                "margin": float(raw["margin"]),
                "correct": int(raw["correct"]),
                "clean_correct": int(raw["clean_correct"]),
            }
        )
    return rows


def percentile_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    if not len(values):
        raise ValueError("Cannot bootstrap an empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def mcnemar_exact_pvalue(lost: int, gained: int) -> float:
    """Two-sided exact McNemar p-value for discordant matched pairs."""
    discordant = lost + gained
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(lost, gained) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def sign_flip_pvalue(values: np.ndarray, seed: int, samples: int) -> float:
    """Two-sided paired sign-flip randomization p-value for mean change."""
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + (null >= observed).sum()) / (samples + 1))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (condition, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * pvalue))
        adjusted[condition] = running
    return adjusted


def comparison(clean: dict[str, dict[str, object]], shifted: dict[str, dict[str, object]], seed: int, samples: int) -> dict[str, object]:
    pair_ids = sorted(clean)
    if set(shifted) != set(pair_ids):
        raise ValueError("Condition does not have exactly the clean pair IDs")
    accuracy_change = np.asarray(
        [int(shifted[pair]["correct"]) - int(clean[pair]["correct"]) for pair in pair_ids],
        dtype=np.float64,
    )
    margin_change = np.asarray(
        [float(shifted[pair]["margin"]) - float(clean[pair]["margin"]) for pair in pair_ids],
        dtype=np.float64,
    )
    lost = int(((accuracy_change < 0)).sum())
    gained = int(((accuracy_change > 0)).sum())
    return {
        "n_pairs": len(pair_ids),
        "accuracy_change": float(accuracy_change.mean()),
        "accuracy_change_ci95": percentile_ci(accuracy_change, seed, samples),
        "correct_to_incorrect": lost,
        "incorrect_to_correct": gained,
        "accuracy_mcnemar_pvalue": mcnemar_exact_pvalue(lost, gained),
        "margin_change_mean": float(margin_change.mean()),
        "margin_change_ci95": percentile_ci(margin_change, seed + 1, samples),
        "margin_change_signflip_pvalue": sign_flip_pvalue(margin_change, seed + 2, samples),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def payload_pair_and_label(payload: dict[str, object]) -> tuple[str, str]:
    metadata = payload["metadata"]
    event_type = str(metadata["type"])
    label = "impossible" if "impossible" in event_type.lower() else "possible"
    member = event_type.split("_", 1)[0].strip()
    return f"{metadata['SceneIndex']}/{member}", label


def temporal_diagnostics(
    matrix_root: Path,
    selection_manifest: Path,
    output_dir: Path,
    endpoints: tuple[str, ...],
) -> dict[str, object]:
    """Compare aligned source windows; no event-time labels are assumed available."""
    selected_names = set(json.loads(selection_manifest.read_text(encoding="utf-8"))["selected_names"])
    payloads: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for condition in ("clean", *endpoints):
        videos: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        for path in sorted((matrix_root / condition / "videos").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["name"] not in selected_names:
                continue
            pair, label = payload_pair_and_label(payload)
            if label in videos[pair]:
                raise ValueError(f"Duplicate {label} video in {condition} pair {pair}")
            videos[pair][label] = payload
        payloads[condition] = videos

    clean_pairs = payloads["clean"]
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for condition in endpoints:
        if set(payloads[condition]) != set(clean_pairs):
            raise ValueError(f"{condition} does not have the clean pair set")
        by_index: dict[int, list[tuple[float, float, float, float, float, float]]] = defaultdict(list)
        aligned_pairs = 0
        for pair in sorted(clean_pairs):
            clean_pair = clean_pairs[pair]
            shifted_pair = payloads[condition][pair]
            if set(clean_pair) != {"possible", "impossible"} or set(shifted_pair) != {"possible", "impossible"}:
                raise ValueError(f"Incomplete pair {pair}")
            aligned_pairs += 1
            for label in ("possible", "impossible"):
                clean_windows = clean_pair[label]["windows"]
                shifted_windows = shifted_pair[label]["windows"]
                if len(clean_windows) != len(shifted_windows):
                    raise ValueError(f"Window count mismatch for {condition} {pair} {label}")
                for index, (baseline, transformed) in enumerate(zip(clean_windows, shifted_windows)):
                    if baseline["target_source_frames_0based"] != transformed["target_source_frames_0based"]:
                        raise ValueError(f"Target-frame mismatch for {condition} {pair} {label} window {index}")
            for index, (clean_possible, shifted_possible, clean_impossible, shifted_impossible) in enumerate(
                zip(
                    clean_pair["possible"]["windows"],
                    shifted_pair["possible"]["windows"],
                    clean_pair["impossible"]["windows"],
                    shifted_pair["impossible"]["windows"],
                )
            ):
                possible_shift = float(shifted_possible["scalar_l1"]) - float(clean_possible["scalar_l1"])
                impossible_shift = float(shifted_impossible["scalar_l1"]) - float(clean_impossible["scalar_l1"])
                by_index[index].append(
                    (
                        possible_shift,
                        impossible_shift,
                        impossible_shift - possible_shift,
                        float(clean_possible["scalar_l1"]),
                        float(clean_impossible["scalar_l1"]),
                        float(clean_possible["target_source_frames_0based"][0]),
                    )
                )
        all_possible = []
        all_impossible = []
        all_margin = []
        for index, values in sorted(by_index.items()):
            array = np.asarray(values, dtype=np.float64)
            all_possible.extend(array[:, 0])
            all_impossible.extend(array[:, 1])
            all_margin.extend(array[:, 2])
            rows.append(
                {
                    "condition": condition,
                    "window_index": index,
                    "n_pairs": len(array),
                    "mean_target_start_frame_0based": float(array[:, 5].mean()),
                    "possible_shift_mean": float(array[:, 0].mean()),
                    "impossible_shift_mean": float(array[:, 1].mean()),
                    "margin_shift_mean": float(array[:, 2].mean()),
                    "possible_shift_ci95_low": float(np.quantile(array[:, 0], 0.025)),
                    "possible_shift_ci95_high": float(np.quantile(array[:, 0], 0.975)),
                    "impossible_shift_ci95_low": float(np.quantile(array[:, 1], 0.025)),
                    "impossible_shift_ci95_high": float(np.quantile(array[:, 1], 0.975)),
                    "margin_shift_ci95_low": float(np.quantile(array[:, 2], 0.025)),
                    "margin_shift_ci95_high": float(np.quantile(array[:, 2], 0.975)),
                }
            )
        possible = np.asarray(all_possible)
        impossible = np.asarray(all_impossible)
        margin = np.asarray(all_margin)
        summaries[condition] = {
            "aligned_pairs": aligned_pairs,
            "aligned_pair_windows": len(margin),
            "possible_shift_mean_over_pair_windows": float(possible.mean()),
            "impossible_shift_mean_over_pair_windows": float(impossible.mean()),
            "margin_shift_mean_over_pair_windows": float(margin.mean()),
            "same_direction_fraction_over_pair_windows": float((np.sign(possible) == np.sign(impossible)).mean()),
            "event_time_annotations": "not released for IntPhys2 Main; no event-localization claim is made",
        }
    write_csv(output_dir / "temporal_common_mode_diagnostics.csv", rows)
    return summaries


def run(
    input_path: Path,
    output_dir: Path,
    seed: int,
    bootstrap_samples: int,
    permutation_samples: int,
    matrix_root: Path | None = None,
    selection_manifest: Path | None = None,
) -> dict[str, object]:
    rows = read_rows(input_path)
    by_condition: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        pair = str(row["pair"])
        if pair in by_condition[condition]:
            raise ValueError(f"Duplicate pair {pair} in {condition}")
        by_condition[condition][pair] = row
    if set(by_condition) != {"clean", *ENDPOINTS}:
        raise ValueError(f"Unexpected conditions: {sorted(by_condition)}")

    clean = by_condition["clean"]
    summaries: dict[str, dict[str, object]] = {}
    for offset, condition in enumerate(ENDPOINTS):
        summaries[condition] = comparison(
            clean, by_condition[condition], seed + 10 * offset, bootstrap_samples
        )
        # Use the requested randomization budget only for the final p-value.
        margins = np.asarray(
            [float(by_condition[condition][pair]["margin"]) - float(clean[pair]["margin"]) for pair in sorted(clean)],
            dtype=np.float64,
        )
        summaries[condition]["margin_change_signflip_pvalue"] = sign_flip_pvalue(
            margins, seed + 10 * offset + 2, permutation_samples
        )

    accuracy_adjusted = holm_adjust({key: float(value["accuracy_mcnemar_pvalue"]) for key, value in summaries.items()})
    margin_adjusted = holm_adjust({key: float(value["margin_change_signflip_pvalue"]) for key, value in summaries.items()})
    flat_rows = []
    flip_rows = []
    breakdown_rows = []
    for condition in ENDPOINTS:
        result = summaries[condition]
        result["accuracy_mcnemar_holm_pvalue"] = accuracy_adjusted[condition]
        result["margin_change_signflip_holm_pvalue"] = margin_adjusted[condition]
        flat_rows.append({"condition": condition, **result})
        for pair in sorted(clean):
            baseline = clean[pair]
            shifted = by_condition[condition][pair]
            change = int(shifted["correct"]) - int(baseline["correct"])
            if change:
                flip_rows.append(
                    {
                        "condition": condition,
                        "pair": pair,
                        "flip": "correct_to_incorrect" if change < 0 else "incorrect_to_correct",
                        "clean_margin": baseline["margin"],
                        "condition_margin": shifted["margin"],
                        "margin_change": float(shifted["margin"]) - float(baseline["margin"]),
                        "condition_name": shifted["condition_name"],
                        "difficulty": shifted["difficulty"],
                        "camera": shifted["camera"],
                    }
                )
        for field in ("condition_name", "difficulty", "camera"):
            groups: dict[str, list[str]] = defaultdict(list)
            for pair in sorted(clean):
                groups[str(by_condition[condition][pair][field])].append(pair)
            for value, pairs in sorted(groups.items()):
                deltas = np.asarray(
                    [int(by_condition[condition][pair]["correct"]) - int(clean[pair]["correct"]) for pair in pairs],
                    dtype=np.float64,
                )
                margins = np.asarray(
                    [float(by_condition[condition][pair]["margin"]) - float(clean[pair]["margin"]) for pair in pairs],
                    dtype=np.float64,
                )
                breakdown_rows.append(
                    {
                        "condition": condition,
                        "stratum_field": field,
                        "stratum_value": value,
                        "n_pairs": len(pairs),
                        "accuracy_change": float(deltas.mean()),
                        "margin_change_mean": float(margins.mean()),
                        "correct_to_incorrect": int((deltas < 0).sum()),
                        "incorrect_to_correct": int((deltas > 0).sum()),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_endpoint_statistics.csv", flat_rows)
    write_csv(output_dir / "correctness_flips.csv", flip_rows)
    write_csv(output_dir / "descriptive_stratified_breakdowns.csv", breakdown_rows)
    temporal = {
        "status": "not run",
        "reason": "No --matrix-root supplied; the committed handoff has only pair-level summaries.",
    }
    if matrix_root is not None:
        if selection_manifest is None:
            raise ValueError("--selection-manifest is required with --matrix-root")
        temporal = {
            "status": "complete without event labels",
            "matrix_root": str(matrix_root),
            "selection_manifest": str(selection_manifest),
            "conditions": temporal_diagnostics(matrix_root, selection_manifest, output_dir, ENDPOINTS),
        }
    qualified = [
        condition
        for condition, result in summaries.items()
        if result["accuracy_mcnemar_holm_pvalue"] < 0.05
        or result["margin_change_signflip_holm_pvalue"] < 0.05
    ]
    report = {
        "input": str(input_path),
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "sign_flip_samples": permutation_samples,
        "n_pairs": len(clean),
        "conditions": summaries,
        "temporal_localization": temporal,
        "gate": {
            "status": "continue" if qualified and matrix_root is not None else "stop",
            "qualified_endpoints_after_holm": qualified,
            "reason": (
                "At least one endpoint has a Holm-qualified paired result; assess its temporal evidence before any new run."
                if qualified and matrix_root is not None
                else "No endpoint is Holm-qualified across the six prespecified tests. Do not expand to remaining Main pairs or medium severities."
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--sign-flip-samples", type=int, default=100000)
    parser.add_argument("--matrix-root", type=Path, help="Extracted R2 matrix root containing condition/video JSON files")
    parser.add_argument("--selection-manifest", type=Path, help="Frozen selected video names; required with --matrix-root")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.input,
                args.output_dir,
                args.seed,
                args.bootstrap_samples,
                args.sign_flip_samples,
                args.matrix_root,
                args.selection_manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
