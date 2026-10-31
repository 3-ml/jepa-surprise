#!/usr/bin/env python3
"""Summarize a selected IntPhys 2 appearance subset against its clean control."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def label_and_pair(payload: dict) -> tuple[str, str]:
    metadata = payload["metadata"]
    value = str(metadata["type"])
    lowered = value.lower()
    label = "impossible" if "impossible" in lowered else "possible"
    member = value.split("_", 1)[0].strip()
    return label, f"{metadata['SceneIndex']}/{member}"


def metrics(rows: list[dict]) -> dict:
    margins = np.asarray([row["margin"] for row in rows])
    return {
        "pairs": len(rows),
        "correct": int((margins > 0).sum()),
        "accuracy": float((margins > 0).mean()),
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
        "mean_possible_surprise": float(np.mean([row["possible"] for row in rows])),
        "mean_impossible_surprise": float(np.mean([row["impossible"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", action="append", required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_manifest.read_text())
    selected_names = set(selection["selected_names"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    observed_contexts: set[int] = set()
    for condition in args.condition:
        for name in sorted(selected_names):
            path = args.matrix_root / condition / "videos" / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            if payload.get("appearance_condition") != condition:
                raise ValueError(
                    f"Condition mismatch in {path}: {payload.get('appearance_condition')}"
                )
            label, pair = label_and_pair(payload)
            observed_contexts.update(int(row["context_length"]) for row in payload["windows"])
            windows = np.asarray([float(row["scalar_l1"]) for row in payload["windows"]])
            if not np.isfinite(windows).all():
                raise ValueError(f"Non-finite values in {path}")
            values[condition][pair][label] = {
                "mean": float(windows.mean()),
                "maximum": float(windows.max()),
                "feasible": bool(payload.get("appearance_feasible", True)),
                "metadata": payload["metadata"],
            }

    condition_rows: dict[str, list[dict]] = {}
    for condition, pairs in values.items():
        rows = []
        for pair, labels in sorted(pairs.items()):
            if set(labels) != {"possible", "impossible"}:
                raise ValueError(f"Incomplete {condition} pair {pair}: {sorted(labels)}")
            possible = labels["possible"]
            impossible = labels["impossible"]
            rows.append(
                {
                    "pair": pair,
                    "possible": possible["mean"],
                    "impossible": impossible["mean"],
                    "margin": impossible["mean"] - possible["mean"],
                    "maximum_margin": impossible["maximum"] - possible["maximum"],
                    "feasible": possible["feasible"] and impossible["feasible"],
                    "metadata": possible["metadata"],
                }
            )
        condition_rows[condition] = rows

    if "clean" not in condition_rows:
        raise ValueError("clean must be included")
    if len(observed_contexts) != 1:
        raise ValueError(f"Expected one frozen context, found {sorted(observed_contexts)}")
    context = next(iter(observed_contexts))
    clean_by_pair = {row["pair"]: row for row in condition_rows["clean"]}
    summary = {
        "model": "released general V-JEPA 2-G",
        "dataset": "IntPhys2 Main",
        "context": context,
        "selection": {
            key: selection[key]
            for key in ("available_pairs", "selected_pairs", "selected_videos", "fraction", "seed", "stratum_fields")
        },
        "conditions": {},
    }
    for condition, rows in condition_rows.items():
        report = metrics(rows)
        if condition != "clean":
            deltas = np.asarray(
                [row["margin"] - clean_by_pair[row["pair"]]["margin"] for row in rows]
            )
            clean_correct = np.asarray(
                [clean_by_pair[row["pair"]]["margin"] > 0 for row in rows]
            )
            transformed_correct = np.asarray([row["margin"] > 0 for row in rows])
            report.update(
                {
                    "mean_margin_change_from_clean": float(deltas.mean()),
                    "median_margin_change_from_clean": float(np.median(deltas)),
                    "correct_to_incorrect_flips": int((clean_correct & ~transformed_correct).sum()),
                    "incorrect_to_correct_flips": int((~clean_correct & transformed_correct).sum()),
                }
            )
        feasible = [row for row in rows if row["feasible"]]
        report["feasible_pairs"] = len(feasible)
        if len(feasible) != len(rows):
            report["feasible_only"] = metrics(feasible)
        summary["conditions"][condition] = report

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "condition", "pair", "possible_surprise", "impossible_surprise",
        "margin", "clean_margin", "margin_change_from_clean", "correct",
        "clean_correct", "appearance_feasible", "condition_name", "difficulty", "camera",
    ]
    with (args.output_dir / "pair_surprises.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, rows in condition_rows.items():
            for row in rows:
                clean = clean_by_pair[row["pair"]]
                metadata = row["metadata"]
                writer.writerow(
                    {
                        "condition": condition,
                        "pair": row["pair"],
                        "possible_surprise": row["possible"],
                        "impossible_surprise": row["impossible"],
                        "margin": row["margin"],
                        "clean_margin": clean["margin"],
                        "margin_change_from_clean": row["margin"] - clean["margin"],
                        "correct": int(row["margin"] > 0),
                        "clean_correct": int(clean["margin"] > 0),
                        "appearance_feasible": int(row["feasible"]),
                        "condition_name": metadata.get("condition"),
                        "difficulty": metadata.get("Difficulty"),
                        "camera": metadata.get("Camera"),
                    }
                )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
