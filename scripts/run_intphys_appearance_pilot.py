#!/usr/bin/env python3
"""Run and minimally summarize the frozen-context IntPhys appearance pilot."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

TRANSFORMS = ("hue", "brightness", "selective_recolor")
SEVERITIES = ("mild", "medium", "strong")
CONDITIONS = ("clean",) + tuple(
    f"{transform}-{severity}"
    for transform in TRANSFORMS
    for severity in SEVERITIES
)


def loss_file(block: str, context: int) -> str:
    return f"intuitive_physics/intphys-default/losses_{block}_2fs_{context}ctxt.pth"


def condition_roots(clean_root: Path, transformed_root: Path) -> dict[str, Path]:
    roots = {"clean": clean_root}
    for transform in TRANSFORMS:
        for severity in SEVERITIES:
            roots[f"{transform}-{severity}"] = transformed_root / transform / severity
    return roots


def require_condition_scenes(
    roots: dict[str, Path], *, scene_count: int, block: str = "O1"
) -> list[str]:
    clean_scenes = sorted(path.name for path in (roots["clean"] / block).iterdir() if path.is_dir())
    selected = clean_scenes[:scene_count]
    if len(selected) != scene_count:
        raise ValueError(
            f"Clean {block} has {len(selected)} scenes, expected at least {scene_count}"
        )
    for condition, root in roots.items():
        available = sorted(path.name for path in (root / block).iterdir() if path.is_dir())
        if available[:scene_count] != selected:
            raise ValueError(
                f"{condition} first {scene_count} {block} scenes differ from clean: "
                f"{available[:scene_count]}"
            )
    return selected


def load_condition_losses(path: Path, pair_ids: list[str]) -> list[dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    losses = data["losses"]
    labels = data["labels"]
    if list(losses.shape) != [len(pair_ids) * 2, 1, 17]:
        raise ValueError(f"Unexpected loss shape in {path}: {list(losses.shape)}")
    if not bool(torch.isfinite(losses).all()):
        raise ValueError(f"Non-finite losses in {path}")

    rows = []
    for index, pair_id in enumerate(pair_ids):
        pair_losses = losses[index * 2 : index * 2 + 2, 0]
        pair_labels = labels[index * 2 : index * 2 + 2]
        if sorted(pair_labels.tolist()) != [0.0, 1.0]:
            raise ValueError(f"{pair_id} does not contain one possible and one impossible video")
        members = {}
        for loss, label in zip(pair_losses, pair_labels):
            members["possible" if label.item() == 1.0 else "impossible"] = loss
        rows.append(
            {
                "pair_id": pair_id,
                "possible": members["possible"].numpy(),
                "impossible": members["impossible"].numpy(),
            }
        )
    return rows


def aggregate_condition(rows: list[dict]) -> dict:
    average_margins = np.array(
        [row["impossible"].mean() - row["possible"].mean() for row in rows]
    )
    maximum_margins = np.array(
        [row["impossible"].max() - row["possible"].max() for row in rows]
    )
    return {
        "n_pairs": len(rows),
        "average": {
            "pairwise_accuracy": float((average_margins > 0).mean()),
            "mean_margin": float(average_margins.mean()),
            "median_margin": float(np.median(average_margins)),
            "margin_std": float(average_margins.std(ddof=1)),
            "mean_possible_surprise": float(
                np.mean([row["possible"].mean() for row in rows])
            ),
            "mean_impossible_surprise": float(
                np.mean([row["impossible"].mean() for row in rows])
            ),
        },
        "maximum": {
            "pairwise_accuracy": float((maximum_margins > 0).mean()),
            "mean_margin": float(maximum_margins.mean()),
            "median_margin": float(np.median(maximum_margins)),
            "margin_std": float(maximum_margins.std(ddof=1)),
            "mean_possible_surprise": float(
                np.mean([row["possible"].max() for row in rows])
            ),
            "mean_impossible_surprise": float(
                np.mean([row["impossible"].max() for row in rows])
            ),
        },
    }


def summarize_outputs(
    output_root: Path,
    scenes: list[str],
    *,
    block: str = "O1",
    context: int = 6,
) -> dict:
    pair_ids = [f"{block}/{scene}/pair-{pair}" for scene in scenes for pair in (0, 1)]
    all_rows = {
        condition: load_condition_losses(
            output_root / "inference" / condition / loss_file(block, context), pair_ids
        )
        for condition in CONDITIONS
    }
    summaries = {
        condition: aggregate_condition(rows) for condition, rows in all_rows.items()
    }

    pair_path = output_root / "pair_surprises.csv"
    with pair_path.open("w", newline="") as handle:
        fields = [
            "condition",
            "pair_id",
            "possible_average",
            "impossible_average",
            "average_margin",
            "average_correct",
            "possible_maximum",
            "impossible_maximum",
            "maximum_margin",
            "maximum_correct",
            "possible_windows",
            "impossible_windows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, rows in all_rows.items():
            for row in rows:
                possible = row["possible"]
                impossible = row["impossible"]
                average_margin = float(impossible.mean() - possible.mean())
                maximum_margin = float(impossible.max() - possible.max())
                writer.writerow(
                    {
                        "condition": condition,
                        "pair_id": row["pair_id"],
                        "possible_average": float(possible.mean()),
                        "impossible_average": float(impossible.mean()),
                        "average_margin": average_margin,
                        "average_correct": int(average_margin > 0),
                        "possible_maximum": float(possible.max()),
                        "impossible_maximum": float(impossible.max()),
                        "maximum_margin": maximum_margin,
                        "maximum_correct": int(maximum_margin > 0),
                        "possible_windows": json.dumps(possible.tolist()),
                        "impossible_windows": json.dumps(impossible.tolist()),
                    }
                )

    clean_by_pair = {row["pair_id"]: row for row in all_rows["clean"]}
    comparisons = {}
    for condition in CONDITIONS[1:]:
        current = all_rows[condition]
        average_deltas = []
        maximum_deltas = []
        for row in current:
            clean = clean_by_pair[row["pair_id"]]
            average_deltas.append(
                (row["impossible"].mean() - row["possible"].mean())
                - (clean["impossible"].mean() - clean["possible"].mean())
            )
            maximum_deltas.append(
                (row["impossible"].max() - row["possible"].max())
                - (clean["impossible"].max() - clean["possible"].max())
            )
        comparisons[condition] = {
            "average_accuracy_change": summaries[condition]["average"]["pairwise_accuracy"]
            - summaries["clean"]["average"]["pairwise_accuracy"],
            "average_margin_delta_mean": float(np.mean(average_deltas)),
            "average_margin_delta_std": float(np.std(average_deltas, ddof=1)),
            "average_fraction_pairs_margin_decreased": float(
                np.mean(np.asarray(average_deltas) < 0)
            ),
            "maximum_accuracy_change": summaries[condition]["maximum"]["pairwise_accuracy"]
            - summaries["clean"]["maximum"]["pairwise_accuracy"],
            "maximum_margin_delta_mean": float(np.mean(maximum_deltas)),
            "maximum_margin_delta_std": float(np.std(maximum_deltas, ddof=1)),
            "maximum_fraction_pairs_margin_decreased": float(
                np.mean(np.asarray(maximum_deltas) < 0)
            ),
        }

    report = {
        "scope": "descriptive pilot; no inferential claims",
        "block": block,
        "frozen_context": context,
        "scenes": scenes,
        "conditions": summaries,
        "comparisons_to_clean": comparisons,
        "pair_surprises": str(pair_path),
    }
    (output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--transformed-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--block", choices=("O1", "O2", "O3"), default="O1")
    parser.add_argument("--context", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if config["data"]["context_lengths"] != [args.context]:
        raise ValueError(
            f"Config contexts {config['data']['context_lengths']} do not match "
            f"--context {args.context}"
        )

    roots = condition_roots(args.clean_root, args.transformed_root)
    scenes = require_condition_scenes(
        roots, scene_count=args.scene_count, block=args.block
    )
    if not args.summarize_only:
        if args.output_root.exists() and any(args.output_root.iterdir()):
            raise ValueError("Output root must be new or empty")
        args.output_root.mkdir(parents=True, exist_ok=True)
        runner = Path(__file__).with_name("run_intphys_upstream.py")
        run_records = []
        for condition, dataset_root in roots.items():
            output = args.output_root / "inference" / condition
            command = [
                sys.executable,
                str(runner),
                "--upstream-root",
                str(args.upstream_root),
                "--config",
                str(args.config),
                "--dataset-root",
                str(dataset_root),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(output),
                "--block",
                args.block,
                "--scene-limit",
                str(args.scene_count),
                "--workers",
                str(args.workers),
                "--losses-only",
            ]
            log_path = args.output_root / f"{condition}.log"
            start = time.monotonic()
            with log_path.open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            run_records.append(
                {
                    "condition": condition,
                    "dataset_root": str(dataset_root.resolve()),
                    "command": shlex.join(command),
                    "elapsed_seconds": time.monotonic() - start,
                    "log": str(log_path),
                }
            )
        (args.output_root / "runs.json").write_text(json.dumps(run_records, indent=2) + "\n")

    report = summarize_outputs(
        args.output_root, scenes, block=args.block, context=args.context
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
