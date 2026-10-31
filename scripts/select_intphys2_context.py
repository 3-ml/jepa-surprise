#!/usr/bin/env python3
"""Select and summarize an IntPhys 2 context from public Debug labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def label_and_pair(metadata: dict) -> tuple[str, str]:
    value = str(metadata["type"])
    lowered = value.lower()
    if "impossible" in lowered:
        label = "impossible"
    elif "possible" in lowered:
        label = "possible"
    else:
        raise ValueError(f"Unknown type: {value}")
    member = value.split("_", 1)[0].strip()
    pair = f"{metadata['SceneIndex']}/{member}"
    return label, pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    grouped: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in sorted((args.output_dir / "videos").glob("*.json")):
        payload = json.loads(path.read_text())
        label, pair = label_and_pair(payload["metadata"])
        for row in payload["windows"]:
            grouped[int(row["context_length"])][pair][label].append(
                float(row["scalar_l1"])
            )

    results = {}
    for context, pairs in sorted(grouped.items()):
        margins = []
        for pair, labels in pairs.items():
            if set(labels) != {"possible", "impossible"}:
                raise ValueError(f"Incomplete pair {pair}: {sorted(labels)}")
            margins.append(
                np.mean(labels["impossible"]) - np.mean(labels["possible"])
            )
        values = np.asarray(margins)
        results[str(context)] = {
            "pairs": len(values),
            "correct": int((values > 0).sum()),
            "accuracy": float((values > 0).mean()),
            "mean_margin": float(values.mean()),
            "median_margin": float(np.median(values)),
        }
    if not results:
        raise ValueError("No completed videos")
    selected = max(
        (int(context) for context in results),
        key=lambda context: (
            results[str(context)]["accuracy"],
            results[str(context)]["mean_margin"],
            -context,
        ),
    )
    report = {
        "selection_split": "Debug",
        "selection_rule": "highest pairwise mean-surprise accuracy; then mean margin; then shorter context",
        "selected_context": selected,
        "contexts": results,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.write:
        args.write.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
