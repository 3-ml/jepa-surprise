#!/usr/bin/env python3
"""Select an original-IntPhys V-JEPA 2-G context from clean pair accuracy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(lambda: defaultdict(dict))
    for path in (args.output_dir / "videos").glob("*.json"):
        payload = json.loads(path.read_text())
        pair = f"{payload['block']}/{payload['scene']}/{payload['pair']}"
        label = "possible" if payload["possible"] else "impossible"
        per_context = defaultdict(list)
        for row in payload["windows"]:
            per_context[int(row["context_length"])].append(float(row["scalar_l1"]))
        for context, values in per_context.items():
            grouped[context][pair][label] = float(np.mean(values))
    results = {}
    for context, pairs in sorted(grouped.items()):
        margins = []
        for pair, labels in pairs.items():
            if set(labels) != {"possible", "impossible"}:
                raise ValueError(f"Incomplete pair {pair}")
            margins.append(labels["impossible"] - labels["possible"])
        margins = np.asarray(margins)
        results[str(context)] = {
            "pairs": len(margins), "correct": int((margins > 0).sum()),
            "accuracy": float((margins > 0).mean()), "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
        }
    selected = max((int(value) for value in results), key=lambda value: (results[str(value)]["accuracy"], results[str(value)]["mean_margin"], -value))
    report = {"selected_context": selected, "selection_rule": "highest clean pairwise mean-surprise accuracy; then mean margin; then shorter context", "contexts": results}
    text = json.dumps(report, indent=2) + "\n"
    if args.write: args.write.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
