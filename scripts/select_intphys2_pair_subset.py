#!/usr/bin/env python3
"""Select a deterministic stratified, pair-preserving IntPhys 2 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

STRATUM_FIELDS = ("condition", "Difficulty", "Camera")


def pair_member(value: str) -> str:
    return str(value).split("_", 1)[0].strip()


def stable_key(seed: str, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--fraction", type=float)
    size.add_argument("--pairs", type=int)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Selection manifest whose pair IDs must be excluded; repeat as needed.",
    )
    parser.add_argument("--seed", default="intphys2-main-endpoints-v1")
    args = parser.parse_args()
    if args.fraction is not None and not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if args.pairs is not None and args.pairs < 1:
        raise ValueError("pairs must be positive")

    frame = pd.read_csv(args.metadata).where(pd.notna, None)
    pairs: list[dict] = []
    for (scene, member), group in frame.groupby(
        ["SceneIndex", frame["type"].map(pair_member)], sort=True
    ):
        rows = group.to_dict("records")
        if len(rows) != 2:
            raise ValueError(f"Pair {scene}/{member} has {len(rows)} videos")
        labels = sorted(
            "impossible" if "impossible" in str(row["type"]).lower() else "possible"
            for row in rows
        )
        if labels != ["impossible", "possible"]:
            raise ValueError(f"Pair {scene}/{member} labels are {labels}")
        for field in STRATUM_FIELDS:
            values = {str(row[field]) for row in rows}
            if len(values) != 1:
                raise ValueError(f"Pair {scene}/{member} differs on {field}: {values}")
        pair_id = f"{scene}/{member}"
        pairs.append(
            {
                "pair_id": pair_id,
                "SceneIndex": int(scene),
                "member": str(member),
                "names": sorted(str(row["name"]) for row in rows),
                "filenames": sorted(str(row["file_name"]) for row in rows),
                "stratum": {field: rows[0][field] for field in STRATUM_FIELDS},
            }
        )

    total_pairs = len(pairs)
    excluded_pair_ids: set[str] = set()
    for manifest_path in args.exclude_manifest:
        manifest = json.loads(manifest_path.read_text())
        excluded_pair_ids.update(str(pair["pair_id"]) for pair in manifest["pairs"])
    pairs = [pair for pair in pairs if pair["pair_id"] not in excluded_pair_ids]
    if not pairs:
        raise ValueError("No pairs remain after exclusions")

    if args.pairs is not None:
        target = args.pairs
        requested_fraction = None
    else:
        requested_fraction = args.fraction if args.fraction is not None else 1 / 3
        target = round(len(pairs) * requested_fraction)
    if target > len(pairs):
        raise ValueError(f"Requested {target} pairs but only {len(pairs)} are available")

    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for pair in pairs:
        key = tuple(str(pair["stratum"][field]) for field in STRATUM_FIELDS)
        strata[key].append(pair)

    exact = {key: len(group) * target / len(pairs) for key, group in strata.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (exact[key] - quotas[key], len(strata[key]), key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1

    selected = []
    stratum_counts = []
    for key in sorted(strata):
        ordered = sorted(
            strata[key], key=lambda pair: stable_key(args.seed, pair["pair_id"])
        )
        chosen = ordered[: quotas[key]]
        selected.extend(chosen)
        stratum_counts.append(
            {
                "stratum": dict(zip(STRATUM_FIELDS, key)),
                "available_pairs": len(ordered),
                "selected_pairs": len(chosen),
            }
        )
    selected.sort(key=lambda pair: pair["pair_id"])
    if len(selected) != target:
        raise AssertionError((len(selected), target))

    payload = {
        "schema_version": 1,
        "dataset": "IntPhys2",
        "split": "Main",
        "metadata": str(args.metadata),
        "selection": "proportional largest-remainder allocation, then seeded SHA-256 ordering within strata",
        "stratum_fields": list(STRATUM_FIELDS),
        "seed": args.seed,
        "fraction": requested_fraction,
        "requested_pairs": args.pairs,
        "available_pairs_total": total_pairs,
        "exclude_manifests": [str(path) for path in args.exclude_manifest],
        "excluded_pairs": total_pairs - len(pairs),
        "available_pairs": len(pairs),
        "selected_pairs": len(selected),
        "selected_videos": len(selected) * 2,
        "selected_names": sorted(name for pair in selected for name in pair["names"]),
        "strata": stratum_counts,
        "pairs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("available_pairs", "selected_pairs", "selected_videos")}))


if __name__ == "__main__":
    main()
