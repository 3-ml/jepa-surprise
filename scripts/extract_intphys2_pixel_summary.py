#!/usr/bin/env python3
"""Extract corrected IntPhys2 PixelSummary features from the locked manifest."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import intphys2_confirmatory_common as common
    import intphys2_confirmatory_extraction as extraction
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common
    from scripts import intphys2_confirmatory_extraction as extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("debug", "development", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def exact_command() -> str:
    return " ".join([sys.executable, *sys.argv])


def load_rows(shared_config: Path, partition: str) -> tuple[dict, list[dict[str, str]]]:
    shared = json.loads(shared_config.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[1]
    paths = extraction.contract_paths(repo_root, shared)
    common.validate_retained_transform_audit(
        paths["transform_full_audit"],
        shared["transform_audit_provenance"]["validation_sha256"],
        paths["transform_metrics"],
    )
    rows = common.read_samples(paths["samples"], partition)
    return shared, rows


def rgb_hist(values: np.ndarray, bins: int = 16) -> list[float]:
    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(values[..., channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        features.extend(hist.tolist())
    return features


def scalar_stats(values: np.ndarray) -> list[float]:
    flat = values.reshape(-1).astype(np.float32)
    return [
        float(np.mean(flat)),
        float(np.std(flat)),
        float(np.percentile(flat, 95)),
        float(np.max(flat)),
    ]


def summarize_frames(frames: np.ndarray) -> np.ndarray:
    rgb = frames.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    diffs = np.abs(np.diff(rgb, axis=0))
    luma_diffs = np.abs(np.diff(luma, axis=0))
    parts: list[float] = []
    parts.extend(np.mean(rgb, axis=(0, 1, 2)).tolist())
    parts.extend(np.std(rgb, axis=(0, 1, 2)).tolist())
    parts.extend(rgb_hist(rgb, bins=16))
    parts.extend([float(np.mean(luma)), float(np.std(luma))])
    parts.extend(np.mean(diffs, axis=(0, 1, 2)).tolist())
    parts.extend(np.std(diffs, axis=(0, 1, 2)).tolist())
    parts.extend(scalar_stats(luma_diffs))
    hist, _ = np.histogram(luma_diffs, bins=16, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    parts.extend(hist.tolist())
    for frame_index in (0, len(rgb) // 2, len(rgb) - 1):
        frame = rgb[frame_index]
        parts.extend(np.mean(frame, axis=(0, 1)).tolist())
        parts.extend(np.std(frame, axis=(0, 1)).tolist())
    if len(parts) != 100:
        raise ValueError(f"Unexpected PixelSummary width {len(parts)}")
    return np.asarray(parts, dtype=np.float32)


def metadata_columns(rows: list[dict[str, str]]) -> list[str]:
    columns = [
        "sample_id",
        "dataset_split",
        "partition",
        "pair_id",
        "pair_label",
        "pair_label_value",
        "name",
        "source_relative_path",
        "source_sha256",
        "source_frame_count",
        "source_frame_indices_0based",
        "appearance_condition",
        "transform_family",
        "transform_value",
        "transform_unit",
        "selective_center_degrees",
        "transform_feasible",
        "infeasible_execution",
        "condition",
        "difficulty",
        "camera",
        "inspected_pair",
        "inspected_scene",
        "debug_source_id_overlap_scene",
    ]
    return [column for column in columns if column == "pair_label" or column in rows[0]]


def main() -> None:
    args = parse_args()
    if args.partition == "test" and not args.allow_test:
        raise ValueError("Refusing sealed test PixelSummary extraction without --allow-test")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    shared, rows = load_rows(args.shared_config.resolve(), args.partition)
    rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_source[row["source_relative_path"]].append(row)

    features: list[np.ndarray] = []
    sample_rows: list[dict[str, str]] = []
    decode_seconds = 0.0
    transform_seconds = 0.0
    dataset_root = args.dataset_root.resolve()
    expected_conditions = [item["name"] for item in shared["appearance_conditions"]]
    ordered_sources = sorted(rows_by_source.items())
    for source_index, (source_relative, source_rows) in enumerate(ordered_sources, start=1):
        source_rows = sorted(
            source_rows, key=lambda row: expected_conditions.index(row["appearance_condition"])
        )
        clean_row = source_rows[0]
        source_path = dataset_root / source_relative
        if common.sha256_file(source_path) != clean_row["source_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {source_path}")
        decode_started = time.perf_counter()
        frames = common.decode_selected_frames(
            source_path, json.loads(clean_row["source_frame_indices_0based"])
        )
        decode_seconds += time.perf_counter() - decode_started
        for row in source_rows:
            transform_started = time.perf_counter()
            transformed = common.apply_appearance(frames, row, shared)
            transform_seconds += time.perf_counter() - transform_started
            features.append(summarize_frames(transformed))
            sample_rows.append(row)
        if source_index == 1 or source_index % 10 == 0 or source_index == len(ordered_sources):
            print(
                json.dumps(
                    {
                        "sources_done": source_index,
                        "sources_total": len(ordered_sources),
                        "samples_done": len(sample_rows),
                    }
                ),
                flush=True,
            )

    x = np.stack(features, axis=0)[:, None, :]
    columns = metadata_columns(sample_rows)
    arrays: dict[str, np.ndarray] = {
        "X": x.astype(np.float32, copy=False),
        "layer_names": np.asarray(["pixel_summary"]),
        "metadata_columns": np.asarray(columns),
    }
    for column in columns:
        if column == "pair_label":
            values = [row["physical_label"] for row in sample_rows]
        else:
            values = [row.get(column, "") for row in sample_rows]
        arrays[column] = np.asarray(values)

    representations = args.output_dir / "representations.npz"
    representations_sha256 = common.write_atomic_npz(representations, arrays)
    with (args.output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in sample_rows:
            record = {column: row.get(column, "") for column in columns}
            record["pair_label"] = row["physical_label"]
            writer.writerow(record)
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "command": exact_command(),
        "shared_config": str(args.shared_config.resolve()),
        "shared_config_sha256": common.sha256_text_file(args.shared_config.resolve()),
        "dataset_root": str(dataset_root),
        "partition": args.partition,
        "representations": str(representations.resolve()),
        "representations_sha256": representations_sha256,
        "samples": int(x.shape[0]),
        "pairs": len({row["pair_id"] for row in sample_rows}),
        "sources": len(rows_by_source),
        "feature_dim": int(x.shape[2]),
        "decode_seconds": decode_seconds,
        "transform_seconds": transform_seconds,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "claim_boundary": "Low-level pixel comparator only; no model-semantics claim is evaluated here.",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "samples": int(x.shape[0]),
                "pairs": len({row["pair_id"] for row in sample_rows}),
                "representations_sha256": representations_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
