from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STRATUM_FIELDS = ("condition", "Difficulty", "Camera")
APPEARANCE_CONDITIONS = {
    "original": {"family": "original", "parameters": {}},
    "grayscale": {
        "family": "color",
        "parameters": {"method": "rgb_luma_0.299_0.587_0.114"},
    },
    "brightness_contrast": {
        "family": "lighting",
        "parameters": {"scale": 0.72, "offset": 20.0},
    },
    "hue_shift": {"family": "color", "parameters": {"degrees": 40.0}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", default="intphys2-confirmatory-v1")
    parser.add_argument("--frame-count", type=int, default=64)
    parser.add_argument("--input-resolution", type=int, default=256)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def pair_member(value: str) -> str:
    return str(value).split("_", 1)[0].strip()


def pair_label(value: str) -> str:
    lowered = str(value).lower()
    if "impossible" in lowered:
        return "impossible"
    if "possible" in lowered:
        return "possible"
    raise ValueError(f"Cannot infer pair label from {value!r}")


def load_pairs(metadata_path: Path, dataset_split: str) -> list[dict]:
    frame = pd.read_csv(metadata_path, dtype={"SceneIndex": str}).where(pd.notna, None)
    pairs: list[dict] = []
    for (scene, member), group in frame.groupby(
        ["SceneIndex", frame["type"].map(pair_member)], sort=True
    ):
        rows = group.to_dict("records")
        if len(rows) != 2:
            raise ValueError(
                f"{dataset_split} pair {scene}/{member} has {len(rows)} videos"
            )
        labels = sorted(pair_label(row["type"]) for row in rows)
        if labels != ["impossible", "possible"]:
            raise ValueError(
                f"{dataset_split} pair {scene}/{member} labels are {labels}"
            )
        for field in STRATUM_FIELDS:
            values = {str(row[field]) for row in rows}
            if len(values) != 1:
                raise ValueError(
                    f"{dataset_split} pair {scene}/{member} differs on {field}: {values}"
                )
        by_label = {pair_label(row["type"]): row for row in rows}
        scene_id = str(scene)
        pair_id = f"{scene_id}/{member}"
        pair = {
            "dataset_split": dataset_split,
            "pair_id": pair_id,
            "scene_index": scene_id,
            "member": str(member),
            "condition": str(rows[0]["condition"]),
            "difficulty": str(rows[0]["Difficulty"]),
            "camera": str(rows[0]["Camera"]),
            "possible_name": str(by_label["possible"]["name"]),
            "possible_file_name": str(by_label["possible"]["file_name"]),
            "impossible_name": str(by_label["impossible"]["name"]),
            "impossible_file_name": str(by_label["impossible"]["file_name"]),
            "game_name": str(rows[0]["game_name"]),
            "env": str(rows[0]["env"]),
            "occluder": "" if rows[0]["occluder"] is None else str(rows[0]["occluder"]),
        }
        pairs.append(pair)
    return pairs


def excluded_pair_sources(manifests: list[Path]) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = defaultdict(list)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        label = manifest_path.as_posix()
        for pair in manifest.get("pairs", []):
            sources[str(pair["pair_id"])].append(label)
    return sources


def select_stratified_test(
    pairs: list[dict],
    *,
    target: int,
    seed: str,
) -> set[str]:
    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for pair in pairs:
        key = tuple(str(pair[field]) for field in ("condition", "difficulty", "camera"))
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
    selected: set[str] = set()
    for key in sorted(strata):
        ordered = sorted(
            strata[key], key=lambda pair: stable_key(seed, pair["pair_id"])
        )
        selected.update(pair["pair_id"] for pair in ordered[: quotas[key]])
    if len(selected) != target:
        raise AssertionError((len(selected), target))
    return selected


def video_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"Could not read positive frame count from {path}")
    return count


def source_indices(frame_count: int, requested: int) -> list[int]:
    if frame_count < requested:
        raise ValueError(
            f"Video has {frame_count} frames, fewer than requested {requested}"
        )
    values = pd.Series(
        [
            round(index * (frame_count - 1) / (requested - 1))
            for index in range(requested)
        ]
    ).astype(int)
    if values.nunique() != requested:
        raise ValueError(
            f"Uniform source-frame rule produced duplicate indices for {frame_count} frames"
        )
    return values.tolist()


def pair_rows_to_sample_rows(
    pairs: list[dict],
    *,
    dataset_root: Path,
    frame_count: int,
    input_resolution: int,
) -> tuple[list[dict], dict[str, dict]]:
    samples: list[dict] = []
    video_cache: dict[str, dict] = {}
    for pair in pairs:
        if str(pair["partition"]).startswith("excluded_"):
            continue
        for label in ("possible", "impossible"):
            name = pair[f"{label}_name"]
            file_name = pair[f"{label}_file_name"]
            video_path = dataset_root / pair["dataset_split"] / file_name
            if name not in video_cache:
                decoded_count = video_frame_count(video_path)
                video_cache[name] = {
                    "video_path": str(video_path),
                    "frame_count": decoded_count,
                    "source_frame_indices_0based": source_indices(
                        decoded_count, frame_count
                    ),
                }
            for appearance_condition, transform in APPEARANCE_CONDITIONS.items():
                samples.append(
                    {
                        "sample_id": f"{pair['dataset_split']}|{pair['partition']}|{pair['pair_id']}|{name}|{appearance_condition}",
                        "dataset_split": pair["dataset_split"],
                        "partition": pair["partition"],
                        "pair_id": pair["pair_id"],
                        "scene_index": pair["scene_index"],
                        "member": pair["member"],
                        "name": name,
                        "file_name": file_name,
                        "video_path": str(video_path),
                        "pair_label": label,
                        "appearance_condition": appearance_condition,
                        "transform_family": transform["family"],
                        "transform_parameters_json": json.dumps(
                            transform["parameters"], sort_keys=True
                        ),
                        "condition": pair["condition"],
                        "difficulty": pair["difficulty"],
                        "camera": pair["camera"],
                        "frame_count": video_cache[name]["frame_count"],
                        "source_frame_indices_0based_json": json.dumps(
                            video_cache[name]["source_frame_indices_0based"],
                            separators=(",", ":"),
                        ),
                        "input_resolution": input_resolution,
                    }
                )
    return samples, video_cache


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in (0, 1)")

    main_metadata = args.dataset_root / "Main" / "metadata.csv"
    debug_metadata = args.dataset_root / "Debug" / "metadata.csv"
    main_pairs = load_pairs(main_metadata, "Main")
    debug_pairs = load_pairs(debug_metadata, "Debug")
    excluded_sources = excluded_pair_sources(args.exclude_manifest)
    debug_pair_ids = {pair["pair_id"] for pair in debug_pairs}

    non_excluded = [
        pair for pair in main_pairs if pair["pair_id"] not in excluded_sources
    ]
    test_eligible = [
        pair for pair in non_excluded if pair["pair_id"] not in debug_pair_ids
    ]
    target_test = round(len(test_eligible) * args.test_fraction)
    test_pair_ids = select_stratified_test(
        test_eligible, target=target_test, seed=args.seed
    )

    pair_rows: list[dict] = []
    for pair in debug_pairs:
        row = dict(pair)
        row.update(
            {
                "partition": "debug",
                "inspected_source": "",
                "debug_overlap": True,
            }
        )
        pair_rows.append(row)
    for pair in main_pairs:
        row = dict(pair)
        if pair["pair_id"] in excluded_sources:
            partition = "excluded_inspected"
            inspected_source = "|".join(excluded_sources[pair["pair_id"]])
        elif pair["pair_id"] in debug_pair_ids:
            partition = "excluded_debug"
            inspected_source = "Debug split smoke overlap"
        elif pair["pair_id"] in test_pair_ids:
            partition = "test"
            inspected_source = ""
        else:
            partition = "development"
            inspected_source = ""
        row.update(
            {
                "partition": partition,
                "inspected_source": inspected_source,
                "debug_overlap": pair["pair_id"] in debug_pair_ids,
            }
        )
        pair_rows.append(row)

    samples, video_cache = pair_rows_to_sample_rows(
        pair_rows,
        dataset_root=args.dataset_root,
        frame_count=args.frame_count,
        input_resolution=args.input_resolution,
    )

    pairs_frame = pd.DataFrame(pair_rows).sort_values(
        ["dataset_split", "partition", "scene_index", "member"]
    )
    samples_frame = pd.DataFrame(samples).sort_values(
        [
            "dataset_split",
            "partition",
            "scene_index",
            "member",
            "pair_label",
            "appearance_condition",
        ]
    )
    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    args.output_samples.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    pairs_frame.to_csv(args.output_pairs, index=False, lineterminator="\n")
    samples_frame.to_csv(args.output_samples, index=False, lineterminator="\n")

    main_counts = (
        pairs_frame[pairs_frame["dataset_split"].eq("Main")]["partition"]
        .value_counts()
        .to_dict()
    )
    debug_counts = (
        pairs_frame[pairs_frame["dataset_split"].eq("Debug")]["partition"]
        .value_counts()
        .to_dict()
    )
    config = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "IntPhys2",
            "root": str(args.dataset_root),
            "artifact_root": str(args.artifact_root),
            "main_metadata": str(main_metadata),
            "main_metadata_sha256": sha256_file(main_metadata),
            "debug_metadata": str(debug_metadata),
            "debug_metadata_sha256": sha256_file(debug_metadata),
        },
        "manifests": {
            "pairs": str(args.output_pairs),
            "samples": str(args.output_samples),
        },
        "split": {
            "seed": args.seed,
            "test_fraction_of_non_inspected_non_debug_main_pairs": args.test_fraction,
            "stratum_fields": ["condition", "Difficulty", "Camera"],
            "excluded_manifests": [str(path) for path in args.exclude_manifest],
            "debug_pairs_for_smoke_only": len(debug_pairs),
            "debug_pair_ids_excluded_from_main_test": sorted(debug_pair_ids),
            "main_partition_counts": main_counts,
            "debug_partition_counts": debug_counts,
        },
        "appearance_conditions": APPEARANCE_CONDITIONS,
        "source_frames": {
            "frame_count": args.frame_count,
            "rule": "round(linspace(0, decoded_frame_count - 1, frame_count)) per source video",
            "same_indices_for_all_models_and_appearances": True,
            "decoded_video_count": len(video_cache),
        },
        "model_input": {
            "resolution": args.input_resolution,
            "relative_depths": [0.0, 0.25, 0.5, 0.75, 1.0],
            "feature_views": ["pooled", "patch_or_token_preserving_summary"],
        },
        "policy": {
            "heldout_labels": "HeldOut physical labels are not used.",
            "test_features": "Run once only after the analysis specification is committed.",
            "large_features": "Keep feature shards in external storage and index by hash.",
        },
    }
    args.output_config.write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            {
                "main_partition_counts": main_counts,
                "debug_partition_counts": debug_counts,
                "samples": len(samples_frame),
                "decoded_video_count": len(video_cache),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
