from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

STRATUM_FIELDS = ("condition", "Difficulty", "Camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pair_member(value: str) -> str:
    return str(value).split("_", 1)[0].strip()


def physical_label(value: str) -> str:
    lowered = str(value).lower()
    if "impossible" in lowered:
        return "impossible"
    if "possible" in lowered:
        return "possible"
    raise ValueError(f"Unknown physical label {value!r}")


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def inspected_manifest_paths(config: dict, repo_root: Path) -> dict[str, Path]:
    paths = {
        key: resolve_repo_path(repo_root, value)
        for key, value in config["manifests"].items()
        if key.startswith("inspected_")
    }
    if not paths:
        raise ValueError("No inspected_* manifests configured")
    return dict(sorted(paths.items()))


def ffprobe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    if str(stream.get("nb_frames", "")).isdigit():
        frame_count = int(stream["nb_frames"])
    else:
        count_command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
        frame_count = int(
            subprocess.run(
                count_command, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    rate = Fraction(str(stream["avg_frame_rate"]))
    return {
        "frame_count": frame_count,
        "fps_numerator": rate.numerator,
        "fps_denominator": rate.denominator,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_seconds": float(stream["duration"]),
    }


def decode_first_frame(path: Path, width: int, height: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    expected = width * height * 3
    if len(raw) != expected:
        raise ValueError(f"Decoded {len(raw)} bytes for {path}; expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def selective_center(frames: list[np.ndarray], config: dict) -> float | None:
    minimum_saturation = int(config["minimum_hsv_saturation_uint8"])
    bins = int(config["hue_histogram_bins"])
    selected = []
    for frame in frames:
        hsv = np.asarray(Image.fromarray(frame).convert("HSV"), dtype=np.uint8)
        selected.append(hsv[..., 0][hsv[..., 1] >= minimum_saturation])
    selected = [values for values in selected if values.size]
    if not selected:
        return None
    hues = np.concatenate(selected).astype(np.int64)
    assignments = np.minimum(hues * bins // 256, bins - 1)
    dominant = int(np.argmax(np.bincount(assignments, minlength=bins)))
    return (dominant + 0.5) * (360.0 / bins)


def allocate_development_scenes(
    scenes: list[dict], *, fraction: float, seed: str
) -> set[str]:
    target = round(len(scenes) * fraction)
    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for scene in scenes:
        strata[tuple(scene[field] for field in STRATUM_FIELDS)].append(scene)
    exact = {key: len(group) * target / len(scenes) for key, group in strata.items()}
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
    for key, group in strata.items():
        ordered = sorted(group, key=lambda row: stable_key(seed, row["scene_id"]))
        selected.update(row["scene_id"] for row in ordered[: quotas[key]])
    if len(selected) != target:
        raise AssertionError((len(selected), target))
    return selected


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "Main/metadata.csv").exists():
        raise FileNotFoundError(
            f"IntPhys2 Main metadata not found under {dataset_root}"
        )

    outputs = {
        key: resolve_repo_path(repo_root, config["manifests"][key])
        for key in ("release_inventory", "pairs", "samples")
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist; pass --overwrite: {existing}")

    archive = (
        args.archive_path.resolve()
        if args.archive_path is not None
        else dataset_root.parent / config["dataset"]["archive_filename"]
    )
    if not archive.exists():
        raise FileNotFoundError(archive)
    if archive.stat().st_size != int(config["dataset"]["archive_bytes"]):
        raise ValueError("IntPhys2 archive byte count differs from config")
    if sha256_file(archive) != config["dataset"]["archive_sha256"]:
        raise ValueError("IntPhys2 archive SHA-256 differs from config")

    raw_files = sorted(
        (path for path in dataset_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    inventory_rows = []
    hashes: dict[str, str] = {}
    for index, path in enumerate(raw_files):
        relative = path.relative_to(dataset_root).as_posix()
        digest = sha256_file(path)
        hashes[relative] = digest
        inventory_rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        if (index + 1) % 200 == 0:
            print(
                json.dumps({"inventory_files": index + 1, "total": len(raw_files)}),
                flush=True,
            )
    write_csv(outputs["release_inventory"], inventory_rows)

    inspected_manifests = {
        key: json.loads(path.read_text())
        for key, path in inspected_manifest_paths(config, repo_root).items()
    }
    inspected_names_by_key = {
        key: set(map(str, manifest["selected_names"]))
        for key, manifest in inspected_manifests.items()
    }
    inspected_names = set().union(*inspected_names_by_key.values())

    metadata: dict[str, pd.DataFrame] = {}
    for split in ("Debug", "Main"):
        metadata[split] = pd.read_csv(
            dataset_root / split / "metadata.csv", dtype=str, keep_default_na=False
        )
    main_names = set(metadata["Main"]["name"])
    if not inspected_names <= main_names:
        raise ValueError(
            f"Inspected manifest has {len(inspected_names - main_names)} unknown names"
        )
    debug_source_id_overlap_names = set(metadata["Debug"]["name"]) & main_names
    debug_source_id_overlap_scenes = set(
        metadata["Main"].loc[
            metadata["Main"]["name"].isin(debug_source_id_overlap_names), "SceneIndex"
        ]
    )

    pair_groups: dict[str, list[dict]] = {}
    pair_split: dict[str, str] = {}
    for split, frame in metadata.items():
        for (scene, member), group in frame.groupby(
            ["SceneIndex", frame["type"].map(pair_member)], sort=True
        ):
            pair_id = f"{split}:{scene}/{member}"
            rows = group.to_dict("records")
            if len(rows) != 2:
                raise ValueError(f"Pair {pair_id} has {len(rows)} videos")
            labels = sorted(physical_label(row["type"]) for row in rows)
            if labels != ["impossible", "possible"]:
                raise ValueError(f"Pair {pair_id} labels are {labels}")
            for field in STRATUM_FIELDS:
                if len({row[field] for row in rows}) != 1:
                    raise ValueError(f"Pair {pair_id} differs on {field}")
            pair_groups[pair_id] = rows
            pair_split[pair_id] = split

    inspected_pairs_by_key = {}
    for key, names in inspected_names_by_key.items():
        inspected_pairs = {
            pair_id
            for pair_id, rows in pair_groups.items()
            if pair_split[pair_id] == "Main" and {row["name"] for row in rows} <= names
        }
        expected_pairs = int(inspected_manifests[key]["selected_pairs"])
        if len(inspected_pairs) != expected_pairs:
            raise ValueError(
                f"{key}: recovered {len(inspected_pairs)} inspected pairs, "
                f"expected {expected_pairs}"
            )
        recovered_names = set().union(
            *(set(row["name"] for row in pair_groups[p]) for p in inspected_pairs)
        )
        if recovered_names != names:
            raise ValueError(f"{key}: selected_names are not exactly pair preserving")
        inspected_pairs_by_key[key] = inspected_pairs
    main_inspected_pairs = set().union(*inspected_pairs_by_key.values())
    inspected_scenes = {
        pair_groups[pair_id][0]["SceneIndex"] for pair_id in main_inspected_pairs
    }

    scene_rows = []
    for scene, group in metadata["Main"].groupby("SceneIndex", sort=True):
        values = {}
        for field in STRATUM_FIELDS:
            unique = sorted(set(group[field]))
            if len(unique) != 1:
                raise ValueError(f"Main scene {scene} differs on {field}: {unique}")
            values[field] = unique[0]
        if scene in inspected_scenes:
            partition = "excluded_inspected_scene"
        elif scene in debug_source_id_overlap_scenes:
            partition = "excluded_debug_source_id_overlap"
        else:
            partition = "fresh"
        scene_rows.append({"scene_id": scene, "partition": partition, **values})
    fresh_scenes = [row for row in scene_rows if row["partition"] == "fresh"]
    development_scenes = allocate_development_scenes(
        fresh_scenes,
        fraction=float(config["partition"]["development_fraction_of_fresh_scenes"]),
        seed=config["partition"]["seed"],
    )
    scene_partition = {}
    for row in scene_rows:
        if row["partition"] == "fresh":
            scene_partition[row["scene_id"]] = (
                "development" if row["scene_id"] in development_scenes else "test"
            )
        else:
            scene_partition[row["scene_id"]] = row["partition"]

    video_info: dict[str, dict] = {}
    pair_rows = []
    samples = []
    for pair_index, pair_id in enumerate(sorted(pair_groups)):
        split = pair_split[pair_id]
        rows = pair_groups[pair_id]
        scene = str(rows[0]["SceneIndex"])
        member = pair_member(rows[0]["type"])
        partition = "debug" if split == "Debug" else scene_partition[scene]
        by_label = {physical_label(row["type"]): row for row in rows}
        member_info = {}
        first_frames = []
        for label in ("possible", "impossible"):
            row = by_label[label]
            relative = f"{split}/{row['file_name']}"
            path = dataset_root / relative
            if not path.exists():
                raise FileNotFoundError(path)
            if relative not in video_info:
                info = ffprobe_video(path)
                if info["frame_count"] < int(config["sampling"]["source_frames"]):
                    raise ValueError(
                        f"{relative} has only {info['frame_count']} frames"
                    )
                indices = np.rint(
                    np.linspace(
                        0,
                        info["frame_count"] - 1,
                        int(config["sampling"]["source_frames"]),
                    )
                ).astype(np.int64)
                if len(np.unique(indices)) != len(indices):
                    raise ValueError(f"Uniform source indices repeat for {relative}")
                info["source_frame_indices"] = indices.tolist()
                info["relative_path"] = relative
                info["sha256"] = hashes[relative]
                video_info[relative] = info
            info = video_info[relative]
            member_info[label] = (row, info)
            first_frames.append(decode_first_frame(path, info["width"], info["height"]))
        center = selective_center(first_frames, config["selective_recolor"])
        selective_feasible = center is not None
        inspected_pair = pair_id in main_inspected_pairs
        inspected_scene = split == "Main" and scene in inspected_scenes
        debug_source_id_overlap_scene = (
            split == "Main" and scene in debug_source_id_overlap_scenes
        )
        pair_rows.append(
            {
                "dataset_split": split,
                "partition": partition,
                "pair_id": pair_id,
                "scene_id": scene,
                "pair_member": member,
                "condition": rows[0]["condition"],
                "difficulty": rows[0]["Difficulty"],
                "camera": rows[0]["Camera"],
                "inspected_pair": str(inspected_pair).lower(),
                "inspected_scene": str(inspected_scene).lower(),
                "debug_source_id_overlap_scene": str(
                    debug_source_id_overlap_scene
                ).lower(),
                "selective_center_degrees": "" if center is None else f"{center:.6f}",
                "selective_transform_feasible": str(selective_feasible).lower(),
                "possible_name": by_label["possible"]["name"],
                "possible_relative_path": member_info["possible"][1]["relative_path"],
                "possible_sha256": member_info["possible"][1]["sha256"],
                "impossible_name": by_label["impossible"]["name"],
                "impossible_relative_path": member_info["impossible"][1][
                    "relative_path"
                ],
                "impossible_sha256": member_info["impossible"][1]["sha256"],
            }
        )
        for label in ("possible", "impossible"):
            row, info = member_info[label]
            indices_json = json.dumps(
                info["source_frame_indices"], separators=(",", ":")
            )
            for condition in config["appearance_conditions"]:
                selective = condition["family"] == "selective_recolor"
                feasible = selective_feasible if selective else True
                samples.append(
                    {
                        "sample_id": f"{split}__{row['name']}__{condition['name']}",
                        "dataset_split": split,
                        "partition": partition,
                        "pair_id": pair_id,
                        "scene_id": scene,
                        "pair_member": member,
                        "physical_label": label,
                        "physical_label_value": 1 if label == "impossible" else 0,
                        "name": row["name"],
                        "source_relative_path": info["relative_path"],
                        "source_sha256": info["sha256"],
                        "source_frame_count": info["frame_count"],
                        "source_fps_numerator": info["fps_numerator"],
                        "source_fps_denominator": info["fps_denominator"],
                        "source_width": info["width"],
                        "source_height": info["height"],
                        "source_frame_indices_0based": indices_json,
                        "appearance_condition": condition["name"],
                        "transform_family": condition["family"],
                        "transform_value": ""
                        if condition["value"] is None
                        else condition["value"],
                        "transform_unit": ""
                        if condition["unit"] is None
                        else condition["unit"],
                        "selective_center_degrees": ""
                        if not selective or center is None
                        else f"{center:.6f}",
                        "transform_feasible": str(feasible).lower(),
                        "infeasible_execution": "" if feasible else "exact_identity",
                        "condition": row["condition"],
                        "difficulty": row["Difficulty"],
                        "camera": row["Camera"],
                        "inspected_pair": str(inspected_pair).lower(),
                        "inspected_scene": str(inspected_scene).lower(),
                        "debug_source_id_overlap_scene": str(
                            debug_source_id_overlap_scene
                        ).lower(),
                    }
                )
        if (pair_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {"pairs_prepared": pair_index + 1, "total": len(pair_groups)}
                ),
                flush=True,
            )

    write_csv(outputs["pairs"], pair_rows)
    write_csv(outputs["samples"], samples)
    counts = defaultdict(int)
    for row in pair_rows:
        counts[(row["dataset_split"], row["partition"])] += 1
    print(
        json.dumps(
            {
                "release_files": len(inventory_rows),
                "release_bytes": sum(int(row["size_bytes"]) for row in inventory_rows),
                "pairs": len(pair_rows),
                "samples": len(samples),
                "pair_counts": {
                    f"{key[0]}:{key[1]}": value for key, value in sorted(counts.items())
                },
                "inspected_pairs": len(main_inspected_pairs),
                "inspected_scenes": len(inspected_scenes),
                "inspected_manifests": {
                    key: {
                        "pairs": len(pairs),
                        "scenes": len(
                            {pair_groups[pair_id][0]["SceneIndex"] for pair_id in pairs}
                        ),
                    }
                    for key, pairs in inspected_pairs_by_key.items()
                },
                "debug_source_id_overlap_names": len(debug_source_id_overlap_names),
                "debug_source_id_overlap_main_scenes": len(
                    debug_source_id_overlap_scenes
                ),
                "selective_infeasible_pairs": sum(
                    row["selective_transform_feasible"] == "false" for row in pair_rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
