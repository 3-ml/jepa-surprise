#!/usr/bin/env python3
"""Validate the frozen IntPhys2 confirmatory data, split, and transform contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument(
        "--transform-audit",
        choices=("none", "active", "all", "reuse"),
        default="active",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty metrics file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def bool_value(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"Expected lowercase boolean, found {value!r}")
    return value == "true"


def pair_member(value: str) -> str:
    return str(value).split("_", 1)[0].strip()


def physical_label(value: str) -> str:
    lowered = str(value).lower()
    if "impossible" in lowered:
        return "impossible"
    if "possible" in lowered:
        return "possible"
    raise ValueError(f"Unknown physical label {value!r}")


def video_frame_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    value = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    if not value.isdigit():
        raise ValueError(f"Missing frame count for {path}")
    return int(value)


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def inspected_manifest_paths(config: dict, repo_root: Path) -> dict[str, Path]:
    paths = {
        key: resolve(repo_root, value)
        for key, value in config["manifests"].items()
        if key.startswith("inspected_")
    }
    if not paths:
        raise ValueError("No inspected_* manifests configured")
    return dict(sorted(paths.items()))


def aggregate_effects(rows: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    result = {}
    for condition, values in sorted(by_condition.items()):
        result[condition] = {
            "videos": len(values),
            "feasible_videos": sum(value["feasible"] for value in values),
            "identity_videos": sum(value["identity"] for value in values),
            "mean_rgb_l1_01": float(np.mean([value["rgb_l1"] for value in values])),
            "min_rgb_l1_01": float(np.min([value["rgb_l1"] for value in values])),
            "max_rgb_l1_01": float(np.max([value["rgb_l1"] for value in values])),
            "mean_affected_pixel_fraction": float(
                np.mean([value["affected"] for value in values])
            ),
            "minimum_luma_standard_deviation_ratio": float(
                np.min([value["luma_std_ratio"] for value in values])
            ),
            "maximum_additional_zero_luma_fraction": float(
                np.max([value["additional_zero_luma_fraction"] for value in values])
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    dataset_root = args.dataset_root
    if dataset_root is None:
        variable = config["dataset"]["root_environment_variable"]
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"Pass --dataset-root or set {variable}")
        dataset_root = Path(value)
    dataset_root = dataset_root.resolve()

    manifest_paths = {
        key: resolve(repo_root, config["manifests"][key])
        for key in ("release_inventory", "pairs", "samples")
    }
    inspected_paths = inspected_manifest_paths(config, repo_root)
    model_config_paths = {
        key: resolve(repo_root, value) for key, value in config["model_configs"].items()
    }
    for path in [*manifest_paths.values(), *inspected_paths.values(), *model_config_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    model_configs = {
        key: json.loads(path.read_text()) for key, path in model_config_paths.items()
    }
    for key, model in model_configs.items():
        if [item["relative_depth"] for item in model["relative_depth_mapping"]] != config[
            "relative_depths"
        ]:
            raise ValueError(f"{key} relative-depth mapping differs from shared config")
        if len({item["native_layer"] for item in model["relative_depth_mapping"]}) != len(
            config["relative_depths"]
        ):
            raise ValueError(f"{key} native layer mapping is not unique")
        if len(model["checkpoint_sha256"]) != 64:
            raise ValueError(f"{key} checkpoint SHA-256 is malformed")
    if model_configs["vjepa2"]["native_token_grid"] != config["cross_model_contract"][
        "native_token_grid_after_concatenation"
    ]:
        raise ValueError("V-JEPA2 native token grid differs from shared contract")
    if model_configs["videomae2"]["concatenated_token_grid"] != config[
        "cross_model_contract"
    ]["native_token_grid_after_concatenation"]:
        raise ValueError("VideoMAE2 concatenated token grid differs from shared contract")
    output = args.output or resolve(repo_root, config["manifests"]["validation"])
    transform_metrics_path = resolve(repo_root, config["manifests"]["transform_metrics"])

    archive = (
        args.archive_path.resolve()
        if args.archive_path is not None
        else dataset_root.parent / config["dataset"]["archive_filename"]
    )
    if archive.stat().st_size != int(config["dataset"]["archive_bytes"]):
        raise ValueError("Archive byte count mismatch")
    archive_hash = common.sha256_file(archive)
    if archive_hash != config["dataset"]["archive_sha256"]:
        raise ValueError("Archive SHA-256 mismatch")

    inventory = read_csv(manifest_paths["release_inventory"])
    inventory_paths = [row["relative_path"] for row in inventory]
    if len(inventory_paths) != len(set(inventory_paths)):
        raise ValueError("Duplicate release-inventory paths")
    actual_paths = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*")
        if path.is_file()
    )
    if inventory_paths != actual_paths:
        raise ValueError("Release inventory path set differs from extracted dataset")
    inventory_by_path = {row["relative_path"]: row for row in inventory}
    for index, row in enumerate(inventory):
        path = dataset_root / row["relative_path"]
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Size mismatch for {path}")
        if common.sha256_file(path) != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        if (index + 1) % 300 == 0:
            print(json.dumps({"inventory_verified": index + 1, "total": len(inventory)}), flush=True)

    pairs = read_csv(manifest_paths["pairs"])
    samples = read_csv(manifest_paths["samples"])
    pair_ids = [row["pair_id"] for row in pairs]
    sample_ids = [row["sample_id"] for row in samples]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair IDs")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs")
    pair_by_id = {row["pair_id"]: row for row in pairs}
    if set(row["pair_id"] for row in samples) != set(pair_by_id):
        raise ValueError("Sample and pair manifests disagree on pair IDs")

    raw_pairs: dict[str, dict] = {}
    raw_sources: dict[tuple[str, str], dict] = {}
    metadata_names: dict[str, set[str]] = {}
    for split in ("Debug", "Main"):
        frame = __import__("pandas").read_csv(
            dataset_root / split / "metadata.csv", dtype=str, keep_default_na=False
        )
        metadata_names[split] = set(frame["name"])
        for record in frame.to_dict("records"):
            raw_sources[(split, record["name"])] = record
        for (scene, member), group in frame.groupby(
            ["SceneIndex", frame["type"].map(pair_member)], sort=True
        ):
            records = group.to_dict("records")
            pair_id = f"{split}:{scene}/{member}"
            if len(records) != 2:
                raise ValueError(f"Raw pair {pair_id} has {len(records)} members")
            by_label = {physical_label(record["type"]): record for record in records}
            if set(by_label) != {"possible", "impossible"}:
                raise ValueError(f"Raw pair {pair_id} labels are {sorted(by_label)}")
            for field in ("condition", "Difficulty", "Camera"):
                if len({record[field] for record in records}) != 1:
                    raise ValueError(f"Raw pair {pair_id} differs on {field}")
            raw_pairs[pair_id] = {
                "dataset_split": split,
                "pair_id": pair_id,
                "scene_id": str(scene),
                "pair_member": str(member),
                "condition": records[0]["condition"],
                "difficulty": records[0]["Difficulty"],
                "camera": records[0]["Camera"],
                "possible_name": by_label["possible"]["name"],
                "possible_relative_path": f"{split}/{by_label['possible']['file_name']}",
                "impossible_name": by_label["impossible"]["name"],
                "impossible_relative_path": f"{split}/{by_label['impossible']['file_name']}",
            }
    if set(pair_by_id) != set(raw_pairs):
        raise ValueError("Pair manifest IDs differ from independently reconstructed metadata pairs")

    inspected_manifests = {
        key: json.loads(path.read_text()) for key, path in inspected_paths.items()
    }
    inspected_names_by_key = {
        key: set(map(str, manifest["selected_names"]))
        for key, manifest in inspected_manifests.items()
    }
    inspected_names = set().union(*inspected_names_by_key.values())
    if not inspected_names <= metadata_names["Main"]:
        raise ValueError("Inspected manifest contains names outside raw Main metadata")
    independently_inspected_pairs_by_key = {}
    for key, names in inspected_names_by_key.items():
        inspected_pairs = {
            pair_id
            for pair_id, raw in raw_pairs.items()
            if raw["dataset_split"] == "Main"
            and {raw["possible_name"], raw["impossible_name"]} <= names
        }
        inspected_names_recovered = {
            name
            for pair_id in inspected_pairs
            for name in (
                raw_pairs[pair_id]["possible_name"],
                raw_pairs[pair_id]["impossible_name"],
            )
        }
        expected_pairs = int(inspected_manifests[key]["selected_pairs"])
        if inspected_names_recovered != names or len(inspected_pairs) != expected_pairs:
            raise ValueError(f"{key} manifest is not exactly complete raw pairs")
        independently_inspected_pairs_by_key[key] = inspected_pairs
    independently_inspected_pairs = set().union(*independently_inspected_pairs_by_key.values())
    independently_inspected_scenes = {
        raw_pairs[pair_id]["scene_id"] for pair_id in independently_inspected_pairs
    }
    source_id_collisions = metadata_names["Debug"] & metadata_names["Main"]
    source_id_collision_scenes = {
        record["SceneIndex"]
        for (split, name), record in raw_sources.items()
        if split == "Main" and name in source_id_collisions
    }

    for pair_id, row in pair_by_id.items():
        raw = raw_pairs[pair_id]
        for field in (
            "dataset_split",
            "pair_id",
            "scene_id",
            "pair_member",
            "condition",
            "difficulty",
            "camera",
            "possible_name",
            "possible_relative_path",
            "impossible_name",
            "impossible_relative_path",
        ):
            if row[field] != raw[field]:
                raise ValueError(f"Pair {pair_id} differs from raw metadata on {field}")
        expected_inspected_pair = pair_id in independently_inspected_pairs
        expected_inspected_scene = (
            raw["dataset_split"] == "Main" and raw["scene_id"] in independently_inspected_scenes
        )
        expected_collision_scene = (
            raw["dataset_split"] == "Main" and raw["scene_id"] in source_id_collision_scenes
        )
        if bool_value(row["inspected_pair"]) != expected_inspected_pair:
            raise ValueError(f"Pair {pair_id} has incorrect inspected_pair")
        if bool_value(row["inspected_scene"]) != expected_inspected_scene:
            raise ValueError(f"Pair {pair_id} has incorrect inspected_scene")
        if bool_value(row["debug_source_id_overlap_scene"]) != expected_collision_scene:
            raise ValueError(f"Pair {pair_id} has incorrect Debug source-ID overlap")
        if raw["dataset_split"] == "Debug":
            expected_partitions = {"debug"}
        elif expected_inspected_scene:
            expected_partitions = {"excluded_inspected_scene"}
        elif expected_collision_scene:
            expected_partitions = {"excluded_debug_source_id_overlap"}
        else:
            expected_partitions = {"development", "test"}
        if row["partition"] not in expected_partitions:
            raise ValueError(f"Pair {pair_id} has invalid partition {row['partition']}")

    expected_conditions = [row["name"] for row in config["appearance_conditions"]]
    samples_by_pair: dict[str, list[dict]] = defaultdict(list)
    samples_by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in samples:
        samples_by_pair[row["pair_id"]].append(row)
        samples_by_source[(row["dataset_split"], row["name"])].append(row)
        source_key = (row["dataset_split"], row["name"])
        if source_key not in raw_sources:
            raise ValueError(f"Unknown raw metadata source {source_key}")
        raw_source = raw_sources[source_key]
        expected_source_fields = {
            "scene_id": raw_source["SceneIndex"],
            "pair_member": pair_member(raw_source["type"]),
            "physical_label": physical_label(raw_source["type"]),
            "physical_label_value": "1" if physical_label(raw_source["type"]) == "impossible" else "0",
            "source_relative_path": f"{row['dataset_split']}/{raw_source['file_name']}",
            "condition": raw_source["condition"],
            "difficulty": raw_source["Difficulty"],
            "camera": raw_source["Camera"],
        }
        for field, expected in expected_source_fields.items():
            if row[field] != expected:
                raise ValueError(f"Sample {row['sample_id']} differs from raw metadata on {field}")
        if row["source_relative_path"] not in inventory_by_path:
            raise ValueError(f"Unknown source {row['source_relative_path']}")
        if row["source_sha256"] != inventory_by_path[row["source_relative_path"]]["sha256"]:
            raise ValueError(f"Source hash mismatch in sample {row['sample_id']}")
        indices = json.loads(row["source_frame_indices_0based"])
        if len(indices) != int(config["sampling"]["source_frames"]):
            raise ValueError(f"Frame budget mismatch in {row['sample_id']}")
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError(f"Frame indices are not sorted and unique in {row['sample_id']}")
        if indices[0] != 0 or indices[-1] != int(row["source_frame_count"]) - 1:
            raise ValueError(f"Frame endpoints are not covered in {row['sample_id']}")
        if bool_value(row["transform_feasible"]) is False:
            if row["transform_family"] != "selective_recolor":
                raise ValueError("Only selective recoloring may be infeasible")
            if row["infeasible_execution"] != "exact_identity":
                raise ValueError("Infeasible transform does not declare exact identity")

    for pair_id, rows in samples_by_pair.items():
        if len(rows) != 2 * len(expected_conditions):
            raise ValueError(f"Pair {pair_id} has {len(rows)} sample rows")
        labels = Counter(row["physical_label"] for row in rows)
        if labels != {"possible": len(expected_conditions), "impossible": len(expected_conditions)}:
            raise ValueError(f"Pair {pair_id} label counts are {labels}")
        for name in {row["name"] for row in rows}:
            conditions = sorted(
                row["appearance_condition"] for row in rows if row["name"] == name
            )
            if conditions != sorted(expected_conditions):
                raise ValueError(f"Source {name} conditions are {conditions}")
        pair = pair_by_id[pair_id]
        fields = ("dataset_split", "partition", "scene_id", "pair_member")
        for field in fields:
            if {row[field] for row in rows} != {pair[field]}:
                raise ValueError(f"Pair {pair_id} differs from samples on {field}")

    for source, rows in samples_by_source.items():
        frame_mappings = {row["source_frame_indices_0based"] for row in rows}
        source_hashes = {row["source_sha256"] for row in rows}
        partitions = {row["partition"] for row in rows}
        if len(frame_mappings) != 1 or len(source_hashes) != 1 or len(partitions) != 1:
            raise ValueError(f"Appearance rows are misaligned for source {source}")
        actual_frame_count = video_frame_count(dataset_root / rows[0]["source_relative_path"])
        if {int(row["source_frame_count"]) for row in rows} != {actual_frame_count}:
            raise ValueError(f"Frame count differs from raw video for source {source}")
        indices = json.loads(rows[0]["source_frame_indices_0based"])
        expected_indices = np.rint(np.linspace(0, actual_frame_count - 1, 64)).astype(int).tolist()
        if indices != expected_indices:
            raise ValueError(f"Source-frame mapping was not independently reproduced for {source}")

    chunks = config["sampling"]["videomae2_native_chunks_in_sample_positions"]
    chunk_positions = [position for start, stop in chunks for position in range(start, stop)]
    if chunk_positions != list(range(int(config["sampling"]["source_frames"]))):
        raise ValueError("VideoMAE2 chunks do not cover each shared source-frame position once")
    native_grid = tuple(config["cross_model_contract"]["native_token_grid_after_concatenation"])
    coarse_grid = tuple(config["output_views"]["coarse_tokens"]["shape"])
    if any(native % coarse for native, coarse in zip(native_grid, coarse_grid)):
        raise ValueError("Coarse-token grid does not divide the native token grid")

    main_pairs = [row for row in pairs if row["dataset_split"] == "Main"]
    debug_pairs = [row for row in pairs if row["dataset_split"] == "Debug"]
    if len(main_pairs) != 506 or len(debug_pairs) != 30:
        raise ValueError(f"Unexpected pair counts: Main={len(main_pairs)}, Debug={len(debug_pairs)}")
    if {
        row["pair_id"] for row in main_pairs if bool_value(row["inspected_pair"])
    } != independently_inspected_pairs:
        raise ValueError("Pair-manifest inspected membership differs from raw reconstruction")

    scenes_by_partition: dict[str, set[str]] = defaultdict(set)
    for row in main_pairs:
        scenes_by_partition[row["partition"]].add(row["scene_id"])
        if row["partition"] in {"development", "test"}:
            if bool_value(row["inspected_scene"]) or bool_value(row["debug_source_id_overlap_scene"]):
                raise ValueError(f"Contaminated scene entered {row['partition']}: {row['scene_id']}")
            if bool_value(row["inspected_pair"]):
                raise ValueError(f"Inspected pair entered {row['partition']}: {row['pair_id']}")
    partition_names = list(scenes_by_partition)
    for i, first in enumerate(partition_names):
        for second in partition_names[i + 1 :]:
            overlap = scenes_by_partition[first] & scenes_by_partition[second]
            if overlap:
                raise ValueError(f"Scene overlap between {first} and {second}: {sorted(overlap)}")
    fresh_scene_count = len(scenes_by_partition["development"]) + len(scenes_by_partition["test"])
    expected_development_scenes = round(
        fresh_scene_count * float(config["partition"]["development_fraction_of_fresh_scenes"])
    )
    if len(scenes_by_partition["development"]) != expected_development_scenes:
        raise ValueError(
            f"Expected {expected_development_scenes} development scenes, found "
            f"{len(scenes_by_partition['development'])}"
        )

    effect_rows: list[dict] = []
    reused_transform_validation: dict | None = None
    reused_transform_validation_sha256: str | None = None
    audit_partitions = {
        "active": {"debug", "development", "test"},
        "all": set(scenes_by_partition) | {"debug"},
        "none": set(),
        "reuse": {"debug", "development", "test"},
    }[args.transform_audit]
    sources_to_audit = [
        (source, rows)
        for source, rows in sorted(samples_by_source.items())
        if rows[0]["partition"] in audit_partitions
    ]
    if args.transform_audit == "reuse":
        provenance = config["transform_audit_provenance"]
        full_validation_path = resolve(repo_root, provenance["validation"])
        if not full_validation_path.exists() or not transform_metrics_path.exists():
            raise FileNotFoundError(
                "Reuse requires the retained full-audit validation and transform metrics"
            )
        reused_transform_validation_sha256 = common.sha256_text_file(full_validation_path)
        full_validation = common.validate_retained_transform_audit(
            full_validation_path,
            provenance["validation_sha256"],
            transform_metrics_path,
        )
        reused_transform_validation = full_validation
        relative_config = config_path.relative_to(repo_root).as_posix()
        audited_config_bytes = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{provenance['config_revision']}:{relative_config}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(
            audited_config_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest() != full_validation.get(
            "config_sha256"
        ):
            raise ValueError("Retained full audit is not tied to its declared config revision")
        audited_config = json.loads(audited_config_bytes)
        transform_contract = {
            "appearance_conditions": config["appearance_conditions"],
            "selective_recolor": config["selective_recolor"],
            "transform_gates": config["transform_gates"],
            "source_frames": config["sampling"]["source_frames"],
        }
        audited_transform_contract = {
            "appearance_conditions": audited_config["appearance_conditions"],
            "selective_recolor": audited_config["selective_recolor"],
            "transform_gates": audited_config["transform_gates"],
            "source_frames": audited_config["sampling"]["source_frames"],
        }
        if transform_contract != audited_transform_contract:
            raise ValueError("Transform contract changed since the full audit")
        metrics = read_csv(transform_metrics_path)
        active_samples = {
            row["sample_id"]: row
            for row in samples
            if row["partition"] in {"debug", "development", "test"}
        }
        if {row["sample_id"] for row in metrics} != set(active_samples):
            raise ValueError("Transform metrics do not exactly cover active locked samples")
        for metric in metrics:
            sample = active_samples[metric["sample_id"]]
            for field in (
                "sample_id",
                "dataset_split",
                "partition",
                "pair_id",
                "name",
                "appearance_condition",
            ):
                metric_field = "condition" if field == "appearance_condition" else field
                if metric[metric_field] != sample[field]:
                    raise ValueError(f"Reused transform metric differs on {field}")
            values = {
                "sample_id": metric["sample_id"],
                "dataset_split": metric["dataset_split"],
                "partition": metric["partition"],
                "pair_id": metric["pair_id"],
                "name": metric["name"],
                "condition": metric["condition"],
                "feasible": metric["feasible"] == "True",
                "identity": metric["identity"] == "True",
                "rgb_l1": float(metric["rgb_l1"]),
                "affected": float(metric["affected"]),
                "luma_std_ratio": float(metric["luma_std_ratio"]),
                "additional_zero_luma_fraction": float(
                    metric["additional_zero_luma_fraction"]
                ),
            }
            numeric = [
                values[key]
                for key in (
                    "rgb_l1",
                    "affected",
                    "luma_std_ratio",
                    "additional_zero_luma_fraction",
                )
            ]
            if not np.isfinite(numeric).all():
                raise ValueError("Reused transform metric contains non-finite values")
            if values["feasible"] != bool_value(sample["transform_feasible"]):
                raise ValueError("Reused transform feasibility differs from manifest")
            if sample["transform_family"] == "brightness":
                gates = config["transform_gates"]
                if values["luma_std_ratio"] < float(
                    gates["brightness_minimum_luma_standard_deviation_ratio"]
                ) or values["additional_zero_luma_fraction"] > float(
                    gates["brightness_maximum_additional_zero_luma_fraction"]
                ):
                    raise ValueError("Reused brightness metric fails locked gates")
            effect_rows.append(values)
        sources_to_audit = []
    for index, (_source, rows) in enumerate(sources_to_audit):
        by_condition = {row["appearance_condition"]: row for row in rows}
        clean = by_condition["clean"]
        indices = json.loads(clean["source_frame_indices_0based"])
        frames = common.decode_selected_frames(
            dataset_root / clean["source_relative_path"], indices
        )
        if frames.shape[0] != int(config["sampling"]["source_frames"]):
            raise ValueError(f"Decoded shape mismatch for {clean['name']}: {frames.shape}")
        for condition in expected_conditions:
            row = by_condition[condition]
            transformed = common.apply_appearance(frames, row, config)
            if transformed.shape != frames.shape or transformed.dtype != np.uint8:
                raise ValueError(f"Transform output mismatch for {row['sample_id']}")
            difference = np.abs(transformed.astype(np.int16) - frames.astype(np.int16))
            rgb_l1 = float(difference.mean() / 255.0)
            affected = float(np.any(difference > 0, axis=-1).mean())
            identity = bool(np.array_equal(transformed, frames))
            feasible = bool_value(row["transform_feasible"])
            clean_luma = np.tensordot(
                frames.astype(np.float32), np.asarray([0.299, 0.587, 0.114]), axes=([-1], [0])
            )
            transformed_luma = np.tensordot(
                transformed.astype(np.float32),
                np.asarray([0.299, 0.587, 0.114]),
                axes=([-1], [0]),
            )
            clean_std = float(clean_luma.std())
            transformed_std = float(transformed_luma.std())
            if clean_std == 0:
                luma_std_ratio = 1.0 if transformed_std == 0 else float("inf")
            else:
                luma_std_ratio = transformed_std / clean_std
            additional_zero_luma = max(
                0.0,
                float((transformed_luma < 0.5).mean() - (clean_luma < 0.5).mean()),
            )
            if condition == "clean" and not identity:
                raise ValueError("Clean transformation changed bytes")
            if feasible and condition != "clean" and identity:
                raise ValueError(f"Feasible transform is identity for {row['sample_id']}")
            if not feasible and not identity:
                raise ValueError(f"Infeasible transform changed bytes for {row['sample_id']}")
            if row["transform_family"] == "brightness":
                gates = config["transform_gates"]
                if luma_std_ratio < float(gates["brightness_minimum_luma_standard_deviation_ratio"]):
                    raise ValueError(f"Brightness contrast-retention gate failed for {row['sample_id']}")
                if additional_zero_luma > float(gates["brightness_maximum_additional_zero_luma_fraction"]):
                    raise ValueError(f"Brightness zero-luma gate failed for {row['sample_id']}")
            effect_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset_split": row["dataset_split"],
                    "partition": row["partition"],
                    "pair_id": row["pair_id"],
                    "name": row["name"],
                    "condition": condition,
                    "feasible": feasible,
                    "identity": identity,
                    "rgb_l1": rgb_l1,
                    "affected": affected,
                    "luma_std_ratio": luma_std_ratio,
                    "additional_zero_luma_fraction": additional_zero_luma,
                }
            )
        if (index + 1) % 25 == 0:
            print(
                json.dumps(
                    {"transform_sources_verified": index + 1, "total": len(sources_to_audit)}
                ),
                flush=True,
            )

    if effect_rows and args.transform_audit != "reuse":
        write_csv(transform_metrics_path, effect_rows)
    pair_counts = Counter((row["dataset_split"], row["partition"]) for row in pairs)
    validation = {
        "schema_version": 1,
        "status": "passed",
        "config": str(args.config),
        "config_sha256": common.sha256_text_file(config_path),
        "model_config_sha256s": {
            key: common.sha256_text_file(path)
            for key, path in sorted(model_config_paths.items())
        },
        "dataset_root": str(dataset_root),
        "archive_sha256": archive_hash,
        "release_files": len(inventory),
        "release_bytes": sum(int(row["size_bytes"]) for row in inventory),
        "inventory_sha256": common.sha256_text_file(manifest_paths["release_inventory"]),
        "pairs_manifest_sha256": common.sha256_text_file(manifest_paths["pairs"]),
        "samples_manifest_sha256": common.sha256_text_file(manifest_paths["samples"]),
        "transform_metrics_sha256": (
            common.sha256_text_file(transform_metrics_path) if effect_rows else None
        ),
        "pairs": len(pairs),
        "samples": len(samples),
        "pair_counts": {
            f"{split}:{partition}": count
            for (split, partition), count in sorted(pair_counts.items())
        },
        "scene_counts": {
            partition: len(scenes) for partition, scenes in sorted(scenes_by_partition.items())
        },
        "inspected_pairs": len(independently_inspected_pairs),
        "inspected_manifests": {
            key: {
                "pairs": len(pairs),
                "scenes": len({raw_pairs[pair_id]["scene_id"] for pair_id in pairs}),
            }
            for key, pairs in independently_inspected_pairs_by_key.items()
        },
        "inspected_scenes": len(
            {row["scene_id"] for row in main_pairs if bool_value(row["inspected_scene"])}
        ),
        "debug_source_id_collisions": len(source_id_collisions),
        "debug_source_id_collision_equal_file_hashes": sum(
            inventory_by_path[f"Debug/{raw_sources[('Debug', name)]['file_name']}"]["sha256"]
            == inventory_by_path[f"Main/{raw_sources[('Main', name)]['file_name']}"]["sha256"]
            for name in source_id_collisions
        ),
        "selective_infeasible_pairs": sum(
            not bool_value(row["selective_transform_feasible"]) for row in pairs
        ),
        "transform_audit": args.transform_audit,
        "transform_sources_verified": (
            reused_transform_validation["transform_sources_verified"]
            if reused_transform_validation is not None
            else len(sources_to_audit)
        ),
        "transform_metrics_reused_from_validation_sha256": reused_transform_validation_sha256,
        "transform_effects": aggregate_effects(effect_rows) if effect_rows else {},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "decoder": "decord if available, otherwise PyAV",
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "This validates data, partition, frame, and transform alignment only; it exposes no model features or confirmatory test result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(validation, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
