from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

import numpy as np

try:
    import intphys2_confirmatory_common as common
    import intphys2_confirmatory_extraction as extraction
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common
    from scripts import intphys2_confirmatory_extraction as extraction


MODEL_SPECIFIC = {
    "model_id",
    "checkpoint_sha256",
    "model_config_sha256",
    "native_layers",
    "shard",
    "shard_bytes",
    "shard_sha256",
}
REQUIRED_SHARED_FIELDS = {
    "sample_id",
    "dataset_split",
    "partition",
    "pair_id",
    "scene_id",
    "physical_label",
    "name",
    "source_relative_path",
    "source_sha256",
    "source_frame_indices_0based",
    "appearance_condition",
    "transform_family",
    "transform_feasible",
    "samples_manifest_sha256",
    "shared_config_sha256",
    "project_revision",
    "relative_depths",
    "pooled_shape",
    "coarse_tokens_shape",
}
INDEX_ONLY = {
    "relative_depths",
    "native_layers",
    "pooled_shape",
    "coarse_tokens_shape",
    "shard",
    "shard_bytes",
    "shard_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-config", type=Path, required=True)
    parser.add_argument("--samples-manifest", type=Path, required=True)
    parser.add_argument("--vjepa-model-config", type=Path, required=True)
    parser.add_argument("--videomae-model-config", type=Path, required=True)
    parser.add_argument("--vjepa-index", type=Path, required=True)
    parser.add_argument("--vjepa-root", type=Path, required=True)
    parser.add_argument("--videomae-index", type=Path, required=True)
    parser.add_argument("--videomae-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_index(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return fields, by_id


def verify_shards(
    root: Path,
    rows: dict[str, dict[str, str]],
    *,
    shared: dict,
    shared_hash: str,
    model: dict,
    model_hash: str,
) -> int:
    logical_bytes = 0
    expected_paths = {(root / row["shard"]).resolve() for row in rows.values()}
    shard_directory = root / "shards"
    actual_paths = (
        {path.resolve() for path in shard_directory.glob("*.npz")}
        if shard_directory.exists()
        else set()
    )
    if actual_paths != expected_paths:
        raise ValueError(
            f"Shard file-set mismatch under {root}: "
            f"missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}"
        )
    relative_depths = shared["relative_depths"]
    native_layers = [item["native_layer"] for item in model["relative_depth_mapping"]]
    layers = len(relative_depths)
    coarse_count = int(np.prod(shared["output_views"]["coarse_tokens"]["shape"]))
    feature_dim = int(model["embedding_dimension"])
    for row in rows.values():
        if row["model_id"] != model["model_id"]:
            raise ValueError(f"Unexpected model ID in index: {row['model_id']}")
        if row["checkpoint_sha256"] != model["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity differs from model config")
        if row["shared_config_sha256"] != shared_hash:
            raise ValueError("Shared-config identity differs from index")
        if row["model_config_sha256"] != model_hash:
            raise ValueError("Model-config identity differs from index")
        if json.loads(row["relative_depths"]) != relative_depths:
            raise ValueError("Relative-depth array differs from shared config")
        if json.loads(row["native_layers"]) != native_layers:
            raise ValueError("Native layers differ from model config")
        if json.loads(row["pooled_shape"]) != [layers, feature_dim]:
            raise ValueError("Pooled shape differs from model config")
        if json.loads(row["coarse_tokens_shape"]) != [
            layers,
            coarse_count,
            feature_dim,
        ]:
            raise ValueError("Coarse-token shape differs from model config")
        path = root / row["shard"]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["shard_bytes"]):
            raise ValueError(f"Shard size mismatch: {path}")
        if common.sha256_file(path) != row["shard_sha256"]:
            raise ValueError(f"Shard SHA-256 mismatch: {path}")
        expected_metadata = {
            key: value for key, value in row.items() if key not in INDEX_ONLY
        }
        common.validate_feature_shard(
            path,
            expected_sample_id=row["sample_id"],
            layers=layers,
            coarse_tokens=coarse_count,
            feature_dim=feature_dim,
            expected_metadata=expected_metadata,
            expected_relative_depths=relative_depths,
            expected_native_layers=native_layers,
        )
        logical_bytes += path.stat().st_size
    return logical_bytes


def validate_index_alignment(
    vjepa_fields: list[str],
    vjepa: dict[str, dict[str, str]],
    videomae_fields: list[str],
    videomae: dict[str, dict[str, str]],
    expected_by_id: dict[str, dict[str, str]],
    samples_manifest_hash: str,
    project_revision: str,
) -> tuple[str, list[str]]:
    partitions = {row["partition"] for row in vjepa.values()} | {
        row["partition"] for row in videomae.values()
    }
    if len(partitions) != 1:
        raise ValueError(f"Expected one package partition, found {sorted(partitions)}")
    partition = next(iter(partitions))
    expected_ids = set(expected_by_id)
    if set(vjepa) != expected_ids or set(videomae) != expected_ids:
        raise ValueError(
            f"Package completeness mismatch against locked manifest: "
            f"expected={len(expected_ids)}, vjepa={len(vjepa)}, videomae={len(videomae)}"
        )
    for sample_id, expected in expected_by_id.items():
        for field, value in expected.items():
            if (
                vjepa[sample_id].get(field) != value
                or videomae[sample_id].get(field) != value
            ):
                raise ValueError(
                    f"Locked-manifest field mismatch for {sample_id} {field}"
                )
        if vjepa[sample_id]["samples_manifest_sha256"] != samples_manifest_hash:
            raise ValueError("V-JEPA index sample-manifest identity mismatch")
        if videomae[sample_id]["samples_manifest_sha256"] != samples_manifest_hash:
            raise ValueError("VideoMAE index sample-manifest identity mismatch")
        if vjepa[sample_id]["project_revision"] != project_revision:
            raise ValueError(
                "V-JEPA index was extracted from a different project revision"
            )
        if videomae[sample_id]["project_revision"] != project_revision:
            raise ValueError(
                "VideoMAE index was extracted from a different project revision"
            )
    if not REQUIRED_SHARED_FIELDS <= set(vjepa_fields):
        raise ValueError(
            f"Index lacks required shared fields: {sorted(REQUIRED_SHARED_FIELDS - set(vjepa_fields))}"
        )
    if set(vjepa_fields) != set(videomae_fields):
        raise ValueError(
            f"Index schema mismatch: V-JEPA-only={sorted(set(vjepa_fields) - set(videomae_fields))}, "
            f"VideoMAE-only={sorted(set(videomae_fields) - set(vjepa_fields))}"
        )
    shared_fields = sorted(set(vjepa_fields) - MODEL_SPECIFIC)
    mismatches = []
    for sample_id in sorted(vjepa):
        for field in shared_fields:
            if vjepa[sample_id][field] != videomae[sample_id][field]:
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "vjepa2": vjepa[sample_id][field],
                        "videomae2": videomae[sample_id][field],
                    }
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    if mismatches:
        raise ValueError(f"Cross-model metadata mismatch: {mismatches}")
    return partition, shared_fields


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    shared = json.loads(args.shared_config.read_text())
    contract = extraction.contract_paths(repo_root, shared)
    required_archive_files = [
        extraction.require_archive_file(repo_root, Path(__file__)),
        extraction.require_archive_file(repo_root, Path(common.__file__)),
        extraction.require_archive_file(repo_root, Path(extraction.__file__)),
        extraction.require_archive_file(repo_root, args.shared_config),
        extraction.require_archive_file(repo_root, args.samples_manifest),
        extraction.require_archive_file(repo_root, args.vjepa_model_config),
        extraction.require_archive_file(repo_root, args.videomae_model_config),
        extraction.require_archive_file(repo_root, contract["transform_metrics"]),
        extraction.require_archive_file(repo_root, contract["transform_full_audit"]),
    ]
    project_revision = extraction.ARCHIVE_REVISION
    vjepa_model = json.loads(args.vjepa_model_config.read_text())
    videomae_model = json.loads(args.videomae_model_config.read_text())
    shared_hash = common.sha256_text_file(args.shared_config)
    samples_manifest_hash = common.sha256_text_file(args.samples_manifest)
    vjepa_model_hash = common.sha256_text_file(args.vjepa_model_config)
    videomae_model_hash = common.sha256_text_file(args.videomae_model_config)
    vjepa_fields, vjepa = read_index(args.vjepa_index)
    videomae_fields, videomae = read_index(args.videomae_index)
    index_partitions = {row["partition"] for row in vjepa.values()} | {
        row["partition"] for row in videomae.values()
    }
    if len(index_partitions) != 1:
        raise ValueError(
            f"Expected one package partition, found {sorted(index_partitions)}"
        )
    expected_rows = common.read_samples(
        args.samples_manifest, next(iter(index_partitions))
    )
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    partition, shared_fields = validate_index_alignment(
        vjepa_fields,
        vjepa,
        videomae_fields,
        videomae,
        expected_by_id,
        samples_manifest_hash,
        project_revision,
    )
    vjepa_bytes = verify_shards(
        args.vjepa_root,
        vjepa,
        shared=shared,
        shared_hash=shared_hash,
        model=vjepa_model,
        model_hash=vjepa_model_hash,
    )
    videomae_bytes = verify_shards(
        args.videomae_root,
        videomae,
        shared=shared,
        shared_hash=shared_hash,
        model=videomae_model,
        model_hash=videomae_model_hash,
    )
    output = {
        "schema_version": 1,
        "status": "passed",
        "command": shlex.join([sys.executable, *sys.argv]),
        "project_revision": project_revision,
        "required_archive_files": required_archive_files,
        "script_sha256": common.sha256_file(Path(__file__).resolve()),
        "samples": len(vjepa),
        "pairs": len({row["pair_id"] for row in vjepa.values()}),
        "partitions": [partition],
        "conditions": sorted({row["appearance_condition"] for row in vjepa.values()}),
        "shared_fields_compared": shared_fields,
        "metadata_mismatches": 0,
        "missing_or_extra_samples": 0,
        "shared_config_sha256": shared_hash,
        "samples_manifest_sha256": samples_manifest_hash,
        "vjepa_model_config_sha256": vjepa_model_hash,
        "videomae_model_config_sha256": videomae_model_hash,
        "vjepa_index_sha256": common.sha256_file(args.vjepa_index),
        "videomae_index_sha256": common.sha256_file(args.videomae_index),
        "vjepa_shard_logical_bytes": vjepa_bytes,
        "videomae_shard_logical_bytes": videomae_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
