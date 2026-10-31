from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None
    import msvcrt
else:
    msvcrt = None

try:
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import intphys2_confirmatory_common as common


FeatureFunction = Callable[[list[np.ndarray]], tuple[np.ndarray, np.ndarray]]
ARCHIVE_REVISION = "anonymous-source-snapshot"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shared-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("debug", "development", "test"), required=True
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repeat-first", action="store_true")
    parser.add_argument("--overwrite-corrupt", action="store_true")
    parser.add_argument("--analysis-spec", type=Path)
    parser.add_argument("--release-tag")


def require_archive_file(archive_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(archive_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Executed contract file is outside the archive: {resolved}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative


def exact_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


def acquire_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, OSError) as error:
        handle.close()
        raise RuntimeError(f"Another extraction process holds {path}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "command": exact_command()}) + "\n")
    handle.flush()
    return handle


def release_process_lock(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    else:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    handle.close()


def write_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def fixed_output_dir(shared: dict, model_key: str, partition: str) -> Path:
    artifact_variable = shared["storage"]["artifact_root_environment_variable"]
    artifact_root = os.environ.get(artifact_variable)
    if not artifact_root:
        raise ValueError(f"Set {artifact_variable} to the external artifact root")
    return Path(artifact_root).resolve() / shared["run_id"] / model_key / partition


def output_process_locked(function):
    @functools.wraps(function)
    def wrapped(args, *, model_key: str, **kwargs):
        shared = json.loads(args.shared_config.resolve().read_text())
        output_dir = fixed_output_dir(shared, model_key, args.partition)
        process_lock = acquire_process_lock(output_dir / ".extraction.lock")
        try:
            return function(args, model_key=model_key, **kwargs)
        finally:
            release_process_lock(process_lock)

    return wrapped


def reserve_external_test(
    shared: dict,
    *,
    model_key: str,
    output_dir: Path,
    lock_identity: dict,
) -> dict:

    local_path = output_dir / "test_reservation.json"
    remote_prefix = (
        shared["storage"]["external_prefix"].rstrip("/")
        + f"/{model_key}/test/reservations"
    )
    readback = output_dir / ".test-reservation-readback.json"
    created = False
    if local_path.exists():
        reservation = json.loads(local_path.read_text())
        for key, value in lock_identity.items():
            if reservation.get(key) != value:
                raise ValueError(
                    "Local sealed-test reservation differs from this invocation"
                )
    else:
        reservation_uuid = str(uuid.uuid4())
        remote_uri = f"{remote_prefix}/{reservation_uuid}.json"
        reservation = {
            **lock_identity,
            "reservation_uuid": reservation_uuid,
            "remote_uri": remote_uri,
        }
        write_atomic_text(local_path, json.dumps(reservation, indent=2) + "\n")
        subprocess.run(
            ["s5cmd", "cp", "--no-clobber", str(local_path), remote_uri],
            check=True,
        )
        created = True

    remote_uri = reservation["remote_uri"]
    result = subprocess.run(
        ["s5cmd", "cp", remote_uri, str(readback)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "External sealed-test reservation is missing or inaccessible"
        )
    if readback.read_bytes() != local_path.read_bytes():
        raise ValueError(
            "External sealed-test reservation differs from local reservation"
        )
    readback.unlink()

    if created:
        time.sleep(5)
    listing = subprocess.run(
        ["s5cmd", "ls", f"{remote_prefix}/*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    candidates = []
    for line in listing:
        parts = line.split(maxsplit=3)
        if len(parts) != 4 or not parts[3].endswith(".json"):
            raise ValueError(f"Malformed sealed-test reservation listing: {line!r}")
        timestamp = time.strptime(f"{parts[0]} {parts[1]}", "%Y/%m/%d %H:%M:%S")
        candidates.append((timestamp, parts[3]))
    if not candidates:
        raise RuntimeError("No external sealed-test reservation candidates found")
    winner = min(candidates)
    local_key = Path(remote_uri).name
    if Path(winner[1]).name != local_key:
        raise ValueError(f"Sealed test is owned by an earlier reservation: {winner[1]}")
    reservation["election"] = {
        "rule": "minimum R2 LastModified second, then lexicographic object key",
        "candidate_count": len(candidates),
        "winner_key": winner[1],
    }
    return reservation


def contract_paths(repo_root: Path, shared: dict) -> dict[str, Path]:
    values = {
        "samples": shared["manifests"]["samples"],
        "validation": shared["manifests"]["validation"],
        "transform_metrics": shared["manifests"]["transform_metrics"],
        "transform_full_audit": shared["transform_audit_provenance"]["validation"],
    }
    return {
        key: path if (path := Path(value)).is_absolute() else repo_root / path
        for key, value in values.items()
    }


def load_locked_contract(
    args: argparse.Namespace, model_key: str
) -> tuple[dict, dict, list[dict]]:
    repo_root = Path(__file__).resolve().parents[1]
    shared_path = args.shared_config.resolve()
    model_path = args.model_config.resolve()
    shared = json.loads(shared_path.read_text())
    model = json.loads(model_path.read_text())
    paths = contract_paths(repo_root, shared)
    validation = json.loads(paths["validation"].read_text())
    if validation["status"] != "passed":
        raise ValueError("Shared manifest validation did not pass")
    if common.sha256_text_file(shared_path) != validation["config_sha256"]:
        raise ValueError("Shared config changed after validation")
    if (
        common.sha256_text_file(model_path)
        != validation["model_config_sha256s"][model_key]
    ):
        raise ValueError("Model config changed after validation")
    if (
        common.sha256_text_file(paths["samples"])
        != validation["samples_manifest_sha256"]
    ):
        raise ValueError("Sample manifest changed after validation")
    full_audit_hash = common.sha256_text_file(paths["transform_full_audit"])
    common.validate_retained_transform_audit(
        paths["transform_full_audit"],
        shared["transform_audit_provenance"]["validation_sha256"],
        paths["transform_metrics"],
    )
    if (
        validation.get("transform_metrics_reused_from_validation_sha256")
        != full_audit_hash
    ):
        raise ValueError(
            "Current validation is not rooted in the retained full transform audit"
        )
    if args.partition == "debug" and not args.repeat_first:
        raise ValueError("Debug extraction requires --repeat-first")
    if args.partition == "test":
        if args.max_pairs is not None:
            raise ValueError("Sealed test extraction forbids --max-pairs")
        if args.analysis_spec is None or args.release_tag is None:
            raise ValueError(
                "Sealed test extraction requires --analysis-spec and --release-tag"
            )
        analysis_spec = args.analysis_spec.resolve()
        if not analysis_spec.exists():
            raise FileNotFoundError(analysis_spec)
        require_archive_file(repo_root, analysis_spec)
    rows = common.read_samples(paths["samples"], args.partition)
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("--max-pairs must be positive")
        selected_pairs = sorted({row["pair_id"] for row in rows})[: args.max_pairs]
        rows = [row for row in rows if row["pair_id"] in set(selected_pairs)]
    return shared, model, rows


def expected_shapes(shared: dict, model: dict) -> tuple[int, int, int, int]:
    layers = len(shared["relative_depths"])
    coarse_tokens = int(np.prod(shared["output_views"]["coarse_tokens"]["shape"]))
    feature_dim = int(model["embedding_dimension"])
    return layers, coarse_tokens, feature_dim, len(shared["appearance_conditions"])


def shard_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "shards" / f"{sample_id}.npz"


def validate_or_remove_existing(
    path: Path,
    row: dict,
    *,
    layers: int,
    coarse_tokens: int,
    feature_dim: int,
    expected_metadata: dict[str, object],
    expected_relative_depths: list[float],
    expected_native_layers: list[int],
    overwrite_corrupt: bool,
) -> bool:
    if not path.exists():
        return False
    try:
        common.validate_feature_shard(
            path,
            expected_sample_id=row["sample_id"],
            layers=layers,
            coarse_tokens=coarse_tokens,
            feature_dim=feature_dim,
            expected_metadata=expected_metadata,
            expected_relative_depths=expected_relative_depths,
            expected_native_layers=expected_native_layers,
        )
        return True
    except Exception:
        if not overwrite_corrupt:
            raise
        path.unlink()
        return False


def metadata_for_shard(
    row: dict,
    *,
    model: dict,
    shared_hash: str,
    model_hash: str,
    checkpoint_hash: str,
    samples_manifest_hash: str,
    project_revision: str,
) -> dict:
    return {
        **row,
        "model_id": model["model_id"],
        "checkpoint_sha256": checkpoint_hash,
        "samples_manifest_sha256": samples_manifest_hash,
        "shared_config_sha256": shared_hash,
        "model_config_sha256": model_hash,
        "project_revision": project_revision,
    }


def write_feature_shard(
    path: Path,
    row: dict,
    pooled: np.ndarray,
    coarse_tokens: np.ndarray,
    *,
    shared: dict,
    model: dict,
    shared_hash: str,
    model_hash: str,
    checkpoint_hash: str,
    samples_manifest_hash: str,
    project_revision: str,
) -> str:
    metadata = metadata_for_shard(
        row,
        model=model,
        shared_hash=shared_hash,
        model_hash=model_hash,
        checkpoint_hash=checkpoint_hash,
        samples_manifest_hash=samples_manifest_hash,
        project_revision=project_revision,
    )
    arrays = {
        "pooled": pooled.astype(np.float16),
        "coarse_tokens": coarse_tokens.astype(np.float16),
        "relative_depths": np.asarray(shared["relative_depths"], dtype=np.float32),
        "native_layers": np.asarray(
            [item["native_layer"] for item in model["relative_depth_mapping"]],
            dtype=np.int64,
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    }
    return common.write_atomic_npz(path, arrays)


def is_oom(error: BaseException) -> bool:
    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


@output_process_locked
def run_extraction(
    args: argparse.Namespace,
    *,
    model_key: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    infer_features: FeatureFunction,
    load_provenance: dict,
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    shared, model, rows = load_locked_contract(args, model_key)
    if checkpoint_hash != model["checkpoint_sha256"]:
        raise ValueError("Loaded checkpoint hash differs from locked model config")
    contract = contract_paths(repo_root, shared)
    required_archive_files = [
        require_archive_file(repo_root, Path(sys.argv[0])),
        require_archive_file(repo_root, Path(__file__)),
        require_archive_file(repo_root, Path(common.__file__)),
        require_archive_file(repo_root, args.shared_config),
        require_archive_file(repo_root, args.model_config),
        require_archive_file(repo_root, contract["samples"]),
        require_archive_file(repo_root, contract["validation"]),
        require_archive_file(repo_root, contract["transform_metrics"]),
        require_archive_file(repo_root, contract["transform_full_audit"]),
    ]
    project_revision = ARCHIVE_REVISION
    shared_hash = common.sha256_text_file(args.shared_config.resolve())
    model_hash = common.sha256_text_file(args.model_config.resolve())
    samples_manifest_hash = common.sha256_text_file(contract["samples"])
    layers, coarse_count, feature_dim, condition_count = expected_shapes(shared, model)
    native_layers = [item["native_layer"] for item in model["relative_depth_mapping"]]

    def expected_shard_metadata(row: dict) -> dict[str, object]:
        return metadata_for_shard(
            row,
            model=model,
            shared_hash=shared_hash,
            model_hash=model_hash,
            checkpoint_hash=checkpoint_hash,
            samples_manifest_hash=samples_manifest_hash,
            project_revision=project_revision,
        )

    if len(rows) % (2 * condition_count):
        raise ValueError("Selected rows are not pair complete")

    output_dir = fixed_output_dir(shared, model_key, args.partition)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    if args.partition == "test" and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "complete":
            raise ValueError("Sealed test extraction is already complete and immutable")
    test_reservation: dict | None = None
    if args.partition == "test":
        test_lock_path = output_dir / "test_lock.json"
        test_lock = {
            "schema_version": 1,
            "project_revision": project_revision,
            "release_tag": args.release_tag,
            "analysis_spec_sha256": common.sha256_text_file(
                args.analysis_spec.resolve()
            ),
            "shared_config_sha256": shared_hash,
            "model_config_sha256": model_hash,
            "checkpoint_sha256": checkpoint_hash,
            "samples_manifest_sha256": samples_manifest_hash,
            "sample_ids_sha256": hashlib.sha256(
                "\n".join(row["sample_id"] for row in rows).encode()
            ).hexdigest(),
            "samples": len(rows),
        }
        if test_lock_path.exists():
            if json.loads(test_lock_path.read_text()) != test_lock:
                raise ValueError(
                    "Existing sealed-test lock differs from this invocation"
                )
        else:
            if (output_dir / "shards").exists() and any(
                (output_dir / "shards").iterdir()
            ):
                raise ValueError("Sealed-test shards exist without a test lock")
            write_atomic_text(test_lock_path, json.dumps(test_lock, indent=2) + "\n")
        test_reservation = reserve_external_test(
            shared,
            model_key=model_key,
            output_dir=output_dir,
            lock_identity=test_lock,
        )
    expected_shards = {
        shard_path(output_dir, row["sample_id"]).resolve() for row in rows
    }
    shard_directory = output_dir / "shards"
    actual_existing_shards = (
        {path.resolve() for path in shard_directory.glob("*.npz")}
        if shard_directory.exists()
        else set()
    )
    extra_existing_shards = actual_existing_shards - expected_shards
    if extra_existing_shards:
        raise ValueError(
            f"Unexpected stale feature shards: {sorted(map(str, extra_existing_shards))[:5]}"
        )
    rows_by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_source[(row["dataset_split"], row["name"])].append(row)
    expected_conditions = [item["name"] for item in shared["appearance_conditions"]]
    for source_rows in rows_by_source.values():
        if sorted(row["appearance_condition"] for row in source_rows) != sorted(
            expected_conditions
        ):
            raise ValueError("A selected source lacks an appearance condition")

    initial_batch = args.batch_size or max(
        int(value) for value in model["batch_candidates"]
    )
    if initial_batch not in [int(value) for value in model["batch_candidates"]]:
        raise ValueError("Requested batch size is outside locked candidates")
    batch_size = initial_batch
    oom_fallbacks: list[dict] = []
    determinism_max_abs: float | None = None
    completed = 0
    skipped = 0
    decode_seconds = 0.0
    inference_seconds = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    if args.repeat_first:
        repeat_row = min(
            (row for row in rows if row["appearance_condition"] == "clean"),
            key=lambda row: row["sample_id"],
        )
        repeat_source = args.dataset_root.resolve() / repeat_row["source_relative_path"]
        if common.sha256_file(repeat_source) != repeat_row["source_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {repeat_source}")
        repeat_decode_started = time.perf_counter()
        repeat_frames = common.decode_selected_frames(
            repeat_source, json.loads(repeat_row["source_frame_indices_0based"])
        )
        repeat_frames = common.apply_appearance(repeat_frames, repeat_row, shared)
        decode_seconds += time.perf_counter() - repeat_decode_started
        repeat_inference_started = time.perf_counter()
        first_pooled, first_coarse = infer_features([repeat_frames])
        second_pooled, second_coarse = infer_features([repeat_frames])
        inference_seconds += time.perf_counter() - repeat_inference_started
        expected_pooled_shape = (1, layers, feature_dim)
        expected_coarse_shape = (1, layers, coarse_count, feature_dim)
        for name, values, shape in (
            ("first pooled", first_pooled, expected_pooled_shape),
            ("second pooled", second_pooled, expected_pooled_shape),
            ("first coarse", first_coarse, expected_coarse_shape),
            ("second coarse", second_coarse, expected_coarse_shape),
        ):
            if values.shape != shape:
                raise ValueError(
                    f"Repeatability {name} shape {values.shape} != {shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Repeatability {name} contains non-finite values")
        determinism_max_abs = float(
            max(
                np.max(
                    np.abs(
                        first_pooled.astype(np.float32)
                        - second_pooled.astype(np.float32)
                    )
                ),
                np.max(
                    np.abs(
                        first_coarse.astype(np.float32)
                        - second_coarse.astype(np.float32)
                    )
                ),
            )
        )
        if determinism_max_abs > float(shared["determinism"]["absolute_tolerance"]):
            raise ValueError(f"Determinism tolerance failed: {determinism_max_abs}")

    for source_index, (_source, source_rows) in enumerate(
        sorted(rows_by_source.items())
    ):
        source_rows = sorted(
            source_rows,
            key=lambda row: expected_conditions.index(row["appearance_condition"]),
        )
        pending = []
        for row in source_rows:
            path = shard_path(output_dir, row["sample_id"])
            expected_metadata = expected_shard_metadata(row)
            if validate_or_remove_existing(
                path,
                row,
                layers=layers,
                coarse_tokens=coarse_count,
                feature_dim=feature_dim,
                expected_metadata=expected_metadata,
                expected_relative_depths=shared["relative_depths"],
                expected_native_layers=native_layers,
                overwrite_corrupt=args.overwrite_corrupt,
            ):
                skipped += 1
            else:
                pending.append(row)
        clean_row = source_rows[0]
        source_path = args.dataset_root.resolve() / clean_row["source_relative_path"]
        if common.sha256_file(source_path) != clean_row["source_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {source_path}")
        if not pending:
            continue

        decode_started = time.perf_counter()
        indices = json.loads(clean_row["source_frame_indices_0based"])
        frames = common.decode_selected_frames(source_path, indices)
        decode_seconds += time.perf_counter() - decode_started
        transformed = [common.apply_appearance(frames, row, shared) for row in pending]

        offset = 0
        while offset < len(pending):
            current_rows = pending[offset : offset + batch_size]
            current_frames = transformed[offset : offset + batch_size]
            inference_started = time.perf_counter()
            try:
                pooled, coarse = infer_features(current_frames)
            except BaseException as error:
                if not is_oom(error) or batch_size == min(model["batch_candidates"]):
                    raise
                next_batch = max(
                    int(value)
                    for value in model["batch_candidates"]
                    if int(value) < batch_size
                )
                oom_fallbacks.append(
                    {
                        "source_index": source_index,
                        "from_batch_size": batch_size,
                        "to_batch_size": next_batch,
                    }
                )
                batch_size = next_batch
                torch.cuda.empty_cache()
                continue
            inference_seconds += time.perf_counter() - inference_started
            if pooled.shape != (len(current_rows), layers, feature_dim):
                raise ValueError(f"Unexpected pooled shape {pooled.shape}")
            if coarse.shape != (len(current_rows), layers, coarse_count, feature_dim):
                raise ValueError(f"Unexpected coarse shape {coarse.shape}")
            if not np.isfinite(pooled).all() or not np.isfinite(coarse).all():
                raise ValueError("Non-finite extracted features")

            for item, row in enumerate(current_rows):
                path = shard_path(output_dir, row["sample_id"])
                write_feature_shard(
                    path,
                    row,
                    pooled[item],
                    coarse[item],
                    shared=shared,
                    model=model,
                    shared_hash=shared_hash,
                    model_hash=model_hash,
                    checkpoint_hash=checkpoint_hash,
                    samples_manifest_hash=samples_manifest_hash,
                    project_revision=project_revision,
                )
                common.validate_feature_shard(
                    path,
                    expected_sample_id=row["sample_id"],
                    layers=layers,
                    coarse_tokens=coarse_count,
                    feature_dim=feature_dim,
                    expected_metadata=expected_shard_metadata(row),
                    expected_relative_depths=shared["relative_depths"],
                    expected_native_layers=native_layers,
                )
                completed += 1
            offset += len(current_rows)
        print(
            json.dumps(
                {
                    "sources_done": source_index + 1,
                    "sources_total": len(rows_by_source),
                    "samples_written": completed,
                    "samples_skipped": skipped,
                    "batch_size": batch_size,
                }
            ),
            flush=True,
        )

    index_rows = []
    for row in rows:
        path = shard_path(output_dir, row["sample_id"])
        common.validate_feature_shard(
            path,
            expected_sample_id=row["sample_id"],
            layers=layers,
            coarse_tokens=coarse_count,
            feature_dim=feature_dim,
            expected_metadata=expected_shard_metadata(row),
            expected_relative_depths=shared["relative_depths"],
            expected_native_layers=native_layers,
        )
        index_rows.append(
            {
                **row,
                "model_id": model["model_id"],
                "checkpoint_sha256": checkpoint_hash,
                "samples_manifest_sha256": samples_manifest_hash,
                "shared_config_sha256": shared_hash,
                "model_config_sha256": model_hash,
                "project_revision": project_revision,
                "relative_depths": json.dumps(
                    shared["relative_depths"], separators=(",", ":")
                ),
                "native_layers": json.dumps(native_layers, separators=(",", ":")),
                "pooled_shape": json.dumps(
                    [layers, feature_dim], separators=(",", ":")
                ),
                "coarse_tokens_shape": json.dumps(
                    [layers, coarse_count, feature_dim], separators=(",", ":")
                ),
                "shard": path.relative_to(output_dir).as_posix(),
                "shard_bytes": path.stat().st_size,
                "shard_sha256": common.sha256_file(path),
            }
        )
    final_shards = {path.resolve() for path in shard_directory.glob("*.npz")}
    if final_shards != expected_shards:
        raise ValueError(
            f"Final shard set mismatch: missing={len(expected_shards - final_shards)}, "
            f"extra={len(final_shards - expected_shards)}"
        )
    index_path = output_dir / "index.csv"
    temporary_index = index_path.with_suffix(".csv.tmp")
    with temporary_index.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(index_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(index_rows)
    temporary_index.replace(index_path)

    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    write_atomic_text(output_dir / "environment.txt", environment)
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "-q"], check=True, capture_output=True, text=True
    ).stdout
    write_atomic_text(output_dir / "nvidia-smi-q.txt", nvidia_smi.rstrip() + "\n")
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "command": exact_command(),
        "project_revision": project_revision,
        "required_archive_files": required_archive_files,
        "artifact_root_environment_variable": shared["storage"][
            "artifact_root_environment_variable"
        ],
        "output_dir": str(output_dir),
        "script": str(Path(sys.argv[0]).resolve()),
        "script_sha256": common.sha256_file(Path(sys.argv[0]).resolve()),
        "runner_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "common_script_sha256": common.sha256_file(Path(common.__file__).resolve()),
        "shared_config": str(args.shared_config.resolve()),
        "shared_config_sha256": shared_hash,
        "model_config": str(args.model_config.resolve()),
        "model_config_sha256": model_hash,
        "samples_manifest_sha256": samples_manifest_hash,
        "dataset_archive_sha256": shared["dataset"]["archive_sha256"],
        "partition": args.partition,
        "max_pairs": args.max_pairs,
        "samples": len(rows),
        "pairs": len({row["pair_id"] for row in rows}),
        "source_hashes_verified": len(rows_by_source),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model": model,
        "initial_batch_size": initial_batch,
        "final_batch_size": batch_size,
        "oom_fallbacks": oom_fallbacks,
        "repeat_first": args.repeat_first,
        "determinism_max_abs": determinism_max_abs,
        "feature_storage_dtype": "float16",
        "index": str(index_path),
        "index_sha256": common.sha256_file(index_path),
        "shard_count": len(index_rows),
        "shard_logical_bytes": sum(int(row["shard_bytes"]) for row in index_rows),
        "decode_seconds": decode_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "release_tag": args.release_tag,
        "test_reservation": test_reservation,
        "analysis_spec": str(args.analysis_spec) if args.analysis_spec else None,
        "analysis_spec_sha256": (
            common.sha256_text_file(args.analysis_spec) if args.analysis_spec else None
        ),
        "load_provenance": load_provenance,
        "claim_boundary": "Feature extraction and alignment only; no probe or paper claim is evaluated here.",
    }
    write_atomic_text(
        output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n"
    )
    write_atomic_text(
        output_dir / "status.json",
        json.dumps(
            {
                "status": "complete",
                "samples": len(rows),
                "pairs": len({row["pair_id"] for row in rows}),
                "shards": len(index_rows),
                "index_sha256": provenance["index_sha256"],
            },
            indent=2,
        )
        + "\n",
    )
    return provenance
