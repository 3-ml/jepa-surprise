import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import intphys2_confirmatory_common as common
from scripts import intphys2_confirmatory_extraction as extraction
from scripts import validate_intphys2_feature_parity as parity
from scripts.extract_videomae2_intphys2_layers import (
    VideoMAE2Extractor,
    validate_checkpoint_key_partition,
)
from scripts.extract_vjepa2_intphys2_layers import clean_state, normalize_batch


ROOT = Path(__file__).resolve().parents[1]


def test_locked_contract_selects_one_complete_debug_pair():
    args = argparse.Namespace(
        shared_config=ROOT / "configs/intphys2_confirmatory_shared.json",
        model_config=ROOT / "configs/vjepa2_intphys2_confirmatory.json",
        partition="debug",
        analysis_spec=None,
        release_tag=None,
        repeat_first=True,
        max_pairs=1,
    )
    shared, model, rows = extraction.load_locked_contract(args, "vjepa2")
    assert model["model_id"] == "vjepa2-g"
    assert len(rows) == 8
    assert len({row["pair_id"] for row in rows}) == 1
    assert {row["appearance_condition"] for row in rows} == {
        condition["name"] for condition in shared["appearance_conditions"]
    }


def test_debug_repeatability_and_sealed_test_guards_are_mandatory():
    base = {
        "shared_config": ROOT / "configs/intphys2_confirmatory_shared.json",
        "model_config": ROOT / "configs/vjepa2_intphys2_confirmatory.json",
        "analysis_spec": None,
        "release_tag": None,
    }
    with pytest.raises(ValueError, match="repeat-first"):
        extraction.load_locked_contract(
            argparse.Namespace(
                **base,
                partition="debug",
                repeat_first=False,
                max_pairs=1,
            ),
            "vjepa2",
        )
    with pytest.raises(ValueError, match="forbids --max-pairs"):
        extraction.load_locked_contract(
            argparse.Namespace(
                **base,
                partition="test",
                repeat_first=False,
                max_pairs=1,
            ),
            "vjepa2",
        )


def test_process_lock_excludes_same_root_and_releases(tmp_path):
    path = tmp_path / ".extraction.lock"
    first = extraction.acquire_process_lock(path)
    with pytest.raises(RuntimeError, match="Another extraction process"):
        extraction.acquire_process_lock(path)
    first.close()
    resumed = extraction.acquire_process_lock(path)
    resumed.close()


def test_output_process_lock_releases_after_error(tmp_path, monkeypatch):
    shared_path = tmp_path / "shared.json"
    shared_path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "storage": {"artifact_root_environment_variable": "TEST_FEATURE_ROOT"},
            }
        )
    )
    monkeypatch.setenv("TEST_FEATURE_ROOT", str(tmp_path / "features"))
    args = argparse.Namespace(shared_config=shared_path, partition="debug")

    @extraction.output_process_locked
    def fail(_args, *, model_key):
        raise ValueError(model_key)

    with pytest.raises(ValueError, match="vjepa2"):
        fail(args, model_key="vjepa2")
    lock_path = tmp_path / "features" / "run" / "vjepa2" / "debug" / ".extraction.lock"
    resumed = extraction.acquire_process_lock(lock_path)
    resumed.close()


def test_sealed_test_reservation_loses_to_earlier_immutable_candidate(
    tmp_path, monkeypatch
):
    state = {}

    def fake_run(command, **_kwargs):
        if command[:3] == ["s5cmd", "cp", "--no-clobber"]:
            state["local"] = Path(command[3])
            state["remote"] = command[4]
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["s5cmd", "cp"]:
            Path(command[3]).write_bytes(state["local"].read_bytes())
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["s5cmd", "ls"]:
            local_key = Path(state["remote"]).name
            listing = (
                "2026/08/23 10:00:00 100 earlier.json\n"
                f"2026/08/23 10:00:01 100 {local_key}\n"
            )
            return subprocess.CompletedProcess(command, 0, listing, "")
        raise AssertionError(command)

    monkeypatch.setattr(extraction.subprocess, "run", fake_run)
    monkeypatch.setattr(extraction.time, "sleep", lambda _seconds: None)
    with pytest.raises(ValueError, match="earlier reservation"):
        extraction.reserve_external_test(
            {"storage": {"external_prefix": "s3://bucket/run/"}},
            model_key="vjepa2",
            output_dir=tmp_path,
            lock_identity={"project_revision": "revision"},
        )


def test_retained_transform_audit_requires_exact_root_and_metrics(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("sample_id,rgb_l1\na,0.1\n")
    validation = tmp_path / "full-audit.json"
    validation.write_text(
        json.dumps(
            {
                "status": "passed",
                "transform_audit": "active",
                "transform_metrics_sha256": common.sha256_text_file(metrics),
            }
        )
        + "\n"
    )
    validation_hash = common.sha256_text_file(validation)
    assert (
        common.validate_retained_transform_audit(validation, validation_hash, metrics)[
            "status"
        ]
        == "passed"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        common.validate_retained_transform_audit(validation, "0" * 64, metrics)
    metrics.write_text("changed\n")
    with pytest.raises(ValueError, match="Transform metrics differ"):
        common.validate_retained_transform_audit(validation, validation_hash, metrics)


def test_feature_shard_validation_and_corruption(tmp_path):
    path = tmp_path / "sample.npz"
    arrays = {
        "pooled": np.zeros((5, 3), dtype=np.float16),
        "coarse_tokens": np.zeros((5, 64, 3), dtype=np.float16),
        "relative_depths": np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32),
        "native_layers": np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        "metadata_json": np.asarray(json.dumps({"sample_id": "sample"})),
    }
    common.write_atomic_npz(path, arrays)
    common.validate_feature_shard(
        path,
        expected_sample_id="sample",
        layers=5,
        coarse_tokens=64,
        feature_dim=3,
    )
    with pytest.raises(ValueError):
        common.validate_feature_shard(
            path,
            expected_sample_id="different",
            layers=5,
            coarse_tokens=64,
            feature_dim=3,
        )
    with pytest.raises(ValueError):
        common.validate_feature_shard(
            path,
            expected_sample_id="sample",
            layers=5,
            coarse_tokens=64,
            feature_dim=3,
            expected_metadata={"checkpoint_sha256": "different"},
        )


def test_model_specific_tensor_contracts():
    state = {"module.backbone.weight": torch.ones(1), "bias": torch.zeros(1)}
    assert set(clean_state(state)) == {"weight", "bias"}

    frames = [np.zeros((64, 8, 8, 3), dtype=np.uint8)]
    normalized = normalize_batch(frames, 16)
    assert normalized.shape == (1, 3, 64, 16, 16)
    assert torch.isfinite(normalized).all()

    extractor = object.__new__(VideoMAE2Extractor)
    extractor.native_chunk_shape = (8, 16, 16)
    extractor.concatenated_shape = (32, 16, 16)

    class PositionalProcessor:
        def __call__(self, chunks, **kwargs):
            assert len(chunks) == 4
            assert kwargs == {"return_tensors": "pt"}
            return {"pixel_values": torch.zeros(4, 16, 3, 224, 224)}

    extractor.processor = PositionalProcessor()
    assert extractor.preprocess([np.zeros((64, 8, 8, 3), dtype=np.uint8)]).shape == (
        4,
        3,
        16,
        224,
        224,
    )
    tokens = torch.stack(
        [torch.full((8 * 16 * 16, 1), float(chunk)) for chunk in range(4)]
    )
    concatenated = extractor._concatenate_chunks(tokens, batch=1)
    temporal = concatenated.reshape(1, 32, 16, 16, 1)[:, :, 0, 0, 0]
    assert temporal.tolist() == [[*[0.0] * 8, *[1.0] * 8, *[2.0] * 8, *[3.0] * 8]]


def test_videomae_checkpoint_partition_requires_exact_encoder_and_known_decoder():
    config = {
        "checkpoint_encoder_tensor_count": 2,
        "checkpoint_excluded_tensor_count": 2,
        "checkpoint_excluded_prefixes": ["model.decoder."],
        "checkpoint_excluded_keys": ["model.mask_token"],
    }
    expected = {"model.blocks.0.weight", "model.norm.weight"}
    checkpoint = expected | {"model.decoder.head.weight", "model.mask_token"}
    assert validate_checkpoint_key_partition(expected, checkpoint, config) == {
        "model.decoder.head.weight",
        "model.mask_token",
    }
    with pytest.raises(ValueError, match="lacks encoder tensors"):
        validate_checkpoint_key_partition(
            expected, checkpoint - {"model.norm.weight"}, config
        )
    with pytest.raises(ValueError, match="unexpected non-encoder"):
        validate_checkpoint_key_partition(
            expected, checkpoint | {"model.unknown.weight"}, config
        )


def _write_package(root: Path, shared_path: Path, model_path: Path, model_id: str):
    shared = json.loads(shared_path.read_text())
    model = {
        "model_id": model_id,
        "checkpoint_sha256": ("a" if model_id == "vjepa2" else "b") * 64,
        "embedding_dimension": 3,
        "relative_depth_mapping": [
            {"relative_depth": 0.0, "native_layer": 0},
            {"relative_depth": 1.0, "native_layer": 2},
        ],
    }
    model_path.write_text(json.dumps(model))
    shard = root / "shards" / "sample.npz"
    shard.parent.mkdir()
    metadata = {
        "sample_id": "sample",
        "dataset_split": "Debug",
        "partition": "debug",
        "pair_id": "Debug:0/1",
        "scene_id": "0",
        "physical_label": "possible",
        "name": "source",
        "source_relative_path": "Debug/Videos/source.mp4",
        "source_sha256": "c" * 64,
        "source_frame_indices_0based": "[0,1]",
        "appearance_condition": "clean",
        "transform_family": "clean",
        "transform_feasible": "true",
        "shared": "same",
        "model_id": model_id,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "samples_manifest_sha256": "d" * 64,
        "shared_config_sha256": common.sha256_file(shared_path),
        "model_config_sha256": common.sha256_file(model_path),
        "project_revision": "revision",
    }
    common.write_atomic_npz(
        shard,
        {
            "pooled": np.zeros((2, 3), dtype=np.float16),
            "coarse_tokens": np.zeros((2, 1, 3), dtype=np.float16),
            "relative_depths": np.asarray(shared["relative_depths"], dtype=np.float32),
            "native_layers": np.asarray([0, 2], dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(metadata)),
        },
    )
    row = {
        **metadata,
        "relative_depths": "[0.0,1.0]",
        "native_layers": "[0,2]",
        "pooled_shape": "[2,3]",
        "coarse_tokens_shape": "[2,1,3]",
        "shard": shard.relative_to(root).as_posix(),
        "shard_bytes": shard.stat().st_size,
        "shard_sha256": common.sha256_file(shard),
    }
    index = root / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return index


def test_cross_model_parity_checks_metadata_and_hashes(tmp_path):
    shared_path = tmp_path / "shared.json"
    shared_path.write_text(
        json.dumps(
            {
                "relative_depths": [0.0, 1.0],
                "output_views": {"coarse_tokens": {"shape": [1, 1, 1]}},
            }
        )
    )
    vjepa_root = tmp_path / "vjepa"
    videomae_root = tmp_path / "videomae"
    vjepa_root.mkdir()
    videomae_root.mkdir()
    vjepa_model = tmp_path / "vjepa-model.json"
    videomae_model = tmp_path / "videomae-model.json"
    vjepa_index = _write_package(vjepa_root, shared_path, vjepa_model, "vjepa2")
    videomae_index = _write_package(
        videomae_root, shared_path, videomae_model, "videomae2"
    )
    with vjepa_index.open(newline="") as handle:
        package_row = next(csv.DictReader(handle))
    manifest_fields = [
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
        "shared",
    ]
    samples_manifest = tmp_path / "samples.csv"
    with samples_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: package_row[field] for field in manifest_fields})
    samples_hash = common.sha256_file(samples_manifest)
    for index_path in (vjepa_index, videomae_index):
        with index_path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["samples_manifest_sha256"] = samples_hash
        with index_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        root = index_path.parent
        shard = root / row["shard"]
        with np.load(shard, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["samples_manifest_sha256"] = samples_hash
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        common.write_atomic_npz(shard, arrays)
        row["shard_bytes"] = str(shard.stat().st_size)
        row["shard_sha256"] = common.sha256_file(shard)
        with index_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
    vjepa_fields, vjepa_rows = parity.read_index(vjepa_index)
    videomae_fields, videomae_rows = parity.read_index(videomae_index)
    expected_rows = common.read_samples(samples_manifest, "debug")
    expected_by_id = {row["sample_id"]: row for row in expected_rows}
    partition, _shared_fields = parity.validate_index_alignment(
        vjepa_fields,
        vjepa_rows,
        videomae_fields,
        videomae_rows,
        expected_by_id,
        samples_hash,
        "revision",
    )
    assert partition == "debug"
    shared = json.loads(shared_path.read_text())
    parity.verify_shards(
        vjepa_root,
        vjepa_rows,
        shared=shared,
        shared_hash=common.sha256_file(shared_path),
        model=json.loads(vjepa_model.read_text()),
        model_hash=common.sha256_file(vjepa_model),
    )
    parity.verify_shards(
        videomae_root,
        videomae_rows,
        shared=shared,
        shared_hash=common.sha256_file(shared_path),
        model=json.loads(videomae_model.read_text()),
        model_hash=common.sha256_file(videomae_model),
    )
    extra = vjepa_root / "shards" / "extra.npz"
    extra.write_bytes(b"stale")
    with pytest.raises(ValueError, match="Shard file-set mismatch"):
        parity.verify_shards(
            vjepa_root,
            vjepa_rows,
            shared=shared,
            shared_hash=common.sha256_file(shared_path),
            model=json.loads(vjepa_model.read_text()),
            model_hash=common.sha256_file(vjepa_model),
        )
    extra.unlink()

    vjepa_rows["sample"]["project_revision"] = "STALE"
    with pytest.raises(ValueError, match="different project revision"):
        parity.validate_index_alignment(
            vjepa_fields,
            vjepa_rows,
            videomae_fields,
            videomae_rows,
            expected_by_id,
            samples_hash,
            "revision",
        )
    vjepa_rows["sample"]["project_revision"] = "revision"

    videomae_rows["sample"]["shared"] = "changed"
    with pytest.raises(ValueError):
        parity.validate_index_alignment(
            vjepa_fields,
            vjepa_rows,
            videomae_fields,
            videomae_rows,
            expected_by_id,
            samples_hash,
            "revision",
        )
