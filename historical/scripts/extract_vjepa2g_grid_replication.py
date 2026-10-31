


from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import benchmark_vjepa2g_grid_replication as bench
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import benchmark_vjepa2g_grid_replication as bench
    from scripts import intphys2_confirmatory_common as common


GRID = {
    "hue": (15.0, 45.0),
    "brightness": (0.85, 0.65),
    "selective_recolor": (45.0, 90.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser.parse_args()


def dose_label(family: str, dose: float) -> str:
    if family == "brightness":
        return f"{dose:.2f}".replace(".", "p")
    return f"{int(dose):03d}"


def job_id(source_id: str, family: str, dose: float) -> str:
    return f"vjepa2-g__{source_id}__{family}_{dose_label(family, dose)}"


def valid_shard(path: Path, expected_job: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["job_id"]) == expected_job
                and data["X"].shape == (5, 1408)
                and data["X"].dtype == np.float16
                and np.isfinite(data["X"]).all()
                and str(data["checkpoint_sha256"]) == bench.CHECKPOINT_SHA256
            )
    except Exception:
        return False


def visual_metrics(clean: np.ndarray, transformed: np.ndarray, family: str, dose: float) -> dict[str, float]:
    difference = transformed.astype(np.float32) - clean.astype(np.float32)
    clipping = 0.0
    if family == "brightness":
        unbounded = clean.astype(np.float32) * dose
        clipping = float(np.mean((unbounded < 0.0) | (unbounded > 255.0)))
    return {
        "rgb_mean_absolute_difference": float(np.mean(np.abs(difference)) / 255.0),
        "rgb_root_mean_square_difference": float(np.sqrt(np.mean(difference**2)) / 255.0),
        "changed_pixel_fraction": float(np.mean(np.any(difference != 0.0, axis=-1))),
        "clipping_fraction": clipping,
    }


def atomic_shard(path: Path, arrays: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    shards = args.cache_root / "shards"
    args.cache_root.mkdir(parents=True, exist_ok=True)
    metadata, references = bench.load_reference(args.reference_cache)
    jobs = [
        (source, family, dose)
        for source in metadata
        for family, doses in GRID.items()
        for dose in doses
    ]
    completed = sum(valid_shard(shards / f"{job_id(s['name'], f, d)}.npz", job_id(s["name"], f, d)) for s, f, d in jobs)
    started = time.perf_counter()
    model, state, payload, load = bench.load_exact_encoder(args.checkpoint, args.vjepa2_root)
    encoder = bench.SequentialOffloadEncoder(model, state)
    shared_config = json.loads(Path("configs/intphys2_confirmatory_shared.json").read_text())
    runtime_path = args.cache_root / "hardware_runtime_log.csv"
    log_exists = runtime_path.exists()
    runtime_handle = runtime_path.open("a", newline="")
    runtime_fields = [
        "timestamp", "job_id", "source_id", "scene_id", "family", "dose", "seconds",
        "peak_allocated_bytes", "peak_reserved_bytes", "rss_start_bytes", "rss_end_bytes",
        "utilization.gpu", "memory.used", "memory.total", "temperature.gpu", "power.draw",
    ]
    writer = csv.DictWriter(runtime_handle, fieldnames=runtime_fields)
    if not log_exists:
        writer.writeheader()
    inferred = 0
    skipped = 0
    repeat_checks = []
    try:
        for source_index, source in enumerate(metadata, 1):
            pending = []
            for family, doses in GRID.items():
                for dose in doses:
                    identifier = job_id(source["name"], family, dose)
                    path = shards / f"{identifier}.npz"
                    if valid_shard(path, identifier):
                        skipped += 1
                    else:
                        pending.append((identifier, path, family, dose))
            if not pending:
                continue
            source_path = args.dataset_root / source["source_relative_path"]
            indices = json.loads(source["source_frame_indices_0based"])
            clean = common.decode_selected_frames(source_path, indices)
            for identifier, path, family, dose in pending:
                row = bench.appearance_row(source, family, dose)
                transformed = common.apply_appearance(clean, row, shared_config)
                image_metrics = visual_metrics(clean, transformed, family, dose)
                normalized = bench.normalize_frames(transformed)
                del transformed
                pooled, inference = encoder.infer(normalized)
                del normalized
                if inferred < 3:
                    repeated_normalized = bench.normalize_frames(
                        common.apply_appearance(clean, row, shared_config)
                    )
                    repeated, _ = encoder.infer(repeated_normalized)
                    repeat_checks.append(
                        {"job_id": identifier, "max_abs_diff": float(np.max(np.abs(pooled - repeated)))}
                    )
                    del repeated_normalized, repeated
                arrays = {
                    "X": pooled.astype(np.float16),
                    "layer_names": np.asarray(bench.LAYERS),
                    "native_layers": np.asarray(bench.NATIVE_LAYERS, dtype=np.int16),
                    "job_id": np.asarray(identifier),
                    "model_id": np.asarray("vjepa2-g"),
                    "checkpoint_sha256": np.asarray(bench.CHECKPOINT_SHA256),
                    "source_id": np.asarray(source["name"]),
                    "source_relative_path": np.asarray(source["source_relative_path"]),
                    "source_sha256": np.asarray(source["source_sha256"]),
                    "scene_id": np.asarray(source["scene_id"]),
                    "pair_id": np.asarray(source["pair_id"]),
                    "physical_label": np.asarray(source["physical_label"]),
                    "transform_family": np.asarray(family),
                    "transform_value": np.asarray(dose, dtype=np.float64),
                    "transform_unit": np.asarray("multiplicative_rgb_factor" if family == "brightness" else "degrees"),
                    "selective_center_degrees": np.asarray(source["selective_center_degrees"]),
                    "transform_feasible": np.asarray(source["transform_feasible"]),
                    **{key: np.asarray(value, dtype=np.float64) for key, value in image_metrics.items()},
                }
                atomic_shard(path, arrays)
                if not valid_shard(path, identifier):
                    raise RuntimeError(f"Post-write shard validation failed: {path}")
                hardware = bench.nvidia_row()
                log_row = {
                    "timestamp": hardware.pop("timestamp"),
                    "job_id": identifier,
                    "source_id": source["name"],
                    "scene_id": source["scene_id"],
                    "family": family,
                    "dose": dose,
                    **inference,
                    **hardware,
                }
                writer.writerow(log_row)
                runtime_handle.flush()
                inferred += 1
                append_jsonl(args.cache_root / "completed_jobs.jsonl", {**log_row, **image_metrics})
                print(json.dumps({"source": source_index, "of_sources": len(metadata), "inferred": inferred, "initially_completed": completed, **log_row}), flush=True)
                del pooled
                gc.collect()
            del clean
        total_valid = sum(valid_shard(shards / f"{job_id(s['name'], f, d)}.npz", job_id(s["name"], f, d)) for s, f, d in jobs)
        summary = {
            "status": "PASS" if total_valid == len(jobs) else "INCOMPLETE",
            "jobs_total": len(jobs),
            "jobs_valid": total_valid,
            "jobs_inferred_this_run": inferred,
            "jobs_skipped_this_run": skipped,
            "wall_seconds": time.perf_counter() - started,
            "model_load": load,
            "repeat_checks": repeat_checks,
            "strategy": "exact sequential transformer-block CPU/GPU offload",
        }
        temporary = args.cache_root / "extraction_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2))
        os.replace(temporary, args.cache_root / "extraction_summary.json")
        print(json.dumps(summary, indent=2))
    finally:
        runtime_handle.close()
        del encoder, model, state, payload
        gc.collect()


if __name__ == "__main__":
    main()
