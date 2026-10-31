#!/usr/bin/env python3
"""Pack reused and newly extracted V-JEPA2-G grid features for CPU analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import benchmark_vjepa2g_grid_replication as bench
    import extract_vjepa2g_grid_replication as extract
    import intphys2_confirmatory_common as common
except ModuleNotFoundError:
    from scripts import benchmark_vjepa2g_grid_replication as bench
    from scripts import extract_vjepa2g_grid_replication as extract
    from scripts import intphys2_confirmatory_common as common


CONDITIONS = (
    ("clean", "clean", 0.0, 0.0),
    ("hue_015", "hue", 15.0, 15.0),
    ("hue_045", "hue", 45.0, 45.0),
    ("hue_090", "hue", 90.0, 90.0),
    ("brightness_0p85", "brightness", 0.85, 0.15),
    ("brightness_0p65", "brightness", 0.65, 0.35),
    ("brightness_0p45", "brightness", 0.45, 0.55),
    ("selective_recolor_045", "selective_recolor", 45.0, 45.0),
    ("selective_recolor_090", "selective_recolor", 90.0, 90.0),
    ("selective_recolor_180", "selective_recolor", 180.0, 180.0),
)
REUSED = {
    "clean": "clean",
    "hue_090": "hue_090",
    "brightness_0p45": "brightness_0p45",
    "selective_recolor_180": "selective_recolor_180",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--new-cache-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def scene_split(scene: str) -> str:
    digest = hashlib.sha256(f"vjepa2g-grid-replication-v1|{scene}".encode()).digest()
    return digest.hex()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata, _ = bench.load_reference(args.reference_cache)
    scenes = sorted({row["scene_id"] for row in metadata}, key=scene_split)
    train_scenes = set(scenes[:23])
    with Path("data/manifests/intphys2_confirmatory_transform_metrics.csv").open(newline="") as handle:
        transform_metrics = {
            (row["name"], row["condition"]): row
            for row in csv.DictReader(handle)
            if row["partition"] == "development"
        }
    with np.load(args.reference_cache, allow_pickle=False) as old:
        old_lookup = {
            (str(name), str(condition)): old["X"][index]
            for index, (name, condition) in enumerate(zip(old["name"], old["appearance_condition"]))
        }
    features = []
    samples = []
    frame_indices = []
    image_rows = []
    for source in metadata:
        indices = json.loads(source["source_frame_indices_0based"])
        for condition, family, nominal, path_dose in CONDITIONS:
            identifier = extract.job_id(source["name"], family, nominal)
            if condition in REUSED:
                x = old_lookup[(source["name"], REUSED[condition])]
                feature_origin = "reused corrected development cache"
                if condition == "clean":
                    image_metrics = {
                        "rgb_mean_absolute_difference": 0.0,
                        "rgb_root_mean_square_difference": 0.0,
                        "changed_pixel_fraction": 0.0,
                        "clipping_fraction": 0.0,
                    }
                else:
                    metric = transform_metrics[(source["name"], condition)]
                    image_metrics = {
                        "rgb_mean_absolute_difference": float(metric["rgb_l1"]),
                        "rgb_root_mean_square_difference": float("nan"),
                        "changed_pixel_fraction": float(metric["affected"]),
                        "clipping_fraction": float(metric["additional_zero_luma_fraction"]),
                    }
            else:
                shard = args.new_cache_root / "shards" / f"{identifier}.npz"
                if not extract.valid_shard(shard, identifier):
                    raise ValueError(f"Missing or invalid shard: {shard}")
                with np.load(shard, allow_pickle=False) as data:
                    x = data["X"].copy()
                    image_metrics = {
                        "rgb_mean_absolute_difference": float(data["rgb_mean_absolute_difference"]),
                        "rgb_root_mean_square_difference": float(data["rgb_root_mean_square_difference"]),
                        "changed_pixel_fraction": float(data["changed_pixel_fraction"]),
                        "clipping_fraction": float(data["clipping_fraction"]),
                    }
                feature_origin = "new exact sequential-offload extraction"
            sample_id = f"{source['name']}__{condition}"
            features.append(x)
            frame_indices.append(indices)
            samples.append(
                {
                    "sample_id": sample_id,
                    "base_video_id": source["name"],
                    "scene": source["scene_id"],
                    "split": "development" if source["scene_id"] in train_scenes else "test",
                    "base_pair_id": source["pair_id"],
                    "possible": source["physical_label"] == "possible",
                    "physical_label": source["physical_label"],
                    "condition": condition,
                    "factor": family,
                    "nominal_dose": nominal,
                    "path_dose": path_dose,
                    "source_relative_path": source["source_relative_path"],
                    "feature_origin": feature_origin,
                    **image_metrics,
                }
            )
            image_rows.append({"sample_id": sample_id, **image_metrics})
    pooled = np.stack(features).astype(np.float16)
    if pooled.shape != (1520, 5, 1408) or not np.isfinite(pooled).all():
        raise ValueError(f"Unexpected packed feature shape/values: {pooled.shape}")
    temporary = args.output_root / "layer_features.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pooled=pooled,
            layers=np.asarray(bench.NATIVE_LAYERS, dtype=np.int16),
            layer_names=np.asarray(bench.LAYERS),
            sampled_source_indices=np.asarray(frame_indices, dtype=np.int16),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output_root / "layer_features.npz")
    (args.output_root / "samples.json").write_text(json.dumps(samples, indent=2))
    split = {
        "seed": "vjepa2g-grid-replication-v1",
        "policy": "23/38 scenes with lowest SHA-256 rank are train; remaining 15 are held out",
        "train_scenes": sorted(train_scenes),
        "heldout_scenes": sorted(set(scenes) - train_scenes),
    }
    (args.output_root / "analysis_split.json").write_text(json.dumps(split, indent=2))
    provenance = {
        "status": "PASS",
        "samples": len(samples),
        "sources": len(metadata),
        "scenes": len(scenes),
        "pairs": len({row["pair_id"] for row in metadata}),
        "reused_samples": sum(row["feature_origin"].startswith("reused") for row in samples),
        "new_samples": sum(row["feature_origin"].startswith("new") for row in samples),
        "feature_shape": list(pooled.shape),
        "checkpoint_sha256": bench.CHECKPOINT_SHA256,
        "layers": list(bench.LAYERS),
        "split": split,
    }
    (args.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
