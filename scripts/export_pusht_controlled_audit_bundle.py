#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

__import__("hdf5plugin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    dataset_sha256 = sha256_file(args.dataset)
    if dataset_sha256 != args.expected_dataset_sha256:
        raise RuntimeError(f"Dataset SHA-256 mismatch: {dataset_sha256}")
    with args.selection_manifest.open(newline="") as handle:
        selected = list(csv.DictReader(handle))
    if not selected:
        raise ValueError("Selection manifest is empty")

    pixels = np.empty((len(selected), 4, 3, 224, 224), dtype=np.uint8)
    actions = np.empty((len(selected), 4, 10), dtype=np.float32)
    states = np.empty((len(selected), 4, 7), dtype=np.float32)
    episode_indices = np.empty((len(selected), 4), dtype=np.int64)
    step_indices = np.empty((len(selected), 4), dtype=np.int64)

    with h5py.File(args.dataset, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024) as source:
        lengths = source["ep_len"][:]
        offsets = source["ep_offset"][:]
        valid_counts = np.maximum(lengths.astype(np.int64) - 20 + 1, 0)
        clip_offsets = np.concatenate(([0], np.cumsum(valid_counts)))
        for index, row in enumerate(selected):
            episode = int(row["episode"])
            start = int(row["start_step"])
            dataset_index = int(row["dataset_index"])
            expected_index = int(clip_offsets[episode] + start)
            if expected_index != dataset_index:
                raise RuntimeError(
                    f"Manifest index mismatch for {row['base_id']}: {dataset_index} != {expected_index}"
                )
            flat_start = int(offsets[episode] + start)
            flat_indices = flat_start + np.arange(20)
            pixels[index] = source["pixels"][flat_indices[::5]].transpose(0, 3, 1, 2)
            actions[index] = source["action"][flat_indices].reshape(4, 10)
            states[index] = source["state"][flat_indices[::5]]
            episode_indices[index] = source["episode_idx"][flat_indices[::5]]
            step_indices[index] = source["step_idx"][flat_indices[::5]]
            if not np.all(episode_indices[index] == episode):
                raise RuntimeError(f"Episode mismatch for {row['base_id']}")
            expected_steps = start + np.arange(4) * 5
            if not np.array_equal(step_indices[index], expected_steps):
                raise RuntimeError(f"Step mismatch for {row['base_id']}")

    base_ids = np.asarray([row["base_id"] for row in selected])
    dataset_indices = np.asarray([int(row["dataset_index"]) for row in selected], dtype=np.int64)
    episodes = np.asarray([int(row["episode"]) for row in selected], dtype=np.int64)
    starts = np.asarray([int(row["start_step"]) for row in selected], dtype=np.int64)
    np.savez_compressed(
        args.output,
        base_id=base_ids,
        dataset_index=dataset_indices,
        episode=episodes,
        start_step=starts,
        pixels=pixels,
        action=actions,
        state=states,
        episode_idx=episode_indices,
        step_idx=step_indices,
    )
    provenance = {
        "command": " ".join(sys.argv),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "bundle": str(args.output.resolve()),
        "bundle_sha256": sha256_file(args.output),
        "samples": len(selected),
        "arrays": {
            "pixels": list(pixels.shape),
            "action": list(actions.shape),
            "state": list(states.shape),
            "episode_idx": list(episode_indices.shape),
            "step_idx": list(step_indices.shape),
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
