#!/usr/bin/env python3
"""Run crossed appearance and temporal-control V-JEPA2 prediction scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import VJEPA2Model, VJEPA2VideoProcessor

from extract_vjepa2_layer_features import build_records, sha256_file
from run_vjepa2_section_surprise import distances, make_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def load_video(video_dir: Path) -> tuple[np.ndarray, list[Path]]:
    paths = sorted((video_dir / "scene").glob("scene_*.png"))
    if not paths:
        raise FileNotFoundError(f"No scene frames under {video_dir}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return np.stack(frames), paths


def uniform_indices(frame_count: int, clip_length: int) -> np.ndarray:
    return np.linspace(0, frame_count - 1, clip_length).round().astype(int)


def lowest_motion_indices(clean_frames: np.ndarray, clip_length: int) -> np.ndarray:
    frame_float = clean_frames.astype(np.float32)
    consecutive = np.abs(np.diff(frame_float, axis=0)).mean(axis=(1, 2, 3)) / 255.0
    window_motion = np.convolve(
        consecutive, np.ones(clip_length - 1, dtype=np.float32), mode="valid"
    ) / (clip_length - 1)
    start = int(np.argmin(window_motion))
    return np.arange(start, start + clip_length, dtype=int)


def make_variants(
    frames: np.ndarray,
    base_indices: np.ndarray,
    low_motion: np.ndarray,
    controls: list[str],
    target_start_fraction: float,
) -> list[dict[str, Any]]:
    original = frames[base_indices]
    target_start = int(round(len(original) * target_start_fraction))
    target_start = max(2, min(len(original) - 2, target_start))
    variants = []
    for control in controls:
        if control == "original":
            clip, mapping = original.copy(), base_indices.copy()
        elif control == "freeze_first":
            clip = np.repeat(original[0:1], len(original), axis=0)
            mapping = np.full(len(original), base_indices[0], dtype=int)
        elif control == "freeze_middle":
            middle = len(original) // 2
            clip = np.repeat(original[middle : middle + 1], len(original), axis=0)
            mapping = np.full(len(original), base_indices[middle], dtype=int)
        elif control == "freeze_last":
            clip = np.repeat(original[-1:], len(original), axis=0)
            mapping = np.full(len(original), base_indices[-1], dtype=int)
        elif control == "target_freeze":
            clip, mapping = original.copy(), base_indices.copy()
            clip[target_start:] = clip[target_start - 1]
            mapping[target_start:] = mapping[target_start - 1]
        elif control == "target_shuffle":
            clip, mapping = original.copy(), base_indices.copy()
            clip[target_start:] = clip[target_start:][::-1]
            mapping[target_start:] = mapping[target_start:][::-1]
        elif control == "natural_low_motion":
            clip, mapping = frames[low_motion].copy(), low_motion.copy()
        else:
            raise ValueError(f"Unknown control: {control}")
        motion = float(
            np.abs(np.diff(clip.astype(np.float32), axis=0)).mean() / 255.0
        )
        variants.append(
            {
                "control": control,
                "clip": clip,
                "mapping": mapping,
                "mean_consecutive_rgb_difference": motion,
            }
        )
    return variants


@torch.inference_mode()
def score_batch(
    variants: list[dict[str, Any]],
    model: VJEPA2Model,
    processor: VJEPA2VideoProcessor,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    device = torch.device(config["device"])
    dtype = torch.float16 if config["dtype"] == "float16" else torch.float32
    videos = [variant["clip"] for variant in variants]
    inputs = processor(videos=videos, return_tensors="pt")["pixel_values_videos"].to(
        device=device, dtype=dtype
    )
    context_mask, target_mask, target_time_ids, target_start, _ = make_masks(
        inputs, model, float(config["target_start_fraction"])
    )
    with torch.autocast(
        "cuda", dtype=dtype, enabled=config["device"].startswith("cuda")
    ):
        output = model(
            inputs,
            context_mask=context_mask,
            target_mask=target_mask,
        ).predictor_output
    token_distances = distances(
        output.last_hidden_state,
        output.target_hidden_state,
        config["metric"],
        config["normalization"],
    )
    unique_time_ids = torch.unique(target_time_ids, sorted=True)
    results = []
    for batch_index, variant in enumerate(variants):
        temporal = [
            float(token_distances[batch_index, target_time_ids == time_id].mean().item())
            for time_id in unique_time_ids
        ]
        frame_ranges = []
        for time_id in unique_time_ids.cpu().tolist():
            offset = int(time_id) * int(model.config.tubelet_size)
            mapped = variant["mapping"][offset : offset + int(model.config.tubelet_size)]
            frame_ranges.append([int(mapped[0]), int(mapped[-1])])
        peak = int(np.argmax(temporal))
        results.append(
            {
                "control": variant["control"],
                "scalar_surprise": float(token_distances[batch_index].mean().item()),
                "temporal_tubelet_surprise": json.dumps(temporal),
                "target_source_frame_ranges_0based": json.dumps(frame_ranges),
                "peak_target_tubelet_index": peak,
                "peak_source_frame_range_0based": json.dumps(frame_ranges[peak]),
                "source_frame_mapping_0based": json.dumps(
                    variant["mapping"].astype(int).tolist()
                ),
                "target_start_tubelet": int(target_start),
                "mean_consecutive_rgb_difference": variant[
                    "mean_consecutive_rgb_difference"
                ],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (args.output_root or Path(config["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config["model_source"]).resolve()
    if sha256_file(checkpoint / "model.safetensors") != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the locked config")
    records = build_records(config, args.max_pairs)
    clean_by_base = {
        record["base_video_id"]: Path(record["video_dir"])
        for record in records
        if record["condition"] == "clean"
    }
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float16 if config["dtype"] == "float16" else torch.float32
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM GiB: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}; "
        f"requested batch_size={config['batch_size']}"
    )
    processor = VJEPA2VideoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = VJEPA2Model.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()

    low_motion_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    requested_batch_size = batch_size
    determinism_max_abs = None
    elapsed = []
    for record_index, record in enumerate(records):
        frames, _ = load_video(Path(record["video_dir"]))
        if record["base_video_id"] not in low_motion_cache:
            clean_frames, _ = load_video(clean_by_base[record["base_video_id"]])
            low_motion_cache[record["base_video_id"]] = lowest_motion_indices(
                clean_frames, int(config["frames_per_clip"])
            )
        base_indices = uniform_indices(len(frames), int(config["frames_per_clip"]))
        variants = make_variants(
            frames,
            base_indices,
            low_motion_cache[record["base_video_id"]],
            config["controls"],
            float(config["target_start_fraction"]),
        )
        record_results = []
        start = time.perf_counter()
        offset = 0
        while offset < len(variants):
            current = min(batch_size, len(variants) - offset)
            try:
                record_results.extend(
                    score_batch(
                        variants[offset : offset + current], model, processor, config
                    )
                )
                offset += current
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current == 1:
                    raise
                batch_size = max(1, current // 2)
                print(f"OOM fallback: batch_size={batch_size}")
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)

        if record_index == 0:
            rerun = score_batch(variants[:1], model, processor, config)[0]
            determinism_max_abs = abs(
                rerun["scalar_surprise"] - record_results[0]["scalar_surprise"]
            )
            if determinism_max_abs > float(config["determinism_atol"]):
                raise ValueError(f"Determinism check failed: {determinism_max_abs}")

        for result in record_results:
            rows.append(
                {
                    **{key: value for key, value in record.items() if key != "video_dir"},
                    **result,
                }
            )
        print(
            f"[{record_index + 1:03d}/{len(records):03d}] {record['sample_id']} "
            f"{elapsed[-1]:.2f}s"
        )

    scores_path = output_root / "counterfactual_scores.csv"
    pd.DataFrame(rows).to_csv(scores_path, index=False)
    provenance = {
        "schema_version": 1,
        "command": (
            f"python scripts/run_vjepa2_counterfactual_pilot.py --config "
            f"{args.config}"
            + (f" --max-pairs {args.max_pairs}" if args.max_pairs is not None else "")
            + (f" --output-root {args.output_root}" if args.output_root is not None else "")
        ),
        "git_revision": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain")),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "checkpoint_origin": config["model_origin"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": batch_size,
        "record_count": len(records),
        "score_row_count": len(rows),
        "elapsed_seconds_total": float(sum(elapsed)),
        "elapsed_seconds_median_per_clip_condition": float(np.median(elapsed)),
        "determinism_max_abs": determinism_max_abs,
        "scores_path": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
