#!/usr/bin/env python3
"""Run the nine on-the-fly IntPhys 2 appearance conditions in one model pass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import run_intphys2_dense as core
except ModuleNotFoundError:  # Imported as scripts.run_intphys2_appearance_matrix
    from scripts import run_intphys2_dense as core

CONDITIONS = (
    ("hue-mild", "hue", 15.0),
    ("hue-medium", "hue", 45.0),
    ("hue-strong", "hue", 90.0),
    ("brightness-mild", "brightness", 0.85),
    ("brightness-medium", "brightness", 0.65),
    ("brightness-strong", "brightness", 0.45),
    ("selective-recolor-mild", "selective_recolor", 45.0),
    ("selective-recolor-medium", "selective_recolor", 90.0),
    ("selective-recolor-strong", "selective_recolor", 180.0),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("Debug", "Main"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--condition",
        action="append",
        choices=("clean", *(condition[0] for condition in CONDITIONS)),
        help="Condition to run; repeat as needed. Defaults to all nine transforms.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="JSON manifest containing selected_names; complete pairs are required.",
    )
    parser.add_argument("--job-tag")
    return parser.parse_args()


def output_paths(root: Path, condition: str, name: str):
    condition_root = root / condition
    videos = condition_root / "videos"
    maps = condition_root / "spatial_maps"
    videos.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    return condition_root, videos / f"{name}.json", maps / f"{name}.npz"


def condition_complete(paths: tuple[Path, Path, Path]) -> bool:
    """A matrix cell is complete only when both atomic component files exist."""
    _, json_path, map_path = paths
    return json_path.exists() and map_path.exists()


def pending_cells(conditions: tuple, paths: list[tuple[Path, Path, Path]]):
    return [
        (condition, cell_paths)
        for condition, cell_paths in zip(conditions, paths)
        if not condition_complete(cell_paths)
    ]


def main():
    from decord import VideoReader, cpu

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = core.MODEL_SPECS["vjepa2-g"]
    records = core.split_records(args.dataset_root / args.split)
    if args.selection_manifest:
        selection = json.loads(args.selection_manifest.read_text())
        selected_names = set(selection["selected_names"])
        records = [record for record in records if str(record.get("name")) in selected_names]
        if len(records) != len(selected_names):
            found = {str(record.get("name")) for record in records}
            raise ValueError(f"Selection names absent from split: {sorted(selected_names - found)}")
    if args.max_videos is not None:
        records = records[: args.max_videos]
    requested = args.condition or [condition[0] for condition in CONDITIONS]
    definitions = {condition[0]: condition for condition in CONDITIONS}
    definitions["clean"] = ("clean", None, None)
    active_conditions = tuple(definitions[name] for name in requested)
    if len({condition[0] for condition in active_conditions}) != len(active_conditions):
        raise ValueError("Conditions must not be repeated")
    needs_selective = any(
        transform == "selective_recolor" for _, transform, _ in active_conditions
    )
    centers = (
        core.selective_centers(records, allow_infeasible=True)
        if needs_selective else {str(record.get("name")): None for record in records}
    )
    sys.path.insert(0, str(args.vjepa2_root))
    from src.masks.utils import apply_masks

    encoder, target_encoder, predictor, load_info = core.load_models(args, spec)
    provenance = {
        "model": "vjepa2-g",
        "model_spec": spec,
        "context": args.context,
        "conditions": [
            {"name": name, "transform": transform, "value": value}
            for name, transform, value in active_conditions
        ],
        "appearance_execution": "on-the-fly after one shared MP4 decode; no re-encode",
        "selective_recolor_infeasible_policy": (
            "paired identity input with explicit infeasible flag when neither anchor frame "
            "contains pixels at saturation >= 64; never infer a replacement hue band"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": core.sha256_file(args.checkpoint),
        "vjepa2_revision": core.revision(args.vjepa2_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "video_count": len(records),
        "frame_step": args.frame_step,
        "window_stride": args.window_stride,
        "condition_batch_size": len(active_conditions),
        "selection_manifest": str(args.selection_manifest) if args.selection_manifest else None,
        "job_tag": args.job_tag,
        "dtype": "bfloat16 autocast",
        "activation_checkpointing": True,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        **load_info,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = (
        args.output_dir / "jobs" / args.job_tag / "provenance.json"
        if args.job_tag else args.output_dir / "provenance.json"
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    core.write_atomic_text(provenance_path, json.dumps(provenance, indent=2) + "\n")
    if args.selection_manifest and args.job_tag:
        selection_copy = provenance_path.parent / "selection-manifest.json"
        core.write_atomic_text(selection_copy, args.selection_manifest.read_text())

    spatial_size = spec["resolution"] // 16
    for video_index, record in enumerate(records):
        name = str(record.get("name") or Path(record["video_path"]).stem)
        paths = [
            output_paths(args.output_dir, condition, name)
            for condition, _, _ in active_conditions
        ]
        pending = pending_cells(active_conditions, paths)
        if not pending:
            continue
        video_conditions = tuple(condition for condition, _ in pending)
        video_paths = [cell_paths for _, cell_paths in pending]
        reader = VideoReader(record["video_path"], ctx=cpu(0))
        fps = float(reader.get_avg_fps())
        total_source_frames = len(reader)
        sampled = np.arange(0, total_source_frames, args.frame_step)
        decoded = reader.get_batch(sampled).asnumpy()
        center = centers[name]
        transformed = [
            (
                decoded.copy()
                if transform is None or (transform == "selective_recolor" and center is None)
                else core.appearance_transform(
                    decoded,
                    transform,
                    value,
                    center if transform == "selective_recolor" else None,
                )
            )
            for _, transform, value in video_conditions
        ]
        normalized = torch.stack([
            core.normalize_video(torch.from_numpy(frames), spec["resolution"])
            for frames in transformed
        ])
        starts = list(range(0, len(sampled) - spec["frames_per_clip"] + 1, args.window_stride))
        rows = [[] for _ in video_conditions]
        maps = [[] for _ in video_conditions]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for window_index, start in enumerate(starts):
                indices = sampled[start : start + spec["frames_per_clip"]]
                x = normalized[:, start : start + spec["frames_per_clip"]]
                x = x.permute(0, 2, 1, 3, 4).contiguous().cuda()
                batch_size = len(video_conditions)
                enc_mask, pred_mask, full_mask = core.token_masks(
                    args.context, spec, batch_size
                )
                targets_all = target_encoder(x, full_mask)
                targets = apply_masks(targets_all, [pred_mask], concat=False)[0]
                encoded = encoder(x, enc_mask)
                predictions = predictor(encoded, enc_mask, pred_mask)
                targets = F.layer_norm(targets, (targets.size(-1),))
                error = (predictions - targets).abs().float()
                temporal_spatial = error.mean(-1).reshape(
                    batch_size,
                    (spec["frames_per_clip"] - args.context) // 2,
                    spatial_size,
                    spatial_size,
                )
                temporal = temporal_spatial.mean((-1, -2))
                if not bool(torch.isfinite(error).all()):
                    raise RuntimeError(f"Non-finite loss for {name}, window {window_index}")
                pairs = core.target_frame_pairs(indices, args.context)
                for condition_index in range(batch_size):
                    rows[condition_index].append({
                        "window_index": window_index,
                        "context_length": args.context,
                        "window_sampled_start": start,
                        "window_source_start_frame_0based": int(indices[0]),
                        "window_source_end_frame_0based": int(indices[-1]),
                        "context_source_frames_0based": indices[: args.context].tolist(),
                        "target_source_frames_0based": indices[args.context :].tolist(),
                        "target_tubelet_source_frame_pairs_0based": pairs,
                        "target_tubelet_source_second_spans": [
                            [pair[0] / fps, pair[1] / fps] for pair in pairs
                        ],
                        "scalar_l1": float(error[condition_index].mean().item()),
                        "temporal_tubelet_l1": temporal[condition_index].cpu().tolist(),
                    })
                    maps[condition_index].append(
                        temporal_spatial[condition_index].to(torch.float16).cpu().numpy()
                    )
        wall_seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        for condition_index, ((condition, transform, value), (condition_root, json_path, map_path)) in enumerate(zip(video_conditions, video_paths)):
            map_array = np.stack(maps[condition_index])
            temporary = map_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **{f"context_{args.context}": map_array})
            temporary.replace(map_path)
            payload = {
                "dataset": "IntPhys2",
                "split": args.split,
                "filename": record.get("file_name"),
                "name": name,
                "metadata": {key: value for key, value in record.items() if key != "video_path"},
                "fps": fps,
                "duration_seconds": total_source_frames / fps,
                "total_source_frames": total_source_frames,
                "all_sampled_source_frame_indices_0based": sampled.tolist(),
                "frame_step": args.frame_step,
                "appearance_condition": condition,
                "appearance_transform": transform,
                "appearance_value": value,
                "selective_source_hue_center_degrees": center if transform == "selective_recolor" else None,
                "appearance_feasible": not (transform == "selective_recolor" and center is None),
                "appearance_infeasible_reason": (
                    "paired anchor frames contain no pixels at saturation >= 64"
                    if transform == "selective_recolor" and center is None else None
                ),
                "appearance_execution": "on-the-fly after shared MP4 decoding; no re-encode",
                "frames_per_clip": spec["frames_per_clip"],
                "window_stride_sampled_frames": args.window_stride,
                "violation_start_frame": None,
                "violation_end_frame": None,
                "violation_start_seconds": None,
                "violation_end_seconds": None,
                "violation_annotation_source": "not released",
                "windows": rows[condition_index],
                "wall_seconds_shared_condition_batch": wall_seconds,
                "peak_allocated_bytes_shared_condition_batch": peak_allocated,
                "peak_reserved_bytes_shared_condition_batch": peak_reserved,
                "shared_condition_batch_size": len(video_conditions),
                "temporal_spatial_l1_npz": str(map_path.relative_to(condition_root)),
                "temporal_spatial_l1_shapes": {str(args.context): list(map_array.shape)},
            }
            core.write_atomic_text(json_path, json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps({
            "done": video_index + 1,
            "total": len(records),
            "name": name,
            "wall_seconds_condition_batch": wall_seconds,
            "peak_allocated_bytes": peak_allocated,
        }), flush=True)

    if args.job_tag:
        status_path = args.output_dir / "jobs" / args.job_tag / "status.json"
        core.write_atomic_text(
            status_path,
            json.dumps({
                "status": "complete",
                "conditions": [condition[0] for condition in active_conditions],
                "selected_videos": len(records),
                "completed_selected_videos": {
                    condition: sum(
                        condition_complete(
                            output_paths(args.output_dir, condition, str(record.get("name")))
                        )
                        for record in records
                    )
                    for condition, _, _ in active_conditions
                },
            }, indent=2) + "\n",
        )
    else:
        for condition, _, _ in active_conditions:
            condition_root = args.output_dir / condition
            core.consolidate(condition_root)
            core.write_atomic_text(
                condition_root / "status.json",
                json.dumps({
                    "status": "complete",
                    "condition": condition,
                    "expected_videos": len(records),
                    "completed_videos": len(list((condition_root / "videos").glob("*.json"))),
                }, indent=2) + "\n",
            )
        core.write_atomic_text(
            args.output_dir / "status.json",
            json.dumps({"status": "complete", "conditions": len(active_conditions), "videos_per_condition": len(records)}, indent=2) + "\n",
        )


if __name__ == "__main__":
    main()
