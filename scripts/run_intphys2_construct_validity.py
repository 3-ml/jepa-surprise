#!/usr/bin/env python3
"""Run locked V-JEPA2-G static and temporal controls on IntPhys 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import run_intphys2_dense as core
except ModuleNotFoundError:  # Imported as scripts.run_intphys2_construct_validity
    from scripts import run_intphys2_dense as core

CONTEXT = 24
FRAMES_PER_CLIP = 48
GRID_SIZE = 16
CONTROL_NAMES = (
    "original",
    "full_start",
    "full_boundary",
    "full_end",
    "target_hold",
    "target_shuffle",
)
MOTION_FIELDS = (
    "motion_global_consecutive_l1_01",
    "motion_patch_p95_consecutive_l1_01",
    "motion_first_last_l1_01",
)
SHUFFLE_VERSION = "target-tubelet-shuffle-v1"
MOTION_VERSION = "resized-rgb-global-patch-span-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vjepa2-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("Debug", "Main"), default="Main")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--shuffle-seed", default="intphys2-construct-validity-v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Pair-preserving smoke limit; omitted for the locked study.",
    )
    return parser.parse_args()


def project_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def pair_identity(record: dict) -> tuple[str, str]:
    if "SceneIndex" not in record or "type" not in record:
        raise ValueError("Construct-validity selection requires released pair metadata")
    member = str(record["type"]).split("_", 1)[0].strip()
    pair_id = f"{int(record['SceneIndex'])}/{member}"
    label = "impossible" if "impossible" in str(record["type"]).lower() else "possible"
    return pair_id, label


def selected_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    selection = json.loads(args.selection_manifest.read_text())
    pairs = selection["pairs"]
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("--max-pairs must be positive")
        pairs = pairs[: args.max_pairs]
    effective_selection = dict(selection)
    effective_selection["pairs"] = pairs
    effective_selection["selected_pairs"] = len(pairs)
    effective_selection["selected_videos"] = 2 * len(pairs)
    effective_selection["selected_names"] = sorted(
        str(name) for pair in pairs for name in pair["names"]
    )
    effective_selection["pair_preserving_smoke_limit"] = args.max_pairs
    selected_names = set(effective_selection["selected_names"])
    records = core.split_records(args.dataset_root / args.split)
    records = [record for record in records if str(record.get("name")) in selected_names]
    found = {str(record.get("name")) for record in records}
    if found != selected_names:
        raise ValueError(f"Selection names absent from split: {sorted(selected_names - found)}")
    observed_pairs: dict[str, set[str]] = defaultdict(set)
    for record in records:
        pair_id, label = pair_identity(record)
        observed_pairs[pair_id].add(label)
    incomplete = {pair: labels for pair, labels in observed_pairs.items() if labels != {"possible", "impossible"}}
    if incomplete or len(records) != 2 * len(pairs):
        raise ValueError(f"Selection does not contain complete pairs: {incomplete}")
    return sorted(records, key=lambda record: str(record["name"])), effective_selection


def resize_rgb_01(frames: np.ndarray, resolution: int = 256) -> torch.Tensor:
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    tensor = F.interpolate(
        tensor, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    return tensor.permute(0, 2, 3, 1).contiguous() / 255.0


def _segment_change(frames: torch.Tensor) -> tuple[float, float]:
    if len(frames) < 2:
        return 0.0, 0.0
    differences = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
    return float(differences.mean()), float(differences.max())


def motion_metrics(frames: torch.Tensor, context: int = CONTEXT) -> dict[str, float]:
    if tuple(frames.shape) != (FRAMES_PER_CLIP, 256, 256, 3):
        raise ValueError(f"Expected [48,256,256,3], found {tuple(frames.shape)}")
    pixel_differences = (frames[1:] - frames[:-1]).abs().mean(dim=-1)
    patch_differences = pixel_differences.reshape(
        FRAMES_PER_CLIP - 1, GRID_SIZE, 16, GRID_SIZE, 16
    ).mean(dim=(2, 4))
    patch_p95 = torch.quantile(
        patch_differences.flatten(1), 0.95, dim=1
    ).mean()
    context_mean, context_max = _segment_change(frames[:context])
    target_mean, target_max = _segment_change(frames[context:])
    return {
        "motion_global_consecutive_l1_01": float(pixel_differences.mean()),
        "motion_global_max_transition_l1_01": float(
            pixel_differences.mean(dim=(1, 2)).max()
        ),
        "motion_patch_p95_consecutive_l1_01": float(patch_p95),
        "motion_patch_max_consecutive_l1_01": float(patch_differences.max()),
        "motion_first_last_l1_01": float((frames[-1] - frames[0]).abs().mean()),
        "motion_context_consecutive_l1_01": context_mean,
        "motion_context_max_transition_l1_01": context_max,
        "motion_target_consecutive_l1_01": target_mean,
        "motion_target_max_transition_l1_01": target_max,
        "motion_context_target_boundary_l1_01": float(
            (frames[context] - frames[context - 1]).abs().mean()
        ),
    }


def deterministic_tubelet_permutation(
    seed: str, video_name: str, window_index: int, tubelets: int = 12
) -> np.ndarray:
    material = f"{SHUFFLE_VERSION}:{seed}:{video_name}:{window_index}"
    value = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    permutation = np.random.default_rng(value).permutation(tubelets)
    if np.array_equal(permutation, np.arange(tubelets)):
        permutation = np.roll(permutation, 1)
    return permutation


def control_slot_mapping(
    control: str,
    *,
    seed: str,
    video_name: str,
    window_index: int,
    context: int = CONTEXT,
) -> tuple[np.ndarray, dict]:
    if control not in CONTROL_NAMES:
        raise ValueError(f"Unknown control: {control}")
    slots = np.arange(FRAMES_PER_CLIP)
    metadata: dict = {
        "control_anchor_sampled_slot_0based": None,
        "target_tubelet_permutation": None,
    }
    if control == "full_start":
        slots[:] = 0
        metadata["control_anchor_sampled_slot_0based"] = 0
    elif control == "full_boundary":
        slots[:] = context - 1
        metadata["control_anchor_sampled_slot_0based"] = context - 1
    elif control == "full_end":
        slots[:] = FRAMES_PER_CLIP - 1
        metadata["control_anchor_sampled_slot_0based"] = FRAMES_PER_CLIP - 1
    elif control == "target_hold":
        slots[context:] = context - 1
        metadata["control_anchor_sampled_slot_0based"] = context - 1
    elif control == "target_shuffle":
        target = slots[context:].reshape(-1, 2)
        permutation = deterministic_tubelet_permutation(
            seed, video_name, window_index, len(target)
        )
        slots[context:] = target[permutation].reshape(-1)
        metadata["target_tubelet_permutation"] = permutation.tolist()
    return slots, metadata


def _percentile_ranks(rows: list[dict], field: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (float(row[field]), row["window_id"]))
    denominator = max(len(ordered) - 1, 1)
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        value = float(ordered[start][field])
        while end < len(ordered) and float(ordered[end][field]) == value:
            end += 1
        average_rank = (start + end - 1) / 2
        for row in ordered[start:end]:
            ranks[row["window_id"]] = average_rank / denominator
        start = end
    return ranks


def assign_motion_ranks(rows: list[dict]) -> None:
    rank_maps = {field: _percentile_ranks(rows, field) for field in MOTION_FIELDS}
    for row in rows:
        ranks = [rank_maps[field][row["window_id"]] for field in MOTION_FIELDS]
        row["motion_composite_max_percentile"] = max(ranks)
        for field, rank in zip(MOTION_FIELDS, ranks):
            row[f"{field}_percentile"] = rank
    selected_count = max(1, math.ceil(0.1 * len(rows)))
    selected = {
        row["window_id"]
        for row in sorted(
            rows,
            key=lambda row: (row["motion_composite_max_percentile"], row["window_id"]),
        )[:selected_count]
    }
    for row in rows:
        row["lowest_visual_change_decile"] = row["window_id"] in selected


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_window_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for field, value in row.items()
                }
            )
    temporary.replace(path)


def prepare_window_manifest(
    args: argparse.Namespace, records: list[dict], selection: dict
) -> list[dict]:
    from decord import VideoReader, cpu

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "window_manifest.jsonl"
    preparation_path = args.output_dir / "window_manifest_provenance.json"
    expected_preparation = {
        "schema_version": 1,
        "motion_version": MOTION_VERSION,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "selection_manifest_sha256": core.sha256_file(args.selection_manifest),
        "effective_selected_pairs": len(selection["pairs"]),
        "effective_selected_names": selection["selected_names"],
        "frame_step": args.frame_step,
        "window_stride": args.window_stride,
        "context": CONTEXT,
        "frames_per_clip": FRAMES_PER_CLIP,
        "model_scores_existed_when_written": False,
    }
    if jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line]
        expected_names = {str(record["name"]) for record in records}
        found_names = {str(row["name"]) for row in rows}
        if expected_names != found_names:
            raise ValueError("Existing window manifest does not match selected videos")
        if not preparation_path.exists() or json.loads(preparation_path.read_text()) != expected_preparation:
            raise ValueError("Existing window manifest provenance does not match this invocation")
        return rows

    rows: list[dict] = []
    for video_index, record in enumerate(records):
        name = str(record["name"])
        pair_id, label = pair_identity(record)
        reader = VideoReader(record["video_path"], ctx=cpu(0))
        fps = float(reader.get_avg_fps())
        total_source_frames = len(reader)
        sampled = np.arange(0, total_source_frames, args.frame_step)
        decoded = reader.get_batch(sampled).asnumpy()
        resized = resize_rgb_01(decoded)
        starts = list(
            range(
                0,
                len(sampled) - FRAMES_PER_CLIP + 1,
                args.window_stride,
            )
        )
        for window_index, start in enumerate(starts):
            source_indices = sampled[start : start + FRAMES_PER_CLIP]
            metrics = motion_metrics(resized[start : start + FRAMES_PER_CLIP])
            rows.append(
                {
                    "window_id": f"{name}:{window_index}",
                    "name": name,
                    "filename": record.get("file_name"),
                    "pair_id": pair_id,
                    "pair_label": label,
                    "condition": record.get("condition"),
                    "difficulty": record.get("Difficulty"),
                    "camera": record.get("Camera"),
                    "window_index": window_index,
                    "window_sampled_start": start,
                    "window_source_start_frame_0based": int(source_indices[0]),
                    "window_source_end_frame_0based": int(source_indices[-1]),
                    "context_source_frames_0based": source_indices[:CONTEXT].tolist(),
                    "target_source_frames_0based": source_indices[CONTEXT:].tolist(),
                    "target_tubelet_source_frame_pairs_0based": core.target_frame_pairs(
                        source_indices, CONTEXT
                    ),
                    "fps": fps,
                    "total_source_frames": total_source_frames,
                    **metrics,
                }
            )
        print(
            json.dumps(
                {
                    "prepared": video_index + 1,
                    "total": len(records),
                    "name": name,
                    "windows": len(starts),
                }
            ),
            flush=True,
        )
    if not rows:
        raise ValueError("Selected videos have no complete 48-frame sampled windows")
    assign_motion_ranks(rows)
    _write_jsonl(jsonl_path, rows)
    _write_window_csv(args.output_dir / "windows.csv", rows)
    low_rows = [row for row in rows if row["lowest_visual_change_decile"]]
    core.write_atomic_text(
        args.output_dir / "lowest_visual_change_manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "definition": (
                    "lowest ceil(10%) by max percentile rank across global consecutive "
                    "RGB L1, p95 patch consecutive RGB L1, and first-last RGB L1; "
                    "ties broken by window_id"
                ),
                "motion_version": MOTION_VERSION,
                "eligible_windows": len(rows),
                "selected_windows": len(low_rows),
                "window_ids": [row["window_id"] for row in low_rows],
            },
            indent=2,
        )
        + "\n",
    )
    core.write_atomic_text(
        args.output_dir / "selection-manifest.json",
        json.dumps(selection, indent=2) + "\n",
    )
    core.write_atomic_text(
        preparation_path, json.dumps(expected_preparation, indent=2) + "\n"
    )
    return rows


def run_identity(provenance: dict) -> str:
    excluded = {"run_id", "load_seconds", "gpu", "gpu_total_bytes"}
    identity = {key: value for key, value in provenance.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _video_complete(
    output_dir: Path,
    name: str,
    run_id: str,
    expected_window_ids: list[str],
) -> bool:
    json_path = output_dir / "videos" / f"{name}.json"
    map_path = output_dir / "spatial_maps" / f"{name}.npz"
    if not json_path.exists() or not map_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text())
        if payload["run_id"] != run_id:
            raise ValueError(f"Run identity mismatch for {name}")
        if payload["spatial_maps_sha256"] != core.sha256_file(map_path):
            raise ValueError(f"Spatial-map checksum mismatch for {name}")
        score_pairs = [(row["window_id"], row["control"]) for row in payload["scores"]]
        expected_pairs = [
            (window_id, control)
            for window_id in expected_window_ids
            for control in CONTROL_NAMES
        ]
        if score_pairs != expected_pairs:
            raise ValueError(f"Score rows do not match expected windows for {name}")
        with np.load(map_path) as maps:
            if set(maps.files) != set(CONTROL_NAMES):
                raise ValueError(f"Spatial-map controls differ for {name}: {maps.files}")
            expected_shape = (len(expected_window_ids), 12, 16, 16)
            if any(tuple(maps[control].shape) != expected_shape for control in CONTROL_NAMES):
                raise ValueError(f"Spatial-map shape mismatch for {name}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid existing output for {name}: {error}") from error
    return True


def score_video(
    args: argparse.Namespace,
    record: dict,
    window_rows: list[dict],
    spec: dict,
    models: tuple,
) -> tuple[dict, dict[str, np.ndarray]]:
    from decord import VideoReader, cpu
    from src.masks.utils import apply_masks

    encoder, target_encoder, predictor = models
    name = str(record["name"])
    reader = VideoReader(record["video_path"], ctx=cpu(0))
    sampled = np.arange(0, len(reader), args.frame_step)
    decoded = reader.get_batch(sampled).asnumpy()
    spatial_size = spec["resolution"] // 16
    score_rows: list[dict] = []
    maps: dict[str, list[np.ndarray]] = {control: [] for control in CONTROL_NAMES}
    predictor_differences = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for map_offset, window in enumerate(
            sorted(window_rows, key=lambda row: row["window_index"])
        ):
            start = int(window["window_sampled_start"])
            original_clip = decoded[start : start + FRAMES_PER_CLIP]
            original_source_indices = sampled[start : start + FRAMES_PER_CLIP]
            control_clips = []
            control_records = []
            for control in CONTROL_NAMES:
                slot_mapping, control_metadata = control_slot_mapping(
                    control,
                    seed=args.shuffle_seed,
                    video_name=name,
                    window_index=int(window["window_index"]),
                )
                input_source_indices = original_source_indices[slot_mapping]
                control_clips.append(original_clip[slot_mapping].copy())
                control_records.append(
                    (control, slot_mapping, input_source_indices, control_metadata)
                )
            normalized = torch.stack(
                [core.normalize_video(torch.from_numpy(clip), spec["resolution"]) for clip in control_clips]
            )
            x = normalized.permute(0, 2, 1, 3, 4).contiguous().to("cuda")
            batch_size = len(CONTROL_NAMES)
            enc_mask, pred_mask, full_mask = core.token_masks(CONTEXT, spec, batch_size)
            targets_all = target_encoder(x, full_mask)
            targets = apply_masks(targets_all, [pred_mask], concat=False)[0]
            encoded = encoder(x, enc_mask)
            predictions = predictor(encoded, enc_mask, pred_mask)
            original_index = CONTROL_NAMES.index("original")
            same_context_indices = [
                CONTROL_NAMES.index("target_hold"),
                CONTROL_NAMES.index("target_shuffle"),
            ]
            maximum_predictor_difference = max(
                float(
                    (predictions[original_index] - predictions[index])
                    .abs()
                    .float()
                    .max()
                    .item()
                )
                for index in same_context_indices
            )
            if maximum_predictor_difference > 1e-4:
                raise RuntimeError(
                    f"Target controls changed predictor output for {window['window_id']}: "
                    f"{maximum_predictor_difference}"
                )
            predictor_differences.append(maximum_predictor_difference)
            targets = F.layer_norm(targets, (targets.size(-1),))
            error = (predictions - targets).abs().float()
            temporal_spatial = error.mean(-1).reshape(
                batch_size,
                (FRAMES_PER_CLIP - CONTEXT) // 2,
                spatial_size,
                spatial_size,
            )
            temporal = temporal_spatial.mean((-1, -2))
            if not bool(torch.isfinite(error).all()):
                raise RuntimeError(f"Non-finite loss for {window['window_id']}")
            for control_index, (
                control,
                slot_mapping,
                input_source_indices,
                control_metadata,
            ) in enumerate(control_records):
                anchor_slot = control_metadata["control_anchor_sampled_slot_0based"]
                score_rows.append(
                    {
                        "window_id": window["window_id"],
                        "window_index": window["window_index"],
                        "pair_id": window["pair_id"],
                        "pair_label": window["pair_label"],
                        "condition": window["condition"],
                        "difficulty": window["difficulty"],
                        "camera": window["camera"],
                        **{
                            field: window[field]
                            for field in (*MOTION_FIELDS, "lowest_visual_change_decile")
                        },
                        "control": control,
                        "scalar_l1": float(error[control_index].mean().item()),
                        "temporal_tubelet_l1": temporal[control_index].cpu().tolist(),
                        "control_input_sampled_slot_mapping_0based": slot_mapping.tolist(),
                        "control_input_source_frames_0based": input_source_indices.tolist(),
                        "prediction_context_source_frames_0based": input_source_indices[:CONTEXT].tolist(),
                        "evaluation_target_source_frames_0based": input_source_indices[CONTEXT:].tolist(),
                        "evaluation_target_tubelet_source_frame_pairs_0based": core.target_frame_pairs(
                            input_source_indices, CONTEXT
                        ),
                        "control_anchor_sampled_slot_0based": anchor_slot,
                        "control_anchor_source_frame_0based": (
                            int(original_source_indices[anchor_slot])
                            if anchor_slot is not None
                            else None
                        ),
                        "target_tubelet_permutation": control_metadata[
                            "target_tubelet_permutation"
                        ],
                        "spatial_map_key": control,
                        "spatial_map_window_offset": map_offset,
                    }
                )
                maps[control].append(
                    temporal_spatial[control_index].to(torch.float16).cpu().numpy()
                )
            del targets_all, targets, encoded, predictions, error, x

    payload = {
        "schema_version": 1,
        "dataset": "IntPhys2",
        "split": args.split,
        "name": name,
        "filename": record.get("file_name"),
        "metadata": {key: value for key, value in record.items() if key != "video_path"},
        "controls": list(CONTROL_NAMES),
        "context": CONTEXT,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_step": args.frame_step,
        "window_stride_sampled_frames": args.window_stride,
        "control_construction": "in-memory after shared MP4 decode; no re-encode",
        "target_shuffle_version": SHUFFLE_VERSION,
        "target_shuffle_seed": args.shuffle_seed,
        "scores": score_rows,
        "same_context_predictor_max_abs_difference": max(predictor_differences),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return payload, {control: np.stack(values) for control, values in maps.items()}


def consolidate_scores(
    output_dir: Path,
    records: list[dict],
    run_id: str,
    windows_by_video: dict[str, list[dict]],
) -> None:
    fields = [
        "name",
        "filename",
        "window_id",
        "window_index",
        "pair_id",
        "pair_label",
        "condition",
        "difficulty",
        "camera",
        *MOTION_FIELDS,
        "lowest_visual_change_decile",
        "control",
        "scalar_l1",
        "temporal_tubelet_l1",
        "control_input_sampled_slot_mapping_0based",
        "control_input_source_frames_0based",
        "prediction_context_source_frames_0based",
        "evaluation_target_source_frames_0based",
        "evaluation_target_tubelet_source_frame_pairs_0based",
        "control_anchor_sampled_slot_0based",
        "control_anchor_source_frame_0based",
        "target_tubelet_permutation",
        "spatial_map_key",
        "spatial_map_window_offset",
    ]
    temporary = output_dir / "scores.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        expected_paths = {
            output_dir / "videos" / f"{record['name']}.json" for record in records
        }
        actual_paths = set((output_dir / "videos").glob("*.json"))
        if actual_paths != expected_paths:
            raise ValueError(
                f"Video JSON set differs from selection: extra={sorted(actual_paths - expected_paths)}, "
                f"missing={sorted(expected_paths - actual_paths)}"
            )
        for path in sorted(expected_paths):
            payload = json.loads(path.read_text())
            name = str(payload["name"])
            expected_ids = [row["window_id"] for row in sorted(
                windows_by_video[name], key=lambda row: row["window_index"]
            )]
            _video_complete(output_dir, name, run_id, expected_ids)
            for row in payload["scores"]:
                writer.writerow(
                    {
                        field: (
                            json.dumps(row.get(field), separators=(",", ":"))
                            if isinstance(row.get(field), list)
                            else row.get(field, payload.get(field))
                        )
                        for field in fields
                    }
                )
    temporary.replace(output_dir / "scores.csv")


def write_control_deltas(output_dir: Path, expected_windows: int) -> None:
    scores = pd.read_csv(output_dir / "scores.csv")
    if len(scores) != expected_windows * len(CONTROL_NAMES):
        raise ValueError("Consolidated score row count does not match the locked manifest")
    counts = scores.groupby("window_id")["control"].agg(list)
    if len(counts) != expected_windows or any(set(values) != set(CONTROL_NAMES) for values in counts):
        raise ValueError("Each expected window must contain each control exactly once")
    pivot = scores.pivot(index="window_id", columns="control", values="scalar_l1")
    rows = []
    for window_id, values in pivot.iterrows():
        row = {"window_id": window_id, "original_scalar_l1": float(values["original"])}
        for control in CONTROL_NAMES[1:]:
            row[f"delta_original_minus_{control}"] = float(
                values["original"] - values[control]
            )
        rows.append(row)
    fields = list(rows[0])
    temporary = output_dir / "control_deltas.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_dir / "control_deltas.csv")


def main() -> None:
    args = parse_args()
    if args.frame_step < 1 or args.window_stride < 1:
        raise ValueError("Frame step and window stride must be positive")
    records, selection = selected_records(args)
    windows = prepare_window_manifest(args, records, selection)
    rows_by_video: dict[str, list[dict]] = defaultdict(list)
    for row in windows:
        rows_by_video[str(row["name"])].append(row)
    expected_names = {str(record["name"]) for record in records}
    if set(rows_by_video) != expected_names:
        raise ValueError(
            f"Prepared window names differ from selection: {sorted(expected_names - set(rows_by_video))}"
        )
    for name, video_windows in rows_by_video.items():
        ordered = sorted(video_windows, key=lambda row: row["window_index"])
        if [row["window_index"] for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"Window indices are not contiguous for {name}")
        if len({row["window_id"] for row in ordered}) != len(ordered):
            raise ValueError(f"Duplicate window IDs for {name}")
    if args.prepare_only:
        core.write_atomic_text(
            args.output_dir / "preparation-status.json",
            json.dumps(
                {
                    "status": "complete",
                    "selected_pairs": len(records) // 2,
                    "selected_videos": len(records),
                    "eligible_windows": len(windows),
                    "model_scores_written": False,
                },
                indent=2,
            )
            + "\n",
        )
        return
    if args.checkpoint is None or args.vjepa2_root is None:
        raise ValueError("--checkpoint and --vjepa2-root are required for scoring")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scoring")

    spec = core.MODEL_SPECS["vjepa2-g"]
    videos_dir = args.output_dir / "videos"
    maps_dir = args.output_dir / "spatial_maps"
    videos_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    encoder, target_encoder, predictor, load_info = core.load_models(args, spec)
    provenance = {
        "schema_version": 1,
        "script": "scripts/run_intphys2_construct_validity.py",
        "script_sha256": core.sha256_file(Path(__file__)),
        "core_script_sha256": core.sha256_file(Path(core.__file__)),
        "project_revision": project_revision(),
        "model": "vjepa2-g",
        "model_spec": spec,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": core.sha256_file(args.checkpoint),
        "vjepa2_root": str(args.vjepa2_root),
        "vjepa2_revision": core.revision(args.vjepa2_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": core.sha256_file(args.selection_manifest),
        "window_manifest_sha256": core.sha256_file(args.output_dir / "window_manifest.jsonl"),
        "selected_pairs": len(records) // 2,
        "selected_videos": len(records),
        "eligible_windows": len(windows),
        "controls": list(CONTROL_NAMES),
        "context": CONTEXT,
        "frames_per_clip": FRAMES_PER_CLIP,
        "frame_step": args.frame_step,
        "window_stride": args.window_stride,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_version": SHUFFLE_VERSION,
        "motion_version": MOTION_VERSION,
        "dtype": "bfloat16 autocast",
        "activation_checkpointing": True,
        "condition_batch_size": len(CONTROL_NAMES),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "load_seconds": time.perf_counter() - load_started,
        **load_info,
    }
    provenance["run_id"] = run_identity(provenance)
    provenance_path = args.output_dir / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text())
        if existing.get("run_id") != provenance["run_id"]:
            raise ValueError("Existing output directory belongs to a different run identity")
    else:
        core.write_atomic_text(
            provenance_path, json.dumps(provenance, indent=2) + "\n"
        )

    for video_index, record in enumerate(records):
        name = str(record["name"])
        expected_ids = [
            row["window_id"]
            for row in sorted(rows_by_video[name], key=lambda row: row["window_index"])
        ]
        if _video_complete(args.output_dir, name, provenance["run_id"], expected_ids):
            continue
        payload, maps = score_video(
            args,
            record,
            rows_by_video[name],
            spec,
            (encoder, target_encoder, predictor),
        )
        map_path = maps_dir / f"{name}.npz"
        temporary_map = map_path.with_suffix(".npz.tmp")
        with temporary_map.open("wb") as handle:
            np.savez_compressed(handle, **maps)
        temporary_map.replace(map_path)
        payload["run_id"] = provenance["run_id"]
        payload["spatial_maps_npz"] = str(map_path.relative_to(args.output_dir))
        payload["spatial_maps_sha256"] = core.sha256_file(map_path)
        payload["spatial_map_shapes"] = {
            control: list(array.shape) for control, array in maps.items()
        }
        core.write_atomic_text(
            videos_dir / f"{name}.json",
            json.dumps(payload, separators=(",", ":")) + "\n",
        )
        print(
            json.dumps(
                {
                    "done": video_index + 1,
                    "total": len(records),
                    "name": name,
                    "windows": len(rows_by_video[name]),
                    "wall_seconds": payload["wall_seconds"],
                    "peak_allocated_bytes": payload["peak_allocated_bytes"],
                }
            ),
            flush=True,
        )

    consolidate_scores(args.output_dir, records, provenance["run_id"], rows_by_video)
    write_control_deltas(args.output_dir, len(windows))
    completed = sum(
        _video_complete(
            args.output_dir,
            str(record["name"]),
            provenance["run_id"],
            [
                row["window_id"]
                for row in sorted(
                    rows_by_video[str(record["name"])],
                    key=lambda row: row["window_index"],
                )
            ],
        )
        for record in records
    )
    core.write_atomic_text(
        args.output_dir / "status.json",
        json.dumps(
            {
                "status": "complete" if completed == len(records) else "incomplete",
                "expected_pairs": len(records) // 2,
                "expected_videos": len(records),
                "completed_videos": completed,
                "eligible_windows": len(windows),
                "score_rows": len(pd.read_csv(args.output_dir / "scores.csv")),
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
