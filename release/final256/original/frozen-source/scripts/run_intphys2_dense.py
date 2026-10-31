#!/usr/bin/env python3
"""Run frame-addressable native V-JEPA surprise inference on IntPhys 2."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_SPECS = {
    "vjepa-h-public": {
        "architecture": "vit_huge",
        "embed_dim": 1280,
        "encoder_heads": 16,
        "resolution": 224,
        "frames_per_clip": 16,
        "contexts": (4, 6, 8, 10, 12, 14),
        "num_mask_tokens": 2,
        "use_rope": False,
    },
    "vjepa2-g": {
        "architecture": "vit_giant_xformers",
        "embed_dim": 1408,
        "encoder_heads": 22,
        "resolution": 256,
        "frames_per_clip": 48,
        "contexts": (12, 18, 24, 30, 36, 42),
        "num_mask_tokens": 10,
        "use_rope": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa2-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("Debug", "Main", "HeldOut"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument(
        "--appearance", choices=("hue", "brightness", "selective_recolor")
    )
    parser.add_argument("--appearance-value", type=float)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(root: Path) -> str:
    safe_root = root.resolve().as_posix()
    return subprocess.run(
        ["git", "-c", f"safe.directory={safe_root}", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clean_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module.backbone."): value for key, value in state.items()}


def load_component(builder, state: dict[str, torch.Tensor], *, allow_pos_resize: bool):
    model = builder()
    cleaned = clean_state(state)
    if allow_pos_resize:
        for key in ("pos_embed", "predictor_pos_embed"):
            if key in cleaned and key in model.state_dict():
                if cleaned[key].shape != model.state_dict()[key].shape:
                    del cleaned[key]
    result = model.load_state_dict(cleaned, strict=not allow_pos_resize)
    if allow_pos_resize:
        missing = set(result.missing_keys)
        if missing not in ({"pos_embed"}, {"predictor_pos_embed"}, set()):
            raise ValueError(f"Unexpected missing keys: {sorted(missing)}")
        if result.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys}")
    model = model.to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, str(result)


def load_models(args: argparse.Namespace, spec: dict):
    sys.path.insert(0, str(args.vjepa2_root))
    from src.models import predictor as predictor_module
    from src.models import vision_transformer as vit

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    common = {
        "img_size": spec["resolution"],
        "num_frames": spec["frames_per_clip"],
        "uniform_power": True,
        "use_activation_checkpointing": True,
        "use_rope": spec["use_rope"],
    }
    architecture = getattr(vit, spec["architecture"])
    allow_pos_resize = not spec["use_rope"]
    encoder, enc_result = load_component(
        lambda: architecture(**common), checkpoint["encoder"],
        allow_pos_resize=allow_pos_resize,
    )
    target_encoder, target_result = load_component(
        lambda: architecture(**common), checkpoint["target_encoder"],
        allow_pos_resize=allow_pos_resize,
    )
    predictor, pred_result = load_component(
        lambda: predictor_module.vit_predictor(
            img_size=spec["resolution"],
            num_frames=spec["frames_per_clip"],
            embed_dim=spec["embed_dim"],
            predictor_embed_dim=384,
            depth=12,
            num_heads=12,
            uniform_power=True,
            use_activation_checkpointing=True,
            use_mask_tokens=True,
            use_rope=spec["use_rope"],
            zero_init_mask_tokens=True,
            num_mask_tokens=spec["num_mask_tokens"],
        ),
        checkpoint["predictor"],
        allow_pos_resize=allow_pos_resize,
    )
    checkpoint_keys = sorted(checkpoint)
    del checkpoint
    gc.collect()
    return encoder, target_encoder, predictor, {
        "checkpoint_keys": checkpoint_keys,
        "strict_load": {
            "encoder": enc_result,
            "target_encoder": target_result,
            "predictor": pred_result,
        },
    }


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    height, width = hsv.shape[:2]
    image = Image.frombytes("HSV", (width, height), hsv.tobytes())
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def appearance_transform(
    frames: np.ndarray,
    transform: str | None,
    value: float | None,
    selective_center_degrees: float | None,
) -> np.ndarray:
    if transform is None:
        return frames
    if value is None:
        raise ValueError("--appearance-value is required with --appearance")
    if transform == "brightness":
        return np.clip(np.rint(frames.astype(np.float32) * value), 0, 255).astype(np.uint8)
    output = []
    shift = int(round(value * 256.0 / 360.0))
    for frame in frames:
        hsv = rgb_to_hsv(frame).copy()
        shifted = (hsv[..., 0].astype(np.uint16) + shift) % 256
        if transform == "hue":
            hsv[..., 0] = shifted
        else:
            if selective_center_degrees is None:
                raise ValueError("Selective recoloring requires a paired hue center")
            hue_degrees = hsv[..., 0].astype(np.float32) * (360.0 / 256.0)
            distance = np.abs(
                (hue_degrees - selective_center_degrees + 180.0) % 360.0 - 180.0
            )
            mask = (distance <= 20.0) & (hsv[..., 1] >= 64)
            hsv[..., 0][mask] = shifted[mask]
        output.append(hsv_to_rgb(hsv))
    return np.stack(output)


def normalize_video(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    frames = frames.float().permute(0, 3, 1, 2)
    frames = F.interpolate(
        frames, size=(resolution, resolution), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (frames / 255.0 - mean) / std


def split_records(split_root: Path) -> list[dict]:
    metadata_path = split_root / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path).where(pd.notna, None)
        records = metadata.to_dict("records")
        for record in records:
            record["video_path"] = str(split_root / record["file_name"])
        return records
    return [
        {
            "name": path.stem,
            "file_name": f"Videos/{path.name}",
            "video_path": str(path),
            "violation_annotation_source": "not released",
        }
        for path in sorted((split_root / "Videos").glob("*.mp4"))
    ]


def selective_centers(
    records: list[dict], *, allow_infeasible: bool = False
) -> dict[str, float | None]:
    from decord import VideoReader, cpu

    pairs: dict[str, list[dict]] = {}
    for record in records:
        if "SceneIndex" not in record or "type" not in record:
            raise ValueError("Selective recoloring is unavailable without released pair metadata")
        member = str(record["type"]).split("_", 1)[0].strip()
        pairs.setdefault(f"{record['SceneIndex']}/{member}", []).append(record)
    centers = {}
    for pair, members in pairs.items():
        if len(members) != 2:
            raise ValueError(f"Expected two videos for {pair}, found {len(members)}")
        samples = []
        for member in members:
            reader = VideoReader(member["video_path"], ctx=cpu(0))
            hsv = rgb_to_hsv(reader[0].asnumpy())
            selected = hsv[..., 0][hsv[..., 1] >= 64]
            if selected.size:
                samples.append(selected)
        if not samples:
            if not allow_infeasible:
                raise ValueError(f"No sufficiently saturated anchor pixels for {pair}")
            center = None
        else:
            hues = np.concatenate(samples).astype(np.int32)
            bins = np.minimum(hues * 36 // 256, 35)
            center = (int(np.argmax(np.bincount(bins, minlength=36))) + 0.5) * 10.0
        for member in members:
            centers[str(member.get("name"))] = center
    return centers


def token_masks(context: int, spec: dict, batch_size: int = 1):
    spatial = spec["resolution"] // 16
    tokens_per_tubelet = spatial * spatial
    total_tubelets = spec["frames_per_clip"] // 2
    context_tokens = context // 2 * tokens_per_tubelet
    full = torch.arange(total_tubelets * tokens_per_tubelet, device="cuda").unsqueeze(0)
    full = full.repeat(batch_size, 1)
    return full[:, :context_tokens], full[:, context_tokens:], full


def target_frame_pairs(indices: np.ndarray, context: int) -> list[list[int]]:
    return [
        [int(indices[offset]), int(indices[offset + 1])]
        for offset in range(context, len(indices), 2)
    ]


def run_video(
    record: dict, contexts: tuple[int, ...], args, spec, models,
    selective_center_degrees: float | None,
):
    from src.masks.utils import apply_masks

    from decord import VideoReader, cpu

    encoder, target_encoder, predictor = models
    path = Path(record["video_path"])
    reader = VideoReader(str(path), ctx=cpu(0))
    fps = float(reader.get_avg_fps())
    total_source_frames = len(reader)
    sampled = np.arange(0, total_source_frames, args.frame_step)
    decoded = reader.get_batch(sampled).asnumpy()
    transformed = appearance_transform(
        decoded, args.appearance, args.appearance_value, selective_center_degrees
    )
    frames = normalize_video(torch.from_numpy(transformed), spec["resolution"])
    starts = list(range(
        0,
        len(sampled) - spec["frames_per_clip"] + 1,
        args.window_stride,
    ))
    spatial = spec["resolution"] // 16
    rows = []
    maps: dict[str, list[np.ndarray]] = {str(context): [] for context in contexts}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_start in range(0, len(starts), args.window_batch_size):
            batch_starts = starts[batch_start : batch_start + args.window_batch_size]
            batch_indices = [
                sampled[start : start + spec["frames_per_clip"]] for start in batch_starts
            ]
            clips = torch.stack([
                frames[start : start + spec["frames_per_clip"]].permute(1, 0, 2, 3)
                for start in batch_starts
            ])
            x = clips.contiguous().to("cuda")
            batch_size = len(batch_starts)
            full_mask = token_masks(contexts[0], spec, batch_size)[2]
            targets_all = target_encoder(x, full_mask)
            for context in contexts:
                enc_mask, pred_mask, _ = token_masks(context, spec, batch_size)
                targets = apply_masks(targets_all, [pred_mask], concat=False)[0]
                encoded = encoder(x, enc_mask)
                predictions = predictor(encoded, enc_mask, pred_mask)
                targets = F.layer_norm(targets, (targets.size(-1),))
                error = (predictions - targets).abs().float()
                temporal_spatial = error.mean(-1).reshape(
                    batch_size,
                    (spec["frames_per_clip"] - context) // 2,
                    spatial,
                    spatial,
                )
                temporal = temporal_spatial.mean((-1, -2))
                if not bool(torch.isfinite(error).all()):
                    raise RuntimeError(f"Non-finite loss for {path}, batch {batch_start}")
                for offset, (start, clip_indices) in enumerate(zip(batch_starts, batch_indices)):
                    window_index = batch_start + offset
                    pairs = target_frame_pairs(clip_indices, context)
                    rows.append({
                        "window_index": window_index,
                        "context_length": context,
                        "window_sampled_start": start,
                        "window_source_start_frame_0based": int(clip_indices[0]),
                        "window_source_end_frame_0based": int(clip_indices[-1]),
                        "context_source_frames_0based": clip_indices[:context].tolist(),
                        "target_source_frames_0based": clip_indices[context:].tolist(),
                        "target_tubelet_source_frame_pairs_0based": pairs,
                        "target_tubelet_source_second_spans": [
                            [pair[0] / fps, pair[1] / fps] for pair in pairs
                        ],
                        "scalar_l1": float(error[offset].mean().item()),
                        "temporal_tubelet_l1": temporal[offset].cpu().tolist(),
                    })
                    maps[str(context)].append(
                        temporal_spatial[offset].to(torch.float16).cpu().numpy()
                    )
            del targets_all, x

    payload = {
        "dataset": "IntPhys2",
        "split": args.split,
        "filename": record.get("file_name"),
        "name": record.get("name", path.stem),
        "metadata": {key: value for key, value in record.items() if key != "video_path"},
        "fps": fps,
        "duration_seconds": total_source_frames / fps,
        "total_source_frames": total_source_frames,
        "all_sampled_source_frame_indices_0based": sampled.tolist(),
        "frame_step": args.frame_step,
        "appearance_transform": args.appearance,
        "appearance_value": args.appearance_value,
        "selective_source_hue_center_degrees": selective_center_degrees,
        "appearance_execution": "on-the-fly after shared MP4 decoding; no re-encode",
        "frames_per_clip": spec["frames_per_clip"],
        "window_stride_sampled_frames": args.window_stride,
        "window_batch_size": args.window_batch_size,
        "violation_start_frame": None,
        "violation_end_frame": None,
        "violation_start_seconds": None,
        "violation_end_seconds": None,
        "violation_annotation_source": "not released",
        "windows": rows,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    return payload, {key: np.stack(value) for key, value in maps.items()}


def write_atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def consolidate(output_dir: Path) -> None:
    fields = [
        "name", "filename", "window_index", "context_length",
        "window_source_start_frame_0based", "window_source_end_frame_0based",
        "scalar_l1", "context_source_frames_0based", "target_source_frames_0based",
        "target_tubelet_source_frame_pairs_0based", "temporal_tubelet_l1",
    ]
    temporary = output_dir / "windows.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path in sorted((output_dir / "videos").glob("*.json")):
            payload = json.loads(path.read_text())
            for row in payload["windows"]:
                writer.writerow({
                    field: (
                        json.dumps(row[field], separators=(",", ":"))
                        if isinstance(row.get(field), list) else row.get(field, payload.get(field))
                    )
                    for field in fields
                })
    temporary.replace(output_dir / "windows.csv")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = MODEL_SPECS[args.model]
    contexts = tuple(args.context or spec["contexts"])
    if any(context <= 0 or context >= spec["frames_per_clip"] or context % 2 for context in contexts):
        raise ValueError("Contexts must be positive even values below frames_per_clip")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError("Output path is not a directory")
    videos_dir = args.output_dir / "videos"
    maps_dir = args.output_dir / "spatial_maps"
    videos_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    records = split_records(args.dataset_root / args.split)
    if args.max_videos is not None:
        records = records[: args.max_videos]
    if not records:
        raise ValueError("No videos found")
    if args.window_batch_size < 1:
        raise ValueError("--window-batch-size must be positive")
    if (args.appearance is None) != (args.appearance_value is None):
        raise ValueError("--appearance and --appearance-value must be supplied together")
    centers = selective_centers(records) if args.appearance == "selective_recolor" else {}

    load_started = time.perf_counter()
    encoder, target_encoder, predictor, load_info = load_models(args, spec)
    provenance = {
        "model": args.model,
        "model_spec": spec,
        "contexts": contexts,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "vjepa2_root": str(args.vjepa2_root),
        "vjepa2_revision": revision(args.vjepa2_root),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "video_count": len(records),
        "frame_step": args.frame_step,
        "window_stride": args.window_stride,
        "window_batch_size": args.window_batch_size,
        "appearance_transform": args.appearance,
        "appearance_value": args.appearance_value,
        "appearance_execution": "on-the-fly after shared MP4 decoding; no re-encode",
        "batch_size": 1,
        "dtype": "bfloat16 autocast",
        "activation_checkpointing": True,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "load_seconds": time.perf_counter() - load_started,
        **load_info,
    }
    write_atomic_text(args.output_dir / "provenance.json", json.dumps(provenance, indent=2) + "\n")

    for index, record in enumerate(records):
        name = record.get("name") or Path(record["video_path"]).stem
        json_path = videos_dir / f"{name}.json"
        map_path = maps_dir / f"{name}.npz"
        if json_path.exists() and map_path.exists():
            continue
        payload, maps = run_video(
            record,
            contexts,
            args,
            spec,
            (encoder, target_encoder, predictor),
            centers.get(str(name)),
        )
        temporary_map = map_path.with_suffix(".npz.tmp")
        with temporary_map.open("wb") as handle:
            np.savez_compressed(handle, **{f"context_{key}": value for key, value in maps.items()})
        temporary_map.replace(map_path)
        payload["temporal_spatial_l1_npz"] = str(map_path.relative_to(args.output_dir))
        payload["temporal_spatial_l1_shapes"] = {
            key: list(value.shape) for key, value in maps.items()
        }
        write_atomic_text(json_path, json.dumps(payload, separators=(",", ":")) + "\n")
        print(json.dumps({
            "done": index + 1,
            "total": len(records),
            "name": name,
            "wall_seconds": payload["wall_seconds"],
            "peak_allocated_bytes": payload["peak_allocated_bytes"],
        }), flush=True)

    consolidate(args.output_dir)
    write_atomic_text(args.output_dir / "status.json", json.dumps({
        "status": "complete",
        "expected_videos": len(records),
        "completed_videos": len(list(videos_dir.glob("*.json"))),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
