#!/usr/bin/env python3
"""Run local V-JEPA2 masked-prediction surprise over video frame sections."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model, VJEPA2VideoProcessor


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    path: Path
    metadata: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute native V-JEPA2 predictor surprise for frame sections."
    )
    parser.add_argument("--input", action="append", default=[], help="Video path or glob; repeat as needed.")
    parser.add_argument("--manifest", type=Path, help="Optional CSV manifest with a video path column.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root for relative manifest paths.")
    parser.add_argument("--path-column", default="path", help="Manifest column containing video paths.")
    parser.add_argument("--id-column", default="", help="Optional manifest column used as video_id.")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="float16" if torch.cuda.is_available() else "float32",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--sections", type=int, default=8)
    parser.add_argument(
        "--segment-frames",
        type=int,
        help=(
            "Use fixed N-frame source segments instead of --sections equal sections. "
            "Only complete N-frame source segments are evaluated."
        ),
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        help="Override model clip length. Defaults to config.frames_per_clip.",
    )
    parser.add_argument(
        "--target-start-fraction",
        type=float,
        default=0.75,
        help="Temporal-token fraction where prediction targets begin.",
    )
    parser.add_argument(
        "--metric",
        default="l1",
        choices=("l1", "l2", "cosine"),
        help="Distance between predictor output and target representation.",
    )
    parser.add_argument(
        "--normalization",
        default="both",
        choices=("none", "target", "both"),
        help="LayerNorm before distance computation.",
    )
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_model_snapshot(model_id: str, local_files_only: bool) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(model_id, local_files_only=local_files_only)
        )
    except Exception:
        return None


def sanitize_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "video"


def resolve_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted({path.resolve() for path in paths})


def read_manifest(args: argparse.Namespace) -> list[VideoSpec]:
    if args.manifest is None:
        return []
    rows: list[VideoSpec] = []
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or args.path_column not in reader.fieldnames:
            raise ValueError(f"Manifest must contain column {args.path_column!r}")
        for index, row in enumerate(reader):
            raw_path = Path(row[args.path_column])
            path = raw_path if raw_path.is_absolute() else args.root / raw_path
            video_id = row.get(args.id_column) if args.id_column else ""
            if not video_id:
                video_id = f"{index:04d}_{path.stem}"
            rows.append(VideoSpec(sanitize_id(video_id), path.resolve(), dict(row)))
    return rows


def collect_videos(args: argparse.Namespace) -> list[VideoSpec]:
    videos = read_manifest(args)
    videos.extend(
        VideoSpec(sanitize_id(path.stem), path, {})
        for path in resolve_inputs(args.input)
    )
    unique: dict[Path, VideoSpec] = {}
    for video in videos:
        unique.setdefault(video.path, video)
    selected = list(unique.values())
    if args.max_videos is not None:
        selected = selected[: args.max_videos]
    if not selected:
        raise ValueError("No input videos were selected")
    missing = [str(video.path) for video in selected if not video.path.exists()]
    if missing:
        raise FileNotFoundError("Missing input videos:\n" + "\n".join(missing))
    return selected


def read_video_rgb(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames), fps


def section_bounds(total_frames: int, sections: int) -> list[tuple[int, int]]:
    if sections < 1:
        raise ValueError("--sections must be positive")
    edges = np.linspace(0, total_frames, sections + 1).round().astype(int)
    bounds = []
    for start, end in zip(edges[:-1], edges[1:]):
        start = int(np.clip(start, 0, total_frames - 1))
        end = int(np.clip(max(end, start + 1), 1, total_frames))
        bounds.append((start, end))
    return bounds


def fixed_segment_bounds(total_frames: int, segment_frames: int) -> list[tuple[int, int]]:
    if segment_frames < 1:
        raise ValueError("--segment-frames must be positive")
    bounds = []
    for start in range(0, total_frames - segment_frames + 1, segment_frames):
        bounds.append((start, start + segment_frames))
    return bounds


def sample_indices(start: int, end: int, frames_per_clip: int) -> np.ndarray:
    if frames_per_clip < 2:
        raise ValueError("frames_per_clip must be at least 2")
    if end <= start:
        end = start + 1
    return np.linspace(start, end - 1, frames_per_clip).round().astype(int)


def make_masks(
    inputs: torch.Tensor,
    model: VJEPA2Model,
    target_start_fraction: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, int, int]:
    config = model.config
    batch_size, frames, _, height, width = inputs.shape
    tubelet = int(config.tubelet_size)
    patch = int(config.patch_size)
    temporal_tokens = frames // tubelet
    spatial_tokens = (height // patch) * (width // patch)
    target_start = int(round(temporal_tokens * target_start_fraction))
    target_start = max(1, min(temporal_tokens - 1, target_start))

    token_ids = torch.arange(temporal_tokens * spatial_tokens, device=inputs.device)
    time_ids = token_ids // spatial_tokens
    context = token_ids[time_ids < target_start].unsqueeze(0).repeat(batch_size, 1)
    target = token_ids[time_ids >= target_start].unsqueeze(0).repeat(batch_size, 1)
    return [context], [target], target[0] // spatial_tokens, target_start, spatial_tokens


def distances(pred: torch.Tensor, target: torch.Tensor, metric: str, normalization: str) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    if normalization in {"target", "both"}:
        target = F.layer_norm(target, (target.shape[-1],))
    if normalization == "both":
        pred = F.layer_norm(pred, (pred.shape[-1],))
    if metric == "l1":
        return (pred - target).abs().mean(-1)
    if metric == "l2":
        return (pred - target).pow(2).mean(-1).sqrt()
    if metric == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1)
    raise ValueError(metric)


def target_frame_ranges(indices: np.ndarray, tubelet_size: int, time_ids: torch.Tensor) -> list[list[int]]:
    ranges = []
    for time_id in torch.unique(time_ids.cpu(), sorted=True).tolist():
        offset = int(time_id) * tubelet_size
        tubelet_indices = indices[offset : offset + tubelet_size]
        ranges.append([int(tubelet_indices[0]), int(tubelet_indices[-1])])
    return ranges


@torch.no_grad()
def score_section(
    model: VJEPA2Model,
    processor: VJEPA2VideoProcessor,
    frames: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    video = frames[indices]
    batch = processor([video], return_tensors="pt")
    inputs = batch["pixel_values_videos"].to(args.device)
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda"):
        inputs = inputs.to(dtype=dtype)
    context_mask, target_mask, target_time_ids, target_start, _ = make_masks(
        inputs, model, args.target_start_fraction
    )
    with torch.autocast("cuda", dtype=dtype, enabled=args.device.startswith("cuda")):
        output = model(
            inputs,
            context_mask=context_mask,
            target_mask=target_mask,
        ).predictor_output
    token_distances = distances(
        output.last_hidden_state,
        output.target_hidden_state,
        args.metric,
        args.normalization,
    )
    temporal = []
    for time_id in torch.unique(target_time_ids, sorted=True):
        temporal.append(token_distances[:, target_time_ids == time_id].mean(dim=1))
    temporal_tensor = torch.stack(temporal, dim=1).squeeze(0)
    temporal_values = temporal_tensor.detach().cpu().tolist()
    peak_index = int(np.argmax(np.asarray(temporal_values)))
    frame_ranges = target_frame_ranges(indices, int(model.config.tubelet_size), target_time_ids)
    return {
        "target_start_tubelet": target_start,
        "target_tubelet_source_frame_ranges_0based": frame_ranges,
        "scalar_surprise": float(token_distances.mean().item()),
        "temporal_tubelet_surprise": temporal_values,
        "peak_target_tubelet_index": peak_index,
        "peak_source_frame_range_0based": frame_ranges[peak_index],
    }


def maybe_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_video.setdefault(str(row["video_id"]), []).append(row)
    for video_id, video_rows in by_video.items():
        video_rows = sorted(video_rows, key=lambda row: int(row["section_index"]))
        xs = [int(row["section_index"]) for row in video_rows]
        ys = [float(row["scalar_surprise"]) for row in video_rows]
        width = max(7.0, 0.45 * len(xs))
        plt.figure(figsize=(width, 3.8))
        plt.plot(xs, ys, marker="o")
        for x, y in zip(xs, ys):
            plt.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                rotation=60 if len(xs) > 18 else 0,
            )
        plt.xticks(xs, [str(x) for x in xs], fontsize=8)
        plt.xlabel("section index")
        plt.ylabel("masked-prediction surprise")
        plt.title(video_id)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{sanitize_id(video_id)}_section_surprise.png", dpi=160)
        plt.close()

        temporal = np.asarray([row["temporal_tubelet_surprise"] for row in video_rows], dtype=float)
        plt.figure(figsize=(7, 3.5))
        plt.imshow(temporal, aspect="auto", interpolation="nearest", cmap="magma")
        plt.xlabel("target tubelet within section")
        plt.ylabel("section index")
        plt.title(video_id)
        plt.colorbar(label="surprise")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{sanitize_id(video_id)}_target_tubelet_heatmap.png", dpi=160)
        plt.close()


def write_outputs(args: argparse.Namespace, rows: list[dict[str, object]], provenance: dict[str, object]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "path",
        "section_index",
        "section_source_start_frame_0based",
        "section_source_end_frame_0based",
        "section_start_seconds",
        "section_end_seconds",
        "sampled_source_frames_0based",
        "target_start_tubelet",
        "target_tubelet_source_frame_ranges_0based",
        "scalar_surprise",
        "temporal_tubelet_surprise",
        "peak_target_tubelet_index",
        "peak_source_frame_range_0based",
    ]
    with (args.output_dir / "section_surprise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], separators=(",", ":"))
                if isinstance(row.get(field), list)
                else row.get(field)
                for field in fields
            })
    video_dir = args.output_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    for video_id in sorted({str(row["video_id"]) for row in rows}):
        video_rows = [row for row in rows if row["video_id"] == video_id]
        payload = {
            "video_id": video_id,
            "path": video_rows[0]["path"],
            "metadata": video_rows[0]["metadata"],
            "sections": video_rows,
        }
        (video_dir / f"{sanitize_id(video_id)}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    scalar = np.asarray([float(row["scalar_surprise"]) for row in rows], dtype=float)
    highest = sorted(rows, key=lambda row: float(row["scalar_surprise"]), reverse=True)[:5]
    summary = {
        **provenance,
        "row_count": len(rows),
        "video_count": len({row["video_id"] for row in rows}),
        "mean_section_surprise": float(scalar.mean()),
        "min_section_surprise": float(scalar.min()),
        "max_section_surprise": float(scalar.max()),
        "highest_sections": [
            {
                "video_id": row["video_id"],
                "section_index": row["section_index"],
                "section_source_start_frame_0based": row["section_source_start_frame_0based"],
                "section_source_end_frame_0based": row["section_source_end_frame_0based"],
                "scalar_surprise": row["scalar_surprise"],
                "peak_source_frame_range_0based": row["peak_source_frame_range_0based"],
            }
            for row in highest
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.plots:
        maybe_plot(args.output_dir, rows)


def main() -> None:
    args = parse_args()
    if args.segment_frames is not None and args.sections < 1:
        raise ValueError("--sections must be positive when --segment-frames is absent")
    videos = collect_videos(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = getattr(torch, args.dtype)

    started = time.perf_counter()
    model_snapshot = cached_model_snapshot(args.model_id, not args.allow_download)
    model_source = str(model_snapshot) if model_snapshot is not None else args.model_id
    processor = VJEPA2VideoProcessor.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
    )
    model = VJEPA2Model.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frames_per_clip = args.frames_per_clip or int(model.config.frames_per_clip)

    rows: list[dict[str, object]] = []
    for video_index, video in enumerate(videos, start=1):
        frames, fps = read_video_rgb(video.path)
        total_frames = int(frames.shape[0])
        bounds = (
            fixed_segment_bounds(total_frames, args.segment_frames)
            if args.segment_frames is not None
            else section_bounds(total_frames, args.sections)
        )
        for section_index, (start, end) in enumerate(bounds):
            indices = sample_indices(start, end, frames_per_clip)
            score = score_section(model, processor, frames, indices, args)
            row = {
                "video_id": video.video_id,
                "path": str(video.path),
                "metadata": video.metadata,
                "section_index": section_index,
                "section_source_start_frame_0based": start,
                "section_source_end_frame_0based": end - 1,
                "section_start_seconds": start / fps if fps > 0 else None,
                "section_end_seconds": (end - 1) / fps if fps > 0 else None,
                "sampled_source_frames_0based": indices.tolist(),
                **score,
            }
            rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"done": video_index, "total": len(videos), "video_id": video.video_id}), flush=True)

    provenance = {
        "script": "scripts/run_vjepa2_section_surprise.py",
        "project_revision": revision(Path.cwd()),
        "model_id": args.model_id,
        "model_snapshot": str(model_snapshot) if model_snapshot is not None else None,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_config": {
            "image_size": int(model.config.image_size),
            "frames_per_clip": int(model.config.frames_per_clip),
            "patch_size": int(model.config.patch_size),
            "tubelet_size": int(model.config.tubelet_size),
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
        },
        "model_safetensors_sha256": (
            sha256_file(model_snapshot / "model.safetensors")
            if model_snapshot is not None and (model_snapshot / "model.safetensors").exists()
            else None
        ),
        "local_files_only": not args.allow_download,
        "device": str(device),
        "dtype": args.dtype,
        "sections": None if args.segment_frames is not None else args.sections,
        "segment_frames": args.segment_frames,
        "sectioning": (
            f"fixed {args.segment_frames}-frame source segments"
            if args.segment_frames is not None
            else f"{args.sections} equal source-frame sections"
        ),
        "frames_per_clip": frames_per_clip,
        "target_start_fraction": args.target_start_fraction,
        "metric": args.metric,
        "normalization": args.normalization,
        "wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    write_outputs(args, rows, provenance)
    print(f"Wrote {args.output_dir / 'section_surprise.csv'}")


if __name__ == "__main__":
    main()
