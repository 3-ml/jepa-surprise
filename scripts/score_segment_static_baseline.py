from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import VJEPA2Model, VJEPA2VideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vjepa2_section_surprise import (
    cached_model_snapshot,
    distances,
    make_masks,
    read_video_rgb,
    revision,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one segment against frozen-frame static V-JEPA2 baselines."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--section-csv", required=True, type=Path)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="float16" if torch.cuda.is_available() else "float32",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument("--target-start-fraction", type=float, default=0.75)
    parser.add_argument("--metric", default="l1", choices=("l1", "l2", "cosine"))
    parser.add_argument(
        "--normalization",
        default="both",
        choices=("none", "target", "both"),
    )
    return parser.parse_args()


def source_ranges_from_mapping(
    frame_mapping: np.ndarray,
    tubelet_size: int,
    target_time_ids: torch.Tensor,
) -> list[list[int]]:
    ranges = []
    for time_id in torch.unique(target_time_ids.cpu(), sorted=True).tolist():
        offset = int(time_id) * tubelet_size
        mapped = frame_mapping[offset : offset + tubelet_size]
        ranges.append([int(mapped[0]), int(mapped[-1])])
    return ranges


@torch.no_grad()
def score_clip(
    name: str,
    clip: np.ndarray,
    source_frame_mapping: np.ndarray,
    model: VJEPA2Model,
    processor: VJEPA2VideoProcessor,
    args: argparse.Namespace,
) -> dict[str, object]:
    batch = processor([clip], return_tensors="pt")
    inputs = batch["pixel_values_videos"].to(args.device)
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda"):
        inputs = inputs.to(dtype=dtype)

    context_mask, target_mask, target_time_ids, target_start, spatial_tokens = make_masks(
        inputs,
        model,
        args.target_start_fraction,
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

    _, _, _, height, width = inputs.shape
    patch = int(model.config.patch_size)
    grid_h = int(height // patch)
    grid_w = int(width // patch)
    if grid_h * grid_w != int(spatial_tokens):
        raise ValueError("Unexpected spatial token shape")

    temporal_values = []
    spatial_maps = []
    for time_id in torch.unique(target_time_ids, sorted=True):
        values = token_distances[0, target_time_ids == time_id]
        temporal_values.append(float(values.mean().item()))
        spatial_maps.append(values.float().detach().cpu().reshape(grid_h, grid_w).numpy())

    frame_ranges = source_ranges_from_mapping(
        source_frame_mapping,
        int(model.config.tubelet_size),
        target_time_ids,
    )
    peak_index = int(np.argmax(np.asarray(temporal_values)))
    return {
        "variant": name,
        "source_frame_mapping": source_frame_mapping.astype(int).tolist(),
        "target_start_tubelet": int(target_start),
        "target_source_frame_ranges_0based": frame_ranges,
        "scalar_surprise": float(token_distances.mean().item()),
        "temporal_tubelet_surprise": temporal_values,
        "peak_target_tubelet_index": peak_index,
        "peak_source_frame_range_0based": frame_ranges[peak_index],
        "spatial_maps": spatial_maps,
    }


def put_label(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    out = frame_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {path}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_visual_clips(
    output_dir: Path,
    original_clip: np.ndarray,
    static_clip: np.ndarray,
    start_frame: int,
    freeze_frame: int,
    fps: float,
) -> None:
    original_labeled = [
        put_label(frame, f"original frame {start_frame + i}") for i, frame in enumerate(original_clip)
    ]
    static_labeled = [
        put_label(frame, f"static frame {freeze_frame} repeated") for frame in static_clip
    ]
    side_by_side = [
        np.concatenate([left, right], axis=1)
        for left, right in zip(original_labeled, static_labeled)
    ]
    write_video(output_dir / "segment11_original.mp4", original_labeled, fps)
    write_video(output_dir / "segment11_static_frame176_repeated.mp4", static_labeled, fps)
    write_video(output_dir / "segment11_original_vs_static_frame176.mp4", side_by_side, fps)


def draw_score_chart(rows: list[dict[str, object]], output_path: Path) -> None:
    variants = [str(row["variant"]) for row in rows]
    scores = [float(row["scalar_surprise"]) for row in rows]
    colors = ["#4f6d7a"] + ["#2a9d8f"] * (len(rows) - 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.84)
    bars = ax.bar(variants, scores, color=colors, edgecolor="#1f2933", linewidth=0.8)
    ax.set_ylabel("V-JEPA2 scalar surprise")
    ax.set_title("Segment 11 static baseline", loc="left", fontsize=15, weight="bold")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    y_min = max(0.0, min(scores) - 0.035)
    y_max = max(scores) + 0.035
    ax.set_ylim(y_min, y_max)
    for bar, value in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if len(rows) > 1:
        residual = scores[0] - scores[1]
        ax.text(
            0.02,
            0.98,
            f"original - static_first = {residual:+.6f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#334155",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_residual_maps(rows: list[dict[str, object]], output_path: Path) -> None:
    maps = [m for row in rows for m in row["spatial_maps"]]
    vmin = min(float(np.min(m)) for m in maps)
    vmax = max(float(np.max(m)) for m in maps)
    cols = len(rows[0]["spatial_maps"])
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.3 * cols, 3.0 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    image = None
    for row_index, row in enumerate(rows):
        ranges = row["target_source_frame_ranges_0based"]
        for col_index, spatial_map in enumerate(row["spatial_maps"]):
            ax = axes[row_index, col_index]
            image = ax.imshow(spatial_map, cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{row['variant']}\nframes {ranges[col_index][0]}-{ranges[col_index][1]}",
                fontsize=9,
            )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="latent residual")
    fig.suptitle("Target-token latent residual maps", fontsize=15, weight="bold", x=0.08, ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, object]],
    original_row: pd.Series,
    args: argparse.Namespace,
    model_snapshot: Path | None,
    model: VJEPA2Model,
    fps: float,
    frame_count: int,
    start_frame: int,
    end_frame: int,
) -> None:
    serializable_rows = []
    for row in rows:
        serializable = {k: v for k, v in row.items() if k != "spatial_maps"}
        serializable_rows.append(serializable)
    df = pd.DataFrame(serializable_rows)
    df.to_csv(output_dir / "baseline_scores.csv", index=False)

    original_score = float(rows[0]["scalar_surprise"])
    static_scores = [float(row["scalar_surprise"]) for row in rows[1:]]
    summary = {
        "script": "scripts/score_segment_static_baseline.py",
        "project_revision": revision(Path.cwd()),
        "video": str(args.video),
        "section_csv": str(args.section_csv),
        "frame_count": frame_count,
        "fps": fps,
        "segment_index": args.segment_index,
        "segment_start_frame_0based": start_frame,
        "segment_end_frame_0based": end_frame,
        "original_full_video_csv_scalar_surprise": float(original_row["scalar_surprise"]),
        "original_in_memory_scalar_surprise": original_score,
        "original_minus_full_video_csv": original_score - float(original_row["scalar_surprise"]),
        "static_first_scalar_surprise": static_scores[0],
        "static_middle_scalar_surprise": static_scores[1],
        "static_last_scalar_surprise": static_scores[2],
        "static_mean_scalar_surprise": float(np.mean(static_scores)),
        "event_residual_original_minus_static_first": original_score - static_scores[0],
        "event_residual_original_minus_static_mean": original_score - float(np.mean(static_scores)),
        "scores": serializable_rows,
        "model_id": args.model_id,
        "model_snapshot": str(model_snapshot) if model_snapshot is not None else None,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_safetensors_sha256": (
            sha256_file(model_snapshot / "model.safetensors")
            if model_snapshot is not None and (model_snapshot / "model.safetensors").exists()
            else None
        ),
        "device": args.device,
        "dtype": args.dtype,
        "target_start_fraction": args.target_start_fraction,
        "metric": args.metric,
        "normalization": args.normalization,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frames, fps = read_video_rgb(args.video)
    section_df = pd.read_csv(args.section_csv)
    matching = section_df[section_df["section_index"] == args.segment_index]
    if len(matching) != 1:
        raise ValueError(f"Expected one row for segment {args.segment_index}, found {len(matching)}")
    original_row = matching.iloc[0]
    start = int(original_row["section_source_start_frame_0based"])
    end = int(original_row["section_source_end_frame_0based"])
    original_clip = frames[start : end + 1]
    if len(original_clip) != 16:
        raise ValueError(f"Expected a 16-frame segment, got {len(original_clip)} frames")

    first = start
    middle = start + len(original_clip) // 2
    last = end
    clips = [
        ("original_segment11", original_clip, np.arange(start, end + 1)),
        (
            "static_first_frame176",
            np.repeat(frames[first : first + 1], len(original_clip), axis=0),
            np.full(len(original_clip), first),
        ),
        (
            "static_middle_frame184",
            np.repeat(frames[middle : middle + 1], len(original_clip), axis=0),
            np.full(len(original_clip), middle),
        ),
        (
            "static_last_frame191",
            np.repeat(frames[last : last + 1], len(original_clip), axis=0),
            np.full(len(original_clip), last),
        ),
    ]

    model_snapshot = cached_model_snapshot(args.model_id, not args.allow_download)
    model_source = str(model_snapshot) if model_snapshot is not None else args.model_id
    processor = VJEPA2VideoProcessor.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
    )
    dtype = getattr(torch, args.dtype)
    model = VJEPA2Model.from_pretrained(
        model_source,
        local_files_only=not args.allow_download,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = [
        score_clip(name, clip, mapping, model, processor, args)
        for name, clip, mapping in clips
    ]
    write_visual_clips(
        args.output_dir,
        original_clip,
        clips[1][1],
        start,
        first,
        fps if fps > 0 else 30.0,
    )
    draw_score_chart(rows, args.output_dir / "segment11_static_baseline_scores.png")
    draw_residual_maps(rows, args.output_dir / "segment11_target_residual_maps.png")
    write_outputs(
        args.output_dir,
        rows,
        original_row,
        args,
        model_snapshot,
        model,
        fps,
        int(frames.shape[0]),
        start,
        end,
    )
    print((args.output_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
