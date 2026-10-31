from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    fixed_segment_bounds,
    read_video_rgb,
    revision,
    sample_indices,
    score_section,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V-JEPA2 segment surprise against pixels and static baselines."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/physionpp/mass_only/testdata_v1"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Video path to exclude; repeat as needed.",
    )
    parser.add_argument("--segment-frames", type=int, default=16)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
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
    parser.add_argument(
        "--static-motion-threshold",
        type=float,
        default=0.01,
        help="Mean consecutive RGB diff below this is treated as essentially static.",
    )
    return parser.parse_args()


def sanitize_id(path: Path) -> str:
    return "__".join(path.with_suffix("").parts[-3:])


def select_videos(root: Path, limit: int, excludes: list[str]) -> list[Path]:
    root = root.resolve()
    exclude_resolved = {Path(value).resolve() for value in excludes}
    videos = sorted(path.resolve() for path in root.rglob("*_img.mp4"))
    videos = [path for path in videos if path not in exclude_resolved]
    if len(videos) <= limit:
        return videos
    positions = np.linspace(0, len(videos) - 1, limit).round().astype(int).tolist()
    selected: list[Path] = []
    seen: set[int] = set()
    for pos in positions:
        if pos not in seen:
            selected.append(videos[pos])
            seen.add(pos)
    cursor = 0
    while len(selected) < limit and cursor < len(videos):
        if cursor not in seen:
            selected.append(videos[cursor])
            seen.add(cursor)
        cursor += 1
    return selected[:limit]


def pair_metrics(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    diff = np.abs(current.astype(np.int16) - previous.astype(np.int16)).astype(
        np.float32
    )
    max_channel = diff.max(axis=2)
    return {
        "mean_abs_rgb": float(diff.mean()),
        "max_abs_rgb": float(diff.max()),
        "changed_pixel_fraction_gt2": float((max_channel > 2).mean()),
        "changed_pixel_fraction_gt5": float((max_channel > 5).mean()),
        "changed_pixel_fraction_gt10": float((max_channel > 10).mean()),
    }


def pixel_summary_for_bounds(
    frames: np.ndarray,
    bounds: list[tuple[int, int]],
) -> list[dict[str, float | int]]:
    pair_rows = []
    for current in range(1, int(frames.shape[0])):
        row = {
            "previous_frame_0based": current - 1,
            "current_frame_0based": current,
        }
        row.update(pair_metrics(frames[current - 1], frames[current]))
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)

    summaries = []
    for section_index, (start, end_exclusive) in enumerate(bounds):
        end = end_exclusive - 1
        within = pair_df[
            (pair_df["current_frame_0based"] >= start + 1)
            & (pair_df["current_frame_0based"] <= end)
        ]
        first_last = pair_metrics(frames[start], frames[end])
        summaries.append(
            {
                "section_index": section_index,
                "within_mean_abs_rgb": float(within["mean_abs_rgb"].mean()),
                "within_max_pair_mean_abs_rgb": float(within["mean_abs_rgb"].max()),
                "within_changed_fraction_gt5": float(
                    within["changed_pixel_fraction_gt5"].mean()
                ),
                "first_last_mean_abs_rgb": float(first_last["mean_abs_rgb"]),
                "first_last_changed_fraction_gt5": float(
                    first_last["changed_pixel_fraction_gt5"]
                ),
            }
        )
    return summaries


def static_variant_scores(
    model: VJEPA2Model,
    processor: VJEPA2VideoProcessor,
    frames: np.ndarray,
    start: int,
    end_exclusive: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    clip_len = end_exclusive - start
    middle = start + clip_len // 2
    last = end_exclusive - 1
    original_clip = frames[start:end_exclusive]
    variants = {
        "rescore_original": original_clip,
        "static_first": np.repeat(frames[start : start + 1], clip_len, axis=0),
        "static_middle": np.repeat(frames[middle : middle + 1], clip_len, axis=0),
        "static_last": np.repeat(frames[last : last + 1], clip_len, axis=0),
    }
    scores = {}
    for name, clip in variants.items():
        score = score_section(
            model,
            processor,
            clip,
            np.arange(clip_len),
            args,
        )
        scores[name] = float(score["scalar_surprise"])
    static_values = [
        scores["static_first"],
        scores["static_middle"],
        scores["static_last"],
    ]
    scores["static_mean"] = float(np.mean(static_values))
    scores["residual_minus_static_first"] = (
        scores["rescore_original"] - scores["static_first"]
    )
    scores["residual_minus_static_mean"] = (
        scores["rescore_original"] - scores["static_mean"]
    )
    return scores


def draw_scatter(segment_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.14)
    ax.scatter(
        segment_df["within_mean_abs_rgb"],
        segment_df["scalar_surprise"],
        c=segment_df["video_ordinal"],
        cmap="viridis",
        s=32,
        alpha=0.82,
        edgecolor="#1f2933",
        linewidth=0.3,
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Mean consecutive RGB pixel difference, symlog scale")
    ax.set_ylabel("V-JEPA2 scalar surprise")
    ax.set_title(
        "Raw surprise versus pixel motion, 10 additional videos",
        loc="left",
        weight="bold",
    )
    ax.grid(True, color="#d8dee4", linewidth=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_residual_chart(static_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = static_df.copy()
    labels = [
        f"v{int(row.video_ordinal)} s{int(row.section_index)}\n{row.candidate_kind}"
        for row in plot_df.itertuples()
    ]
    colors = [
        "#d95f02" if value <= 0 else "#2a9d8f"
        for value in plot_df["residual_minus_static_first"]
    ]
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(plot_df)), 5.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.32)
    ax.bar(
        np.arange(len(plot_df)),
        plot_df["residual_minus_static_first"],
        color=colors,
        edgecolor="#1f2933",
        linewidth=0.6,
    )
    ax.axhline(0, color="#1f2933", linewidth=1.0)
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Original surprise - static first-frame surprise")
    ax.set_title(
        "Static-baseline residuals for audited segments", loc="left", weight="bold"
    )
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    selected = select_videos(args.root, args.limit, args.exclude)
    if len(selected) != args.limit:
        raise ValueError(f"Selected {len(selected)} videos, expected {args.limit}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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

    selected_rows = []
    segment_rows = []
    static_rows = []
    for video_ordinal, video_path in enumerate(selected):
        frames, fps = read_video_rgb(video_path)
        total_frames = int(frames.shape[0])
        bounds = fixed_segment_bounds(total_frames, args.segment_frames)
        pixel_rows = pixel_summary_for_bounds(frames, bounds)
        video_id = sanitize_id(video_path.relative_to(args.root))
        selected_rows.append(
            {
                "video_ordinal": video_ordinal,
                "video_id": video_id,
                "path": str(video_path),
                "frame_count": total_frames,
                "fps": fps,
                "complete_segment_count": len(bounds),
            }
        )

        for section_index, (start, end_exclusive) in enumerate(bounds):
            indices = sample_indices(start, end_exclusive, args.frames_per_clip)
            score = score_section(model, processor, frames, indices, args)
            row = {
                "video_ordinal": video_ordinal,
                "video_id": video_id,
                "path": str(video_path),
                "section_index": section_index,
                "section_source_start_frame_0based": start,
                "section_source_end_frame_0based": end_exclusive - 1,
                "sampled_source_frames_0based": indices.tolist(),
                **score,
                **pixel_rows[section_index],
            }
            segment_rows.append(row)

        video_df = pd.DataFrame(
            [row for row in segment_rows if row["video_id"] == video_id]
        )
        video_df["surprise_rank_desc"] = video_df["scalar_surprise"].rank(
            method="min",
            ascending=False,
        )
        video_df["pixel_motion_rank_desc"] = video_df["within_mean_abs_rgb"].rank(
            method="min",
            ascending=False,
        )
        for row in video_df.to_dict(orient="records"):
            index = next(
                i
                for i, segment_row in enumerate(segment_rows)
                if segment_row["video_id"] == video_id
                and segment_row["section_index"] == row["section_index"]
            )
            segment_rows[index]["surprise_rank_desc"] = int(row["surprise_rank_desc"])
            segment_rows[index]["pixel_motion_rank_desc"] = int(
                row["pixel_motion_rank_desc"]
            )

        top = video_df.sort_values("scalar_surprise", ascending=False).iloc[0]
        candidates = [("top_raw_surprise", top)]
        static_segments = video_df[
            video_df["within_mean_abs_rgb"] < args.static_motion_threshold
        ]
        if len(static_segments):
            suspicious = static_segments.sort_values(
                "scalar_surprise", ascending=False
            ).iloc[0]
            if int(suspicious["section_index"]) != int(top["section_index"]):
                candidates.append(("highest_surprise_static_segment", suspicious))

        for candidate_kind, candidate in candidates:
            start = int(candidate["section_source_start_frame_0based"])
            end_exclusive = int(candidate["section_source_end_frame_0based"]) + 1
            static_scores = static_variant_scores(
                model,
                processor,
                frames,
                start,
                end_exclusive,
                args,
            )
            static_rows.append(
                {
                    "video_ordinal": video_ordinal,
                    "video_id": video_id,
                    "path": str(video_path),
                    "candidate_kind": candidate_kind,
                    "section_index": int(candidate["section_index"]),
                    "section_source_start_frame_0based": start,
                    "section_source_end_frame_0based": end_exclusive - 1,
                    "raw_csv_scalar_surprise": float(candidate["scalar_surprise"]),
                    "within_mean_abs_rgb": float(candidate["within_mean_abs_rgb"]),
                    "within_changed_fraction_gt5": float(
                        candidate["within_changed_fraction_gt5"]
                    ),
                    "surprise_rank_desc": int(candidate["surprise_rank_desc"]),
                    "pixel_motion_rank_desc": int(candidate["pixel_motion_rank_desc"]),
                    **static_scores,
                }
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "done": video_ordinal + 1,
                    "total": len(selected),
                    "video_id": video_id,
                    "segments": len(bounds),
                }
            ),
            flush=True,
        )

    selected_df = pd.DataFrame(selected_rows)
    segment_df = pd.DataFrame(segment_rows)
    static_df = pd.DataFrame(static_rows)

    selected_df.to_csv(args.output_dir / "selected_videos.csv", index=False)
    segment_df.to_csv(
        args.output_dir / "segment_surprise_pixel_summary.csv", index=False
    )
    static_df.to_csv(args.output_dir / "static_baseline_candidates.csv", index=False)
    draw_scatter(segment_df, args.output_dir / "surprise_vs_pixel_motion_scatter.png")
    draw_residual_chart(static_df, args.output_dir / "static_baseline_residuals.png")

    top_df = segment_df[segment_df["surprise_rank_desc"] == 1]
    static_candidate_df = static_df[
        static_df["candidate_kind"] == "highest_surprise_static_segment"
    ]
    summary = {
        "script": "scripts/batch_static_surprise_audit.py",
        "project_revision": revision(Path.cwd()),
        "root": str(args.root),
        "selected_video_count": int(len(selected_df)),
        "segment_frames": args.segment_frames,
        "frames_per_clip": args.frames_per_clip,
        "total_segments": int(len(segment_df)),
        "static_motion_threshold": args.static_motion_threshold,
        "top_raw_segments_with_mean_rgb_lt_0_01": int(
            (top_df["within_mean_abs_rgb"] < 0.01).sum()
        ),
        "top_raw_segments_with_mean_rgb_lt_0_1": int(
            (top_df["within_mean_abs_rgb"] < 0.1).sum()
        ),
        "top_raw_segments_with_mean_rgb_ge_1": int(
            (top_df["within_mean_abs_rgb"] >= 1.0).sum()
        ),
        "top_raw_mean_pixel_motion": float(top_df["within_mean_abs_rgb"].mean()),
        "top_raw_median_pixel_motion": float(top_df["within_mean_abs_rgb"].median()),
        "all_segment_surprise_pixel_corr": float(
            segment_df["scalar_surprise"].corr(segment_df["within_mean_abs_rgb"])
        ),
        "all_segment_surprise_first_last_pixel_corr": float(
            segment_df["scalar_surprise"].corr(segment_df["first_last_mean_abs_rgb"])
        ),
        "static_low_motion_candidate_count": int(len(static_candidate_df)),
        "static_low_motion_candidates_positive_residual": int(
            (static_candidate_df["residual_minus_static_first"] > 0).sum()
        ),
        "static_low_motion_candidates_nonpositive_residual": int(
            (static_candidate_df["residual_minus_static_first"] <= 0).sum()
        ),
        "static_low_motion_candidate_mean_residual": (
            float(static_candidate_df["residual_minus_static_first"].mean())
            if len(static_candidate_df)
            else None
        ),
        "all_static_candidate_count": int(len(static_df)),
        "all_static_candidates_positive_residual": int(
            (static_df["residual_minus_static_first"] > 0).sum()
        ),
        "all_static_candidates_nonpositive_residual": int(
            (static_df["residual_minus_static_first"] <= 0).sum()
        ),
        "model_id": args.model_id,
        "model_snapshot": str(model_snapshot) if model_snapshot is not None else None,
        "model_safetensors_sha256": (
            sha256_file(model_snapshot / "model.safetensors")
            if model_snapshot is not None
            and (model_snapshot / "model.safetensors").exists()
            else None
        ),
        "device": args.device,
        "dtype": args.dtype,
        "target_start_fraction": args.target_start_fraction,
        "metric": args.metric,
        "normalization": args.normalization,
        "wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    with (args.output_dir / "summary_key_tables.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("Top raw-surprise segments per video\n")
        handle.write(
            top_df[
                [
                    "video_ordinal",
                    "video_id",
                    "section_index",
                    "section_source_start_frame_0based",
                    "section_source_end_frame_0based",
                    "scalar_surprise",
                    "within_mean_abs_rgb",
                    "within_changed_fraction_gt5",
                    "pixel_motion_rank_desc",
                ]
            ].to_string(index=False)
        )
        handle.write("\n\nStatic baseline candidates\n")
        handle.write(
            static_df[
                [
                    "video_ordinal",
                    "video_id",
                    "candidate_kind",
                    "section_index",
                    "raw_csv_scalar_surprise",
                    "within_mean_abs_rgb",
                    "within_changed_fraction_gt5",
                    "residual_minus_static_first",
                    "residual_minus_static_mean",
                ]
            ].to_string(index=False)
        )
        handle.write("\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
