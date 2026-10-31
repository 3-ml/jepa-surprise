from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare video pixels against V-JEPA2 segment surprise scores."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--section-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--focus-segment", type=int, default=11)
    parser.add_argument(
        "--diff-scale",
        type=float,
        default=10.0,
        help="Multiplier for visualizing absolute pixel differences.",
    )
    return parser.parse_args()


def read_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames, fps


def pair_metrics(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    diff = np.abs(current.astype(np.int16) - previous.astype(np.int16)).astype(np.float32)
    max_channel = diff.max(axis=2)
    luma_previous = (
        0.299 * previous[:, :, 0] + 0.587 * previous[:, :, 1] + 0.114 * previous[:, :, 2]
    )
    luma_current = (
        0.299 * current[:, :, 0] + 0.587 * current[:, :, 1] + 0.114 * current[:, :, 2]
    )
    luma_diff = np.abs(luma_current.astype(np.float32) - luma_previous.astype(np.float32))
    return {
        "mean_abs_rgb": float(diff.mean()),
        "rmse_rgb": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_rgb": float(diff.max()),
        "mean_abs_luma": float(luma_diff.mean()),
        "changed_pixel_fraction_gt2": float((max_channel > 2).mean()),
        "changed_pixel_fraction_gt5": float((max_channel > 5).mean()),
        "changed_pixel_fraction_gt10": float((max_channel > 10).mean()),
    }


def compute_pair_diffs(frames: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for current_index in range(1, len(frames)):
        row = {
            "previous_frame_0based": current_index - 1,
            "current_frame_0based": current_index,
        }
        row.update(pair_metrics(frames[current_index - 1], frames[current_index]))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_frame_range(value: object) -> list[int]:
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def compute_segment_summary(
    frames: list[np.ndarray], pair_df: pd.DataFrame, section_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for row in section_df.sort_values("section_index").itertuples(index=False):
        segment_index = int(row.section_index)
        start = int(row.section_source_start_frame_0based)
        end = int(row.section_source_end_frame_0based)
        within = pair_df[
            (pair_df["current_frame_0based"] >= start + 1)
            & (pair_df["current_frame_0based"] <= end)
        ]
        boundary = pair_df[pair_df["current_frame_0based"] == start]
        first_last = pair_metrics(frames[start], frames[end])
        peak_range = parse_frame_range(row.peak_source_frame_range_0based)
        peak_pair = None
        if len(peak_range) == 2 and peak_range[1] > peak_range[0]:
            peak_pair = pair_df[pair_df["current_frame_0based"] == peak_range[1]]

        rows.append(
            {
                "segment_index": segment_index,
                "start_frame_0based": start,
                "end_frame_0based": end,
                "scalar_surprise": float(row.scalar_surprise),
                "within_pair_count": int(len(within)),
                "within_mean_abs_rgb": float(within["mean_abs_rgb"].mean()),
                "within_median_abs_rgb": float(within["mean_abs_rgb"].median()),
                "within_max_pair_mean_abs_rgb": float(within["mean_abs_rgb"].max()),
                "within_mean_abs_luma": float(within["mean_abs_luma"].mean()),
                "within_changed_fraction_gt5": float(
                    within["changed_pixel_fraction_gt5"].mean()
                ),
                "boundary_in_mean_abs_rgb": (
                    float(boundary["mean_abs_rgb"].iloc[0]) if len(boundary) else None
                ),
                "first_last_mean_abs_rgb": first_last["mean_abs_rgb"],
                "first_last_changed_fraction_gt5": first_last[
                    "changed_pixel_fraction_gt5"
                ],
                "peak_source_frame_range_0based": str(peak_range),
                "peak_pair_mean_abs_rgb": (
                    float(peak_pair["mean_abs_rgb"].iloc[0])
                    if peak_pair is not None and len(peak_pair)
                    else None
                ),
                "peak_pair_changed_fraction_gt5": (
                    float(peak_pair["changed_pixel_fraction_gt5"].iloc[0])
                    if peak_pair is not None and len(peak_pair)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_segment_chart(segment_df: pd.DataFrame, output_path: Path, focus_segment: int) -> None:
    fig, ax1 = plt.subplots(figsize=(15, 6))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.2)
    x = np.arange(len(segment_df))
    colors = ["#4f6d7a"] * len(segment_df)
    if focus_segment in set(segment_df["segment_index"]):
        focus_pos = int(segment_df.index[segment_df["segment_index"] == focus_segment][0])
        colors[focus_pos] = "#d95f02"
    peak_pos = int(segment_df["scalar_surprise"].idxmax())
    colors[peak_pos] = "#7b5ea7" if segment_df.loc[peak_pos, "segment_index"] != focus_segment else "#d95f02"

    ax1.bar(x, segment_df["scalar_surprise"], color=colors, edgecolor="#1f2933", alpha=0.85)
    ax1.set_ylabel("V-JEPA2 scalar surprise")
    ax1.set_ylim(
        max(0.0, float(segment_df["scalar_surprise"].min()) - 0.03),
        float(segment_df["scalar_surprise"].max()) + 0.04,
    )
    ax1.grid(axis="y", color="#d8dee4", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        segment_df["within_mean_abs_rgb"],
        marker="o",
        color="#2a9d8f",
        linewidth=2.0,
        label="mean consecutive pixel delta",
    )
    ax2.plot(
        x,
        segment_df["first_last_mean_abs_rgb"],
        marker="s",
        color="#e76f51",
        linewidth=1.6,
        label="first to last pixel delta",
    )
    ax2.set_ylabel("Mean absolute RGB pixel difference, 0-255 scale")

    labels = [
        f"{int(row.segment_index)}\n{int(row.start_frame_0based)}-{int(row.end_frame_0based)}"
        for row in segment_df.itertuples()
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("Segment index and source frame range")
    fig.suptitle(
        "Pixel change versus V-JEPA2 surprise by 16-frame segment",
        x=0.07,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.07,
        0.89,
        f"Focus segment {focus_segment} is orange; purple is the maximum surprise segment.",
        ha="left",
        color="#334155",
    )
    handles, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels2, loc="upper right", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_pair_chart(
    pair_df: pd.DataFrame,
    section_df: pd.DataFrame,
    output_path: Path,
    focus_segment: int,
) -> None:
    focus = section_df[section_df["section_index"] == focus_segment].iloc[0]
    start = int(focus["section_source_start_frame_0based"])
    end = int(focus["section_source_end_frame_0based"])
    window_start = max(1, start - 32)
    window_end = min(int(pair_df["current_frame_0based"].max()), end + 32)
    plot_df = pair_df[
        (pair_df["current_frame_0based"] >= window_start)
        & (pair_df["current_frame_0based"] <= window_end)
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18)
    ax.plot(
        plot_df["current_frame_0based"],
        plot_df["mean_abs_rgb"],
        marker="o",
        markersize=3.5,
        color="#2a9d8f",
        linewidth=1.8,
    )
    ax.axvspan(start, end, color="#d95f02", alpha=0.16, label=f"segment {focus_segment}")
    peak_range = parse_frame_range(focus["peak_source_frame_range_0based"])
    if len(peak_range) == 2:
        ax.axvspan(
            peak_range[0],
            peak_range[1],
            color="#7b5ea7",
            alpha=0.2,
            label=f"model peak target {peak_range[0]}-{peak_range[1]}",
        )
    for row in section_df.itertuples():
        segment_start = int(row.section_source_start_frame_0based)
        if window_start <= segment_start <= window_end:
            ax.axvline(segment_start, color="#94a3b8", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Current frame in consecutive pair: previous frame to current frame")
    ax.set_ylabel("Mean absolute RGB pixel difference")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle(
        f"Consecutive pixel changes around segment {focus_segment}",
        x=0.08,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.08,
        0.86,
        f"Segment {focus_segment}: frames {start}-{end}; V-JEPA2 surprise {float(focus['scalar_surprise']):.6f}.",
        ha="left",
        color="#334155",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def put_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        label,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_sheet(cells: list[np.ndarray], cols: int, pad: int = 8) -> np.ndarray:
    if not cells:
        raise ValueError("No cells for contact sheet")
    cell_h, cell_w = cells[0].shape[:2]
    rows = int(np.ceil(len(cells) / cols))
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r = i // cols
        c = i % cols
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y : y + cell_h, x : x + cell_w] = cell
    return sheet


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def draw_focus_sheets(
    frames: list[np.ndarray],
    pair_df: pd.DataFrame,
    section_df: pd.DataFrame,
    output_dir: Path,
    focus_segment: int,
    diff_scale: float,
) -> None:
    focus = section_df[section_df["section_index"] == focus_segment].iloc[0]
    start = int(focus["section_source_start_frame_0based"])
    end = int(focus["section_source_end_frame_0based"])

    frame_cells = []
    for frame_index in range(start, end + 1):
        frame_cells.append(put_label(frames[frame_index], f"frame {frame_index}"))
    save_rgb(
        output_dir / f"segment{focus_segment}_frames_contact_sheet.png",
        make_sheet(frame_cells, cols=4),
    )

    diff_cells = []
    for current_index in range(start + 1, end + 1):
        diff = np.abs(
            frames[current_index].astype(np.int16)
            - frames[current_index - 1].astype(np.int16)
        ).astype(np.float32)
        gray = np.clip(diff.max(axis=2) * diff_scale, 0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        mae = float(
            pair_df[pair_df["current_frame_0based"] == current_index][
                "mean_abs_rgb"
            ].iloc[0]
        )
        diff_cells.append(
            put_label(
                heat_rgb,
                f"{current_index - 1} to {current_index}, mae {mae:.3f}, x{diff_scale:g}",
            )
        )
    save_rgb(
        output_dir / f"segment{focus_segment}_consecutive_pixel_diffs.png",
        make_sheet(diff_cells, cols=3),
    )


def write_summary(
    output_dir: Path,
    video_path: Path,
    section_csv: Path,
    frames: list[np.ndarray],
    fps: float,
    segment_df: pd.DataFrame,
    focus_segment: int,
) -> None:
    focus = segment_df[segment_df["segment_index"] == focus_segment].iloc[0]
    surprise_rank = (
        segment_df["scalar_surprise"].rank(method="min", ascending=False).astype(int)
    )
    within_pixel_rank = (
        segment_df["within_mean_abs_rgb"].rank(method="min", ascending=False).astype(int)
    )
    first_last_rank = (
        segment_df["first_last_mean_abs_rgb"].rank(method="min", ascending=False).astype(int)
    )
    focus_position = int(segment_df.index[segment_df["segment_index"] == focus_segment][0])
    summary = {
        "video": str(video_path),
        "section_csv": str(section_csv),
        "frame_count": len(frames),
        "fps": fps,
        "focus_segment": focus_segment,
        "focus_start_frame_0based": int(focus["start_frame_0based"]),
        "focus_end_frame_0based": int(focus["end_frame_0based"]),
        "focus_scalar_surprise": float(focus["scalar_surprise"]),
        "focus_surprise_rank_desc": int(surprise_rank.iloc[focus_position]),
        "focus_within_mean_abs_rgb": float(focus["within_mean_abs_rgb"]),
        "focus_within_pixel_rank_desc": int(within_pixel_rank.iloc[focus_position]),
        "focus_first_last_mean_abs_rgb": float(focus["first_last_mean_abs_rgb"]),
        "focus_first_last_pixel_rank_desc": int(first_last_rank.iloc[focus_position]),
        "focus_peak_pair_mean_abs_rgb": float(focus["peak_pair_mean_abs_rgb"]),
        "segment_count": len(segment_df),
        "surprise_pixel_correlation_within_mean_abs_rgb": float(
            segment_df["scalar_surprise"].corr(segment_df["within_mean_abs_rgb"])
        ),
        "surprise_pixel_correlation_first_last_mean_abs_rgb": float(
            segment_df["scalar_surprise"].corr(segment_df["first_last_mean_abs_rgb"])
        ),
        "top_surprise_segments": segment_df.sort_values(
            "scalar_surprise", ascending=False
        )
        .head(5)
        .to_dict(orient="records"),
        "top_within_pixel_segments": segment_df.sort_values(
            "within_mean_abs_rgb", ascending=False
        )
        .head(5)
        .to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = read_frames(args.video)
    section_df = pd.read_csv(args.section_csv).sort_values("section_index").reset_index(drop=True)
    if args.focus_segment not in set(section_df["section_index"]):
        raise ValueError(f"Focus segment {args.focus_segment} not present in section CSV")

    pair_df = compute_pair_diffs(frames)
    segment_df = compute_segment_summary(frames, pair_df, section_df)

    pair_df.to_csv(args.output_dir / "pair_pixel_diffs.csv", index=False)
    segment_df.to_csv(args.output_dir / "segment_pixel_summary.csv", index=False)

    draw_segment_chart(
        segment_df,
        args.output_dir / "segment_surprise_vs_pixel_change.png",
        args.focus_segment,
    )
    draw_pair_chart(
        pair_df,
        section_df,
        args.output_dir / f"segment{args.focus_segment}_pair_pixel_diffs.png",
        args.focus_segment,
    )
    draw_focus_sheets(
        frames,
        pair_df,
        section_df,
        args.output_dir,
        args.focus_segment,
        args.diff_scale,
    )
    write_summary(
        args.output_dir,
        args.video,
        args.section_csv,
        frames,
        fps,
        segment_df,
        args.focus_segment,
    )

    focus = segment_df[segment_df["segment_index"] == args.focus_segment].iloc[0]
    print(
        json.dumps(
            {
                "focus_segment": args.focus_segment,
                "frames": [
                    int(focus["start_frame_0based"]),
                    int(focus["end_frame_0based"]),
                ],
                "scalar_surprise": float(focus["scalar_surprise"]),
                "within_mean_abs_rgb": float(focus["within_mean_abs_rgb"]),
                "first_last_mean_abs_rgb": float(focus["first_last_mean_abs_rgb"]),
                "peak_pair_mean_abs_rgb": float(focus["peak_pair_mean_abs_rgb"]),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
