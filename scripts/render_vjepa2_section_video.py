#!/usr/bin/env python3
"""Render a source video with V-JEPA2 section-surprise scores beside it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated MP4 from a video and section_surprise.csv."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--section-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", help="Select a specific video_id from the CSV.")
    parser.add_argument("--title", default="V-JEPA2-L section surprise")
    parser.add_argument("--source-width", type=int, default=512)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--fps", type=float, help="Override output FPS.")
    parser.add_argument(
        "--trim-to-scored",
        action="store_true",
        help="Stop rendering after the last scored segment instead of showing an unscored tail.",
    )
    return parser.parse_args()


def load_rows(path: Path, video: Path, video_id: str | None) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["section_index"] = int(row["section_index"])
            row["section_source_start_frame_0based"] = int(row["section_source_start_frame_0based"])
            row["section_source_end_frame_0based"] = int(row["section_source_end_frame_0based"])
            row["scalar_surprise"] = float(row["scalar_surprise"])
            row["peak_source_frame_range_0based"] = json.loads(row["peak_source_frame_range_0based"])
            rows.append(row)
    if video_id:
        selected = [row for row in rows if row["video_id"] == video_id]
    else:
        resolved = str(video.resolve())
        selected = [row for row in rows if str(Path(str(row["path"])).resolve()) == resolved]
        if not selected:
            ids = sorted({str(row["video_id"]) for row in rows})
            selected = rows if len(ids) == 1 else []
    if not selected:
        raise ValueError("Could not select matching rows; pass --video-id explicitly.")
    return sorted(selected, key=lambda row: int(row["section_index"]))


def current_section(rows: list[dict[str, object]], frame_index: int) -> dict[str, object]:
    for row in rows:
        start = int(row["section_source_start_frame_0based"])
        end = int(row["section_source_end_frame_0based"])
        if start <= frame_index <= end:
            return row
    return rows[-1] if frame_index > int(rows[-1]["section_source_end_frame_0based"]) else rows[0]


def text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (235, 238, 244),
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(
    height: int,
    width: int,
    rows: list[dict[str, object]],
    active: dict[str, object],
    frame_index: int,
    total_frames: int,
    title: str,
) -> np.ndarray:
    panel = np.full((height, width, 3), (27, 31, 38), dtype=np.uint8)
    pad = 34
    plot_top = 126
    table_needed = len(rows) > 12
    table_top = height - 126 if table_needed else None
    plot_bottom = (table_top - 45) if table_needed else (height - 100)
    plot_left = pad
    plot_right = width - pad
    values = np.asarray([float(row["scalar_surprise"]) for row in rows], dtype=float)
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1e-6)

    text(panel, title, (pad, 34), scale=0.62, color=(245, 247, 252), thickness=1)
    text(panel, f"frame {frame_index}/{total_frames - 1}", (pad, 62), scale=0.56, color=(201, 210, 224))
    text(
        panel,
        "segment {0}: frames {1}-{2}".format(
            active["section_index"],
            active["section_source_start_frame_0based"],
            active["section_source_end_frame_0based"],
        ),
        (pad, 90),
        scale=0.56,
        color=(201, 210, 224),
    )
    text(panel, f"surprise {float(active['scalar_surprise']):.6f}", (pad, 118), scale=0.68, color=(98, 214, 255), thickness=2)

    cv2.rectangle(panel, (plot_left, plot_top), (plot_right, plot_bottom), (77, 86, 101), 1)
    for fraction in (0.25, 0.5, 0.75):
        y = int(plot_bottom - fraction * (plot_bottom - plot_top))
        cv2.line(panel, (plot_left, y), (plot_right, y), (49, 56, 68), 1)

    count = len(rows)
    bar_gap = 2 if count > 18 else 8
    available = plot_right - plot_left - bar_gap * max(0, count - 1)
    bar_width = max(4, available // max(1, count))
    active_index = int(active["section_index"])
    for index, row in enumerate(rows):
        value = float(row["scalar_surprise"])
        x0 = plot_left + index * (bar_width + bar_gap)
        x1 = min(x0 + bar_width, plot_right)
        y1 = plot_bottom
        y0 = int(plot_bottom - ((value - low) / span) * (plot_bottom - plot_top - 18) - 8)
        color = (40, 180, 255) if int(row["section_index"]) == active_index else (103, 130, 171)
        cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (14, 18, 24), 1)
        label_x = x0 + 2
        if count <= 24 or bar_width >= 18:
            text(panel, str(row["section_index"]), (label_x, plot_bottom + 22), scale=0.38, color=(221, 226, 235))
        if count <= 18 or bar_width >= 24:
            text(panel, f"{value:.3f}", (max(plot_left, x0 - 5), max(plot_top + 14, y0 - 8)), scale=0.32, color=(231, 235, 242))

    timeline_y = plot_bottom + 32
    cv2.line(panel, (plot_left, timeline_y), (plot_right, timeline_y), (93, 103, 120), 2)
    marker_x = int(plot_left + (plot_right - plot_left) * frame_index / max(1, total_frames - 1))
    cv2.circle(panel, (marker_x, timeline_y), 7, (98, 214, 255), -1)
    for row in rows:
        start = int(row["section_source_start_frame_0based"])
        x = int(plot_left + (plot_right - plot_left) * start / max(1, total_frames - 1))
        cv2.line(panel, (x, timeline_y - 9), (x, timeline_y + 9), (93, 103, 120), 1)

    if table_needed and table_top is not None:
        columns = min(4, max(2, int(np.ceil(count / 8))))
        rows_per_column = int(np.ceil(count / columns))
        column_width = (plot_right - plot_left) // columns
        for index, row in enumerate(rows):
            column = index // rows_per_column
            row_in_column = index % rows_per_column
            x = plot_left + column * column_width
            y = table_top + 16 * row_in_column
            is_active = int(row["section_index"]) == active_index
            color = (98, 214, 255) if is_active else (216, 222, 232)
            label = f"{int(row['section_index']):02d} {float(row['scalar_surprise']):.3f}"
            text(panel, label, (x, y), scale=0.38, color=color, thickness=1)

    peak = active["peak_source_frame_range_0based"]
    text(panel, f"peak target frames {peak[0]}-{peak[1]}", (pad, height - 20), scale=0.53, color=(201, 210, 224))
    return panel


def main() -> None:
    args = parse_args()
    rows = load_rows(args.section_csv, args.video, args.video_id)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = args.fps or float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scored_frames = max(int(row["section_source_end_frame_0based"]) for row in rows) + 1
    render_frames = min(total_frames, scored_frames) if args.trim_to_scored else total_frames
    source_width = int(args.source_width)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_height = int(round(height * source_width / max(1, width)))
    output_width = source_width + int(args.panel_width)
    output_height = source_height
    output_width += output_width % 2
    output_height += output_height % 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    frame_index = 0
    try:
        while True:
            if frame_index >= render_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (source_width, output_height), interpolation=cv2.INTER_AREA)
            active = current_section(rows, frame_index)
            start = int(active["section_source_start_frame_0based"])
            end = int(active["section_source_end_frame_0based"])
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (source_width, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.42, frame, 0.58, 0)
            text(frame, f"segment {active['section_index']} frames {start}-{end}", (16, 30), scale=0.66, color=(255, 255, 255), thickness=2)
            text(frame, f"surprise {float(active['scalar_surprise']):.6f}", (16, 62), scale=0.66, color=(98, 214, 255), thickness=2)
            panel = draw_panel(output_height, int(args.panel_width), rows, active, frame_index, render_frames, args.title)
            writer.write(np.hstack([frame, panel]))
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
