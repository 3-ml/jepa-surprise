#!/usr/bin/env python3
"""Render an IntPhys pair with synchronized context, target, and surprise dots."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from PIL import Image

SAMPLED_FRAME_STEP = 2
WINDOW_SAMPLED_FRAMES = 16
CONTEXT_SAMPLED_FRAMES = 4
WINDOW_STRIDE_SAMPLED_FRAMES = 2


@dataclass(frozen=True)
class WindowSpec:
    context: tuple[int, ...]
    targets: tuple[int, ...]

    @property
    def start(self) -> int:
        return self.context[0]

    @property
    def end(self) -> int:
        return self.targets[-1]


def window_spec(index: int) -> WindowSpec:
    """Map one zero-based evaluator window to one-based source-frame numbers."""
    if index < 0:
        raise ValueError("Window index must be non-negative")
    start = 1 + index * WINDOW_STRIDE_SAMPLED_FRAMES * SAMPLED_FRAME_STEP
    sampled = tuple(
        start + offset * SAMPLED_FRAME_STEP
        for offset in range(WINDOW_SAMPLED_FRAMES)
    )
    return WindowSpec(
        context=sampled[:CONTEXT_SAMPLED_FRAMES],
        targets=sampled[CONTEXT_SAMPLED_FRAMES:],
    )


def active_window_index(frame: int, window_count: int) -> int:
    """Return the next window to finish, retaining the last after completion."""
    if frame < 1 or window_count < 1:
        raise ValueError("Frame and window count must be positive")
    endings = np.asarray([window_spec(i).end for i in range(window_count)])
    return int(np.clip(np.searchsorted(endings, frame, side="left"), 0, window_count - 1))


def completed_window_count(frame: int, window_count: int) -> int:
    endings = np.asarray([window_spec(i).end for i in range(window_count)])
    return int(np.searchsorted(endings, frame, side="right"))


def find_pair(manifest: list[dict], block: str, scene: str, pair: int) -> dict:
    matches = [
        row
        for row in manifest
        if row["block"] == block
        and str(row["scene"]) == scene
        and int(row["pair"]) == pair
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest row for {block}/{scene}/pair-{pair}, "
            f"found {len(matches)}"
        )
    return matches[0]


def pair_members(pair: dict) -> list[tuple[str, bool]]:
    members = [
        (str(pair["video_a"]), bool(pair["video_a_possible"])),
        (str(pair["video_b"]), bool(pair["video_b_possible"])),
    ]
    if members[0][1] == members[1][1]:
        raise ValueError("Pair must contain one possible and one impossible video")
    return members


def load_surprises(path: Path, condition: str, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["condition"] == condition and row["pair_id"] == pair_id
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one surprise row for {condition} {pair_id}, found {len(rows)}"
        )
    possible = np.asarray(json.loads(rows[0]["possible_windows"]), dtype=float)
    impossible = np.asarray(json.loads(rows[0]["impossible_windows"]), dtype=float)
    if possible.shape != impossible.shape or possible.ndim != 1:
        raise ValueError("Possible and impossible surprise arrays must be matching vectors")
    return possible, impossible


def video_frames(video_root: Path, block: str, scene: str, video: str) -> list[Path]:
    frames = sorted((video_root / block / scene / video / "scene").glob("scene_*.png"))
    if len(frames) != 100:
        raise ValueError(f"Expected 100 RGB frames for video {video}, found {len(frames)}")
    return frames


def render(
    *,
    video_root: Path,
    pairs_manifest: Path,
    pair_surprises: Path,
    block: str,
    scene: str,
    pair_number: int,
    condition: str,
    output: Path,
    fps: float,
    left_video: str | None,
) -> None:
    manifest = json.loads(pairs_manifest.read_text())
    pair = find_pair(manifest, block, scene, pair_number)
    members = pair_members(pair)
    if left_video is None:
        if secrets.randbits(1):
            members.reverse()
    elif left_video == members[1][0]:
        members.reverse()
    elif left_video != members[0][0]:
        raise ValueError(f"--left-video {left_video} is not a member of the selected pair")

    pair_id = f"{block}/{scene}/pair-{pair_number}"
    possible, impossible = load_surprises(pair_surprises, condition, pair_id)
    window_count = len(possible)
    endings = np.asarray([window_spec(i).end for i in range(window_count)])
    left_surprise = possible if members[0][1] else impossible
    right_surprise = possible if members[1][1] else impossible
    left_frames = video_frames(video_root, block, scene, members[0][0])
    right_frames = video_frames(video_root, block, scene, members[1][0])

    fig = plt.figure(figsize=(9.6, 8.6), dpi=100, facecolor="#080b11")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1.0), hspace=0.12, wspace=0.025)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    ax = fig.add_subplot(grid[1, :])
    for current in (ax_left, ax_right):
        current.set_axis_off()
    left_image = ax_left.imshow(np.asarray(Image.open(left_frames[0])))
    right_image = ax_right.imshow(np.asarray(Image.open(right_frames[0])))
    ax_left.set_title("A", color="white", fontsize=20, fontweight="bold", pad=5)
    ax_right.set_title("B", color="white", fontsize=20, fontweight="bold", pad=5)

    all_surprises = np.concatenate((left_surprise, right_surprise))
    surprise_low = float(all_surprises.min())
    surprise_high = float(all_surprises.max())
    surprise_span = max(surprise_high - surprise_low, 0.01)
    context_y = surprise_low - 0.20 * surprise_span
    target_y = surprise_low - 0.42 * surprise_span
    plot_low = surprise_low - 0.56 * surprise_span
    plot_high = surprise_high + 0.08 * surprise_span

    ax.set_facecolor("#10151e")
    ax.set_xlim(1, 100)
    ax.set_ylim(plot_low, plot_high)
    ax.set_xticks([1, 15, 31, 47, 63, 79, 95, 100])
    numeric_ticks = np.linspace(surprise_low, surprise_high, 3)
    ax.set_yticks([target_y, context_y, *numeric_ticks])
    ax.set_yticklabels(
        ["TARGET\n(error)", "CONTEXT", *[f"{value:.3f}" for value in numeric_ticks]]
    )
    ax.tick_params(colors="#aeb9c8", labelsize=9)
    ax.grid(axis="x", color="#4b586d", alpha=0.35)
    ax.grid(axis="y", color="#283242", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#293344")

    context_dots = ax.scatter([], [], s=58, color="#2fa8ff", zorder=5, label="context seen")
    target_dots = ax.scatter([], [], s=58, color="#ff9d3f", zorder=5, label="predicted + graded")
    left_line, = ax.plot([], [], "o-", color="#2fa8ff", linewidth=2.5, markersize=5, label="A surprise")
    right_line, = ax.plot([], [], "o-", color="#ff4e88", linewidth=2.5, markersize=5, label="B surprise")
    cursor = ax.axvline(1, color="#ffe16a", linewidth=2.2, label="current frame")
    for ending in endings:
        ax.axvline(ending, color="#667286", alpha=0.13, linewidth=1)
    legend = ax.legend(loc="upper left", ncol=5, framealpha=0.68, fontsize=9)
    for text in legend.get_texts():
        text.set_color("white")
    status = ax.text(
        0.99,
        0.96,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel(
        "source frame — orange dots contribute to the highlighted window's one surprise dot",
        color="#c9d3df",
        fontsize=10,
    )
    fig.suptitle(
        "4 context frames → 12 targets predicted jointly → mean latent error",
        color="white",
        fontsize=14,
        y=0.986,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(output), dpi=100):
        for source_frame, (left_path, right_path) in enumerate(
            zip(left_frames, right_frames), start=1
        ):
            with Image.open(left_path) as image:
                left_image.set_data(np.asarray(image))
            with Image.open(right_path) as image:
                right_image.set_data(np.asarray(image))

            index = active_window_index(source_frame, window_count)
            spec = window_spec(index)
            context_dots.set_offsets(
                np.column_stack((spec.context, np.full(len(spec.context), context_y)))
            )
            target_dots.set_offsets(
                np.column_stack((spec.targets, np.full(len(spec.targets), target_y)))
            )
            completed = completed_window_count(source_frame, window_count)
            left_line.set_data(endings[:completed], left_surprise[:completed])
            right_line.set_data(endings[:completed], right_surprise[:completed])
            cursor.set_xdata([source_frame, source_frame])
            status.set_text(
                f"frame {source_frame}  |  context {spec.context[0]}–{spec.context[-1]}  |  "
                f"targets {spec.targets[0]}–{spec.targets[-1]}  |  score dot {spec.end}"
            )
            writer.grab_frame(facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--pair-surprises", type=Path, required=True)
    parser.add_argument("--block", choices=("O1", "O2", "O3"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--left-video")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    render(
        video_root=args.video_root,
        pairs_manifest=args.pairs_manifest,
        pair_surprises=args.pair_surprises,
        block=args.block,
        scene=args.scene,
        pair_number=args.pair,
        condition=args.condition,
        output=args.output,
        fps=args.fps,
        left_video=args.left_video,
    )


if __name__ == "__main__":
    main()
