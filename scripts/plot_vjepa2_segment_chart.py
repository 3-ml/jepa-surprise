from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot V-JEPA2 segment surprise from section_surprise.csv."
    )
    parser.add_argument("--section-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--svg-output", type=Path)
    parser.add_argument("--title", default="V-JEPA2 segment surprise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.section_csv)
    if df.empty:
        raise ValueError(f"No rows found in {args.section_csv}")

    df = df.sort_values("section_index").reset_index(drop=True)
    segment_count = len(df)
    segment_frames = (
        df["section_source_end_frame_0based"] - df["section_source_start_frame_0based"] + 1
    ).astype(int)
    if segment_frames.nunique() != 1:
        raise ValueError("Expected all plotted segments to have the same frame count")

    scores = df["scalar_surprise"].astype(float)
    peak_pos = int(scores.idxmax())
    mean_score = float(scores.mean())

    x = list(range(segment_count))
    colors = ["#4f6d7a"] * segment_count
    for i in scores.nlargest(min(3, segment_count)).index:
        colors[int(i)] = "#7b5ea7"
    colors[peak_pos] = "#d95f02"

    width = max(12.0, segment_count * 0.52)
    fig, ax = plt.subplots(figsize=(width, 6.4))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.19, top=0.82)
    bars = ax.bar(x, scores, color=colors, edgecolor="#1f2933", linewidth=0.7)

    ax.axhline(
        mean_score,
        color="#2a9d8f",
        linestyle="--",
        linewidth=1.6,
        label=f"mean {mean_score:.6f}",
    )
    ax.scatter(
        [peak_pos],
        [scores.iloc[peak_pos]],
        s=95,
        color="#d95f02",
        edgecolor="#1f2933",
        zorder=4,
        label=(
            f"peak segment {int(df.loc[peak_pos, 'section_index'])}, "
            f"{int(df.loc[peak_pos, 'section_source_start_frame_0based'])}-"
            f"{int(df.loc[peak_pos, 'section_source_end_frame_0based'])}"
        ),
    )

    y_min = max(0.0, float(scores.min()) - 0.025)
    y_max = float(scores.max()) + 0.035
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.7, segment_count - 0.3)

    labels = [
        f"{int(row.section_index)}\n"
        f"{int(row.section_source_start_frame_0based)}-"
        f"{int(row.section_source_end_frame_0based)}"
        for row in df.itertuples()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.tick_params(axis="x", length=0, pad=8)

    ax.set_ylabel("Scalar surprise")
    ax.set_xlabel(
        f"Segment index and source frame range ({int(segment_frames.iloc[0])} frames each)"
    )
    subtitle = (
        f"{segment_count} complete segments, frames "
        f"{int(df['section_source_start_frame_0based'].min())}-"
        f"{int(df['section_source_end_frame_0based'].max())}. "
        f"Peak score {scores.iloc[peak_pos]:.6f}."
    )
    fig.suptitle(args.title, x=0.06, y=0.97, ha="left", fontsize=18, weight="bold")
    fig.text(
        0.06,
        0.925,
        subtitle,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#334155",
    )

    for bar, value in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            color="#1f2933",
        )

    ax.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    if args.svg_output is not None:
        args.svg_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.svg_output)
    plt.close(fig)


if __name__ == "__main__":
    main()
