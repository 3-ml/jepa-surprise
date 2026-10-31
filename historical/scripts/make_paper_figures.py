from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAYERS = ["block_00", "block_10", "block_20", "block_30", "final"]
LAYER_LABELS = ["b00", "b10", "b20", "b30", "final"]
MODELS = ["V-JEPA2-G", "VideoMAE2-G"]
TRANSFORMS = ["hue_090", "brightness_0p45", "selective_recolor_180"]
MODEL_COLORS = {
    "V-JEPA2-G": "#2f6fbb",
    "VideoMAE2-G": "#c44e52",
    "PixelSummary": "#5a9b68",
}
TRANSFORM_LABELS = {
    "hue_090": "Hue 90 deg",
    "brightness_0p45": "Brightness 0.45",
    "selective_recolor_180": "Recolor 180 deg",
    "physics": "Physical pair",
}
TRANSFORM_COLORS = {
    "hue_090": "#0072B2",
    "brightness_0p45": "#D55E00",
    "selective_recolor_180": "#009E73",
    "physics": "#222222",
}
TRANSFORM_MARKERS = {
    "hue_090": "o",
    "brightness_0p45": "s",
    "selective_recolor_180": "^",
    "physics": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"Creator": None, "Producer": None, "CreationDate": None},
    )
    plt.close(figure)


def errorbar(axis, x, mean, low, high, **kwargs) -> None:
    axis.errorbar(
        x,
        mean,
        yerr=np.vstack(
            [np.asarray(mean) - np.asarray(low), np.asarray(high) - np.asarray(mean)]
        ),
        capsize=3,
        **kwargs,
    )


def make_intphys_crossfit(root: Path, output: Path) -> None:
    stats = root / "results/intphys2/scene_crossfit_probe_audit/sealed/statistics"
    accuracy = pd.read_csv(stats / "scene_clustered_accuracy_ci.csv")
    deltas = pd.read_csv(stats / "scene_clustered_model_delta_ci.csv")
    order = ["V-JEPA2-G", "VideoMAE2-G", "PixelSummary"]
    physics = accuracy[accuracy.task.eq("physics_all")].set_index("model").loc[order]
    selected = (
        deltas[deltas.task.eq("physics_all")]
        .set_index("right_model")
        .loc[["VideoMAE2-G", "PixelSummary"]]
    )
    figure, axes = plt.subplots(1, 2, figsize=(6.55, 2.50))
    x = np.arange(3)
    means = physics["mean"].to_numpy()
    axes[0].bar(x, means, color=[MODEL_COLORS[item] for item in order], width=0.8)
    axes[0].errorbar(
        x,
        means,
        yerr=np.vstack([means - physics["ci_low"], physics["ci_high"] - means]),
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    axes[0].set_xticks(x, order, rotation=0, fontsize=6)
    axes[0].set_ylim(0.40, 0.66)
    axes[0].set_ylabel("physics accuracy")
    axes[0].set_title("A. Five-fold scene cross-fit")
    axes[0].legend(loc="upper right")
    for index, value in enumerate(means):
        axes[0].annotate(
            f"{value:.3f}",
            (index, value + 0.012),
            xytext=(12.5, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    x = np.arange(2)
    means = selected["mean"].to_numpy()
    axes[1].bar(
        x,
        means,
        color=[MODEL_COLORS["V-JEPA2-G"], MODEL_COLORS["PixelSummary"]],
        width=0.8,
    )
    axes[1].errorbar(
        x,
        means,
        yerr=np.vstack([means - selected["ci_low"], selected["ci_high"] - means]),
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    axes[1].set_xticks(
        x,
        ["V-JEPA2-G - VideoMAE2-G", "V-JEPA2-G - PixelSummary"],
        rotation=0,
        fontsize=6,
    )
    axes[1].set_ylim(0.0, 0.15)
    axes[1].set_ylabel("global scene-bootstrap delta")
    axes[1].set_title("B. Paired model deltas")
    for index, value in enumerate(means):
        axes[1].annotate(
            f"{value:.3f}",
            (index, value + 0.006),
            xytext=(12.5, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for axis in axes:
        axis.grid(axis="both", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    save(figure, output)


def make_lewm_controls(root: Path, output: Path) -> None:
    base = root / "results/leworldmodel"
    tasks = {
        "A. TwoRoom": base / "tworoom_action_consistency",
        "B. PushT": base / "pusht_action_consistency",
    }
    line_styles = {
        "zero": {"color": "#55A868", "marker": "o", "linestyle": "-"},
        "opposite": {"color": "#E17C05", "marker": "s", "linestyle": "--"},
    }
    appearance_color = "#8172B2"
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.25))
    for axis, (title, path) in zip(axes, tasks.items()):
        actions = pd.read_csv(path / "same_target_contrasts.csv")
        appearances = pd.read_csv(path / "construct_comparisons.csv")
        for mode in ["zero", "opposite"]:
            frame = actions[actions.action_mode.eq(mode)].sort_values(
                "intended_displacement_scale"
            )
            errorbar(
                axis,
                frame.intended_displacement_scale.to_numpy(),
                frame.inconsistent_minus_lawful_mean.to_numpy(),
                frame.ci95_low.to_numpy(),
                frame.ci95_high.to_numpy(),
                label=f"{mode} action",
                markerfacecolor="white",
                markersize=5,
                linewidth=1.5,
                **line_styles[mode],
            )
        errorbar(
            axis,
            appearances.matched_displacement_scale.to_numpy(),
            appearances.appearance_minus_static_mean.to_numpy(),
            appearances.appearance_ci95_low.to_numpy(),
            appearances.appearance_ci95_high.to_numpy(),
            marker="D",
            markerfacecolor="white",
            markersize=5,
            linestyle="none",
            label="appearance-only",
            color=appearance_color,
        )
        for row in appearances.itertuples(index=False):
            offset = (-20, 6)
            if row.hue_degrees == 30:
                offset = (4, 6)
            elif row.hue_degrees == 180:
                offset = (-20, -10)
            elif title == "B. PushT" and row.hue_degrees == 120:
                offset = (-24, -12)
            axis.annotate(
                f"{int(row.hue_degrees)}°",
                (row.matched_displacement_scale, row.appearance_minus_static_mean),
                xytext=offset,
                textcoords="offset points",
                color=appearance_color,
                fontsize=8,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks([0.25, 0.5, 0.75, 1.0])
        axis.set_xlabel("Matched rendered-change scale")
        axis.set_ylabel("Prediction-error increase")
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(8.5)
        axis.yaxis.label.set_size(8.5)
        if title == "B. PushT":
            axis.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylim(0.0, 2.25)
    axes[1].set_ylim(0.0, 1.7)
    figure.tight_layout(pad=0.35, w_pad=1.0)
    save(figure, output)


def setup_geometry_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_controlled_geometry(root: Path, output: Path) -> None:
    source = root / "results/intphys2/controlled_geometry/sealed"
    layerwise = pd.read_csv(source / "SEALED_LAYERWISE_METRICS.csv")
    pao = pd.read_csv(source / "PAO_K_SENSITIVITY_SEALED.csv")
    figure, axes = plt.subplots(2, 3, figsize=(7.05, 3.55), sharex="col")
    x = np.arange(len(LAYERS))
    for row_index, model in enumerate(MODELS):
        current = layerwise[layerwise.model.eq(model)]
        for transform in TRANSFORMS:
            values = (
                current[current["transform"].eq(transform)]
                .set_index("layer_name")
                .loc[LAYERS]
            )
            axes[row_index, 0].plot(
                x,
                values.appearance_distance_global.to_numpy(),
                color=TRANSFORM_COLORS[transform],
                marker=TRANSFORM_MARKERS[transform],
                linewidth=1.35,
                markersize=3.5,
                label=TRANSFORM_LABELS[transform],
            )
        physical = (
            current[current["transform"].eq("hue_090")]
            .set_index("layer_name")
            .loc[LAYERS]
            .physical_distance_global.to_numpy()
        )
        axes[row_index, 0].plot(
            x,
            physical,
            color=TRANSFORM_COLORS["physics"],
            marker=TRANSFORM_MARKERS["physics"],
            linewidth=1.35,
            markersize=3.3,
            linestyle="--",
            label=TRANSFORM_LABELS["physics"],
        )
        for transform in TRANSFORMS:
            values = (
                current[current["transform"].eq(transform)]
                .set_index("layer_name")
                .loc[LAYERS]
            )
            axes[row_index, 1].plot(
                x,
                values.tdc_excess.to_numpy(),
                color=TRANSFORM_COLORS[transform],
                marker=TRANSFORM_MARKERS[transform],
                linewidth=1.35,
                markersize=3.5,
                label=TRANSFORM_LABELS[transform],
            )
        physics_frame = (
            current[current["transform"].eq("hue_090")]
            .set_index("layer_name")
            .loc[LAYERS]
        )
        if "physics_tdc_excess" in physics_frame:
            physics_direction = physics_frame.physics_tdc_excess.to_numpy()
        else:
            paper_table = pd.read_csv(source / "PAPER_GEOMETRY_TABLE.csv")
            physics_direction = (
                paper_table[paper_table.model.eq(model)]
                .set_index("layer_name")
                .loc[LAYERS]
                .physics_tdc_excess.to_numpy()
            )
        axes[row_index, 1].plot(
            x,
            physics_direction,
            color=TRANSFORM_COLORS["physics"],
            marker=TRANSFORM_MARKERS["physics"],
            linewidth=1.35,
            markersize=3.3,
            linestyle="--",
            label=TRANSFORM_LABELS["physics"],
        )
        axes[row_index, 1].axhline(0, color="#777777", linewidth=0.7, zorder=0)
        projection = (
            pao[pao.model.eq(model) & pao.k.eq(4)]
            .set_index("layer_name")
            .loc[LAYERS]
        )
        values = projection.pao_raw.to_numpy()
        low = projection.pao_ci_low.to_numpy()
        high = projection.pao_ci_high.to_numpy()
        random = projection.pao_random_mean.to_numpy()
        axes[row_index, 2].errorbar(
            x,
            values,
            yerr=np.vstack([values - low, high - values]),
            color="#CC79A7",
            marker="o",
            linewidth=1.35,
            markersize=3.5,
            capsize=2,
            label="Observed",
        )
        axes[row_index, 2].plot(
            x,
            random,
            color="#666666",
            linewidth=1.1,
            linestyle=":",
            label="Random k=4",
        )
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel(f"{model}\nrelative magnitude")
        axes[row_index, 1].set_ylabel(r"$\Delta$MRL")
        axes[row_index, 2].set_ylabel("physical projection")
        for axis in axes[row_index]:
            axis.set_xticks(x, LAYER_LABELS)
            axis.grid(axis="y", color="#dddddd", linewidth=0.45)
    for column, tag in enumerate(("A", "B", "C")):
        axes[0, column].text(
            -0.17,
            1.03,
            tag,
            transform=axes[0, column].transAxes,
            weight="bold",
            fontsize=9,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    axes[0, 2].legend(frameon=False, loc="upper right", borderaxespad=0.1)
    figure.supxlabel("encoder depth", y=0.005, fontsize=8)
    figure.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.12,
        top=0.88,
        wspace=0.38,
        hspace=0.17,
    )
    save(figure, output)


def make_exploratory_trajectories(root: Path, output: Path) -> None:
    source = root / "results/intphys2/multidose_geometry"
    straight = pd.read_csv(source / "STRAIGHTNESS_METRICS.csv")
    rank = pd.read_csv(source / "EFFECTIVE_RANK.csv")
    projection = pd.read_csv(source / "PHYSICS_APPEARANCE_MANIFOLD.csv")
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    x = np.arange(len(LAYERS))
    factors = ["hue", "brightness", "selective_recolor"]
    colors = [
        TRANSFORM_COLORS["hue_090"],
        TRANSFORM_COLORS["brightness_0p45"],
        TRANSFORM_COLORS["selective_recolor_180"],
    ]
    markers = ["o", "s", "^"]
    labels = ["Hue", "Brightness", "Recolor"]
    for factor, color, marker, label in zip(factors, colors, markers, labels):
        current = straight[straight.factor.eq(factor)].set_index("layer").loc[LAYERS]
        values = current.straightness_mean.to_numpy()
        axes[0].errorbar(
            x,
            values,
            yerr=np.vstack(
                [
                    values - current.ci_low.to_numpy(),
                    current.ci_high.to_numpy() - values,
                ]
            ),
            color=color,
            marker=marker,
            linewidth=1.25,
            markersize=3.3,
            capsize=1.5,
            label=label,
        )
        current_rank = rank[rank.factor.eq(factor)].set_index("layer").loc[LAYERS]
        axes[1].plot(
            x,
            current_rank.effective_rank,
            color=color,
            marker=marker,
            linewidth=1.25,
            markersize=3.3,
            label=label,
        )
    final_projection = projection[projection.k.eq(4)]
    styles = {
        "endpoint_strong": ("Endpoint", "o"),
        "all_dose": ("All dose", "s"),
        "local_tangent": ("Local tangent", "^"),
    }
    for space, (label, marker) in styles.items():
        current = (
            final_projection[final_projection.space.eq(space)]
            .set_index("layer")
            .loc[LAYERS]
        )
        values = current.projection_mean.to_numpy()
        axes[2].errorbar(
            x,
            values,
            yerr=np.vstack(
                [
                    values - current.ci_low.to_numpy(),
                    current.ci_high.to_numpy() - values,
                ]
            ),
            marker=marker,
            linewidth=1.25,
            markersize=3.3,
            capsize=1.5,
            label=label,
        )
    random = (
        final_projection[final_projection.space.eq("local_tangent")]
        .set_index("layer")
        .loc[LAYERS]
        .random_mean.to_numpy()
    )
    axes[2].plot(
        x,
        random,
        color="#666666",
        linestyle=":",
        linewidth=1,
        label="Random k=4",
    )
    axes[0].set_ylabel("straightness")
    axes[1].set_ylabel("entropy effective rank")
    axes[2].set_ylabel("physical energy fraction")
    for index, axis in enumerate(axes):
        axis.set_xticks(x, LAYER_LABELS)
        axis.set_xlabel("encoder depth")
        axis.grid(axis="y", color="#dddddd", linewidth=0.45)
        axis.text(
            -0.18,
            1.06,
            chr(ord("A") + index),
            transform=axis.transAxes,
            weight="bold",
            fontsize=10,
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.36, 1.03),
        fontsize=6.5,
    )
    axes[2].legend(frameon=False, fontsize=6.5)
    figure.subplots_adjust(wspace=0.42, top=0.88)
    save(figure, output)


def main() -> None:
    args = parse_args()
    root = args.archive_root.resolve()
    output = args.output_dir.resolve()
    make_intphys_crossfit(root, output / "figure4_intphys_scene_crossfit.pdf")
    make_lewm_controls(root, output / "figure3_lewm_construct_controls.pdf")
    setup_geometry_style()
    make_controlled_geometry(root, output / "figure2_controlled_geometry_compact.pdf")
    make_exploratory_trajectories(root, output / "appendix_exploratory_trajectories.pdf")


if __name__ == "__main__":
    main()
