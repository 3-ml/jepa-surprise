#!/usr/bin/env python3
"""Generate the final controlled geometry and exploratory trajectory figures."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "manuscript_final_integration"
PAPER_FIGURES = ROOT / "paper" / "figures"
LAYERS = ["block_00", "block_10", "block_20", "block_30", "final"]
LAYER_LABELS = ["b00", "b10", "b20", "b30", "final"]
MODELS = ["V-JEPA2-G", "VideoMAE2-G"]
TRANSFORMS = ["hue_090", "brightness_0p45", "selective_recolor_180"]
LABELS = {
    "hue_090": "Hue 90 deg",
    "brightness_0p45": "Brightness 0.45",
    "selective_recolor_180": "Recolor 180 deg",
    "physics": "Physical pair",
}
COLORS = {
    "hue_090": "#0072B2",
    "brightness_0p45": "#D55E00",
    "selective_recolor_180": "#009E73",
    "physics": "#222222",
}
MARKERS = {"hue_090": "o", "brightness_0p45": "s", "selective_recolor_180": "^", "physics": "D"}


def setup() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
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


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        path = REPORT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=320, bbox_inches="tight")
        if suffix == "pdf":
            shutil.copy2(path, PAPER_FIGURES / path.name)
    plt.close(fig)


def controlled_geometry() -> None:
    layerwise = pd.read_csv(ROOT / "reports" / "geometry_sealed" / "SEALED_LAYERWISE_METRICS.csv")
    pao = pd.read_csv(ROOT / "reports" / "geometry_sealed" / "PAO_K_SENSITIVITY_SEALED.csv")
    records: list[dict] = []
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.55), sharex="col")
    x = np.arange(len(LAYERS))
    for row_index, model in enumerate(MODELS):
        current = layerwise[layerwise.model == model]
        for transform in TRANSFORMS:
            values = current[current["transform"] == transform].set_index("layer_name").loc[LAYERS]
            y = values.appearance_distance_global.to_numpy()
            axes[row_index, 0].plot(x, y, color=COLORS[transform], marker=MARKERS[transform], lw=1.35, ms=3.5, label=LABELS[transform])
            for layer, value in zip(LAYERS, y):
                records.append({"model": model, "layer": layer, "panel": "magnitude", "series": transform, "estimate": value, "ci_low": np.nan, "ci_high": np.nan, "reference": "cross-scene unrelated clean scale"})
        physical = current[current["transform"] == "hue_090"].set_index("layer_name").loc[LAYERS].physical_distance_global.to_numpy()
        axes[row_index, 0].plot(x, physical, color=COLORS["physics"], marker=MARKERS["physics"], lw=1.35, ms=3.3, ls="--", label=LABELS["physics"])
        for layer, value in zip(LAYERS, physical):
            records.append({"model": model, "layer": layer, "panel": "magnitude", "series": "physics", "estimate": value, "ci_low": np.nan, "ci_high": np.nan, "reference": "cross-scene unrelated clean scale"})

        for transform in TRANSFORMS:
            values = current[current["transform"] == transform].set_index("layer_name").loc[LAYERS]
            y = values.tdc_excess.to_numpy()
            axes[row_index, 1].plot(x, y, color=COLORS[transform], marker=MARKERS[transform], lw=1.35, ms=3.5, label=LABELS[transform])
            for layer, value in zip(LAYERS, y):
                records.append({"model": model, "layer": layer, "panel": "direction", "series": transform, "estimate": value, "ci_low": np.nan, "ci_high": np.nan, "reference": "observed MRL minus shuffled-correspondence mean"})
        physics_direction = current[current["transform"] == "hue_090"].set_index("layer_name").loc[LAYERS]
        y = physics_direction.physics_tdc_excess.to_numpy() if "physics_tdc_excess" in physics_direction else np.full(len(LAYERS), np.nan)
        if np.all(np.isnan(y)):
            table = pd.read_csv(ROOT / "reports" / "geometry_sealed" / "PAPER_GEOMETRY_TABLE.csv")
            y = table[table.model == model].set_index("layer_name").loc[LAYERS].physics_tdc_excess.to_numpy()
        axes[row_index, 1].plot(x, y, color=COLORS["physics"], marker=MARKERS["physics"], lw=1.35, ms=3.3, ls="--", label=LABELS["physics"])
        axes[row_index, 1].axhline(0, color="#777777", lw=0.7, zorder=0)
        for layer, value in zip(LAYERS, y):
            records.append({"model": model, "layer": layer, "panel": "direction", "series": "physics", "estimate": value, "ci_low": np.nan, "ci_high": np.nan, "reference": "observed MRL minus shuffled-correspondence mean"})

        projection = pao[(pao.model == model) & (pao.k == 4)].set_index("layer_name").loc[LAYERS]
        y = projection.pao_raw.to_numpy()
        low = projection.pao_ci_low.to_numpy()
        high = projection.pao_ci_high.to_numpy()
        random = projection.pao_random_mean.to_numpy()
        axes[row_index, 2].errorbar(x, y, yerr=np.vstack([y - low, high - y]), color="#CC79A7", marker="o", lw=1.35, ms=3.5, capsize=2, label="Observed")
        axes[row_index, 2].plot(x, random, color="#666666", lw=1.1, ls=":", label="Random k=4")
        for layer, value, lo, hi, null in zip(LAYERS, y, low, high, random):
            records.append({"model": model, "layer": layer, "panel": "projection", "series": "appearance k=4", "estimate": value, "ci_low": lo, "ci_high": hi, "reference": null})

        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel(f"{model}\nrelative magnitude")
        axes[row_index, 1].set_ylabel(r"$\Delta$MRL")
        axes[row_index, 2].set_ylabel("physical projection")
        for axis in axes[row_index]:
            axis.set_xticks(x, LAYER_LABELS)
            axis.grid(axis="y", color="#dddddd", lw=0.45)
    for col, tag in enumerate(("A", "B", "C")):
        axes[0, col].text(-0.17, 1.03, tag, transform=axes[0, col].transAxes, weight="bold", fontsize=9)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    axes[0, 2].legend(frameon=False, loc="upper right", borderaxespad=0.1)
    fig.supxlabel("frozen depth", y=0.005, fontsize=8)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.12, top=0.88, wspace=0.38, hspace=0.17)
    pd.DataFrame(records).to_csv(REPORT / "figure2_source_data.csv", index=False)
    save(fig, "figure2_controlled_geometry_compact")


def exploratory_trajectories() -> None:
    source = ROOT / "reports" / "vjepa2g_grid_replication"
    straight = pd.read_csv(source / "STRAIGHTNESS.csv")
    rank = pd.read_csv(source / "EFFECTIVE_RANK.csv")
    projection = pd.read_csv(source / "PHYSICS_APPEARANCE_PROJECTIONS.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    x = np.arange(len(LAYERS))
    factors = ["hue", "brightness", "selective_recolor"]
    colors = [COLORS["hue_090"], COLORS["brightness_0p45"], COLORS["selective_recolor_180"]]
    markers = ["o", "s", "^"]
    labels = ["Hue", "Brightness", "Recolor"]
    for factor, color, marker, factor_label in zip(factors, colors, markers, labels):
        current = straight[straight.factor == factor].set_index("layer").loc[LAYERS]
        y = current.straightness_mean.to_numpy()
        axes[0].errorbar(x, y, yerr=np.vstack([y-current.ci_low.to_numpy(), current.ci_high.to_numpy()-y]), color=color, marker=marker, lw=1.25, ms=3.3, capsize=1.5, label=factor_label)
        current_rank = rank[rank.factor == factor].set_index("layer").loc[LAYERS]
        axes[1].plot(x, current_rank.effective_rank, color=color, marker=marker, lw=1.25, ms=3.3, label=factor_label)
    final_projection = projection[projection.k == 4]
    styles = {"endpoint_strong": ("Endpoint", "o"), "all_dose": ("All dose", "s"), "local_tangent": ("Local tangent", "^")}
    for space, (space_label, marker) in styles.items():
        current = final_projection[final_projection.space == space].set_index("layer").loc[LAYERS]
        y = current.projection_mean.to_numpy()
        axes[2].errorbar(x, y, yerr=np.vstack([y-current.ci_low.to_numpy(), current.ci_high.to_numpy()-y]), marker=marker, lw=1.25, ms=3.3, capsize=1.5, label=space_label)
    random = final_projection[final_projection.space == "local_tangent"].set_index("layer").loc[LAYERS].random_mean.to_numpy()
    axes[2].plot(x, random, color="#666666", ls=":", lw=1, label="Random k=4")
    axes[0].set_ylabel("straightness")
    axes[1].set_ylabel("entropy effective rank")
    axes[2].set_ylabel("physical energy fraction")
    for index, axis in enumerate(axes):
        axis.set_xticks(x, LAYER_LABELS)
        axis.set_xlabel("frozen depth")
        axis.grid(axis="y", color="#dddddd", lw=0.45)
        axis.text(-0.18, 1.06, chr(ord("A") + index), transform=axis.transAxes, weight="bold", fontsize=10)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.36, 1.03), fontsize=6.5)
    axes[2].legend(frameon=False, fontsize=6.5)
    figure_data = straight.merge(rank, on=["model", "layer", "factor", "status"], how="outer")
    figure_data.to_csv(REPORT / "exploratory_trajectory_source_data.csv", index=False)
    projection.to_csv(REPORT / "exploratory_projection_source_data.csv", index=False)
    fig.subplots_adjust(wspace=0.42, top=0.88)
    save(fig, "appendix_exploratory_trajectories")


if __name__ == "__main__":
    setup()
    controlled_geometry()
    exploratory_trajectories()
