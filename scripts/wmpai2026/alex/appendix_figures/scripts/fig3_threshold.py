"""Figure 3: score distributions by class with the frozen threshold t*.

Run from the work folder with: python scripts/fig3_threshold.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kubric_common import (  # noqa: E402
    APPS, COLOR_INCONS, COLOR_LAWFUL, FIGS, JITTER_SEED, T_STAR,
    auroc, lawful_and_inconsistent, load, rates)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


Y_BOTTOM, Y_TOP = 0.3, 3.5  # data range per panel; the top band holds the stats text


def main():
    S, scenes, info = load()
    rng = np.random.default_rng(JITTER_SEED)
    all_vals = np.concatenate(list(S.values()))
    xlo, xhi = float(all_vals.min()) - 0.005, float(all_vals.max()) + 0.005

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=True, sharey=True,
                             layout="constrained")
    for row, app in enumerate(APPS):
        for col, world in enumerate(("solid", "gap")):
            ax = axes[row, col]
            lawful, inc = lawful_and_inconsistent(S, world, app)
            fa, miss = rates(lawful, inc)
            auc = auroc(inc, lawful)

            parts = ax.violinplot([lawful, inc], positions=[1, 2], vert=False,
                                  widths=0.8, showextrema=False)
            for body, color in zip(parts["bodies"], (COLOR_LAWFUL, COLOR_INCONS)):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.45)

            for y, vals, color in ((1, lawful, COLOR_LAWFUL), (2, inc, COLOR_INCONS)):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(vals, y + jitter, s=6, color=color, alpha=0.35, linewidths=0,
                           zorder=3)

            # The line stops at data y = 2.95, below the stats text in the headroom above.
            ax.axvline(T_STAR, ymin=0.0, ymax=(2.95 - Y_BOTTOM) / (Y_TOP - Y_BOTTOM),
                       color="black", ls="--", lw=1.0, zorder=4)
            if row == 0 and col == 0:
                ax.text(T_STAR + 0.002, 0.42, "t* = 0.486", ha="left", va="bottom",
                        fontsize=8)

            ax.text(0.02, 0.98,
                    f"false alarms {fa:.1f}%, misses {miss:.1f}%, AUROC {auc:.3f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

            ax.set_ylim(Y_BOTTOM, Y_TOP)
            ax.set_xlim(xlo, xhi)
            ax.set_yticks([1, 2])
            # Axes share y, so hide labels on the right column with tick_params
            # instead of set_yticklabels, which would also clear the left column.
            if col == 0:
                ax.set_yticklabels(["lawful", "inconsistent"])
                ax.set_ylabel(app)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == 0:
                ax.set_title("solid support" if world == "solid" else "gap")

    for ax in axes[-1, :]:
        ax.set_xlabel("native score (mean absolute error)")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig3_threshold_distributions.png", dpi=200, facecolor="white")
    fig.savefig(FIGS / "fig3_threshold_distributions.pdf", facecolor="white",
                metadata={"CreationDate": None})
    print(f"scenes={len(scenes)} dropped={info['records_dropped_not_complete']} "
          f"x_range=({xlo:.4f}, {xhi:.4f}) jitter_seed={JITTER_SEED}")
    print("wrote", FIGS / "fig3_threshold_distributions.png", "and",
          FIGS / "fig3_threshold_distributions.pdf")


if __name__ == "__main__":
    main()
