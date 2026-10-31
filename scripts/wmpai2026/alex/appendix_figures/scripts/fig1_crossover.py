"""Figure 1: crossover plots, two rows by three appearance conditions.

Row 1 groups clips by trajectory (stop, fall). Row 2 groups clips by physical validity
(lawful, inconsistent). The validity lines cross between solid support and gap.

Run from the work folder with: python scripts/fig1_crossover.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kubric_common import (  # noqa: E402
    APPS, COLOR_FALL, COLOR_INCONS, COLOR_LAWFUL, COLOR_STOP, FIGS, N_SUBSET, SUBSET_SEED,
    boot_ci, boot_indices, lawful_and_inconsistent, load, margin)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def draw_line_pair(ax, idx, subset, pairs, colors, labels):
    """pairs: list of (solid_vector, gap_vector) per series; draws faint per-scene lines,
    then thick mean lines with 95 percent scene-bootstrap error bars."""
    for (solid, gap), color in zip(pairs, colors):
        for i in subset:
            ax.plot([0, 1], [solid[i], gap[i]], color=color, alpha=0.12, lw=0.8, zorder=1)
    handles = []
    for (solid, gap), color, label in zip(pairs, colors, labels):
        stats = [boot_ci(solid, idx), boot_ci(gap, idx)]
        means = np.array([s[0] for s in stats])
        lo = np.array([s[1] for s in stats])
        hi = np.array([s[2] for s in stats])
        handles.append(ax.errorbar([0, 1], means, yerr=np.vstack([means - lo, hi - means]),
                                   color=color, lw=2.0, marker="o", ms=5, capsize=3,
                                   zorder=3, label=label))
    return handles


def main():
    S, scenes, info = load()
    n = len(scenes)
    idx = boot_indices(n)
    subset = np.sort(np.random.default_rng(SUBSET_SEED).choice(n, size=N_SUBSET, replace=False))

    all_vals = np.concatenate(list(S.values()))
    ylo, yhi = float(all_vals.min()) - 0.002, float(all_vals.max()) + 0.002

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.085,
                        wspace=0.10, hspace=0.42)

    for col, app in enumerate(APPS):
        # Row 1: by trajectory. Series are stop and fall, each as (solid, gap).
        ax = axes[0, col]
        stop_pair = (S[("solid", "stop", app)], S[("gap", "stop", app)])
        fall_pair = (S[("solid", "fall", app)], S[("gap", "fall", app)])
        handles = draw_line_pair(ax, idx, subset, [stop_pair, fall_pair],
                                 [COLOR_STOP, COLOR_FALL], ["stop trajectory", "fall trajectory"])
        m_solid = margin(S, "solid", app)
        m_gap = margin(S, "gap", app)
        ax.text(0.5, 0.04,
                f"inconsistent minus lawful:\nsolid {m_solid:+.4f}, gap {m_gap:+.4f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        ax.set_title(app)
        if col == 0:
            ax.legend(handles=handles, loc="upper right", frameon=False)

        # Row 2: by physical validity. Lawful = solid stop to gap fall;
        # inconsistent = solid fall to gap stop.
        ax = axes[1, col]
        lawful_solid, inc_solid = lawful_and_inconsistent(S, "solid", app)
        lawful_gap, inc_gap = lawful_and_inconsistent(S, "gap", app)
        handles = draw_line_pair(ax, idx, subset,
                                 [(lawful_solid, lawful_gap), (inc_solid, inc_gap)],
                                 [COLOR_LAWFUL, COLOR_INCONS], ["lawful", "inconsistent"])
        ax.text(0.5, 0.04,
                "lawful higher on solid support,\ninconsistent higher over the gap",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=5)
        if col == 0:
            ax.legend(handles=handles, loc="upper right", frameon=False)

        for row in (0, 1):
            a = axes[row, col]
            a.set_xticks([0, 1])
            a.set_xticklabels(["solid support", "gap"])
            a.set_xlim(-0.35, 1.35)
            a.set_ylim(ylo, yhi)

    axes[0, 0].set_ylabel("native score (mean absolute error)")
    axes[1, 0].set_ylabel("native score (mean absolute error)")
    fig.text(0.012, 0.72, "by trajectory", rotation=90, ha="left", va="center",
             fontsize=9, fontweight="bold")
    fig.text(0.012, 0.28, "by physical validity", rotation=90, ha="left", va="center",
             fontsize=9, fontweight="bold")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig1_crossover.png", dpi=200, facecolor="white")
    fig.savefig(FIGS / "fig1_crossover.pdf", facecolor="white",
                metadata={"CreationDate": None})
    print(f"scenes={n} dropped={info['records_dropped_not_complete']} "
          f"subset_size={len(subset)} subset_first5={subset[:5].tolist()} "
          f"ylim=({ylo:.4f}, {yhi:.4f})")
    print("wrote", FIGS / "fig1_crossover.png", "and", FIGS / "fig1_crossover.pdf")


if __name__ == "__main__":
    main()
