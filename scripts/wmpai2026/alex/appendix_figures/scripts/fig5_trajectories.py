"""Figure 5 (redesign): per-factor PCA trajectories of dose-dependent displacements (exploratory).

One PCA per (layer, factor), fitted on the train-scene displacements of that factor only
(all three doses pooled, mean-centered SVD). Projections are uncentered, so clean sits at the origin.

Run:  python -I scripts/fig5_trajectories.py   (from the WORK folder)
Writes figs/fig5_dose_trajectories.{png,pdf} (block_30 and final),
figs/fig5_all_layers.{png,pdf} (all five layers, same design), and figs/dose_stats.json.
"""
import os
import site
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# python -I skips the user site directory, where numpy and matplotlib are installed.
sys.path.append(site.getusersitepackages())

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import dose_common as dc  # noqa: E402

COLORS = {"hue": "#E69F00", "brightness": "#56B4E9", "selective_recolor": "#CC79A7"}
FACTOR_NAME = {"hue": "hue", "brightness": "brightness", "selective_recolor": "recolor"}
DOSE_STR = {"hue": "15, 45, 90 deg", "brightness": "0.85, 0.65, 0.45",
            "selective_recolor": "45, 90, 180 deg"}
MARKER_MS = [4.0, 6.5, 9.0]  # mild, medium, strong
N_SAMPLE = 40
SAMPLE_SEED = 1
NOTE = "exploratory, post-sealed; PCA fitted per factor on 23 train scenes"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})


def factor_info(Z, design, f, sample_idx, straight, rank):
    disp = dc.displacements(Z, design)
    train = design["train_mask"]
    rows = np.concatenate([disp[(f, d)][train] for d in dc.DOSES[f]], axis=0)
    V, evr, mu = dc.pca_fit(rows)
    proj = [disp[(f, d)] @ V.T for d in dc.DOSES[f]]  # each (n_sources, 2), uncentered
    mean_path = np.array([np.zeros(2)] + [p.mean(axis=0) for p in proj])  # (4, 2)
    sample_paths = np.stack([np.zeros((len(sample_idx), 2))] +
                            [p[sample_idx] for p in proj], axis=1)  # (N, 4, 2)
    return {
        "factor": f,
        "pc1": float(100 * evr[0]),
        "pc2": float(100 * evr[1]),
        "straightness": float(straight[f]),
        "effective_rank": float(rank[f]),
        "n_train_rows": int(rows.shape[0]),
        "mean_path": mean_path,
        "sample_paths": sample_paths,
    }


def draw_panel(ax, info, layer):
    f = info["factor"]
    c = COLORS[f]
    for path in info["sample_paths"]:
        ax.plot(path[:, 0], path[:, 1], "-", color=c, lw=0.8, alpha=0.18, zorder=2)
    mp = info["mean_path"]
    ax.plot(mp[:, 0], mp[:, 1], "-", color=c, lw=2.4, zorder=4, solid_capstyle="round")
    for k, ms in zip((1, 2, 3), MARKER_MS):
        ax.plot(mp[k, 0], mp[k, 1], "o", color=c, ms=ms, mec="none", zorder=5)
    ax.plot(0, 0, "k+", ms=7, mew=1.2, zorder=6)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({info['pc1']:.1f}%)")
    ax.set_ylabel(f"PC2 ({info['pc2']:.1f}%)")
    ax.set_title(f"{FACTOR_NAME[f]}, {layer}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Headroom above the data for the statistics text, so it never sits on a trajectory.
    # An invisible point extends the autoscaled data range (a fixed ylim would be overridden by the aspect).
    y0, y1 = ax.get_ylim()
    ax.scatter([0.0], [y1 + 0.5 * (y1 - y0)], s=0, alpha=0, linewidths=0)
    ax.autoscale_view()
    lines = [f"doses {DOSE_STR[f]}",
             f"straightness {info['straightness']:.2f}",
             f"effective rank {info['effective_rank']:.2f}"]
    ax.text(0.03, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=8, linespacing=1.25,
            bbox=dict(facecolor="white", edgecolor="0.7", linewidth=0.5, pad=2.5, alpha=0.9),
            zorder=7)


def legend_handles():
    h = [Line2D([0], [0], color="0.3", lw=0.8, alpha=0.5, label="40 sampled sources"),
         Line2D([0], [0], color="0.3", lw=2.4, label="mean path (152 sources)"),
         Line2D([0], [0], color="k", marker="+", ms=7, mew=1.2, lw=0, label="clean (origin)")]
    for name, ms in zip(("mild", "medium", "strong"), MARKER_MS):
        h.append(Line2D([0], [0], color="0.3", marker="o", ms=ms, mec="none", lw=0,
                        label=f"marker size = dose: {name}"))
    return h


def build_figure(layers, infos, out_stem, figsize):
    nrows = len(layers)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(nrows + 1, 3, height_ratios=[1.0] * nrows + [0.45])
    for r, layer in enumerate(layers):
        for c, f in enumerate(dc.FACTORS):
            draw_panel(fig.add_subplot(gs[r, c]), infos[(layer, f)], layer)
    ax_leg = fig.add_subplot(gs[nrows, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False,
                  handlelength=2.2, columnspacing=1.5, handletextpad=0.5, borderaxespad=0.0,
                  bbox_to_anchor=(0.5, 1.0))
    ax_leg.text(0.5, 0.0, NOTE, ha="center", va="bottom", fontsize=8, transform=ax_leg.transAxes)
    for ext in ("png", "pdf"):
        out = dc.FIGS / f"{out_stem}.{ext}"
        if ext == "png":
            fig.savefig(out, dpi=200, facecolor="white")
        else:
            fig.savefig(out, facecolor="white")
        print("wrote", out)
    return fig


def layout_check(fig):
    """Return visible text overlaps and text extents outside the figure, measured by the renderer."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    # Tick labels for ticks outside the view limits are never drawn, so only drawn ones are checked.
    all_tick_texts, drawn_tick_texts = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                all_tick_texts.add(tick.label1)
                if lo - 1e-9 <= tick.get_loc() <= hi + 1e-9 and tick.label1.get_visible():
                    drawn_tick_texts.add(tick.label1)
    texts = [t for t in fig.findobj(mtext.Text)
             if t.get_visible() and t.get_text().strip()
             and (t not in all_tick_texts or t in drawn_tick_texts)]
    boxes = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        boxes.append((t.get_text().replace("\n", " | "), bb))
    clipped = [name for name, bb in boxes
               if bb.x0 < fig_box.x0 - 0.5 or bb.y0 < fig_box.y0 - 0.5
               or bb.x1 > fig_box.x1 + 0.5 or bb.y1 > fig_box.y1 + 0.5]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b):
                ov_w = min(a.x1, b.x1) - max(a.x0, b.x0)
                ov_h = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ov_w > 1.0 and ov_h > 1.0:
                    overlaps.append((boxes[i][0], boxes[j][0]))
    return {"n_texts": len(boxes), "clipped": clipped, "overlaps": overlaps}


def main():
    dc.FIGS.mkdir(parents=True, exist_ok=True)
    samples, split, feats = dc.load_inputs()
    design = dc.build_design(samples, split)
    n = len(design["sources"])
    sample_idx = np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, size=N_SAMPLE, replace=False))

    infos, per_layer = {}, {}
    for li, layer in enumerate(dc.LAYERS):
        Z = dc.l2norm_rows(feats[:, li, :])
        straight = dc.straightness_by_factor(Z, design)
        rank = dc.effective_rank_by_factor(Z, design)
        per_layer[layer] = {"straightness": straight, "effective_rank": rank}
        for f in dc.FACTORS:
            infos[(layer, f)] = factor_info(Z, design, f, sample_idx, straight, rank)

    # Final-layer validation against the paper values (both variants, unchanged definitions).
    Zf = dc.l2norm_rows(feats[:, 4, :])
    Xf = feats[:, 4, :].astype(np.float64)
    st = {"zhat": per_layer["final"]["straightness"], "raw": dc.straightness_by_factor(Xf, design)}
    rk = {"zhat": per_layer["final"]["effective_rank"], "raw": dc.effective_rank_by_factor(Xf, design)}
    acc = dc.acceptance(st, rk)
    assert np.allclose([st["zhat"][f] for f in dc.FACTORS], [dc.straightness_by_factor(Zf, design)[f] for f in dc.FACTORS])

    fig_main = build_figure(["block_30", "final"], infos, "fig5_dose_trajectories", (7.4, 4.6))
    chk_main = layout_check(fig_main)
    plt.close(fig_main)
    fig_all = build_figure(list(dc.LAYERS), infos, "fig5_all_layers", (7.4, 8.4))
    chk_all = layout_check(fig_all)
    plt.close(fig_all)

    stats = {
        "python": sys.version.split()[0],
        "design": "one PCA per (layer, factor) on train-scene displacements of that factor, all doses pooled, "
                  "mean-centered SVD, uncentered projection (clean at origin)",
        "sampled_source_indices_seed1": sample_idx.tolist(),
        "variance_metric": "straightness and effective rank use zhat (L2-normalized features)",
        "layers": {
            layer: {
                f: {k: v for k, v in infos[(layer, f)].items() if k not in ("mean_path", "sample_paths")}
                for f in dc.FACTORS
            }
            for layer in dc.LAYERS
        },
        "mean_paths_pc_coords": {
            f"{layer}|{f}": infos[(layer, f)]["mean_path"].round(6).tolist()
            for layer in dc.LAYERS for f in dc.FACTORS
        },
        "final_layer_acceptance": acc,
        "clip_counts_by_factor_dose": dc.clip_counts(samples, split),
        "layout_check_main": chk_main,
        "layout_check_all_layers": chk_all,
        "n_sources": n,
        "n_train_sources": int(design["train_mask"].sum()),
        "n_heldout_sources": int((~design["train_mask"]).sum()),
    }
    (dc.FIGS / "dose_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("final-layer acceptance:")
    for v in ("zhat", "raw"):
        for r in acc[v]["rows"]:
            print(f"  {v:5s} {r['factor']:18s} {r['metric']:15s} computed {r['computed']:.4f} "
                  f"paper {r['paper']} {'PASS' if r['match'] else 'FAIL'}")
    print("per-factor PCA (PC1 %, PC2 %, straightness, effective rank):")
    for layer in dc.LAYERS:
        for f in dc.FACTORS:
            i = infos[(layer, f)]
            print(f"  {layer:9s} {f:18s} PC1 {i['pc1']:6.2f}  PC2 {i['pc2']:6.2f}  "
                  f"straight {i['straightness']:.4f}  eff rank {i['effective_rank']:.3f}  rows {i['n_train_rows']}")
    for name, chk in (("main", chk_main), ("all_layers", chk_all)):
        print(f"layout check {name}: texts {chk['n_texts']}, clipped {len(chk['clipped'])}, "
              f"overlaps {len(chk['overlaps'])}")
        for a, b in chk["overlaps"]:
            print("   OVERLAP:", a, "<->", b)
        for c_ in chk["clipped"]:
            print("   CLIPPED:", c_)


if __name__ == "__main__":
    main()
