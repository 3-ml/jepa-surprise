"""Figure 7: cosine similarity between mean displacement directions (exploratory).

Run:  python -I scripts/fig7_cosines.py   (from the WORK folder)
Writes figs/fig7_displacement_cosines.{png,pdf} and figs/fig7_cosines.json.
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
import numpy as np  # noqa: E402

import dose_common as dc  # noqa: E402

LABELS = ["hue 90", "brightness 0.45", "recolor 180", "physical"]
NOTE = "descriptive summary of mean directions, not one of the frozen paper metrics"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def mean_vectors(Z, design, samples_pairs=None):
    clean = Z[design["clean_idx"]]
    vecs = []
    for f in dc.FACTORS:
        strong = Z[design["cond_idx"][(f, dc.DOSES[f][-1])]]
        vecs.append((strong - clean).mean(axis=0))
    phys = (Z[design["pair_impossible"]] - Z[design["pair_possible"]]).mean(axis=0)
    vecs.append(phys)
    return np.stack(vecs)  # (4, D)


def cosine_matrix(V):
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    return U @ U.T


def main():
    dc.FIGS.mkdir(parents=True, exist_ok=True)
    samples, split, feats = dc.load_inputs()
    design = dc.build_design(samples, split)
    assert len(design["pair_possible"]) == 76

    mats = {}
    for li, layer in enumerate(dc.LAYERS):
        Z = dc.l2norm_rows(feats[:, li, :])
        mats[layer] = cosine_matrix(mean_vectors(Z, design))

    out = {
        "space": "zhat (L2-normalized per clip), mean over 152 sources (factors) and 76 pairs (physical)",
        "labels": LABELS,
        "cosines": {layer: np.round(M, 6).tolist() for layer, M in mats.items()},
    }
    (dc.FIGS / "fig7_cosines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(7.4, 2.2))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.1], wspace=0.12,
                          left=0.14, right=0.93, top=0.9, bottom=0.5)
    im = None
    for k, layer in enumerate(dc.LAYERS):
        ax = fig.add_subplot(gs[0, k])
        M = mats[layer]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(4):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS, rotation=90, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(LABELS if k == 0 else [], fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(layer)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(gs[0, 5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cosine similarity", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    fig.text(0.5, 0.02, NOTE, ha="center", va="bottom", fontsize=8)

    for ext in ("png", "pdf"):
        target = dc.FIGS / f"fig7_displacement_cosines.{ext}"
        if ext == "png":
            fig.savefig(target, dpi=200, facecolor="white")
        else:
            fig.savefig(target, facecolor="white")
        print("wrote", target)
    plt.close(fig)

    for layer in dc.LAYERS:
        print(layer)
        for i, lab in enumerate(LABELS):
            print("  ", f"{lab:14s}", " ".join(f"{v:6.2f}" for v in mats[layer][i]))


if __name__ == "__main__":
    main()
