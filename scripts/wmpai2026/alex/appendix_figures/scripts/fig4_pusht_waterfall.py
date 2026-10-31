"""Waterfall decomposition of the PushT cross-target contrast Sh - So.

Data: mean native latent MSE at hue 60 degrees and action scale 0.75, 256 episodes
(hard-coded from a paper table).

Decomposition:
    Sh - So = (Sh - S0) - (So - Sl) - (Sl - S0)
"""
import hashlib
import json
import math
import os
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
FIGS = os.path.join(WORK, "figs")
STEM = "fig4_pusht_waterfall"
SCRIPT_PATH = os.path.abspath(__file__)
PNG_PATH = os.path.join(FIGS, STEM + ".png")
PDF_PATH = os.path.join(FIGS, STEM + ".pdf")
PROV_PATH = os.path.join(FIGS, "pusht_fig_provenance.json")

TOL = 5e-6
BAR_W = 0.52

# S0 = static lawful, Sh = hue-modified static, Sl = moving lawful, So = moving target, opposite action
DATA = {
    "LeWM": dict(S0=0.008728, Sh=0.895323, Sl=0.008358, So=0.253505),
    "DINO-WM": dict(S0=0.061985, Sh=0.111752, Sl=0.218553, So=0.221626),
    "JEPA-WM": dict(S0=0.196626, Sh=0.214008, Sl=0.335983, So=0.338820),
}
EXPECTED_CONTRAST = {"LeWM": 0.641818, "DINO-WM": -0.109874, "JEPA-WM": -0.124812}
PAPER_CONTRAST = {"LeWM": 0.642, "DINO-WM": -0.10987, "JEPA-WM": -0.12481}
PAPER_DECIMALS = {"LeWM": 3, "DINO-WM": 5, "JEPA-WM": 5}

COLORS = {
    "hue": "#E69F00",
    "action": "#56B4E9",
    "baseline": "#999999",
    "total": "#000000",
}
# "inc." and "diff." are abbreviations: the full words are wider than one bar slot
# in a 7.2 in wide three-panel figure, and rotated labels then overlap.
X_LABELS = ["hue\ninc.", "minus\naction\ninc.", "minus\nbaseline\ndiff.", "Sh -\nSo"]


def decompose(d):
    hue = d["Sh"] - d["S0"]          # hue increment over own lawful baseline
    action = d["So"] - d["Sl"]       # action increment over own lawful baseline
    baseline = d["Sl"] - d["S0"]     # difference in lawful target difficulty
    direct = d["Sh"] - d["So"]       # reported contrast
    three_term = hue - action - baseline
    return dict(hue=hue, action=action, baseline=baseline, direct=direct,
                three_term=three_term, residual=three_term - direct)


def fmt3(v):
    """Signed value with 3 significant digits."""
    if v == 0:
        return "0"
    decimals = 2 - math.floor(math.log10(abs(v)))
    return f"{v:+.{decimals}f}"


def draw_panel(ax, name, d):
    t = decompose(d)
    contrib = [t["hue"], -t["action"], -t["baseline"]]
    fills = [COLORS["hue"], COLORS["action"], COLORS["baseline"]]

    bars = []   # (low, high, signed value, facecolor)
    ends = []   # running level after each step, used for connectors
    level = 0.0
    for c, col in zip(contrib, fills):
        start, end = level, level + c
        bars.append((min(start, end), max(start, end), c, col))
        ends.append(end)
        level = end
    bars.append((min(0.0, t["direct"]), max(0.0, t["direct"]), t["direct"], COLORS["total"]))

    ymin = min(0.0, min(b[0] for b in bars))
    ymax = max(0.0, max(b[1] for b in bars))
    span = ymax - ymin
    ax.set_ylim(ymin - 0.22 * span, ymax + 0.16 * span)
    pad = 0.02 * span

    ax.axhline(0.0, color="black", lw=0.6, zorder=1)
    for i, y in enumerate(ends):
        ax.plot([i + BAR_W / 2, i + 1 - BAR_W / 2], [y, y], color="0.25", lw=0.6, zorder=2)

    for i, (lo, hi, v, col) in enumerate(bars):
        if i == len(bars) - 1:
            ax.bar(i, hi - lo, bottom=lo, width=BAR_W, color=col, edgecolor="white",
                   hatch="////", linewidth=0, zorder=3)
        else:
            ax.bar(i, hi - lo, bottom=lo, width=BAR_W, color=col, linewidth=0, zorder=3)
        # Value label side. The hue step sits above its bar. The action step goes
        # below its bar, unless the baseline step has its bottom edge at the same
        # level (LeWM), in which case both labels would collide, so it goes above.
        # The baseline step goes below. The total follows its sign.
        is_total = i == len(bars) - 1
        if i == 0:
            above = v >= 0
        elif i == 1:
            above = abs(bars[1][0] - bars[2][0]) < 1e-3 * span
        elif is_total:
            above = v >= 0
        else:
            above = False
        if above:
            ax.text(i, hi + pad, fmt3(v), ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i, lo - pad, fmt3(v), ha="center", va="top", fontsize=8)

    ax.set_title(name, fontsize=9)
    ax.set_xticks(range(len(X_LABELS)))
    ax.set_xticklabels(X_LABELS, rotation=20, ha="center", va="top", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(-0.6, len(X_LABELS) - 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def main():
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "hatch.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, (name, d) in zip(axes, DATA.items()):
        draw_panel(ax, name, d)
    # Explicit margins: the three panels are narrow, so the default tight layout
    # leaves too little room for the rotated four-bar tick labels.
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.90, wspace=0.24)
    fig.text(0.5, 0.012,
             "Each panel has its own y axis (native latent MSE, own scale). "
             "Magnitudes are not comparable across models.",
             ha="center", va="bottom", fontsize=8)

    fig.savefig(PNG_PATH, dpi=200, facecolor="white", metadata={"Software": None})
    fig.savefig(PDF_PATH, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

    # Acceptance checks
    checks = {}
    max_residual = 0.0
    all_pass = True
    print("Acceptance checks (tolerance %.1e)" % TOL)
    for name, d in DATA.items():
        t = decompose(d)
        diff_expected = abs(t["direct"] - EXPECTED_CONTRAST[name])
        paper_diff = abs(t["direct"] - PAPER_CONTRAST[name])
        paper_within_rounding = paper_diff <= 0.5 * 10 ** (-PAPER_DECIMALS[name]) + 1e-12
        ok = diff_expected <= TOL
        all_pass = all_pass and ok
        max_residual = max(max_residual, abs(t["residual"]))
        print(f"{name:9s} Sh-So computed {t['direct']:+.9f} expected {EXPECTED_CONTRAST[name]:+.6f} "
              f"|diff| {diff_expected:.2e} {'PASS' if ok else 'FAIL'}")
        print(f"{'':9s} three-term sum {t['three_term']:+.12f} residual {t['residual']:+.3e}")
        print(f"{'':9s} paper {PAPER_CONTRAST[name]:+.5f} |computed - paper| {paper_diff:.2e} "
              f"(within paper rounding: {paper_within_rounding})")
        checks[name] = dict(
            hue_increment=t["hue"], action_increment=t["action"], baseline_difference=t["baseline"],
            sh_minus_so=t["direct"], expected=EXPECTED_CONTRAST[name],
            abs_diff_expected=diff_expected, pass_tolerance=ok,
            three_term_sum=t["three_term"], residual=t["residual"],
            paper_value=PAPER_CONTRAST[name], abs_diff_paper=paper_diff,
            paper_within_rounding=paper_within_rounding,
        )
    print(f"max |three-term residual| {max_residual:.3e}")
    print("ALL EXPECTED-VALUE CHECKS PASS" if all_pass else "SOME EXPECTED-VALUE CHECKS FAILED")

    provenance = {
        "script": {"path": SCRIPT_PATH, "sha256": sha256(SCRIPT_PATH)},
        "outputs": {
            os.path.relpath(PNG_PATH, WORK).replace(os.sep, "/"): sha256(PNG_PATH),
            os.path.relpath(PDF_PATH, WORK).replace(os.sep, "/"): sha256(PDF_PATH),
        },
        "python_version": sys.version.split()[0],
        "matplotlib_version": matplotlib.__version__,
        "platform": platform.platform(),
        "command": " ".join([sys.executable, *sys.argv]),
        "data": DATA,
        "checks": checks,
        "max_abs_three_term_residual": max_residual,
        "tolerance": TOL,
        "all_expected_checks_pass": all_pass,
    }
    with open(PROV_PATH, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    print("wrote", PNG_PATH)
    print("wrote", PDF_PATH)
    print("wrote", PROV_PATH)


if __name__ == "__main__":
    main()
