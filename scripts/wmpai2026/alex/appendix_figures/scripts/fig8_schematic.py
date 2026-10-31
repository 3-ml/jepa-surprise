"""Schematic (no data) for fig8: native prediction-error score and crossed 2x2 test design.

Writes, under the work folder that contains this scripts/ folder:
  figs/fig8_score_pipeline.png   (200 dpi)
  figs/fig8_score_pipeline.pdf   (vector)
  figs/schematic_provenance.json
"""
import datetime
import hashlib
import json
import math
import os
import platform
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
FIGS = os.path.join(WORK, "figs")
os.makedirs(FIGS, exist_ok=True)
PNG = os.path.join(FIGS, "fig8_score_pipeline.png")
PDF = os.path.join(FIGS, "fig8_score_pipeline.pdf")
PROV = os.path.join(FIGS, "schematic_provenance.json")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

EDGE = "#333333"
BOX = "#f2f2f2"
BLUE = "#a6cee3"      # context frames (colorblind-safe pastel)
ORANGE = "#fdbf6f"    # target frames
GREEN = "#c7e9c0"     # lawful cells
PURPLE = "#d9cce8"    # inconsistent cells
SCORE = "#fff2cc"
PRED_PAL = ["#a6cee3", "#cfe6f4", "#8fbfdd"]
TGT_PAL = ["#fdbf6f", "#fed9a6", "#fdc58a"]
BAR = "#8c8c8c"

FIG_W, FIG_H = 7.4, 4.6
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.set_autoscale_on(False)
ax.axis("off")

CONTAINED = []  # (text artist, (x0, y0, x1, y1) container in data units, label)


def txt(x, y, s, size=9, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    return ax.text(x, y, s, fontsize=size, color="black", linespacing=1.25, zorder=6, **kw)


def box(x0, y0, w, h, s=None, fc=BOX, size=9):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=EDGE, lw=0.9, zorder=2))
    if s is not None:
        t = txt(x0 + w / 2, y0 + h / 2, s, size=size)
        CONTAINED.append((t, (x0, y0, x0 + w, y0 + h), s))


def rect(x0, y0, w, h, fc, lw=0.6, z=3):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=EDGE, lw=lw, zorder=z))


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(posA=(x0, y0), posB=(x1, y1), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color=EDGE,
                                 shrinkA=0, shrinkB=0, zorder=4))


def on_circle(cx, cy, r, px, py):
    d = math.hypot(px - cx, py - cy)
    return cx + r * (px - cx) / d, cy + r * (py - cy) / d


def token_grid(x0, y0, pal):
    """18 squares in 3 rows x 6 columns; (x0, y0) is the bottom-left corner."""
    c, g = 1.3, 0.25
    for r in range(3):
        for k in range(6):
            fc = pal[(2 * r + 3 * k + r * k) % len(pal)]
            rect(x0 + k * (c + g), y0 + (2 - r) * (c + g), c, c, fc, lw=0.5)


# ---------------------------------------------------------------- PART A
txt(0.6, 60.2, "A. How the native score is computed", size=9.5,
    fontweight="bold", ha="left")

# 1. film strip: 48 frames in two rows of 24 (frames 0-23 top, 24-47 bottom)
rect(0.6, 39.4, 19.8, 9.4, "#e6e6e6", lw=0.9, z=2)
for i in range(48):
    row, col = (0, i) if i < 24 else (1, i - 24)
    rect(1.2 + col * 0.78, 44.6 if row == 0 else 40.4, 0.62, 3.0,
         BLUE if i < 12 else ORANGE, lw=0.4)
txt(10.5, 50.6, "context frames 0 to 11", size=8)
txt(10.5, 37.6, "target frames 12 to 47", size=8)
txt(10.5, 34.6, "clip: 48 frames at 12 fps", size=8)

# 2. context branch: frozen encoder f -> predictor p (frozen) -> predicted tokens
arrow(20.5, 47.8, 23.5, 52.0)
box(23.5, 50.5, 10.0, 7.0, "frozen\nencoder f")
arrow(33.5, 54.0, 37.0, 54.0)
box(37.0, 50.5, 12.0, 7.0, "predictor p\n(frozen)")
arrow(49.0, 54.0, 51.8, 54.0)
token_grid(51.8, 51.8, PRED_PAL)
txt(56.3, 48.6, "predicted\ntarget tokens", size=8)

# 3. target branch: EMA target encoder (stop-gradient) + layer norm -> target tokens
arrow(20.5, 40.0, 23.5, 37.0)
box(23.5, 33.0, 20.0, 8.0, "EMA target encoder\n(stop-gradient)\n+ layer norm")
arrow(43.5, 37.0, 46.8, 37.0)
token_grid(46.8, 34.8, TGT_PAL)
txt(51.3, 31.6, "target tokens", size=8)

# 4. both token sets feed |p - t|, then mean, then the native score
CX, CY, R = 69.5, 45.5, 4.8
ax.add_patch(Circle((CX, CY), R, fc="white", ec=EDGE, lw=1.0, zorder=3))
t_circ = txt(CX, CY, "| p - t |")
CONTAINED.append((t_circ, (CX - R, CY - R, CX + R, CY + R), "| p - t |"))
sx, sy = on_circle(CX, CY, R, 60.9, 54.0)
arrow(60.9, 54.0, sx, sy)
sx, sy = on_circle(CX, CY, R, 56.1, 38.0)
arrow(56.1, 38.0, sx, sy)
arrow(CX + R, CY, 77.6, CY)
box(77.8, 41.5, 18.4, 8.0, "mean over tokens\nand 18 tubelets")
arrow(87.0, 41.5, 87.0, 37.6)
box(77.8, 28.5, 18.4, 9.0, "native score S\n(higher means more\nprediction error)", fc=SCORE)

txt(0.6, 29.6, "frozen V-JEPA2-G, bfloat16 autocast, 256 px", size=8, ha="left")

# ---------------------------------------------------------------- PART B
txt(0.6, 26.6, "B. Crossed test: same motion, opposite physical validity", size=9.5,
    fontweight="bold", ha="left")
txt(26.0, 23.8, "solid support")
txt(42.0, 23.8, "gap (no support)")
txt(9.5, 17.75, "object stops at\nplatform height")
txt(9.5, 7.25, "object falls\nthrough")

COL_X = {0: 19.0, 1: 35.0}     # column 0 solid, column 1 gap
ROW_Y = {0: 13.0, 1: 2.5}      # row 0 stops, row 1 falls
CELL_W, CELL_H = 14.0, 9.5
CELLS = {
    (0, 0): (GREEN, "lawful"),        # solid support, stops at platform height
    (0, 1): (PURPLE, "inconsistent"), # gap, stops (object hovers)
    (1, 0): (PURPLE, "inconsistent"), # solid support, falls through (passes the bar)
    (1, 1): (GREEN, "lawful"),        # gap, falls through
}
for (row, col), (fc, name) in CELLS.items():
    x0, y0 = COL_X[col], ROW_Y[row]
    rect(x0, y0, CELL_W, CELL_H, fc, lw=0.9, z=1)
    t = txt(x0 + CELL_W / 2, y0 + 8.1, name)
    CONTAINED.append((t, (x0, y0, x0 + CELL_W, y0 + CELL_H), name))
    cx = x0 + CELL_W / 2
    py = y0 + 3.6                       # platform level
    bars = [(x0 + 1.2, 11.6)] if col == 0 else [(x0 + 1.2, 4.4), (x0 + 8.4, 4.4)]
    for bx, bw in bars:
        rect(bx, py - 0.6, bw, 0.6, BAR, lw=0.6)
    if row == 0:                        # object rests at platform height
        ocy = py + 1.0
    else:                               # object below, dashed trail from platform level
        ocy = y0 + 1.6
        ax.plot([cx, cx], [py, ocy + 1.0], ls="--", color=EDGE, lw=0.9, zorder=3.5)
    ax.add_patch(Circle((cx, ocy), 1.0, fc="white", ec=EDGE, lw=1.0, zorder=5))

TX = 51.5
txt(TX, 19.0, "M_solid = S(fall) - S(stop)  on solid support", ha="left")
txt(TX, 15.0, "M_gap = S(stop) - S(fall)  over the gap", ha="left")
txt(TX, 11.0, "physical-surprise reading needs both margins > 0", ha="left")
txt(TX, 5.4, "also crossed with appearance:", size=8, ha="left")
txt(TX, 3.2, "clean, persistent hue, abrupt hue", size=8, ha="left")


# ---------------------------------------------------------------- CHECKS
def run_checks():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def dbox(t):
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return x0, y0, x1, y1

    problems = []
    texts = [t for t in ax.texts if t.get_text()]
    bbs = {id(t): dbox(t) for t in texts}
    for t, (x0, y0, x1, y1), label in CONTAINED:
        a, b, c, d = bbs[id(t)]
        if a < x0 + 0.15 or c > x1 - 0.15 or b < y0 + 0.15 or d > y1 - 0.15:
            problems.append(f"does not fit: {label!r} text {c - a:.1f}x{d - b:.1f}, "
                            f"box {x1 - x0:.1f}x{y1 - y0:.1f}")
    for i, t in enumerate(texts):
        a, b, c, d = bbs[id(t)]
        if a < 0 or b < 0 or c > 100 or d > 62:
            problems.append(f"outside canvas: {t.get_text()!r}")
        for u in texts[i + 1:]:
            e, f, g, h = bbs[id(u)]
            if min(c, g) - max(a, e) > 0.05 and min(d, h) - max(b, f) > 0.05:
                problems.append(f"overlap: {t.get_text()!r} / {u.get_text()!r}")
    return problems, len(texts)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


problems, n_text = run_checks()
for p in problems:
    print("LAYOUT:", p)
print(f"checked {n_text} text items; layout problems: {len(problems)}")

stamp = datetime.datetime(2026, 10, 31)
fig.savefig(PNG, dpi=200)
fig.savefig(PDF, metadata={"Title": "fig8 score pipeline schematic",
                           "CreationDate": stamp, "ModDate": stamp})

pixels = mpimg.imread(PNG).shape  # (rows, cols, channels)
prov = {
    "figure": "fig8_score_pipeline",
    "script": "scripts/fig8_schematic.py",
    "script_sha256": sha256(os.path.abspath(__file__)),
    "command": "python " + " ".join(sys.argv) + "   (current directory: work folder)",
    "python_version": sys.version.split()[0],
    "matplotlib_version": matplotlib.__version__,
    "platform": platform.platform(),
    "figure_size_in": [FIG_W, FIG_H],
    "png": {"path": "figs/fig8_score_pipeline.png", "dpi": 200,
            "pixel_size_wh": [int(pixels[1]), int(pixels[0])],
            "sha256": sha256(PNG)},
    "pdf": {"path": "figs/fig8_score_pipeline.pdf", "sha256": sha256(PDF)},
    "layout_checks": {"text_items": n_text, "problems": problems},
}
with open(PROV, "w", encoding="utf-8") as fh:
    json.dump(prov, fh, indent=2)
print("wrote", PNG, PDF, PROV, sep="\n")
print("png pixel size (w x h):", prov["png"]["pixel_size_wh"])
