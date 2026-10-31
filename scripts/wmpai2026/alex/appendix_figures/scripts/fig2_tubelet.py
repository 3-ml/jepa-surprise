"""Figure 2: tubelet time course of the native prediction error (f256 scores).

Run from the work folder as:  python -I scripts/fig2_tubelet.py
Reads  f256/original/scores/all-scores.json  (read only).
Writes figs/fig2_tubelet_timecourse.png, figs/fig2_tubelet_timecourse.pdf,
       figs/tubelet_fig_provenance.json.
"""
import hashlib
import json
import os
import platform
import site
import sys
from collections import Counter

# Under `python -I` the user site-packages (where numpy and matplotlib live
# on this machine) is not on sys.path, so add it back explicitly.
_USER_SITE = site.getusersitepackages()
if os.path.isdir(_USER_SITE) and _USER_SITE not in sys.path:
    sys.path.append(_USER_SITE)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WORK, "f256", "original", "scores", "all-scores.json")
FIGS = os.path.join(WORK, "figs")
SCRIPT = os.path.abspath(__file__)
PNG_PATH = os.path.join(FIGS, "fig2_tubelet_timecourse.png")
PDF_PATH = os.path.join(FIGS, "fig2_tubelet_timecourse.pdf")
PROV_PATH = os.path.join(FIGS, "tubelet_fig_provenance.json")
COMMAND = "python -I scripts/fig2_tubelet.py (run from the work folder)"

SEED = 20261031
N_BOOT = 5000
FPS = 12.0
N_TUBELETS = 18
WINDOW = (1.0, 2.0)  # frames 12 to 24
THRESH = 0.003

STOP_C = "#D55E00"
FALL_C = "#0072B2"
DIFF_SOLID_C = "black"
DIFF_GAP_C = "0.45"

WORLDS = ["solid", "gap"]
TRAJS = ["stop", "fall"]
APPS = ["clean", "persistent", "abrupt"]
APPEAR_MAP = {
    "clean": "clean",
    "persistent": "persistent",
    "persistent_hue": "persistent",
    "abrupt": "abrupt",
    "abrupt_event": "abrupt",
}
EXPECTED_FRAMES = [[12 + 2 * i, 13 + 2 * i] for i in range(N_TUBELETS)]
EXPECTED_CLEAN_SF_MINUS_SS = -0.006734183640219271
EXPECTED_CLEAN_GS_MINUS_GF = 0.006907887756824493


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_check(results):
    with open(DATA, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    results["records_total"] = len(records)
    results["output_status_counts"] = dict(
        Counter(str(r.get("output_status")) for r in records)
    )
    kept = [r for r in records if r.get("output_status") == "complete"]
    results["records_dropped_non_complete"] = len(records) - len(kept)
    results["raw_appearance_values"] = dict(
        Counter(str(r.get("appearance_condition")) for r in kept)
    )
    results["raw_physical_values"] = dict(
        Counter(str(r.get("physical_condition")) for r in kept)
    )
    results["tubelets_not_18"] = sum(
        1 for r in kept if len(r["temporal_tubelet_l1"]) != N_TUBELETS
    )
    results["source_frames_not_expected"] = sum(
        1 for r in kept if r["target_tubelet_source_frames"] != EXPECTED_FRAMES
    )

    # (b) clip score equals mean of its tubelets
    max_abs = 0.0
    for r in kept:
        tub = np.asarray(r["temporal_tubelet_l1"], dtype=np.float64)
        max_abs = max(max_abs, float(abs(tub.mean() - float(r["score"]))))
    results["max_abs_mean_tubelets_minus_score"] = max_abs
    results["check_b_pass"] = bool(max_abs <= 1e-6)

    # Group clips by cell and scene.
    cells = {}
    duplicates = 0
    for r in kept:
        app = APPEAR_MAP[r["appearance_condition"]]
        key = (r["world"], r["trajectory"], app)
        d = cells.setdefault(key, {})
        if r["scene_id"] in d:
            duplicates += 1
        d[r["scene_id"]] = (
            np.asarray(r["temporal_tubelet_l1"], dtype=np.float64),
            float(r["score"]),
        )
    results["duplicate_cell_scene_entries"] = duplicates

    # (a) counts
    cell_counts = {
        "%s|%s|%s" % k: len(v) for k, v in sorted(cells.items())
    }
    results["complete_clips"] = len(kept)
    results["cell_counts"] = cell_counts
    results["check_a_pass"] = bool(
        len(kept) == 3072
        and len(cells) == 12
        and all(c == 256 for c in cell_counts.values())
    )

    scene_sets = [set(v.keys()) for v in cells.values()]
    all_scenes = sorted(set().union(*scene_sets))
    results["distinct_scenes"] = len(all_scenes)
    results["all_cells_same_scene_set"] = all(s == set(all_scenes) for s in scene_sets)

    tub = {}
    score = {}
    for key, d in cells.items():
        tub[key] = np.stack([d[s][0] for s in all_scenes])  # (n_scenes, 18)
        score[key] = np.array([d[s][1] for s in all_scenes])
    frames = [f[0] for f in EXPECTED_FRAMES]
    t = np.array(frames, dtype=np.float64) / FPS
    return kept, all_scenes, tub, score, t


def bootstrap_counts(n_scenes):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_scenes, size=(N_BOOT, n_scenes))
    counts = np.zeros((N_BOOT, n_scenes), dtype=np.float64)
    rows = np.repeat(np.arange(N_BOOT), n_scenes)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts


def main():
    results = {}
    kept, all_scenes, tub, score, t = load_and_check(results)
    n_scenes = len(all_scenes)
    results["time_axis_s"] = [round(float(x), 6) for x in t]

    counts = bootstrap_counts(n_scenes)
    mean = {}
    boot = {}
    band = {}
    for key, M in tub.items():
        mean[key] = M.mean(axis=0)
        b = (counts @ M) / n_scenes  # (N_BOOT, 18), scenes resampled together
        boot[key] = b
        band[key] = np.percentile(b, [2.5, 97.5], axis=0)

    diff_point = {}
    diff_band = {}
    for world in WORLDS:
        for app in APPS:
            d_pt = mean[(world, "stop", app)] - mean[(world, "fall", app)]
            d_bt = boot[(world, "stop", app)] - boot[(world, "fall", app)]
            diff_point[(world, app)] = d_pt
            diff_band[(world, app)] = np.percentile(d_bt, [2.5, 97.5], axis=0)

    # (c) scene-averaged clip score differences, clean appearance
    sf_ss = float(np.mean(score[("solid", "fall", "clean")] - score[("solid", "stop", "clean")]))
    gs_gf = float(np.mean(score[("gap", "stop", "clean")] - score[("gap", "fall", "clean")]))
    results["check_c_solid_fall_minus_solid_stop_clean"] = sf_ss
    results["check_c_expected_solid"] = EXPECTED_CLEAN_SF_MINUS_SS
    results["check_c_abs_error_solid"] = abs(sf_ss - EXPECTED_CLEAN_SF_MINUS_SS)
    results["check_c_gap_stop_minus_gap_fall_clean"] = gs_gf
    results["check_c_expected_gap"] = EXPECTED_CLEAN_GS_MINUS_GF
    results["check_c_abs_error_gap"] = abs(gs_gf - EXPECTED_CLEAN_GS_MINUS_GF)
    results["check_c_pass"] = bool(
        results["check_c_abs_error_solid"] <= 1e-9
        and results["check_c_abs_error_gap"] <= 1e-9
    )

    # (d) first tubelet where |solid stop minus fall| exceeds 0.003, clean panel
    d_solid_clean = diff_point[("solid", "clean")]
    over = np.where(np.abs(d_solid_clean) > THRESH)[0]
    first = int(over[0]) if over.size else None
    results["check_d_threshold"] = THRESH
    results["check_d_first_index_0based"] = first
    results["check_d_first_time_s"] = (
        round(float(t[first]), 6) if first is not None else None
    )
    results["check_d_first_source_frames"] = (
        EXPECTED_FRAMES[first] if first is not None else None
    )
    results["check_d_first_value"] = (
        float(d_solid_clean[first]) if first is not None else None
    )
    results["check_d_last_index"] = N_TUBELETS - 1
    results["check_d_last_value"] = float(d_solid_clean[-1])
    results["check_d_all_values_clean_solid"] = [
        round(float(x), 9) for x in d_solid_clean
    ]

    # ---- figure ----
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True, sharey="row")

    for j, app in enumerate(APPS):
        top = axes[0, j]
        bot = axes[1, j]
        for ax in (top, bot):
            ax.axvspan(WINDOW[0], WINDOW[1], facecolor="0.88", edgecolor="none", zorder=0)
            ax.set_xlim(t[0] - 0.05, t[-1] + 0.05)
            ax.set_xticks(np.arange(1.0, 3.6, 0.5))

        for world in WORLDS:
            ls = "-" if world == "solid" else "--"
            for traj in TRAJS:
                key = (world, traj, app)
                color = STOP_C if traj == "stop" else FALL_C
                lo, hi = band[key]
                top.fill_between(t, lo, hi, facecolor=color, alpha=0.2,
                                 edgecolor="none", zorder=2)
                top.plot(t, mean[key], color=color, linestyle=ls, lw=1.5, zorder=3)

        for world in WORLDS:
            if world == "solid":
                ls, color = "-", DIFF_SOLID_C
            else:
                ls, color = "--", DIFF_GAP_C
            lo, hi = diff_band[(world, app)]
            bot.fill_between(t, lo, hi, facecolor=color, alpha=0.18,
                             edgecolor="none", zorder=2)
            bot.plot(t, diff_point[(world, app)], color=color, linestyle=ls,
                     lw=1.5, zorder=3)
        bot.axhline(0.0, color="0.3", lw=0.7, zorder=1)

        top.set_title("%s appearance" % app)
        bot.set_xlabel("time of target tubelet (s)")

    # Shared y limits within each row, with headroom in the top row for the label.
    top_lo = min(float(band[k][0].min()) for k in band if k[0] in WORLDS)
    top_hi = max(float(band[k][1].max()) for k in band)
    pad = 0.04 * (top_hi - top_lo)
    axes[0, 0].set_ylim(top_lo - pad, top_hi + 0.18 * (top_hi - top_lo) + pad)
    bot_lo = min(float(v[0].min()) for v in diff_band.values())
    bot_hi = max(float(v[1].max()) for v in diff_band.values())
    bot_lo = min(bot_lo, 0.0)
    bot_hi = max(bot_hi, 0.0)
    bpad = 0.06 * (bot_hi - bot_lo)
    axes[1, 0].set_ylim(bot_lo - bpad, bot_hi + bpad)

    axes[0, 0].set_ylabel("native error per tubelet")
    axes[1, 0].set_ylabel("stop minus fall")
    axes[0, 0].text(
        1.03, 0.97, "branches diverge (frames 12 to 24)",
        transform=axes[0, 0].get_xaxis_transform(),
        ha="left", va="top", fontsize=8, color="0.2", zorder=5,
    )

    handles = [
        Line2D([0], [0], color=STOP_C, lw=1.5, ls="-", label="stop, solid support"),
        Line2D([0], [0], color=STOP_C, lw=1.5, ls="--", label="stop, gap"),
        Line2D([0], [0], color=FALL_C, lw=1.5, ls="-", label="fall, solid support"),
        Line2D([0], [0], color=FALL_C, lw=1.5, ls="--", label="fall, gap"),
        Line2D([0], [0], color=DIFF_SOLID_C, lw=1.5, ls="-",
               label="stop minus fall, solid support"),
        Line2D([0], [0], color=DIFF_GAP_C, lw=1.5, ls="--",
               label="stop minus fall, gap"),
        Patch(facecolor="0.5", edgecolor="none", alpha=0.3,
              label="95% scene bootstrap band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=200, facecolor="white",
                metadata={"Software": "matplotlib"})
    fig.savefig(PDF_PATH, facecolor="white",
                metadata={"Creator": "fig2_tubelet.py", "CreationDate": None})
    plt.close(fig)

    results["bootstrap"] = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "method": "scene resampling with replacement, clips of a scene kept together, percentile 2.5 and 97.5",
        "n_scenes": n_scenes,
    }

    prov = {
        "script": {"path": SCRIPT, "sha256": sha256_file(SCRIPT)},
        "command": COMMAND,
        "python": sys.version.split()[0],
        "python_executable_platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": {"numpy_default_rng_bootstrap": SEED},
        "input": {
            "path": DATA,
            "sha256": sha256_file(DATA),
            "records": results["records_total"],
        },
        "outputs": {
            "png": {"path": PNG_PATH, "sha256": sha256_file(PNG_PATH),
                    "bytes": os.path.getsize(PNG_PATH), "dpi": 200},
            "pdf": {"path": PDF_PATH, "sha256": sha256_file(PDF_PATH),
                    "bytes": os.path.getsize(PDF_PATH)},
        },
        "notes": [
            "appearance_condition values in the data are persistent_hue and abrupt_event; mapped to persistent and abrupt for labels.",
            "physical_condition is fixed by world and trajectory (solid-stop and gap-stop differ), so it is not a separate factor in the cells.",
            "No data values were altered; all acceptance checks are reported as computed.",
        ],
        "results": results,
    }
    with open(PROV_PATH, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, ensure_ascii=True)
        fh.write("\n")

    print(json.dumps(results, indent=2, ensure_ascii=True))
    print("PNG", PNG_PATH, prov["outputs"]["png"]["sha256"])
    print("PDF", PDF_PATH, prov["outputs"]["pdf"]["sha256"])
    print("PROVENANCE", PROV_PATH)


if __name__ == "__main__":
    main()
