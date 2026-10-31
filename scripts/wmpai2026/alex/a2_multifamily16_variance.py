"""A2 planning input from the archived 16-scene Kubric study (EXPLORATORY, not confirmatory).

Reads the archived per-scene responses and matched comparisons, recomputes the physical-minus-
appearance contrast per scene, checks it against the means the archive reports, and then reports
scene-level SD and the expected scene-bootstrap 95% half-width for candidate scene counts.

The contrast is an RGB-L1-matched direct-edit comparison, NOT the lawful-versus-inconsistent margin
the primary test will use. It is a rough variance proxy only; the real SD comes from Vasi's development
scenes. No effect size or target half-width is chosen here.

Usage: python a2_multifamily16_variance.py --record <dir with analysis/> --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from a2_power_study import simulate

SIZES = (16, 32, 48, 64, 96, 128, 192, 256)


def scene_contrasts(responses, matches):
    r = responses.set_index(["scene_id", "condition"]).response_from_clean
    rows = []
    for _, m in matches.iterrows():
        for scene in sorted(responses.scene_id.unique()):
            rows.append({"scene_id": scene, "physical": m.physical_condition, "appearance": m.appearance_condition,
                         "contrast": float(r[(scene, m.physical_condition)] - r[(scene, m.appearance_condition)])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = Path(a.record)
    responses = pd.read_csv(rec / "analysis" / "per_scene_responses.csv")
    matches = pd.read_csv(rec / "analysis" / "matched_comparisons.csv")
    sc = scene_contrasts(responses, matches)
    check = []
    for _, m in matches.iterrows():
        mine = sc[(sc.physical == m.physical_condition) & (sc.appearance == m.appearance_condition)].contrast
        check.append({"physical": m.physical_condition, "appearance": m.appearance_condition,
                      "recomputed_mean": float(mine.mean()), "archived_mean": float(m.mean_physical_minus_appearance_response),
                      "abs_diff": float(abs(mine.mean() - m.mean_physical_minus_appearance_response)),
                      "scene_sd": float(mine.std(ddof=1)), "n_scenes": int(len(mine)),
                      "wins": int((mine > 0).sum())})
    per_scene_mean = sc.groupby("scene_id").contrast.mean()
    sds = {"pooled_across_pairs_scene_mean": float(per_scene_mean.std(ddof=1)),
           "median_pair_sd": float(np.median([c["scene_sd"] for c in check])),
           "min_pair_sd": float(min(c["scene_sd"] for c in check)),
           "max_pair_sd": float(max(c["scene_sd"] for c in check))}
    grid = []
    for label in ("min_pair_sd", "median_pair_sd", "pooled_across_pairs_scene_mean", "max_pair_sd"):
        for n in SIZES:
            half, _ = simulate(n, 0.0, sds[label], n_sim=400, n_boot=300, seed=1)
            grid.append({"sd_label": label, "sd": sds[label], "n_scenes": n, "expected_half_width": float(half.mean())})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc.to_csv(out / "scene_contrasts.csv", index=False)
    json.dump({"status": "EXPLORATORY variance proxy, 16 scenes, RGB-L1-matched direct-edit contrasts",
               "n_scenes": int(responses.scene_id.nunique()), "reproduction_check": check,
               "max_abs_diff_vs_archive": max(c["abs_diff"] for c in check), "scene_sd": sds,
               "expected_half_width_grid": grid,
               "note": "Half-width is for the mean of N scene-level contrasts with the stated SD; it does not choose N."},
              open(out / "variance_and_precision.json", "w"), indent=1)
    print("max |recomputed - archived|:", max(c["abs_diff"] for c in check))
    print("scene SD:", {k: round(v, 5) for k, v in sds.items()})
    for label in ("median_pair_sd",):
        print(label, [(g["n_scenes"], round(g["expected_half_width"], 5)) for g in grid if g["sd_label"] == label])


if __name__ == "__main__":
    main()
