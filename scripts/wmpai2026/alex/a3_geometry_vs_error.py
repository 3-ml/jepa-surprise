"""A3 on archived exploratory scenes: does feature displacement from the clean clip predict the native score change?

EXPLORATORY: 8 or 16 scenes, direct edits (not lawful-versus-inconsistent pairs). It exercises the A3 analysis path
on real V-JEPA2-G features; it is not a held-out result and says nothing about causal mediation.

For each layer: relative displacement = ||f(clip) - f(clean clip of the same scene)|| / ||f(clean clip)||.
Scene-held-out linear fit of the native score change on that displacement, with a permutation null that shuffles
displacement across scenes within each condition. Nested comparison: gain from the displacement over a physical-edit
flag (and, where stored, over the rendered RGB change).

Usage: python a3_geometry_vs_error.py --features <dir> --source pilot8|multifamily16 --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import a3_a4_lib as lib

PILOT_SCORES = Path(__file__).resolve().parents[3] / "results/23-05-00-vjepa2g-kubric-construct-validity-pilot8/model/scores.csv"
MF_RESPONSES = Path(__import__("os").path.join(__import__("os").environ['JEPA_MULTIFAMILY16_RECORD'], 'run/analysis/per_scene_responses.csv'))


def responses(source):
    if source == "pilot8":
        s = pd.read_csv(PILOT_SCORES)
        clean = s[s.condition == "clean"].set_index("scene_id").scalar_l1
        s["delta"] = s.scalar_l1 - clean.loc[s.scene_id].values
        s["rgb_effect"] = np.nan
        s["physical"] = (s.family == "physical").astype(float)
    else:
        s = pd.read_csv(MF_RESPONSES)
        s["delta"] = s.response_from_clean
        s["rgb_effect"] = s.full_frame_rgb_l1_01
        s["physical"] = s.family.str.startswith("physical").astype(float)
    return s[["scene_id", "condition", "family", "delta", "rgb_effect", "physical"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--source", choices=("pilot8", "multifamily16"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=5000)
    a = ap.parse_args()
    fdir = Path(a.features)
    meta = json.load(open(fdir / "feature_meta.json"))
    X = np.load(fdir / "features.npy")
    idx = pd.read_csv(fdir / "feature_index.csv")
    parts = idx.clip_id.str.split("/", expand=True)
    idx["scene_id"], idx["condition"] = parts[0], parts[1]
    assert not idx.missing.any(), "missing feature rows"
    key = {(r.scene_id, r.condition): i for i, r in idx.iterrows()}
    resp = responses(a.source)
    rows = resp[(resp.condition != "clean") & resp.apply(lambda r: (r.scene_id, r.condition) in key, axis=1)].copy()
    rows = rows.reset_index(drop=True)
    layers = meta["layers"]
    for li, layer in enumerate(layers):
        disp = []
        for r in rows.itertuples():
            f, c = X[key[(r.scene_id, r.condition)], li], X[key[(r.scene_id, "clean")], li]
            disp.append(np.linalg.norm(f - c) / np.linalg.norm(c))
        rows[f"disp_{layer}"] = disp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "displacement_vs_response.csv", index=False)
    y, scene, cond = rows.delta.values, rows.scene_id.values, rows.condition.values
    res = {}
    for layer in layers:
        x = rows[f"disp_{layer}"].values
        obs, null_mean, p = lib.permutation_null_loso(x, y, scene, cond, a.perm)
        b, f, g = lib.nested_gain(x, rows.physical.values, y, scene)
        entry = {"loso_r2": obs, "perm_null_mean": null_mean, "perm_p": p,
                 "spearman_pooled": lib.spearman(x, y), "flag_only_r2": lib.loso_r2(rows.physical.values, y, scene),
                 "displacement_plus_flag_r2": f,
                 "mean_displacement_physical": float(x[rows.physical.values == 1].mean()),
                 "mean_displacement_appearance": float(x[rows.physical.values == 0].mean())}
        if rows.rgb_effect.notna().all():
            base, full, gain = lib.nested_gain(rows.rgb_effect.values, x, y, scene)
            entry.update({"rgb_effect_only_r2": base, "rgb_effect_plus_displacement_r2": full,
                          "gain_from_displacement_over_rgb": gain})
        res[layer] = entry
    json.dump({"status": f"EXPLORATORY, {a.source}, {rows.scene_id.nunique()} scenes, {len(rows)} scene x condition rows",
               "layers": layers, "n_rows": len(rows), "features_dir": str(fdir), "results": res},
              open(out / "summary.json", "w"), indent=1)
    for layer, e in res.items():
        print(layer, {k: round(v, 4) for k, v in e.items()})


if __name__ == "__main__":
    main()
