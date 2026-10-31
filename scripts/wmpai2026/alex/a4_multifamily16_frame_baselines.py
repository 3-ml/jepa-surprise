'A4 on the archived 16-scene Kubric study (EXPLORATORY): do trivial frame statistics predict the native score\nchange, and does a physical-edit flag add anything beyond them on held-out scenes?\n\nSame method as a4_pilot8_frame_baselines.py. Frames: ${JEPA_MULTIFAMILY16_FRAMES}\\<scene>\\<cond>\\frames\\*.png\n(48 frames, 12 context). Outcome: the archived native response from the clean clip. Only the 10 rendered conditions\nper scene have frames; the in-memory appearance conditions of the archive are not covered.\n\nUsage: python a4_multifamily16_frame_baselines.py --out <dir>\n'
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import a3_a4_lib as lib

FR = Path(__import__("os").environ['JEPA_MULTIFAMILY16_FRAMES'])
RESP = Path(__import__("os").path.join(__import__("os").environ['JEPA_MULTIFAMILY16_RECORD'], 'run/analysis/per_scene_responses.csv'))


def load_clip(d):
    files = sorted(d.glob("*.png"))
    assert len(files) == 48, (d, len(files))
    clip = np.stack([cv2.resize(cv2.imread(str(f), cv2.IMREAD_COLOR), (64, 64), interpolation=cv2.INTER_AREA)
                     for f in files]).astype(np.float32) / 255
    return clip, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resp = pd.read_csv(RESP)
    resp = resp[resp.condition.map(lambda c: (FR / "seed-231601" / c / "frames").is_dir())]
    rows, hashes = [], {}
    for (scene, cond), _ in resp.groupby(["scene_id", "condition"]):
        clip, files = load_clip(FR / scene / cond / "frames")
        rows.append({"scene_id": scene, "condition": cond, **lib.frame_features(clip, 12, np.random.default_rng(0))})
        hashes[f"{scene}/{cond}"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    ft = pd.DataFrame(rows)
    clean = ft[ft.condition == "clean"].set_index("scene_id")
    d = ft[ft.condition != "clean"].merge(resp[["scene_id", "condition", "family", "response_from_clean",
                                                 "full_frame_rgb_l1_01"]], on=["scene_id", "condition"])
    d["delta"] = d.response_from_clean
    for k in ("static", "motion", "shuffled"):
        d[f"d_{k}"] = d[k].values - clean.loc[d.scene_id, k].values
    d["is_physical"] = d.family.str.startswith("physical").astype(float)
    d.to_csv(out / "frame_features_vs_response.csv", index=False)
    y, scene, cond = d.delta.values, d.scene_id.values, d.condition.values
    res = {}
    for k in ("static", "motion", "shuffled"):
        x = d[f"d_{k}"].values
        obs, nm, p = lib.permutation_null_loso(x, y, scene, cond, a.perm)
        b, f, g = lib.nested_gain(x, d.is_physical.values, y, scene)
        res[k] = {"loso_r2": obs, "perm_null_mean": nm, "perm_p": p, "spearman_pooled": lib.spearman(x, y),
                  "base_r2": b, "with_physical_flag_r2": f, "flag_gain": g}
    x = d.full_frame_rgb_l1_01.values
    obs, nm, p = lib.permutation_null_loso(x, y, scene, cond, a.perm)
    b, f, g = lib.nested_gain(x, d.is_physical.values, y, scene)
    res["archived_rgb_l1"] = {"loso_r2": obs, "perm_null_mean": nm, "perm_p": p, "spearman_pooled": lib.spearman(x, y),
                              "base_r2": b, "with_physical_flag_r2": f, "flag_gain": g}
    json.dump({"status": f"EXPLORATORY, {d.scene_id.nunique()} scenes, {len(d)} scene x condition rows",
               "n_rows": len(d), "clip_sha256": hashes, "results": res}, open(out / "summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {a_: round(b_, 3) for a_, b_ in v.items()})


if __name__ == "__main__":
    main()
