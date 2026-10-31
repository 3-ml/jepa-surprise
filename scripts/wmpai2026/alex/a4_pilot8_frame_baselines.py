'A4 on the eight-scene Kubric pilot frames (EXPLORATORY): do trivial frame statistics predict the\nnative score change, and does the physical flag add anything beyond them on held-out scenes?\n\nFrames: ${JEPA_PILOT8_FRAMES}\\<scene>\\<condition>\\*.png (48 frames, 12 context).\nUsage: python a4_pilot8_frame_baselines.py --out <dir>\n'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lib", HERE / "a3_a4_lib.py")
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)

FR = Path(__import__("os").environ['JEPA_PILOT8_FRAMES'])
P = Path("results/23-05-00-vjepa2g-kubric-construct-validity-pilot8")


def load_clip(d):
    files = sorted(d.glob("*.png"))
    assert len(files) == 48, (d, len(files))
    return np.stack([cv2.resize(cv2.imread(str(f), cv2.IMREAD_COLOR), (64, 64), interpolation=cv2.INTER_AREA)
                     for f in files]).astype(np.float32) / 255, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=5000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(P / "model/scores.csv")
    sc = sc[sc.condition.isin(["clean"] + [f"hue-{h}" for h in ("030", "060", "120", "180")] + [f"physical-z-{z}" for z in ("0p25", "0p5", "1p0", "1p5")])]
    rows, hashes = [], {}
    for (scene, cond), g in sc.groupby(["scene_id", "condition"]):
        clip, files = load_clip(FR / scene / cond / "frames")
        ctx_last_equal = None
        f = lib.frame_features(clip, 12, np.random.default_rng(0))
        hashes[f"{scene}/{cond}"] = hashlib.sha256(b"".join(open(p, "rb").read() for p in files)).hexdigest()
        rows.append({"scene_id": scene, "condition": cond, **f})
    ft = pd.DataFrame(rows)
    clean_f = ft[ft.condition == "clean"].set_index("scene_id")
    ft = ft.merge(sc[["scene_id", "condition", "family", "scalar_l1"]], on=["scene_id", "condition"])
    clean_s = sc[sc.condition == "clean"].set_index("scene_id").scalar_l1
    d = ft[ft.condition != "clean"].copy()
    d["delta"] = d.scalar_l1.values - clean_s.loc[d.scene_id].values
    for k in ("static", "motion", "shuffled"):
        d[f"d_{k}"] = d[k].values - clean_f.loc[d.scene_id, k].values   # feature change vs the scene's clean clip
    d["is_physical"] = (d.family == "physical").astype(float)
    d.to_csv(out / "frame_features_vs_score.csv", index=False)
    y, scene, cond = d.delta.values, d.scene_id.values, d.condition.values
    res = {}
    for k in ("static", "motion", "shuffled"):
        x = d[f"d_{k}"].values
        obs, nm, p = lib.permutation_null_loso(x, y, scene, cond, a.perm)
        b, f, g = lib.nested_gain(x, d.is_physical.values, y, scene)
        res[k] = {"loso_r2": obs, "perm_null_mean": nm, "perm_p": p, "spearman_pooled": lib.spearman(x, y),
                  "base_r2": b, "with_physical_flag_r2": f, "flag_gain": g}
    json.dump({"status": "EXPLORATORY, 8 scenes, 64 scene x condition rows", "n_rows": len(d),
               "feature_definitions": "d_* = feature of the intervened clip minus feature of the same scene's clean clip; "
                                      "static/motion/shuffled from a3_a4_lib.frame_features, 64x64 colour, context=12 frames",
               "clip_sha256": hashes, "results": res}, open(out / "summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, {a_: round(b_, 3) for a_, b_ in v.items()})


if __name__ == "__main__":
    main()
