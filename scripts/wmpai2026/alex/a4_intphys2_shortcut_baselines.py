"""A4: shortcut baselines on the inspected 169-pair IntPhys2 Main endpoint set (EXPLORATORY).

Question: how well do trivial video statistics rank the impossible member above the possible member
of a matched pair, compared with V-JEPA2-G's native score on the same pairs (clean pair-ordering
accuracy 0.633)? Direction is fixed before running: a higher feature value means "impossible".
No learning, so no train/test leakage; scene-clustered bootstrap over SceneIndex.

Features per video (grayscale, 64x64, all frames): n_frames, motion (mean |frame diff|),
static (mean |frame - first frame|), shuffled (motion after random frame permutation).
Usage: python a4_intphys2_shortcut_baselines.py --out <dir> --tmp <scratch dir>
"""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

SRC = Path("results/21-08-30-intphys2-main-one-third-endpoints-vjepa2-g")
ZIP = Path(__import__("os").environ['JEPA_INTPHYS2_ZIP'])
FEATS = ["n_frames", "motion", "static", "shuffled"]


def features(path, rng):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    cap.release()
    a = np.stack(fr)
    return {"n_frames": float(len(a)), "motion": float(np.abs(np.diff(a, axis=0)).mean()),
            "static": float(np.abs(a - a[0]).mean()),
            "shuffled": float(np.abs(np.diff(a[rng.permutation(len(a))], axis=0)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    out, tmp = Path(a.out), Path(a.tmp)
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    man = json.load(open(SRC / "selection-manifest.json"))
    meta = {}
    for r in csv.DictReader(open(SRC / "pair_surprises.csv")):
        if r["condition"] == "clean":
            meta[r["pair"]] = r
    rng = np.random.default_rng(0)
    z = zipfile.ZipFile(ZIP)
    rows = []
    for p in man["pairs"]:
        # which of the two names is the impossible one comes from the stored metadata json
        vids = {}
        for nm in p["names"]:
            info = json.load(open(__import__("os").path.join(__import__("os").environ['JEPA_INTPHYS2_ENDPOINT'], 'extracted/intphys2-appearance-completion/main/clean/videos/%s.json') % nm))
            typ = info["metadata"]["type"]
            f = tmp / (nm + ".mp4")
            f.write_bytes(z.read(f"Main/Videos/{nm}.mp4"))
            vids[typ] = features(f, rng)
            vids[typ]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            f.unlink()
        imp = next(v for k, v in vids.items() if "Impossible" in k)
        pos = next(v for k, v in vids.items() if "Impossible" not in k)
        rows.append({"pair": p["pair_id"], "scene_index": p["SceneIndex"],
                     **{f"imp_{k}": imp[k] for k in FEATS}, **{f"pos_{k}": pos[k] for k in FEATS},
                     "imp_sha256": imp["sha256"], "pos_sha256": pos["sha256"]})
    with open(out / "per_pair_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    scenes = sorted({r["scene_index"] for r in rows})
    by = {s: [r for r in rows if r["scene_index"] == s] for s in scenes}
    brng = np.random.default_rng(20261024)
    res = {}
    for k in FEATS:
        def acc(sel):
            rr = [r for s in sel for r in by[s]]
            return float(np.mean([(r[f"imp_{k}"] > r[f"pos_{k}"]) for r in rr]))
        ties = int(sum(r[f"imp_{k}"] == r[f"pos_{k}"] for r in rows))
        boots = [acc([scenes[i] for i in brng.integers(0, len(scenes), len(scenes))]) for _ in range(a.boot)]
        res[k] = {"pair_ordering_acc": acc(scenes), "ci95_scene_boot": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "ties": ties}
    json.dump({"status": "OBSERVED, exploratory, inspected data", "n_pairs": len(rows), "n_scenes": len(scenes),
               "direction": "higher feature => impossible (fixed before running)", "boot": a.boot,
               "reference_vjepa2g_clean_acc": 0.633, "results": res}, open(out / "summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k, round(v["pair_ordering_acc"], 3), [round(x, 3) for x in v["ci95_scene_boot"]], "ties", v["ties"])


if __name__ == "__main__":
    main()
