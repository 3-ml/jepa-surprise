"""A1: secondary paired reanalysis of the stored 169-pair IntPhys2 appearance matrix.

Reads stored scores only (no inference). Exploratory, not confirmatory evidence.
Usage: python scripts/wmpai2026/alex/a1_intphys2_paired_margins.py --out <dir>
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SRC = Path("results/21-08-30-intphys2-main-one-third-endpoints-vjepa2-g")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    man = json.load(open(SRC / "selection-manifest.json"))
    scene = {p["pair_id"]: p["SceneIndex"] for p in man["pairs"]}
    rows = list(csv.DictReader(open(SRC / "pair_surprises.csv")))
    for r in rows:
        r["scene"] = scene[r["pair"]]
        r["poss"] = float(r["possible_surprise"])
        r["imp"] = float(r["impossible_surprise"])
        r["m"] = r["imp"] - r["poss"]

    with open(out / "per_pair_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "scene_index", "condition", "possible", "impossible", "margin", "correct"])
        for r in rows:
            w.writerow([r["pair"], r["scene"], r["condition"], r["poss"], r["imp"], r["m"], int(r["m"] > 0)])

    clean = {r["pair"]: r for r in rows if r["condition"] == "clean"}
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != "clean", c))
    rng = random.Random(a.seed)
    scenes = sorted({r["scene"] for r in rows})
    by_cs = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cs[r["condition"]][r["scene"]].append(r)

    def stats(cs, picked):
        ms, acc, shared, n_sh = [], [], 0, 0
        for s in picked:
            for r in cs.get(s, []):
                c = clean[r["pair"]]
                ms.append(r["m"])
                acc.append(r["m"] > 0)  # ties (m==0) count as incorrect
                if r["condition"] != "clean":
                    dp, di = r["poss"] - c["poss"], r["imp"] - c["imp"]
                    n_sh += 1
                    shared += (dp > 0) == (di > 0)
        n = len(ms)
        return (sum(ms) / n, sum(acc) / n, shared / n_sh if n_sh else float("nan"))

    res = []
    for c in conds:
        pt = stats(by_cs[c], scenes)
        boots = [stats(by_cs[c], [rng.choice(scenes) for _ in scenes]) for _ in range(a.boot)]
        ci = lambda i: tuple(sorted(b[i] for b in boots)[int(q * a.boot)] for q in (0.025, 0.975 - 1e-9))
        res.append({"condition": c, "n_pairs": sum(len(v) for v in by_cs[c].values()),
                    "mean_margin": pt[0], "margin_ci95_scene_boot": ci(0),
                    "pair_ordering_acc": pt[1], "acc_ci95_scene_boot": ci(1),
                    "frac_shift_same_sign_both_members": pt[2]})
    json.dump({"scene_resample_units": len(scenes), "bootstrap": a.boot, "seed": a.seed,
               "tie_rule": "margin == 0 counts as incorrect",
               "status": "OBSERVED, exploratory secondary; scene_index not independently verified as true scene id",
               "source_hashes": {str(SRC / n): sha(SRC / n) for n in ("pair_surprises.csv", "selection-manifest.json", "summary.json")},
               "results": res}, open(out / "a1_summary.json", "w"), indent=1)
    for r in res:
        print(r["condition"], f'm={r["mean_margin"]:+.5f} acc={r["pair_ordering_acc"]:.3f} same_sign={r["frac_shift_same_sign_both_members"]:.3f}')


if __name__ == "__main__":
    main()
