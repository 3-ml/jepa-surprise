"""A2 power/precision study for the number of confirmatory scenes.

Input: per-scene margins M(scene, a) from DEVELOPMENT scenes only (a2_scene_stats.margins).
Method: parametric scene-level simulation. Draw N scene margins from a normal with the
development mean/SD (optionally shrunk toward the declared smallest meaningful effect), compute the
scene-bootstrap 95% interval, repeat, and report the expected half-width and the probability the
lower bound clears 0 and the declared margin. The pilot-size SD is itself uncertain, so a
sensitivity grid over SD is always reported.

Usage: python a2_power_study.py --dev-margins dev_margins.csv --effect 0.0005 --target-half-width 0.0003 --out dir
The effect and target half-width are inputs the authors declare jointly; none is chosen here.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def simulate(n, mu, sd, n_sim=500, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    half, lo = [], []
    for _ in range(n_sim):
        m = rng.normal(mu, sd, n)
        idx = rng.integers(0, n, (n_boot, n))
        means = m[idx].mean(1)
        l, h = np.percentile(means, [2.5, 97.5])
        half.append((h - l) / 2)
        lo.append(l)
    return np.array(half), np.array(lo)


def study(mu, sd, sizes, effect, target, sd_mult=(0.75, 1.0, 1.5), **kw):
    rows = []
    for k in sd_mult:
        for n in sizes:
            half, lo = simulate(n, effect, sd * k, **kw)
            rows.append({"sd_multiplier": k, "sd": sd * k, "n_scenes": n,
                         "expected_half_width": float(half.mean()),
                         "p_half_width_below_target": float((half <= target).mean()),
                         "p_lower_bound_above_zero": float((lo > 0).mean())})
    return rows


def smallest_n(rows, k, assurance=0.8):
    ok = [r["n_scenes"] for r in rows if r["sd_multiplier"] == k and r["p_half_width_below_target"] >= assurance]
    return min(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-margins", required=True, help="csv with columns scene_id,margin")
    ap.add_argument("--effect", type=float, required=True, help="declared smallest meaningful mean margin")
    ap.add_argument("--target-half-width", type=float, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--assurance", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = np.array([float(r["margin"]) for r in csv.DictReader(open(a.dev_margins))])
    rows = study(m.mean(), m.std(ddof=1), a.sizes, a.effect, a.target_half_width)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"n_dev_scenes": len(m), "dev_mean": float(m.mean()), "dev_sd": float(m.std(ddof=1)),
               "effect": a.effect, "target_half_width": a.target_half_width, "assurance": a.assurance,
               "smallest_n_by_sd_multiplier": {str(k): smallest_n(rows, k, a.assurance) for k in (0.75, 1.0, 1.5)},
               "rows": rows,
               "note": "Planning aid from development variance; the chosen N must be frozen before confirmatory scoring."}
    json.dump(summary, open(out / "power_study.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
