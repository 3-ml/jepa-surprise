"""Shared loading and statistics for the kubric figure scripts.

The score file is parsed with the json module as data only. Nothing from it is executed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
DATA = Path(__import__("os").environ.get("JEPA_FINAL256_SCORES", str(Path(__file__).resolve().parents[5] / "release/final256/original/scores/all-scores.json")))
FIGS = WORK / "figs"

T_STAR = 0.4858863055706024
BOOT_SEED = 20261031
N_BOOT = 10000
SUBSET_SEED = 1
N_SUBSET = 60
JITTER_SEED = 3

COLOR_STOP = "#D55E00"
COLOR_FALL = "#0072B2"
COLOR_LAWFUL = "#009E73"
COLOR_INCONS = "#CC79A7"

APPS = ["clean", "persistent", "abrupt"]
# Labels as stored in the file; mapped to the short names used in the brief.
APP_MAP = {
    "clean": "clean",
    "persistent_hue": "persistent",
    "persistent": "persistent",
    "abrupt_event": "abrupt",
    "abrupt": "abrupt",
}
LAWFUL_PAIRS = {("solid", "stop"), ("gap", "fall")}

MARGIN_EXPECTED = {
    ("clean", "solid"): -0.006734183640219271,
    ("clean", "gap"): 0.006907887756824493,
    ("persistent", "solid"): -0.007839271682314575,
    ("persistent", "gap"): 0.007495353929698467,
}
# (false alarm %, miss %)
RATE_EXPECTED = {
    ("clean", "solid"): (100.00, 45.70),
    ("clean", "gap"): (37.89, 0.78),
    ("persistent", "solid"): (100.00, 59.38),
    ("persistent", "gap"): (30.47, 0.39),
    ("abrupt", "solid"): (100.00, 65.62),
    ("abrupt", "gap"): (28.52, 0.00),
}
AUROC_EXPECTED = {
    ("clean", "solid"): 0.083,
    ("clean", "gap"): 0.882,
    ("persistent", "solid"): 0.048,
    ("persistent", "gap"): 0.944,
    ("abrupt", "solid"): 0.042,
    ("abrupt", "gap"): 0.948,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    """Return (S, scenes, info). S maps (world, trajectory, appearance) to a 256-vector
    of scores aligned with the sorted scene list."""
    with open(DATA, encoding="utf-8") as fh:
        records = json.load(fh)
    kept = [r for r in records if r.get("output_status") == "complete"]
    scenes = sorted({r["scene_id"] for r in kept})
    pos = {s: i for i, s in enumerate(scenes)}
    n = len(scenes)
    S = {}
    for r in kept:
        app = APP_MAP[r["appearance_condition"]]
        world, traj = r["world"], r["trajectory"]
        lawful_pair = (world, traj) in LAWFUL_PAIRS
        if (r["physical_condition"] == "lawful") != lawful_pair:
            raise ValueError(f"lawful flag inconsistent with cell for {r['clip_id']}")
        key = (world, traj, app)
        arr = S.setdefault(key, np.full(n, np.nan))
        i = pos[r["scene_id"]]
        if not np.isnan(arr[i]):
            raise ValueError(f"duplicate cell {key} for scene {r['scene_id']}")
        arr[i] = float(r["score"])
    if len(S) != 12 or any(np.isnan(v).any() for v in S.values()):
        raise ValueError("expected 12 complete cells with no missing scenes")
    info = {
        "records_total": len(records),
        "records_dropped_not_complete": len(records) - len(kept),
        "records_used": len(kept),
        "scenes": n,
        "input_sha256": sha256(DATA),
    }
    return S, scenes, info


def lawful_and_inconsistent(S, world, app):
    """Return (lawful, inconsistent) score vectors for a world and appearance."""
    if world == "solid":
        return S[("solid", "stop", app)], S[("solid", "fall", app)]
    return S[("gap", "fall", app)], S[("gap", "stop", app)]


def margin_vector(S, world, app):
    lawful, inc = lawful_and_inconsistent(S, world, app)
    return inc - lawful


def margin(S, world, app):
    return float(margin_vector(S, world, app).mean())


def rates(lawful, inc, t=T_STAR):
    """Return (false alarm %, miss %) with score >= t flagged as inconsistent."""
    fa = 100.0 * float(np.mean(lawful >= t))
    miss = 100.0 * float(np.mean(inc < t))
    return fa, miss


def auroc(pos, neg):
    """Mann-Whitney AUROC with the inconsistent class as positive; ties count 0.5."""
    gt = float((pos[:, None] > neg[None, :]).mean())
    eq = float((pos[:, None] == neg[None, :]).mean())
    return gt + 0.5 * eq


def boot_indices(n):
    """Scene-level bootstrap draws; every row resamples whole scenes (all 12 clips)."""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def boot_ci(vec, idx):
    """Mean and 95 percent percentile interval of a per-scene vector."""
    boot = vec[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vec.mean()), float(lo), float(hi)


def run_checks(S):
    """Return a list of acceptance-check rows comparing computed values to the brief."""
    rows = []
    for (app, world), expected in MARGIN_EXPECTED.items():
        computed = margin(S, world, app)
        rows.append(dict(
            check=f"margin {app} {world}", computed=computed, expected=expected,
            tol=1e-9, ok=abs(computed - expected) <= 1e-9))
    for (app, world), (e_fa, e_miss) in RATE_EXPECTED.items():
        lawful, inc = lawful_and_inconsistent(S, world, app)
        fa, miss = rates(lawful, inc)
        rows.append(dict(
            check=f"false alarm % {app} {world}", computed=fa, expected=e_fa,
            tol=0.01, ok=abs(fa - e_fa) <= 0.01))
        rows.append(dict(
            check=f"miss % {app} {world}", computed=miss, expected=e_miss,
            tol=0.01, ok=abs(miss - e_miss) <= 0.01))
    for (app, world), expected in AUROC_EXPECTED.items():
        lawful, inc = lawful_and_inconsistent(S, world, app)
        value = auroc(inc, lawful)
        rows.append(dict(
            check=f"AUROC {app} {world}", computed=value, expected=expected,
            tol=0.0005, ok=abs(value - expected) <= 0.0005))
    # Identity: lawful mean minus inconsistent mean equals minus the margin, per world.
    for app in APPS:
        for world in ("solid", "gap"):
            lawful, inc = lawful_and_inconsistent(S, world, app)
            computed = float(lawful.mean() - inc.mean())
            expected = -margin(S, world, app)
            rows.append(dict(
                check=f"identity lawful-inc {app} {world}", computed=computed,
                expected=expected, tol=1e-9, ok=abs(computed - expected) <= 1e-9))
    return rows


def group_means(S):
    """Per appearance: mean score of the lawful and inconsistent clips at each world."""
    out = {}
    for app in APPS:
        out[app] = {}
        for world in ("solid", "gap"):
            lawful, inc = lawful_and_inconsistent(S, world, app)
            out[app][world] = {"lawful_mean": float(lawful.mean()),
                               "inconsistent_mean": float(inc.mean())}
    return out


def format_table(rows):
    lines = [f"{'check':<28}{'computed':>22}{'expected':>22}{'tol':>8}  pass"]
    for r in rows:
        lines.append(
            f"{r['check']:<28}{r['computed']:>22.12f}{r['expected']:>22.12f}"
            f"{r['tol']:>8g}  {'yes' if r['ok'] else 'NO'}")
    return "\n".join(lines)
