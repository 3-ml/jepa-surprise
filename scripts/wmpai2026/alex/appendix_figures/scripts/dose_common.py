"""Shared loading and statistics for the exploratory dose figures (fig5, fig7).

Read-only with respect to the cache. Inputs are loaded with allow_pickle=False.
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path(__import__("os").environ['JEPA_MULTIDOSE_CACHE'])
SCRIPTS = Path(__file__).resolve().parent
WORK = SCRIPTS.parent
FIGS = WORK / "figs"
NPZ = CACHE / "layer_features.npz"
SAMPLES = CACHE / "samples.json"
SPLIT = CACHE / "analysis_split.json"

LAYERS = ["block_00", "block_10", "block_20", "block_30", "final"]
FACTORS = ["hue", "brightness", "selective_recolor"]
# Doses ordered from mildest to strongest. Brightness strength is 1 - multiplier,
# so 0.85 (strength 0.15) is mildest and 0.45 (strength 0.55) is strongest.
DOSES = {
    "hue": [15.0, 45.0, 90.0],
    "brightness": [0.85, 0.65, 0.45],
    "selective_recolor": [45.0, 90.0, 180.0],
}
CONDS = [(f, d) for f in FACTORS for d in DOSES[f]]

# Values printed in the paper for the final layer (acceptance checks).
PAPER = {
    "straightness": {"hue": 0.692, "brightness": 0.907, "selective_recolor": 0.553},
    "effective_rank": {"hue": 9.83, "brightness": 5.43, "selective_recolor": 13.72},
}
TOL = {"straightness": 0.01, "effective_rank": 0.1}


def load_inputs():
    samples = json.loads(SAMPLES.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    with np.load(NPZ, allow_pickle=False) as npz:
        pooled = npz["pooled"]
        layer_names = [str(x) for x in npz["layer_names"]]
    assert layer_names == LAYERS, layer_names
    assert pooled.shape == (len(samples), len(LAYERS), 1408), pooled.shape
    feats = pooled.astype(np.float32)
    return samples, split, feats


def build_design(samples, split):
    train_scenes = set(split["train_scenes"])
    clean_of, trans_of, pair_clean = {}, {}, {}
    for i, s in enumerate(samples):
        vid = s["base_video_id"]
        if s["factor"] == "clean":
            assert vid not in clean_of
            clean_of[vid] = i
            pair_clean.setdefault(s["base_pair_id"], {})[bool(s["possible"])] = i
        else:
            key = (s["factor"], round(float(s["nominal_dose"]), 6))
            trans_of.setdefault(vid, {})
            assert key not in trans_of[vid], (vid, key)
            trans_of[vid][key] = i
    sources = list(clean_of)
    clean_idx = np.array([clean_of[v] for v in sources])
    cond_idx = {}
    for f, d in CONDS:
        cond_idx[(f, d)] = np.array([trans_of[v][(f, round(d, 6))] for v in sources])
    scene = np.array([samples[clean_of[v]]["scene"] for v in sources])
    train_mask = np.array([sc in train_scenes for sc in scene])
    pairs = sorted(pair_clean)
    for p in pairs:
        assert set(pair_clean[p]) == {True, False}, p
    pair_possible = np.array([pair_clean[p][True] for p in pairs])
    pair_impossible = np.array([pair_clean[p][False] for p in pairs])
    return {
        "sources": sources,
        "clean_idx": clean_idx,
        "cond_idx": cond_idx,
        "scene": scene,
        "train_mask": train_mask,
        "pair_possible": pair_possible,
        "pair_impossible": pair_impossible,
    }


def l2norm_rows(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def displacements(Z, design):
    c = Z[design["clean_idx"]]
    return {cond: Z[idx] - c for cond, idx in design["cond_idx"].items()}


def straightness_by_factor(Z, design):
    """Mean over sources of ||z(strong)-z(clean)|| / sum of step lengths along clean, mild, medium, strong."""
    out = {}
    c = Z[design["clean_idx"]]
    for f in FACTORS:
        pts = [c] + [Z[design["cond_idx"][(f, d)]] for d in DOSES[f]]
        P = np.stack(pts, axis=1).astype(np.float64)  # (n, 4, D)
        direct = np.linalg.norm(P[:, 3] - P[:, 0], axis=1)
        steps = np.linalg.norm(np.diff(P, axis=1), axis=2).sum(axis=1)
        out[f] = float(np.mean(direct / steps))
    return out


def effective_rank_by_factor(Z, design):
    """exp(entropy) of normalized squared singular values of all displacement rows of a factor."""
    out = {}
    disp = displacements(Z, design)
    for f in FACTORS:
        M = np.concatenate([disp[(f, d)] for d in DOSES[f]], axis=0)
        s = np.linalg.svd(M, compute_uv=False)
        p = s ** 2 / np.sum(s ** 2)
        p = p[p > 0]
        out[f] = float(np.exp(-np.sum(p * np.log(p))))
    return out


def pca_fit(D):
    """Mean-centered SVD, no whitening. Returns top-2 axes, their explained variance ratio, and the mean."""
    mu = D.mean(axis=0)
    _, S, Vt = np.linalg.svd(D - mu, full_matrices=False)
    evr = S ** 2 / np.sum(S ** 2)
    return Vt[:2], evr[:2], mu


def clip_counts(samples, split):
    train = set(split["train_scenes"])
    counts = {}
    for s in samples:
        key = f"{s['factor']}|{float(s['nominal_dose']):g}"
        row = counts.setdefault(key, {"all": 0, "train_scenes": 0, "heldout_scenes": 0})
        row["all"] += 1
        row["train_scenes" if s["scene"] in train else "heldout_scenes"] += 1
    return counts


def acceptance(strait_by_variant, rank_by_variant):
    """Compare both variants with the paper values. Returns per-variant rows and matches."""
    result = {}
    for variant, strait, rank in [
        ("zhat", strait_by_variant["zhat"], rank_by_variant["zhat"]),
        ("raw", strait_by_variant["raw"], rank_by_variant["raw"]),
    ]:
        rows = []
        n_match = 0
        for f in FACTORS:
            for name, val in [("straightness", strait[f]), ("effective_rank", rank[f])]:
                paper = PAPER[name][f]
                ok = abs(val - paper) <= TOL[name]
                n_match += int(ok)
                rows.append({"factor": f, "metric": name, "computed": val, "paper": paper,
                             "abs_diff": abs(val - paper), "tolerance": TOL[name], "match": bool(ok)})
        result[variant] = {"rows": rows, "n_match": n_match, "n_total": len(rows)}
    return result
