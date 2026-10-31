"""A3/A4 helpers: scene-held-out association tests, nested shortcut regression, frame baselines.

All inputs are plain numpy arrays so the same code runs on the archived eight-scene pilot
(exploratory) and, later, on Vasi's fresh scenes. Nothing here is a result by itself.
"""
import numpy as np


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _fit_predict(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ beta


def loso_r2(X, y, scene):
    """Leave-one-scene-out R^2 (1 - SSE/SST around the training mean). Scenes are never split."""
    X = np.asarray(X, float).reshape(len(y), -1)
    y, scene = np.asarray(y, float), np.asarray(scene)
    sse = sst = 0.0
    for s in np.unique(scene):
        te = scene == s
        pred = _fit_predict(X[~te], y[~te], X[te])
        sse += float(((y[te] - pred) ** 2).sum())
        sst += float(((y[te] - y[~te].mean()) ** 2).sum())
    return 1.0 - sse / sst


def permutation_null_loso(x, y, scene, group, n_perm=2000, seed=0):
    """Null for loso_r2(x -> y): shuffle x across scenes *within* each group (e.g. condition),
    breaking the scene-level link while keeping each group's marginal distribution."""
    rng = np.random.default_rng(seed)
    x, group = np.asarray(x, float), np.asarray(group)
    obs = loso_r2(x, y, scene)
    null = []
    for _ in range(n_perm):
        xp = x.copy()
        for g in np.unique(group):
            idx = np.where(group == g)[0]
            xp[idx] = x[rng.permutation(idx)]
        null.append(loso_r2(xp, y, scene))
    null = np.array(null)
    return obs, float(null.mean()), float((1 + (null >= obs).sum()) / (1 + n_perm))


def nested_gain(base_X, extra_X, y, scene):
    """LOSO R^2 gain from adding extra_X to base_X (shortcut test: does the physical flag
    explain score changes beyond rendered change?)."""
    base = loso_r2(base_X, y, scene)
    full = loso_r2(np.c_[np.asarray(base_X, float).reshape(len(y), -1),
                         np.asarray(extra_X, float).reshape(len(y), -1)], y, scene)
    return base, full, full - base


def frame_features(frames, n_context, rng=None):
    """Shortcut baselines from a (T,H,W,C) float clip in [0,1]:
    static   = mean |target frame - last context frame|
    motion   = mean |frame_t - frame_{t-1}| over target frames (frame-difference flow proxy)
    shuffled = motion after randomly permuting the target frames (temporal-shuffle control)."""
    f = np.asarray(frames, float)
    ctx_last, tgt = f[n_context - 1], f[n_context:]
    rng = rng or np.random.default_rng(0)
    seq = np.concatenate([ctx_last[None], tgt])
    shuf = np.concatenate([ctx_last[None], tgt[rng.permutation(len(tgt))]])
    return {"static": float(np.abs(tgt - ctx_last).mean()),
            "motion": float(np.abs(np.diff(seq, axis=0)).mean()),
            "shuffled": float(np.abs(np.diff(shuf, axis=0)).mean())}
