"""A2: schema validation and scene-clustered statistics for the WMPAI scores table.

Input rows (dicts) need: scene_id, clip_id, split, physical_condition (lawful|inconsistent),
appearance_condition, score ('' or None = missing), context_sha, state_sha.
Optional column output_status (complete | failed_render | failed_score | excluded_score_blind): when it is
present, only complete cells may carry a score and every other cell must have a blank score, and a cell that
is not complete is read as missing whatever its score field holds.
Missing cells stay NaN; they are never imputed as zero or silently dropped from the audit.
Status of anything computed here from a fixture: tooling check, not evidence.
"""
import csv
import math
import random
from collections import defaultdict

REQUIRED = ["scene_id", "clip_id", "split", "physical_condition", "appearance_condition",
            "score", "context_sha", "state_sha"]
STATUSES = ("complete", "failed_render", "failed_score", "excluded_score_blind")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _score(r):
    status = r.get("output_status")
    if status not in (None, "") and status != "complete":
        return float("nan")  # a failed or excluded cell is missing, whatever its score field holds
    v = r["score"]
    return float("nan") if v in ("", None) else float(v)


def validate(rows):
    """Return a list of error strings; empty list means the table is acceptable."""
    errs = []
    for k in REQUIRED:
        if rows and k not in rows[0]:
            errs.append(f"missing column {k}")
    if errs:
        return errs
    if rows and "output_status" in rows[0]:
        for r in rows:
            status, value = r["output_status"], r["score"]
            if status not in STATUSES:
                errs.append(f"bad output_status {status!r} in {r['clip_id']}")
            elif status == "complete":
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    errs.append(f"complete cell lacks a finite score in {r['clip_id']}")
            elif value not in ("", None):
                errs.append(f"{status} cell must have a blank score in {r['clip_id']}")
    splits = defaultdict(set)
    cells = defaultdict(int)
    for r in rows:
        splits[r["scene_id"]].add(r["split"])
        cells[(r["scene_id"], r["physical_condition"], r["appearance_condition"])] += 1
        if r["physical_condition"] not in ("lawful", "inconsistent"):
            errs.append(f"bad physical_condition {r['physical_condition']!r} in {r['clip_id']}")
    errs += [f"scene {s} in several splits {sorted(v)}" for s, v in splits.items() if len(v) > 1]
    errs += [f"duplicate cell {k}" for k, n in cells.items() if n > 1]
    ctx = defaultdict(set)   # same scene+appearance: lawful and inconsistent share context
    st = defaultdict(set)    # same scene+physical branch: appearance variants share state
    for r in rows:
        ctx[(r["scene_id"], r["appearance_condition"])].add(r["context_sha"])
        st[(r["scene_id"], r["physical_condition"])].add(r["state_sha"])
    errs += [f"context hash mismatch {k}" for k, v in ctx.items() if len(v) > 1]
    errs += [f"state hash mismatch {k}" for k, v in st.items() if len(v) > 1]
    return errs


def margins(rows, split, appearance):
    """{scene_id: S(inconsistent) - S(lawful)}; scenes with a missing/NaN branch are omitted
    and reported by missing_scenes(), never counted as zero."""
    s = {}
    for r in rows:
        if r["split"] == split and r["appearance_condition"] == appearance:
            s[(r["scene_id"], r["physical_condition"])] = _score(r)
    out = {}
    for sc in {k[0] for k in s}:
        a, b = s.get((sc, "inconsistent"), float("nan")), s.get((sc, "lawful"), float("nan"))
        if not (math.isnan(a) or math.isnan(b)):
            out[sc] = a - b
    return out


def missing_scenes(rows, split, appearance):
    scenes = {r["scene_id"] for r in rows if r["split"] == split}
    return sorted(scenes - set(margins(rows, split, appearance)))


def pair_stats(m, n_boot=10000, seed=0):
    """Fraction M>0 (ties count as not >0 and are reported), mean/median margin with
    scene-clustered 95% bootstrap intervals (each scene is the resampling unit)."""
    ids = sorted(m)
    if not ids:
        return {"n_scenes": 0}
    rng = random.Random(seed)
    vals = [m[i] for i in ids]
    med = lambda x: sorted(x)[len(x) // 2] if len(x) % 2 else sum(sorted(x)[len(x) // 2 - 1:len(x) // 2 + 1]) / 2
    boots = []
    for _ in range(n_boot):
        d = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append((sum(x > 0 for x in d) / len(d), sum(d) / len(d), med(d)))
    ci = lambda j: (sorted(b[j] for b in boots)[int(0.025 * n_boot)], sorted(b[j] for b in boots)[int(0.975 * n_boot) - 1])
    return {"n_scenes": len(ids), "ties": sum(x == 0 for x in vals),
            "frac_pos": sum(x > 0 for x in vals) / len(vals), "frac_pos_ci": ci(0),
            "mean": sum(vals) / len(vals), "mean_ci": ci(1),
            "median": med(vals), "median_ci": ci(2)}


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); pos = inconsistent clips, neg = lawful clips."""
    pos = [x for x in pos if not math.isnan(x)]
    neg = [x for x in neg if not math.isnan(x)]
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def calibrate_threshold(rows, appearances=None):
    """Development-only threshold maximizing balanced accuracy over single clips."""
    dev = [r for r in rows if r["split"] == "development" and not math.isnan(_score(r))
           and (appearances is None or r["appearance_condition"] in appearances)]
    pos = [_score(r) for r in dev if r["physical_condition"] == "inconsistent"]
    neg = [_score(r) for r in dev if r["physical_condition"] == "lawful"]
    best = (-1.0, None)
    for t in sorted(set(pos + neg)):
        ba = 0.5 * (sum(p >= t for p in pos) / len(pos) + sum(n < t for n in neg) / len(neg))
        if ba > best[0]:
            best = (ba, t)
    return best[1]


def threshold_errors(rows, t, split="confirmatory"):
    te = [r for r in rows if r["split"] == split and not math.isnan(_score(r))]
    pos = [_score(r) for r in te if r["physical_condition"] == "inconsistent"]
    neg = [_score(r) for r in te if r["physical_condition"] == "lawful"]
    return {"miss_rate": sum(p < t for p in pos) / len(pos) if pos else float("nan"),
            "false_alarm_rate": sum(n >= t for n in neg) / len(neg) if neg else float("nan"),
            "auroc": auroc(pos, neg)}
