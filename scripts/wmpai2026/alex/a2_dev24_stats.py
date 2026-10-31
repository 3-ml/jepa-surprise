"""A2 development design: configuration-clustered statistics for the 2x2 crossed table.

Design: 24 configurations, each with two worlds (solid, gap) and two trajectories (stop, fall).
In the solid world stop is lawful and fall is inconsistent; in the gap world it is the reverse.
Appearance conditions are handled one at a time, as in a2_scene_stats.py.

World margin = S(inconsistent) - S(lawful) for one configuration, world and appearance.
Configuration margin = mean of its two world margins. The configuration is the resampling unit.

Rows are dicts with: configuration_id, world, trajectory, appearance_condition, score (blank = missing),
output_status (complete, failed_render, failed_score or excluded_score_blind), and the constant
factor columns radius, fall_time and camera_x. A cell that is not complete is read as missing.
Missing worlds are omitted and listed, never treated as zero.
Status of anything computed here from a fixture: tooling check, not evidence.
"""
import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "a2_scene_stats", Path(__file__).resolve().parent / "a2_scene_stats.py")
_a2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a2)

WORLDS = ("solid", "gap")
FACTORS = ("radius", "fall_time", "camera_x")
_NAN = float("nan")


def _inconsistent(world):
    """Trajectory that is inconsistent with the physics in this world."""
    return "fall" if world == "solid" else "stop"


def _lawful(world):
    """Trajectory that is lawful in this world."""
    return "stop" if world == "solid" else "fall"


def _configs(rows):
    """All configuration ids in the table, whatever the appearance."""
    return sorted({r["configuration_id"] for r in rows})


def _cells(rows, appearance):
    """{(configuration_id, world, trajectory): score}; a duplicate cell is an error."""
    cells = {}
    for r in rows:
        if r["appearance_condition"] != appearance:
            continue
        key = (r["configuration_id"], r["world"], r["trajectory"])
        if key in cells:
            raise ValueError(f"duplicate cell {key} for appearance {appearance!r}")
        cells[key] = _a2._score(r)
    return cells


def world_margins(rows, appearance):
    """{(configuration_id, world): S(inconsistent) - S(lawful)}.
    A world with a missing or NaN score is omitted; see missing_worlds()."""
    cells = _cells(rows, appearance)
    out = {}
    for cfg in _configs(rows):
        for world in WORLDS:
            s_bad = cells.get((cfg, world, _inconsistent(world)), _NAN)
            s_law = cells.get((cfg, world, _lawful(world)), _NAN)
            if not (math.isnan(s_bad) or math.isnan(s_law)):
                out[(cfg, world)] = s_bad - s_law
    return out


def missing_worlds(rows, appearance):
    """Sorted (configuration_id, world) pairs whose world margin could not be computed."""
    wm = world_margins(rows, appearance)
    return sorted((cfg, w) for cfg in _configs(rows) for w in WORLDS if (cfg, w) not in wm)


def config_margins(rows, appearance):
    """{configuration_id: mean of its solid and gap world margins}, only for configurations
    where both world margins exist."""
    wm = world_margins(rows, appearance)
    return {cfg: (wm[(cfg, "solid")] + wm[(cfg, "gap")]) / 2
            for cfg in _configs(rows) if (cfg, "solid") in wm and (cfg, "gap") in wm}


def incomplete_configurations(rows, appearance):
    """Sorted configuration ids with a missing world; they are left out of the primary statistic."""
    cm = config_margins(rows, appearance)
    return sorted(c for c in _configs(rows) if c not in cm)


def _summary(vals):
    """n, mean and fraction strictly positive; mean and fraction are NaN when n is 0."""
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": _NAN, "frac_pos": _NAN}
    return {"n": n, "mean": sum(vals) / n, "frac_pos": sum(v > 0 for v in vals) / n}


def cluster_stats(rows, appearance, n_boot=10000, seed=20261025):
    """Primary configuration-clustered statistic plus per-world summaries.

    primary: pair_stats over configuration margins (configuration = bootstrap unit).
    solid, gap: summaries of the world margins that exist for that world (descriptive only).
    """
    cm = config_margins(rows, appearance)
    wm = world_margins(rows, appearance)
    return {
        "n_units": len(cm),
        "primary": _a2.pair_stats(cm, n_boot, seed),
        "solid": _summary([v for (c, w), v in wm.items() if w == "solid"]),
        "gap": _summary([v for (c, w), v in wm.items() if w == "gap"]),
        "incomplete_configurations": incomplete_configurations(rows, appearance),
    }


def _factor_by_config(rows, factor):
    """{configuration_id: level}; a factor that varies within a configuration is an error."""
    levels = {}
    for r in rows:
        cfg, lev = r["configuration_id"], r[factor]
        if cfg in levels and levels[cfg] != lev:
            raise ValueError(f"factor {factor} varies within configuration {cfg}")
        levels[cfg] = lev
    return levels


def _sorted_levels(levels):
    try:
        return sorted(levels, key=float)
    except (TypeError, ValueError):
        return sorted(levels, key=str)


def leave_one_level_out(rows, appearance, factor):
    """For each level of one factor, drop every configuration at that level and summarise
    the configuration margins that remain. Returns a list of dicts, one per level."""
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}")
    level_of = _factor_by_config(rows, factor)
    cm = config_margins(rows, appearance)
    out = []
    for lev in _sorted_levels(set(level_of.values())):
        vals = [v for c, v in cm.items() if level_of[c] != lev]
        s = _summary(vals)
        out.append({"factor": factor, "dropped_level": lev, "n_configurations": s["n"],
                    "mean": s["mean"], "frac_pos": s["frac_pos"]})
    return out


def naive_world_stats(rows, appearance, n_boot=10000, seed=20261025):
    """pair_stats applied to the world margins as if each world were an independent unit.

    For comparison only. It is anti-conservative, because the two worlds of one configuration
    share configuration-level variation. It must not be used for inference.
    """
    wm = world_margins(rows, appearance)
    return _a2.pair_stats(wm, n_boot, seed)
