import importlib.util
import itertools
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a2 = _load("a2_scene_stats_for_dev24_test", "a2_scene_stats.py")
dev = _load("a2_dev24", "a2_dev24_stats.py")

RADII = ("0.60", "0.65", "0.70")
FALL_TIMES = ("1.18", "1.24")
CAMERAS = ("-8", "-4", "4", "8")
APPEARANCES = ("clean", "persistent_hue", "abrupt_event")


def row(cfg, world, traj, ap="clean", score="", status="complete",
        radius="0.60", fall_time="1.18", camera_x="-8"):
    return dict(configuration_id=cfg, world=world, trajectory=traj, appearance_condition=ap,
                score=score, output_status=status, radius=radius, fall_time=fall_time,
                camera_x=camera_x)


def make_rows(delta=0.5, margin_sd=0.0, noise=0.02, seed=1):
    """24 configurations x 2 worlds x 2 trajectories x 3 appearances.

    Score = base + configuration effect + 0.5 * (+1 if inconsistent else -1) * (delta + u_c) + noise.
    The planted inconsistent-minus-lawful effect is delta. u_c is a configuration-level random
    margin shared by both worlds of that configuration (sd = margin_sd)."""
    rng = random.Random(seed)
    rows = []
    for radius, ft, cam in itertools.product(RADII, FALL_TIMES, CAMERAS):
        cfg = f"r{radius}_t{ft}_c{cam}"
        a_c = rng.gauss(0, 0.3)
        u_c = rng.gauss(0, margin_sd) if margin_sd else 0.0
        for world in dev.WORLDS:
            for traj in ("stop", "fall"):
                inc = traj == _inc_traj(world)
                for ap in APPEARANCES:
                    sign = 1 if inc else -1
                    s = 0.4 + a_c + sign * 0.5 * (delta + u_c) + rng.gauss(0, noise)
                    rows.append(row(cfg, world, traj, ap, f"{s:.6f}", "complete",
                                    radius, ft, cam))
    return rows


def _inc_traj(world):
    return "fall" if world == "solid" else "stop"


def fixture_with_failed_cell(cfg="r0.60_t1.18_c-8", world="solid", traj="fall"):
    rows = make_rows(seed=3)
    for r in rows:
        if r["configuration_id"] == cfg and r["world"] == world and r["trajectory"] == traj \
                and r["appearance_condition"] == "clean":
            r["output_status"], r["score"] = "failed_score", "0"
    return rows


def test_hand_computed_sign_conventions_for_both_worlds():
    rows = [
        row("c0", "solid", "stop", score="0.2"),   # lawful in solid world
        row("c0", "solid", "fall", score="0.9"),   # inconsistent in solid world
        row("c0", "gap", "stop", score="0.3"),     # inconsistent in gap world
        row("c0", "gap", "fall", score="0.9"),     # lawful in gap world
    ]
    wm = dev.world_margins(rows, "clean")
    assert math.isclose(wm[("c0", "solid")], 0.9 - 0.2)     # S(fall) - S(stop)
    assert math.isclose(wm[("c0", "gap")], 0.3 - 0.9)       # S(stop) - S(fall)
    cm = dev.config_margins(rows, "clean")
    assert math.isclose(cm["c0"], ((0.9 - 0.2) + (0.3 - 0.9)) / 2)
    assert dev.missing_worlds(rows, "clean") == []


def test_planted_effect_is_positive_in_both_worlds():
    rows = make_rows(delta=0.5, margin_sd=0.0, noise=0.02, seed=5)
    wm = dev.world_margins(rows, "persistent_hue")
    assert len(wm) == 48
    for v in wm.values():
        assert abs(v - 0.5) < 0.1


def test_failed_cell_makes_world_and_configuration_missing_not_zero():
    cfg = "r0.60_t1.18_c-8"
    rows = fixture_with_failed_cell(cfg, "solid", "fall")
    assert (cfg, "solid") not in dev.world_margins(rows, "clean")
    assert (cfg, "solid") in dev.missing_worlds(rows, "clean")
    assert cfg not in dev.config_margins(rows, "clean")
    assert dev.incomplete_configurations(rows, "clean") == [cfg]
    # the other appearances are untouched
    assert (cfg, "solid") in dev.world_margins(rows, "persistent_hue")
    s = dev.cluster_stats(rows, "clean", n_boot=200)
    assert s["n_units"] == 23
    assert s["incomplete_configurations"] == [cfg]


def test_n_units_is_24_configurations_not_48_worlds():
    rows = make_rows(seed=2)
    s = dev.cluster_stats(rows, "clean", n_boot=200)
    assert s["n_units"] == 24
    assert s["primary"]["n_scenes"] == 24
    assert s["solid"]["n"] == 24 and s["gap"]["n"] == 24
    assert s["incomplete_configurations"] == []


def test_leave_one_level_out_row_counts():
    rows = make_rows(seed=4)
    for factor, n_levels, n_left in (("radius", 3, 16), ("fall_time", 2, 12), ("camera_x", 4, 18)):
        out = dev.leave_one_level_out(rows, "clean", factor)
        assert len(out) == n_levels
        assert all(r["n_configurations"] == n_left for r in out), factor
        assert all(r["factor"] == factor for r in out)


def test_leave_one_level_out_drops_the_right_configurations():
    rows = make_rows(seed=4)
    out = dev.leave_one_level_out(rows, "clean", "radius")
    cm = dev.config_margins(rows, "clean")
    for r in out:
        kept = [v for c, v in cm.items() if f"r{r['dropped_level']}_" not in c]
        assert math.isclose(r["mean"], sum(kept) / len(kept))


def test_naive_interval_is_narrower_when_worlds_share_a_random_margin():
    rows = make_rows(delta=0.5, margin_sd=1.0, noise=0.05, seed=11)
    cluster = dev.cluster_stats(rows, "clean", n_boot=2000, seed=7)["primary"]
    naive = dev.naive_world_stats(rows, "clean", n_boot=2000, seed=7)
    cw = cluster["mean_ci"][1] - cluster["mean_ci"][0]
    nw = naive["mean_ci"][1] - naive["mean_ci"][0]
    assert nw < cw
    assert math.isclose(naive["mean"], cluster["mean"], rel_tol=1e-9)


def test_incomplete_configuration_listed_and_excluded_from_n_units():
    rows = make_rows(seed=6)
    cfg = "r0.70_t1.24_c8"
    for r in rows:
        if r["configuration_id"] == cfg and r["world"] == "gap" and r["appearance_condition"] == "clean":
            r["score"] = ""
            r["output_status"] = "failed_render"
    s = dev.cluster_stats(rows, "clean", n_boot=200)
    assert s["incomplete_configurations"] == [cfg]
    assert s["n_units"] == 23
    assert cfg not in dev.config_margins(rows, "clean")
    assert (cfg, "gap") in dev.missing_worlds(rows, "clean")


def test_factor_varying_within_configuration_is_rejected():
    rows = make_rows(seed=8)
    rows[0]["radius"] = "0.99"
    try:
        dev.leave_one_level_out(rows, "clean", "radius")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_unknown_factor_is_rejected():
    rows = make_rows(seed=8)
    try:
        dev.leave_one_level_out(rows, "clean", "fall")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
