import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pw", Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex/a2_power_study.py")
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)


def test_half_width_shrinks_with_more_scenes():
    h32 = pw.simulate(32, 0.0, 1.0, n_sim=200, n_boot=200)[0].mean()
    h128 = pw.simulate(128, 0.0, 1.0, n_sim=200, n_boot=200)[0].mean()
    assert h128 < 0.65 * h32  # about half for 4x scenes


def test_lower_bound_clears_zero_only_with_real_effect():
    big = pw.simulate(64, 1.0, 1.0, n_sim=200, n_boot=200)[1]
    none = pw.simulate(64, 0.0, 1.0, n_sim=200, n_boot=200)[1]
    assert (big > 0).mean() > 0.95 and (none > 0).mean() < 0.15


def test_smallest_n_and_missing_case():
    rows = pw.study(0.0, 1.0, [16, 64, 256], 0.3, 0.2, sd_mult=(1.0,), n_sim=100, n_boot=100)
    n = pw.smallest_n(rows, 1.0, 0.8)
    assert n in (None, 16, 64, 256)
    assert pw.smallest_n(rows, 1.0, 1.01) is None
