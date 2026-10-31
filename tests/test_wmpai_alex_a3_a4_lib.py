import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "lib", Path(__file__).resolve().parents[1] / "scripts/wmpai2026/alex/a3_a4_lib.py")
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)


def toy(n_scene=8, slope=1.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    scene = np.repeat(np.arange(n_scene), 4)
    cond = np.tile(np.arange(4), n_scene)
    x = rng.normal(size=len(scene)) + cond
    y = slope * x + noise * rng.normal(size=len(scene))
    return x, y, scene, cond


def test_loso_r2_high_when_signal_and_low_when_none():
    x, y, s, c = toy()
    assert lib.loso_r2(x, y, s) > 0.9
    x2, y2, s2, _ = toy(slope=0.0, noise=1.0, seed=3)
    assert lib.loso_r2(x2, y2, s2) < 0.2


def test_permutation_null_detects_real_link():
    x, y, s, c = toy()
    obs, null_mean, p = lib.permutation_null_loso(x, y, s, c, n_perm=200)
    assert obs > null_mean and p < 0.05


def test_permutation_null_does_not_flag_noise():
    x, y, s, c = toy(slope=0.0, noise=1.0, seed=5)
    assert lib.permutation_null_loso(x, y, s, c, n_perm=200)[2] > 0.05


def test_nested_gain_zero_for_irrelevant_flag_positive_for_relevant():
    x, y, s, _ = toy(slope=0.0, noise=0.1)
    flag = np.tile([0, 1], len(y) // 2).astype(float)
    y = y + 2.0 * flag
    assert lib.nested_gain(x, flag, y, s)[2] > 0.5
    y_nf = toy(slope=1.0, noise=0.05)[1]
    assert abs(lib.nested_gain(toy()[0], flag, y_nf, s)[2]) < 0.05


def test_frame_features_static_motion_shuffle():
    T, H = 10, 4
    ramp = np.linspace(0, 1, T)[:, None, None, None] * np.ones((T, H, H, 1))
    f = lib.frame_features(ramp, n_context=4, rng=np.random.default_rng(1))
    assert f["static"] > 0 and f["motion"] > 0
    assert f["shuffled"] >= f["motion"] - 1e-12  # shuffling a smooth ramp cannot smooth it
    still = np.zeros((T, H, H, 1))
    assert lib.frame_features(still, 4) == {"static": 0.0, "motion": 0.0, "shuffled": 0.0}
