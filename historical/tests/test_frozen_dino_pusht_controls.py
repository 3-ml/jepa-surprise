import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DINO = ROOT / "results/dinowm/pusht_controlled_displacement"
JEPA = ROOT / "results/jepa_wm/pusht_controlled_displacement"


def test_frozen_dino_controls_have_identical_complete_support():
    dino = pd.read_csv(DINO / "scores.csv")
    jepa = pd.read_csv(JEPA / "scores.csv")
    for frame in (dino, jepa):
        assert len(frame) == 256 * 18
        assert frame.base_id.nunique() == 256
        assert frame.condition.nunique() == 18
        assert not frame.duplicated(["base_id", "condition"]).any()
        assert frame.groupby("base_id").size().eq(18).all()
        assert np.isfinite(frame.prediction_mse).all()
    merged = jepa.merge(
        dino[["base_id", "condition", "target_sha256"]],
        on=["base_id", "condition"],
        suffixes=("_jepa", "_dino"),
        validate="one_to_one",
    )
    assert len(merged) == len(jepa)
    assert merged.target_sha256_jepa.eq(merged.target_sha256_dino).all()


def test_jepa_wm_paper_values_and_gates():
    action = pd.read_csv(JEPA / "same_target_contrasts.csv")
    construct = pd.read_csv(JEPA / "construct_comparisons.csv")
    expected_action = {
        ("zero", 0.25): 0.0004887237737420946,
        ("zero", 1.0): 0.0017889772716443986,
        ("opposite", 0.25): 0.0009758902015164495,
        ("opposite", 1.0): 0.0035125026770401746,
    }
    for (mode, scale), expected in expected_action.items():
        row = action[(action.action_mode == mode) & np.isclose(action.intended_displacement_scale, scale)].iloc[0]
        assert np.isclose(row.inconsistent_minus_lawful_mean, expected, rtol=0.0, atol=1e-15)
    expected_construct = {30.0: -0.03670352019253187, 60.0: -0.12481243821093813}
    for degrees, expected in expected_construct.items():
        row = construct[np.isclose(construct.hue_degrees, degrees)].iloc[0]
        assert np.isclose(row.appearance_minus_opposite_mean, expected, rtol=0.0, atol=1e-15)
    validation = json.loads((JEPA / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert validation["target_byte_identity_with_dino_wm"]
    assert validation["all_action_intervals_positive"]
    assert validation["reported_hue_minus_opposite_intervals_negative"]
