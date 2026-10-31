import numpy as np
import pandas as pd
import pytest

from scripts.analyze_intphys2_grouped_probe_statistics import (
    coverage_audit,
    global_scene_values,
    pair_level,
    split_audit,
    validate_cross_model_support,
    validate_crossfit_predictions,
)
from scripts.train_intphys2_grouped_probes import (
    deterministic_crossfit_splits,
    deterministic_group_split,
    scene_id_from_pair_id,
)


def test_scene_id_from_pair_id_preserves_string_scene_names():
    assert scene_id_from_pair_id("Main:154_0/2") == "154_0"
    assert scene_id_from_pair_id("Main:10/1") == "10"


def test_deterministic_group_split_holds_out_whole_scenes():
    groups = np.repeat(np.array(["scene-a", "scene-b", "scene-c", "scene-d", "scene-e"]), 4)
    split = deterministic_group_split(groups, seed=3, train_fraction=0.6, val_fraction=0.2)
    assert split["train"].isdisjoint(split["val"])
    assert split["train"].isdisjoint(split["test"])
    assert split["val"].isdisjoint(split["test"])
    assert split["train"] | split["val"] | split["test"] == set(groups.tolist())


def test_crossfit_splits_cover_every_scene_once_for_test_and_validation():
    groups = np.repeat(np.array([f"scene-{index}" for index in range(13)]), 4)
    splits = deterministic_crossfit_splits(groups, folds=5, seed=7)
    role_counts = {
        role: {group: 0 for group in set(groups.tolist())}
        for role in ("train", "val", "test")
    }
    for index, split in enumerate(splits):
        assert split["train"].isdisjoint(split["val"])
        assert split["train"].isdisjoint(split["test"])
        assert split["val"].isdisjoint(split["test"])
        assert split["train"] | split["val"] | split["test"] == set(groups.tolist())
        assert split["val"] == splits[(index + 1) % len(splits)]["test"]
        for role in role_counts:
            for group in split[role]:
                role_counts[role][group] += 1
    assert set(role_counts["test"].values()) == {1}
    assert set(role_counts["val"].values()) == {1}
    assert set(role_counts["train"].values()) == {3}
    assert max(map(lambda split: len(split["test"]), splits)) - min(
        map(lambda split: len(split["test"]), splits)
    ) <= 1
    assert splits == deterministic_crossfit_splits(groups[::-1], folds=5, seed=7)


def test_split_audit_reports_scene_clusters():
    frame = pd.DataFrame(
        {
            "model": ["m", "m", "m", "m"],
            "task": ["physics_all"] * 4,
            "seed": [0, 0, 0, 0],
            "scene_id": ["s1", "s1", "s2", "s2"],
            "pair_id": ["s1/1", "s1/2", "s2/1", "s2/2"],
            "sample_id": ["a", "b", "c", "d"],
        }
    )
    audit = split_audit(frame).iloc[0]
    assert audit.test_scenes == 2
    assert audit.test_pairs == 4
    assert audit.min_pairs_per_scene == 2
    assert audit.max_pairs_per_scene == 2


def test_coverage_and_global_scene_values_require_unique_test_folds():
    frame = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "task": ["physics_all"] * 4,
            "seed": [0, 0, 1, 1],
            "scene_id": ["s1", "s1", "s2", "s2"],
            "pair_id": ["s1/1", "s1/2", "s2/1", "s2/2"],
            "sample_id": ["a", "b", "c", "d"],
            "correct": [1.0, 0.0, 1.0, 1.0],
        }
    )
    row = coverage_audit(frame).iloc[0]
    assert row.unique_test_scenes == 2
    assert row.unique_test_pairs == 4
    assert row.test_folds_per_scene_min == row.test_folds_per_scene_max == 1
    assert np.allclose(global_scene_values(frame), [0.5, 1.0])
    repeated = pd.concat(
        [frame, pd.DataFrame({"seed": [2], "scene_id": ["s1"], "correct": [1.0]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="one test fold per scene"):
        global_scene_values(repeated)


def test_crossfit_prediction_validation_enforces_complete_support():
    rows = []
    for fold, scene in enumerate(("s1", "s2")):
        for pair in ("1", "2"):
            for sample in range(8):
                rows.append(
                    {
                        "split_mode": "crossfit",
                        "split_index": fold,
                        "task": "physics_all",
                        "seed": fold,
                        "scene_id": scene,
                        "pair_id": f"{scene}/{pair}",
                        "sample_id": f"{scene}/{pair}/{sample}",
                        "y_true": sample % 2,
                        "y_true_label": str(sample % 2),
                        "appearance_condition": "clean",
                    }
                )
    frame = pd.DataFrame(rows)
    validate_crossfit_predictions(frame, expected_scenes=2, expected_pairs=4, expected_folds=2)
    with pytest.raises(ValueError, match="incomplete pair sample support"):
        validate_crossfit_predictions(
            frame.iloc[:-1], expected_scenes=2, expected_pairs=4, expected_folds=2
        )
    changed = frame.copy()
    changed.loc[0, "y_true"] = 9
    with pytest.raises(ValueError, match="support or true labels differ"):
        validate_cross_model_support([frame, changed])


def test_pair_level_keeps_scene_key_for_delta_merges():
    frame = pd.DataFrame(
        {
            "model": ["m", "m"],
            "task": ["physics_all", "physics_all"],
            "layer_name": ["b", "b"],
            "seed": [0, 0],
            "scene_id": ["s1", "s1"],
            "pair_id": ["p1", "p1"],
            "correct": [1.0, 0.0],
            "condition": ["c", "c"],
            "difficulty": ["d", "d"],
            "camera": ["cam", "cam"],
        }
    )
    row = pair_level(frame).iloc[0]
    assert row.scene_id == "s1"
    assert row.pair_id == "p1"
    assert row.correct == 0.5
