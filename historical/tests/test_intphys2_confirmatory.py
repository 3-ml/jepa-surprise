import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts import intphys2_confirmatory_common as common
from scripts.prepare_intphys2_confirmatory_manifest import (
    allocate_development_scenes,
)


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: str):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_confirmatory_manifests_preserve_pairs_scenes_and_conditions():
    pairs = _rows("manifests/intphys2/intphys2_confirmatory_pairs.csv")
    samples = _rows("manifests/intphys2/intphys2_confirmatory_samples.csv")
    config = json.loads(
        (ROOT / "configs/intphys2_confirmatory_shared.json").read_text()
    )

    assert Counter((row["dataset_split"], row["partition"]) for row in pairs) == {
        ("Debug", "debug"): 30,
        ("Main", "development"): 76,
        ("Main", "test"): 114,
        ("Main", "excluded_inspected_scene"): 316,
    }
    assert sum(row["inspected_pair"] == "true" for row in pairs) == 193
    assert sum(row["selective_transform_feasible"] == "false" for row in pairs) == 1
    assert any(row["scene_id"] == "154_0" for row in pairs)

    scenes = defaultdict(set)
    for row in pairs:
        if row["dataset_split"] == "Main":
            scenes[row["partition"]].add(row["scene_id"])
    partitions = list(scenes)
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            assert scenes[first].isdisjoint(scenes[second])
    assert len(scenes["development"]) == 38
    assert len(scenes["test"]) == 57

    expected_conditions = sorted(row["name"] for row in config["appearance_conditions"])
    by_source = defaultdict(list)
    for row in samples:
        by_source[(row["dataset_split"], row["name"])].append(row)
    assert len(by_source) == 1072
    for rows in by_source.values():
        assert (
            sorted(row["appearance_condition"] for row in rows) == expected_conditions
        )
        assert len({row["source_frame_indices_0based"] for row in rows}) == 1
        indices = json.loads(rows[0]["source_frame_indices_0based"])
        assert len(indices) == 64
        assert indices == sorted(set(indices))


def test_scene_allocation_is_deterministic_and_string_safe():
    scenes = [
        {
            "scene_id": scene,
            "condition": "continuity" if index % 2 else "solidity",
            "Difficulty": "Hard",
            "Camera": "Moving",
        }
        for index, scene in enumerate(["154_0", "1540", "2", "10", "11", "12"])
    ]
    first = allocate_development_scenes(scenes, fraction=0.5, seed="locked")
    second = allocate_development_scenes(
        list(reversed(scenes)), fraction=0.5, seed="locked"
    )
    assert first == second
    assert len(first) == 3
    assert {"154_0", "1540"} <= {row["scene_id"] for row in scenes}


def test_appearance_identity_policy_and_coarse_grid():
    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    config = json.loads(
        (ROOT / "configs/intphys2_confirmatory_shared.json").read_text()
    )
    clean = {
        "transform_family": "clean",
        "transform_feasible": "true",
        "transform_value": "",
        "selective_center_degrees": "",
    }
    infeasible = {
        "transform_family": "selective_recolor",
        "transform_feasible": "false",
        "transform_value": "180",
        "selective_center_degrees": "",
    }
    assert np.array_equal(common.apply_appearance(frames, clean, config), frames)
    assert np.array_equal(common.apply_appearance(frames, infeasible, config), frames)

    colors = np.asarray([[[[255, 0, 0], [255, 255, 255]]]], dtype=np.uint8)
    hue = {
        "transform_family": "hue",
        "transform_feasible": "true",
        "transform_value": "90",
        "selective_center_degrees": "",
    }
    brightness = {
        "transform_family": "brightness",
        "transform_feasible": "true",
        "transform_value": "0.45",
        "selective_center_degrees": "",
    }
    selective = {
        "transform_family": "selective_recolor",
        "transform_feasible": "true",
        "transform_value": "180",
        "selective_center_degrees": "5",
    }
    assert not np.array_equal(common.apply_appearance(colors, hue, config), colors)
    assert np.array_equal(
        common.apply_appearance(colors, brightness, config),
        np.rint(colors.astype(np.float32) * 0.45).astype(np.uint8),
    )
    selectively_changed = common.apply_appearance(colors, selective, config)
    assert not np.array_equal(selectively_changed[..., 0, :], colors[..., 0, :])
    assert np.array_equal(selectively_changed[..., 1, :], colors[..., 1, :])

    tokens = torch.arange(2 * 32 * 16 * 16 * 3, dtype=torch.float32).reshape(
        2, 32 * 16 * 16, 3
    )
    summary = common.coarse_token_summary(tokens, (32, 16, 16), (4, 4, 4))
    assert summary.shape == (2, 64, 3)
    assert torch.equal(summary.mean(dim=1), tokens.mean(dim=1))
