import numpy as np

from scripts.train_vjepa2_layerwise_probes import (
    deterministic_group_split,
    run_probe_tasks,
    summarize,
)


def synthetic_payload():
    rng = np.random.default_rng(0)
    appearances = ["original", "grayscale", "brightness_contrast"]
    rows = []
    features = []
    for pair_index in range(12):
        pair_id = f"pair-{pair_index:02d}"
        for physics_label in ("possible", "impossible"):
            physics_value = 1.0 if physics_label == "impossible" else -1.0
            for appearance_index, appearance in enumerate(appearances):
                x = rng.normal(0.0, 0.03, size=(3, 6)).astype(np.float32)
                x[0, appearance_index] += 3.0
                x[1, 0] += physics_value * 3.0
                features.append(x)
                rows.append((pair_id, physics_label, appearance))
    pair_ids, pair_labels, appearance_conditions = zip(*rows)
    return {
        "X": np.stack(features),
        "layer_names": np.array(["appearance_layer", "physics_layer", "noise_layer"]),
        "pair_id": np.array(pair_ids),
        "pair_label": np.array(pair_labels),
        "appearance_condition": np.array(appearance_conditions),
    }


def test_deterministic_group_split_keeps_pairs_disjoint():
    pair_ids = np.array([f"pair-{index // 3:02d}" for index in range(36)])

    split = deterministic_group_split(pair_ids, seed=7)

    assert split["train"].isdisjoint(split["val"])
    assert split["train"].isdisjoint(split["test"])
    assert split["val"].isdisjoint(split["test"])
    assert split["train"] | split["val"] | split["test"] == set(pair_ids.tolist())


def test_probe_tasks_find_synthetic_physics_and_appearance_layers():
    rows, split_records, skipped = run_probe_tasks(
        synthetic_payload(),
        seeds=[0, 1],
        train_fraction=0.6,
        val_fraction=0.2,
        max_iter=1000,
        c_value=1.0,
    )

    summary, payload = summarize(rows)
    assert not skipped
    assert set(split_records) == {"0", "1"}
    best = {row["task"]: row for row in payload["best_test_layers"]}
    assert best["appearance"]["best_test_layer_name"] == "appearance_layer"
    assert best["appearance"]["best_test_accuracy_mean"] >= 0.95
    assert best["physics_all"]["best_test_layer_name"] == "physics_layer"
    assert best["physics_all"]["best_test_accuracy_mean"] >= 0.95
    assert {"train", "val", "test"} <= set(summary["split"])
