from __future__ import annotations

import numpy as np

from scripts.run_intphys2_geometry_audit import (
    explained_energy_curve,
    mean_pairwise_cosine_from_unit,
    orthonormal_basis_from_rows,
    principal_overlap_score,
)
from scripts.run_intphys2_geometry_freeze import (
    entropy_effective_rank,
    random_pao_null_draws,
    tdc_from_unit_vectors,
)
from scripts.run_intphys2_geometry_sealed import LAYER_ORDER, TRANSFORMS, validate_unordered_labels


def test_same_direction_tdc_is_one() -> None:
    direction = np.zeros((16,), dtype=np.float32)
    direction[0] = 1.0
    vectors = np.stack([direction for _ in range(8)], axis=0)
    assert mean_pairwise_cosine_from_unit(vectors) > 0.999


def test_random_direction_tdc_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((512, 32), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert abs(mean_pairwise_cosine_from_unit(vectors)) < 0.05


def test_principal_overlap_orthogonal_and_identical() -> None:
    a = np.eye(6, 2, dtype=np.float32)
    b = np.eye(6, 2, dtype=np.float32)
    c = np.eye(6, 2, k=2, dtype=np.float32)
    assert principal_overlap_score(a, b, 2) > 0.999
    assert principal_overlap_score(a, c, 2) < 1e-6


def test_circular_trajectory_prefers_two_dimensions() -> None:
    angles = np.deg2rad(np.arange(0, 360, 45))
    trajectory = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
    ).astype(np.float32)
    _, singular_values = orthonormal_basis_from_rows(trajectory)
    energy = explained_energy_curve(singular_values, [1, 2])
    assert energy[2] > 0.99
    assert energy[1] < 0.8


def test_random_subspace_pao_matches_k_over_d() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((48, 64), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    scene_ids = np.asarray([f"s{i // 2:02d}" for i in range(len(vectors))], dtype=object)
    scores = random_pao_null_draws(vectors, scene_ids, ambient_dim=64, k=4, draws=2000, rng=rng)
    assert abs(float(np.mean(scores)) - (4.0 / 64.0)) < 0.01


def test_tdc_ignores_magnitude_variation_when_direction_matches() -> None:
    direction = np.zeros((8,), dtype=np.float32)
    direction[0] = 1.0
    scales = np.asarray([0.1, 1.0, 3.0, 10.0, 0.3, 2.0], dtype=np.float32)
    vectors = scales[:, None] * direction[None, :]
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assert tdc_from_unit_vectors(unit_vectors) > 0.999


def test_opposite_directions_have_low_tdc_but_one_dimensional_subspace() -> None:
    direction = np.zeros((6,), dtype=np.float32)
    direction[0] = 1.0
    vectors = np.stack([direction] * 8 + [-direction] * 8, axis=0)
    tdc_value = tdc_from_unit_vectors(vectors)
    _, singular_values = orthonormal_basis_from_rows(vectors)
    energy = explained_energy_curve(singular_values, [1, 2])
    assert tdc_value < 1e-6
    assert energy[1] > 0.999


def test_random_angles_in_plane_have_low_tdc_but_rank_two() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    vectors = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    tdc_value = tdc_from_unit_vectors(vectors)
    _, singular_values = orthonormal_basis_from_rows(vectors)
    energy = explained_energy_curve(singular_values, [1, 2])
    assert tdc_value < 0.05
    assert energy[2] > 0.999
    assert energy[1] < 0.8


def test_two_event_directions_raise_conditional_but_not_aggregate_tdc() -> None:
    basis_a = np.zeros((5,), dtype=np.float32)
    basis_b = np.zeros((5,), dtype=np.float32)
    basis_a[0] = 1.0
    basis_b[1] = 1.0
    event_a = np.stack([basis_a] * 8, axis=0)
    event_b = np.stack([basis_b] * 8, axis=0)
    aggregate = np.concatenate([event_a, event_b], axis=0)
    assert tdc_from_unit_vectors(event_a) > 0.999
    assert tdc_from_unit_vectors(event_b) > 0.999
    assert tdc_from_unit_vectors(aggregate) < 0.8


def test_condition_validation_accepts_expected_order() -> None:
    assert validate_unordered_labels(TRANSFORMS, TRANSFORMS)


def test_condition_validation_accepts_permuted_order() -> None:
    permuted = ["brightness_0p45", "clean", "selective_recolor_180", "hue_090"]
    assert validate_unordered_labels(permuted, TRANSFORMS)


def test_condition_validation_rejects_missing_condition() -> None:
    missing = ["clean", "hue_090", "brightness_0p45"]
    assert not validate_unordered_labels(missing, TRANSFORMS)


def test_condition_validation_rejects_extra_condition() -> None:
    extra = TRANSFORMS + ["extra_condition"]
    assert not validate_unordered_labels(extra, TRANSFORMS)


def test_condition_validation_rejects_duplicate_replacing_expected_condition() -> None:
    duplicate = ["clean", "hue_090", "brightness_0p45", "brightness_0p45"]
    assert not validate_unordered_labels(duplicate, TRANSFORMS)


def test_layer_order_still_requires_exact_five_layers() -> None:
    assert LAYER_ORDER == ["block_00", "block_10", "block_20", "block_30", "final"]
