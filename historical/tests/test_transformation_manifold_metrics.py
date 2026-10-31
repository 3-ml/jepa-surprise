from __future__ import annotations

import numpy as np

from scripts.transformation_manifold_metrics import (
    curvature_metrics,
    direction_kernel,
    interaction_fraction,
    mean_resultant_length,
    orthonormal_basis,
    projection_fraction,
    ray_reconstruction_error,
    svd_energy_rank,
    tangent_vectors,
    trajectory_straightness,
)


def test_perfect_ray() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(12, 16))
    direction = rng.normal(size=16)
    direction /= np.linalg.norm(direction)
    doses = np.array([0.0, 0.2, 0.6, 1.0])
    points = base[:, None, :] + doses[None, :, None] * direction
    delta = points[:, 1:] - points[:, :1]
    assert np.nanmin(direction_kernel(delta)) > 1 - 1e-10
    assert np.allclose(trajectory_straightness(points), 1.0)
    assert svd_energy_rank(delta.reshape(-1, 16))["energy"][1] > 1 - 1e-10
    turns, curvature = curvature_metrics(points)
    assert np.nanmin(turns) > 1 - 1e-10
    assert np.nanmax(curvature) < 1e-10


def test_two_dimensional_circle_beats_ray() -> None:
    theta = np.deg2rad(np.arange(0, 361, 30))
    u = np.eye(8)[0]
    v = np.eye(8)[1]
    points = (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)[None, :, :]
    centered = points[0] - points[0].mean(axis=0)
    spectrum = svd_energy_rank(centered)
    assert spectrum["energy"][2] > 1 - 1e-10
    assert spectrum["energy"][1] < 0.7
    ray = centered[0]
    assert np.nanmean(ray_reconstruction_error(centered, ray)) > 0.35
    assert np.linalg.norm(points[0, 0] - points[0, -1]) < 1e-10


def test_random_dose_directions_have_high_rank() -> None:
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(80, 6, 32))
    spectrum = svd_energy_rank(delta.reshape(-1, 32))
    assert spectrum["effective_rank"] > 20
    assert mean_resultant_length(delta[:, 0]) < 0.25


def test_saturating_ray_stays_one_dimensional() -> None:
    doses = np.array([0.0, 0.2, 0.5, 1.0])
    response = 1 - np.exp(-4 * doses)
    direction = np.eye(10)[2]
    points = response[None, :, None] * direction
    spectrum = svd_energy_rank(points[:, 1:].reshape(-1, 10))
    assert spectrum["energy"][1] > 1 - 1e-10
    assert np.allclose(trajectory_straightness(points), 1.0)
    assert not np.allclose(np.diff(response), np.diff(response)[0])


def test_additive_factors_have_zero_interaction() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(20, 12))
    b = rng.normal(size=(20, 12))
    assert np.nanmax(interaction_fraction(a, b, a + b)) < 1e-20


def test_interacting_factors_have_large_interaction() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(size=(20, 12))
    b = rng.normal(size=(20, 12))
    interaction = 2.5 * rng.normal(size=(20, 12))
    assert np.nanmedian(interaction_fraction(a, b, a + b + interaction)) > 0.5


def test_curved_path_has_structured_tangents_but_rotating_endpoints() -> None:
    theta = np.linspace(0, np.pi / 2, 7)
    scenes = 24
    points = np.repeat(
        np.stack([np.cos(theta), np.sin(theta)], axis=1)[None, :, :], scenes, axis=0
    )
    tangents = tangent_vectors(points, theta)
    assert mean_resultant_length(tangents[:, 3]) > 0.999
    delta = points[:, 1:] - points[:, :1]
    kernel = direction_kernel(delta)
    assert kernel[0, -1] < 0.85


def test_physics_inside_tangent_space_projects_high() -> None:
    basis = np.eye(16)[:, :2]
    physics = np.array([[2.0, -1.0] + [0.0] * 14])
    assert projection_fraction(physics, basis)[0] > 1 - 1e-12


def test_physics_orthogonal_to_tangent_space_projects_zero() -> None:
    appearance = np.eye(16)[:2]
    basis = orthonormal_basis(appearance, 2)
    physics = np.array([[0.0, 0.0, 1.0] + [0.0] * 13])
    assert projection_fraction(physics, basis)[0] < 1e-12
