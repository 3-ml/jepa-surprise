#!/usr/bin/env python3
"""Numerically stable metrics for transformation-dose latent trajectories."""

from __future__ import annotations

import numpy as np


EPS = np.finfo(np.float64).eps


def l2_normalize(values: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(norms) & (norms > floor)
    result = np.full_like(values, np.nan, dtype=np.float64)
    result[valid] = values[valid] / norms[valid, None]
    return result, valid


def cosine_rows(left: np.ndarray, right: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    left_u, left_valid = l2_normalize(left, floor)
    right_u, right_valid = l2_normalize(right, floor)
    valid = left_valid & right_valid
    result = np.full(left.shape[0], np.nan, dtype=np.float64)
    result[valid] = np.einsum("ij,ij->i", left_u[valid], right_u[valid])
    return np.clip(result, -1.0, 1.0)


def mean_resultant_length(vectors: np.ndarray, floor: float = 1e-12) -> float:
    units, valid = l2_normalize(vectors, floor)
    if not np.any(valid):
        return float("nan")
    return float(np.linalg.norm(np.mean(units[valid], axis=0)))


def direction_kernel(trajectories: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Mean matched-source cosine between dose displacements.

    ``trajectories`` has shape (source, dose, feature), with dose zero omitted.
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)
    sources, doses, _ = trajectories.shape
    units, valid = l2_normalize(trajectories.reshape(sources * doses, -1), floor)
    units = units.reshape(sources, doses, -1)
    valid = valid.reshape(sources, doses)
    kernel = np.full((doses, doses), np.nan, dtype=np.float64)
    for a in range(doses):
        for b in range(doses):
            keep = valid[:, a] & valid[:, b]
            if np.any(keep):
                kernel[a, b] = float(
                    np.mean(np.einsum("ij,ij->i", units[keep, a], units[keep, b]))
                )
    return np.clip(kernel, -1.0, 1.0)


def trajectory_straightness(points: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Endpoint distance divided by piecewise-linear path length per source."""
    points = np.asarray(points, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=1), axis=-1)
    path_length = segment_lengths.sum(axis=1)
    endpoint = np.linalg.norm(points[:, -1] - points[:, 0], axis=-1)
    result = np.full(points.shape[0], np.nan, dtype=np.float64)
    valid = path_length > floor
    result[valid] = endpoint[valid] / path_length[valid]
    return np.clip(result, 0.0, 1.0)


def tangent_vectors(points: np.ndarray, doses: np.ndarray) -> np.ndarray:
    """Finite-difference tangents on a possibly irregular dose grid."""
    points = np.asarray(points, dtype=np.float64)
    doses = np.asarray(doses, dtype=np.float64)
    if points.ndim != 3 or points.shape[1] != len(doses):
        raise ValueError("points must be source x dose x feature")
    if len(doses) < 3 or np.any(np.diff(doses) <= 0):
        raise ValueError("at least three strictly increasing doses are required")
    return np.gradient(points, doses, axis=1, edge_order=2)


def curvature_metrics(points: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent-segment turn cosine and normalized direction change."""
    segments = np.diff(np.asarray(points, dtype=np.float64), axis=1)
    sources, count, _ = segments.shape
    if count < 2:
        return np.empty((sources, 0)), np.empty((sources, 0))
    cosines = np.full((sources, count - 1), np.nan)
    normalized = np.full_like(cosines, np.nan)
    for index in range(count - 1):
        first, first_valid = l2_normalize(segments[:, index], floor)
        second, second_valid = l2_normalize(segments[:, index + 1], floor)
        valid = first_valid & second_valid
        cosines[valid, index] = np.einsum(
            "ij,ij->i", first[valid], second[valid]
        )
        normalized[valid, index] = np.linalg.norm(
            second[valid] - first[valid], axis=1
        )
    return np.clip(cosines, -1.0, 1.0), normalized


def svd_energy_rank(matrix: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 4, 8)) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.size == 0:
        return {"energy": {k: float("nan") for k in ks}, "effective_rank": float("nan")}
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= EPS:
        return {"energy": {k: float("nan") for k in ks}, "effective_rank": float("nan")}
    probabilities = energy / total
    positive = probabilities > 0
    effective_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    cumulative = np.cumsum(probabilities)
    return {
        "energy": {k: float(cumulative[min(k, len(cumulative)) - 1]) for k in ks},
        "effective_rank": effective_rank,
        "singular_values": singular,
    }


def projection_fraction(vectors: np.ndarray, basis: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    denominator = np.einsum("ij,ij->i", vectors, vectors)
    numer = np.sum((vectors @ basis) ** 2, axis=1)
    result = np.full(len(vectors), np.nan)
    valid = denominator > floor**2
    result[valid] = numer[valid] / denominator[valid]
    return np.clip(result, 0.0, 1.0)


def orthonormal_basis(matrix: np.ndarray, k: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if not 1 <= k <= min(matrix.shape):
        raise ValueError(f"invalid subspace dimension k={k} for {matrix.shape}")
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return vh[:k].T


def interaction_fraction(delta_a: np.ndarray, delta_b: np.ndarray, delta_ab: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    residual = np.asarray(delta_ab) - np.asarray(delta_a) - np.asarray(delta_b)
    denominator = np.sum(np.asarray(delta_ab, dtype=np.float64) ** 2, axis=-1)
    result = np.full(denominator.shape, np.nan)
    valid = denominator > floor**2
    result[valid] = np.sum(residual[valid] ** 2, axis=-1) / denominator[valid]
    return result


def ray_reconstruction_error(displacements: np.ndarray, direction: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), floor)
    values = np.asarray(displacements, dtype=np.float64)
    projection = np.outer(values @ direction, direction)
    denominator = np.sum(values**2, axis=1)
    result = np.full(len(values), np.nan)
    valid = denominator > floor**2
    result[valid] = np.sum((values[valid] - projection[valid]) ** 2, axis=1) / denominator[valid]
    return result
