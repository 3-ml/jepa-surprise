#!/usr/bin/env python3
"""Reproduce and extend the IntPhys2 latent geometry audit on cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from analyze_intphys2_layerwise_geometry import (
        compute_layer_geometry as legacy_compute_layer_geometry,
    )
    from analyze_intphys2_layerwise_geometry import metadata_from_payload as legacy_metadata_from_payload
    from analyze_intphys2_layerwise_geometry import load_probe_curves as legacy_load_probe_curves
except ModuleNotFoundError:
    from scripts.analyze_intphys2_layerwise_geometry import (
        compute_layer_geometry as legacy_compute_layer_geometry,
    )
    from scripts.analyze_intphys2_layerwise_geometry import (
        load_probe_curves as legacy_load_probe_curves,
    )
    from scripts.analyze_intphys2_layerwise_geometry import (
        metadata_from_payload as legacy_metadata_from_payload,
    )


PRIMARY_K_VALUES = (1, 2, 4, 8, 16)
SCENE_BOOTSTRAP_DRAWS = 10_000
NULL_DRAWS = 2_000
RANDOM_SUBSPACE_DRAWS = 1_000
PAIRWISE_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
PLOT_DPI = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/geometry_audit"),
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=SCENE_BOOTSTRAP_DRAWS,
    )
    parser.add_argument(
        "--null-draws",
        type=int,
        default=NULL_DRAWS,
    )
    parser.add_argument(
        "--random-subspace-draws",
        type=int,
        default=RANDOM_SUBSPACE_DRAWS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = l2_normalize(x)
    y_norm = l2_normalize(y)
    return 1.0 - np.sum(x_norm * y_norm, axis=1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sum(values * weights) / max(np.sum(weights), 1e-12))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * np.sum(weights)
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, len(values) - 1)
    return float(values[index])


def ci_from_draws(draws: np.ndarray) -> tuple[float, float]:
    if draws.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def mean_pairwise_cosine_from_unit(unit_vectors: np.ndarray) -> float:
    n = unit_vectors.shape[0]
    if n < 2:
        return float("nan")
    summed = unit_vectors.sum(axis=0)
    return float((np.dot(summed, summed) - n) / (n * (n - 1)))


def mean_resultant_length(unit_vectors: np.ndarray) -> float:
    n = unit_vectors.shape[0]
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(unit_vectors.sum(axis=0)) / n)


def implied_pairwise_from_resultant(c_value: float, n: int) -> float:
    if n < 2:
        return float("nan")
    return float((n * (c_value**2) - 1.0) / (n - 1.0))


def pairwise_cosine_distribution(unit_vectors: np.ndarray) -> np.ndarray:
    n = unit_vectors.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float32)
    gram = unit_vectors @ unit_vectors.T
    upper = np.triu_indices(n, k=1)
    return gram[upper].astype(np.float32, copy=False)


def principal_overlap_score(basis_a: np.ndarray, basis_b: np.ndarray, k: int) -> float:
    if basis_a.shape[1] < k or basis_b.shape[1] < k or k <= 0:
        return float("nan")
    singular_values = np.linalg.svd(
        basis_a[:, :k].T @ basis_b[:, :k], compute_uv=False
    )
    return float(np.mean(singular_values[:k] ** 2))


def orthonormal_basis_from_rows(matrix: np.ndarray, max_rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.empty((matrix.shape[1], 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    if max_rank is not None:
        singular_values = singular_values[:max_rank]
        vt = vt[:max_rank]
    basis = vt.T.astype(np.float32, copy=False)
    return basis, singular_values.astype(np.float32, copy=False)


def participation_ratio(singular_values: np.ndarray) -> float:
    energies = singular_values.astype(np.float64) ** 2
    if energies.size == 0 or float(np.sum(energies)) <= 0.0:
        return float("nan")
    return float((np.sum(energies) ** 2) / np.sum(energies**2))


def explained_energy_curve(singular_values: np.ndarray, ks: Iterable[int]) -> dict[int, float]:
    energies = singular_values.astype(np.float64) ** 2
    total = float(np.sum(energies))
    result: dict[int, float] = {}
    if total <= 0.0:
        return {int(k): float("nan") for k in ks}
    cumulative = np.cumsum(energies)
    for k in ks:
        kk = min(int(k), len(cumulative))
        if kk <= 0:
            result[int(k)] = float("nan")
        else:
            result[int(k)] = float(cumulative[kk - 1] / total)
    return result


def random_orthonormal_basis(rng: np.random.Generator, dim: int, k: int) -> np.ndarray:
    matrix = rng.standard_normal((dim, k), dtype=np.float32)
    q, _ = np.linalg.qr(matrix)
    return q[:, :k].astype(np.float32, copy=False)


def scene_bootstrap_stat(
    frame: pd.DataFrame,
    value_column: str,
    *,
    scene_column: str = "scene_id",
    stat: str = "mean",
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    if frame.empty:
        return float("nan"), np.empty(0, dtype=np.float32)
    scenes = sorted(frame[scene_column].astype(str).unique().tolist())
    if not scenes:
        return float("nan"), np.empty(0, dtype=np.float32)
    per_scene = []
    for scene in scenes:
        values = frame.loc[frame[scene_column].astype(str) == scene, value_column].to_numpy(
            dtype=np.float64
        )
        per_scene.append(values)
    observed_values = np.concatenate(per_scene)
    if stat == "mean":
        observed = float(np.mean(observed_values))
    elif stat == "median":
        weights = np.concatenate(
            [np.ones(len(values), dtype=np.float64) for values in per_scene]
        )
        observed = weighted_median(observed_values, weights)
    else:
        raise ValueError(stat)
    indices = np.arange(len(scenes))
    draws_out = np.empty(draws, dtype=np.float32)
    for draw_index in range(draws):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sample_values = np.concatenate([per_scene[index] for index in sampled])
        if stat == "mean":
            draws_out[draw_index] = float(np.mean(sample_values))
        else:
            draws_out[draw_index] = float(
                weighted_median(
                    sample_values,
                    np.ones(len(sample_values), dtype=np.float64),
                )
            )
    return observed, draws_out


def tdc_scene_bootstrap(
    unit_vectors: np.ndarray,
    scenes: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    if len(unit_vectors) < 2:
        return float("nan"), np.empty(0), float("nan"), np.empty(0)
    unique_scenes = sorted({str(scene) for scene in scenes.tolist()})
    scene_vectors = {
        scene: unit_vectors[scenes == scene] for scene in unique_scenes
    }
    observed_mean = mean_pairwise_cosine_from_unit(unit_vectors)
    observed_c = mean_resultant_length(unit_vectors)
    indices = np.arange(len(unique_scenes))
    mean_draws = np.empty(draws, dtype=np.float32)
    c_draws = np.empty(draws, dtype=np.float32)
    for draw_index in range(draws):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        sample = np.concatenate([scene_vectors[unique_scenes[index]] for index in sampled], axis=0)
        mean_draws[draw_index] = mean_pairwise_cosine_from_unit(sample)
        c_draws[draw_index] = mean_resultant_length(sample)
    return observed_mean, mean_draws, observed_c, c_draws


def estimate_quantization_threshold(
    base_vectors: np.ndarray,
    moved_vectors: np.ndarray,
    *,
    sigma_multiplier: float = 5.0,
) -> np.ndarray:
    # Float16 roundoff lower bound: unit roundoff / sqrt(12) times vector norm.
    half_ulp = 2.0 ** -11
    scale = half_ulp / math.sqrt(12.0)
    sigma = scale * np.sqrt(
        np.sum(base_vectors.astype(np.float64) ** 2, axis=1)
        + np.sum(moved_vectors.astype(np.float64) ** 2, axis=1)
    )
    return (sigma_multiplier * sigma).astype(np.float32, copy=False)


@dataclass
class LoadedModel:
    key: str
    label: str
    representations_path: Path
    probe_summary_path: Path
    samples_path: Path
    payload: dict[str, np.ndarray]
    metadata: pd.DataFrame
    x_raw: np.ndarray
    x_norm: np.ndarray
    layer_names: list[str]
    transforms: list[str]
    clean_name: str


def load_model(key: str, label: str, base_dir: Path) -> LoadedModel:
    representations_path = base_dir / key / "representations.npz"
    probe_summary_path = base_dir / key / "probes" / "probe_summary.csv"
    samples_path = base_dir / key / "samples.csv"
    payload = load_npz(representations_path)
    metadata = legacy_metadata_from_payload(payload)
    x_raw = payload["X"].astype(np.float32, copy=False)
    x_norm = np.stack([l2_normalize(x_raw[:, layer_index, :]) for layer_index in range(x_raw.shape[1])], axis=1)
    transforms = sorted(
        value
        for value in metadata["appearance_condition"].astype(str).unique().tolist()
        if value != "clean"
    )
    return LoadedModel(
        key=key,
        label=label,
        representations_path=representations_path,
        probe_summary_path=probe_summary_path,
        samples_path=samples_path,
        payload=payload,
        metadata=metadata,
        x_raw=x_raw,
        x_norm=x_norm,
        layer_names=[str(value) for value in payload["layer_names"].tolist()],
        transforms=transforms,
        clean_name="clean",
    )


def reproduce_legacy_geometry(model: LoadedModel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    layer_geometry, _, _ = legacy_compute_layer_geometry(model.payload, model.metadata)
    probe_summary = pd.read_csv(model.probe_summary_path)
    probe_curves = legacy_load_probe_curves(probe_summary)
    layer_metrics = layer_geometry.merge(
        probe_curves, on=["layer_index", "layer_name"], how="left"
    )
    expected_path = model.representations_path.parent / "geometry" / "layerwise_geometry_metrics.csv"
    expected = pd.read_csv(expected_path)
    merged = expected.merge(
        layer_metrics,
        on=["layer_index", "layer_name"],
        suffixes=("_expected", "_reproduced"),
    )
    numeric_failures = []
    notes = []
    max_abs_diff = 0.0
    checked_columns = []
    for column in expected.columns:
        if column in {"layer_index", "layer_name"}:
            continue
        candidate = f"{column}_expected"
        reproduced = f"{column}_reproduced"
        if candidate not in merged or reproduced not in merged:
            continue
        checked_columns.append(column)
        if pd.api.types.is_numeric_dtype(expected[column]):
            expected_values = merged[candidate].to_numpy(dtype=np.float64)
            reproduced_values = merged[reproduced].to_numpy(dtype=np.float64)
            diffs = np.abs(expected_values - reproduced_values)
            if diffs.size:
                max_abs_diff = max(max_abs_diff, float(np.nanmax(diffs)))
            relative = diffs / np.maximum(np.abs(expected_values), 1e-12)
            if column == "appearance_retrieval_top1":
                tolerance = 2.5 / 456.0
                if float(np.nanmax(diffs)) > tolerance:
                    numeric_failures.append(column)
                elif float(np.nanmax(diffs)) > 0.0:
                    notes.append(
                        f"{column}: <= {float(np.nanmax(diffs)):.6f} absolute drift, consistent with one-to-two query tie-break changes out of 456 transformed queries"
                    )
            else:
                if not np.allclose(
                    expected_values,
                    reproduced_values,
                    atol=1e-4,
                    rtol=7.5e-3,
                    equal_nan=True,
                ):
                    numeric_failures.append(column)
                elif float(np.nanmax(diffs)) > 0.0:
                    notes.append(
                        f"{column}: max_abs={float(np.nanmax(diffs)):.3e}, max_rel={float(np.nanmax(relative)):.3e}"
                    )
        else:
            expected_values = merged[candidate].astype(str).to_numpy()
            reproduced_values = merged[reproduced].astype(str).to_numpy()
            if not np.array_equal(expected_values, reproduced_values):
                numeric_failures.append(column)
    summary = pd.DataFrame(
        [
            {
                "model": model.label,
                "rows": int(len(expected)),
                "checked_columns": len(checked_columns),
                "max_abs_diff": max_abs_diff,
                "pass": not numeric_failures,
                "failed_columns": ",".join(numeric_failures),
                "notes": " | ".join(notes),
                "expected_path": str(expected_path),
            }
        ]
    )
    return summary, expected, layer_metrics


def build_sample_maps(metadata: pd.DataFrame) -> dict[str, dict]:
    samples = metadata.reset_index().rename(columns={"index": "row_index"}).copy()
    samples["appearance_condition"] = samples["appearance_condition"].astype(str)
    samples["scene_id"] = samples["scene_id"].astype(str)
    name_groups = {}
    for name, group in samples.groupby("name", sort=True):
        group = group.sort_values("appearance_condition")
        name_groups[str(name)] = {
            "scene_id": str(group["scene_id"].iloc[0]),
            "pair_id": str(group["pair_id"].iloc[0]),
            "pair_label": str(group["pair_label"].iloc[0]),
            "rows_by_condition": {
                str(row["appearance_condition"]): int(row["row_index"])
                for _, row in group.iterrows()
            },
        }
    physical_groups = {}
    for (pair_id, appearance_condition), group in samples.groupby(
        ["pair_id", "appearance_condition"], sort=True
    ):
        group = group.sort_values("pair_label")
        if set(group["pair_label"].astype(str).tolist()) != {"impossible", "possible"}:
            continue
        physical_groups[(str(pair_id), str(appearance_condition))] = {
            "scene_id": str(group["scene_id"].iloc[0]),
            "possible_index": int(
                group.loc[group["pair_label"].eq("possible"), "row_index"].iloc[0]
            ),
            "impossible_index": int(
                group.loc[group["pair_label"].eq("impossible"), "row_index"].iloc[0]
            ),
            "possible_name": str(
                group.loc[group["pair_label"].eq("possible"), "name"].iloc[0]
            ),
            "impossible_name": str(
                group.loc[group["pair_label"].eq("impossible"), "name"].iloc[0]
            ),
        }
    return {"by_name": name_groups, "physical": physical_groups}


def prepare_distance_tables(model: LoadedModel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = build_sample_maps(model.metadata)
    clean_indices = np.asarray(
        [
            info["rows_by_condition"][model.clean_name]
            for _, info in sorted(maps["by_name"].items())
        ],
        dtype=np.int64,
    )
    clean_names = np.asarray(sorted(maps["by_name"].keys()))
    clean_scenes = np.asarray(
        [maps["by_name"][name]["scene_id"] for name in clean_names], dtype=object
    )
    clean_pairs = np.asarray(
        [maps["by_name"][name]["pair_id"] for name in clean_names], dtype=object
    )
    appearance_rows: list[dict] = []
    physical_rows: list[dict] = []
    global_rows: list[dict] = []
    for layer_index, layer_name in enumerate(model.layer_names):
        clean_z = model.x_norm[clean_indices, layer_index, :]
        clean_distances = 1.0 - clean_z @ clean_z.T
        different_scene_mask = clean_scenes[:, None] != clean_scenes[None, :]
        valid = different_scene_mask & ~np.eye(len(clean_indices), dtype=bool)
        global_scale = float(np.median(clean_distances[valid]))
        local_scales = {}
        for idx, name in enumerate(clean_names.tolist()):
            mask = valid[idx]
            local_scales[name] = float(np.median(clean_distances[idx, mask]))
        global_rows.append(
            {
                "model": model.label,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "global_unrelated_clean_scale": global_scale,
                "n_unrelated_pairs": int(np.sum(valid)),
            }
        )
        for name, info in sorted(maps["by_name"].items()):
            base_index = info["rows_by_condition"][model.clean_name]
            base = model.x_norm[base_index, layer_index, :][None, :]
            local_scale = local_scales[name]
            for transform in model.transforms:
                moved_index = info["rows_by_condition"][transform]
                moved = model.x_norm[moved_index, layer_index, :][None, :]
                distance = float(cosine_distance(base, moved)[0])
                appearance_rows.append(
                    {
                        "model": model.label,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "scene_id": info["scene_id"],
                        "pair_id": info["pair_id"],
                        "source_id": name,
                        "transform": transform,
                        "dose": str(
                            model.metadata.loc[moved_index, "transform_value"]
                        ),
                        "distance": distance,
                        "global_scale": global_scale,
                        "local_scale": local_scale,
                        "global_normalized_distance": distance / max(global_scale, 1e-12),
                        "local_normalized_distance": distance / max(local_scale, 1e-12),
                    }
                )
        for (pair_id, appearance), info in sorted(maps["physical"].items()):
            possible = model.x_norm[info["possible_index"], layer_index, :][None, :]
            impossible = model.x_norm[info["impossible_index"], layer_index, :][None, :]
            distance = float(cosine_distance(possible, impossible)[0])
            local_scale = float(
                np.median(
                    [
                        local_scales[info["possible_name"]],
                        local_scales[info["impossible_name"]],
                    ]
                )
            )
            physical_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "scene_id": info["scene_id"],
                    "pair_id": pair_id,
                    "appearance_condition": appearance,
                    "distance": distance,
                    "global_scale": global_scale,
                    "local_scale": local_scale,
                    "global_normalized_distance": distance / max(global_scale, 1e-12),
                    "local_normalized_distance": distance / max(local_scale, 1e-12),
                }
            )
    return (
        pd.DataFrame(appearance_rows),
        pd.DataFrame(physical_rows),
        pd.DataFrame(global_rows),
    )


def same_source_displacements(model: LoadedModel, variant: str) -> pd.DataFrame:
    maps = build_sample_maps(model.metadata)
    x = model.x_norm if variant == "normalized" else model.x_raw
    rows = []
    for layer_index, layer_name in enumerate(model.layer_names):
        for name, info in sorted(maps["by_name"].items()):
            base_index = info["rows_by_condition"][model.clean_name]
            for transform in model.transforms:
                moved_index = info["rows_by_condition"][transform]
                base = x[base_index, layer_index, :]
                moved = x[moved_index, layer_index, :]
                delta = moved - base
                magnitude = float(np.linalg.norm(delta))
                floor = float(
                    estimate_quantization_threshold(
                        base[None, :], moved[None, :]
                    )[0]
                )
                rows.append(
                    {
                        "model": model.label,
                        "embedding_variant": variant,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "scene_id": info["scene_id"],
                        "pair_id": info["pair_id"],
                        "source_id": name,
                        "transform": transform,
                        "dose": str(model.metadata.loc[moved_index, "transform_value"]),
                        "base_index": base_index,
                        "moved_index": moved_index,
                        "magnitude": magnitude,
                        "noise_floor": floor,
                        "is_numerically_valid": bool(magnitude > floor),
                        "delta_index": len(rows),
                    }
                )
    frame = pd.DataFrame(rows)
    deltas = []
    for row in frame.itertuples(index=False):
        source = x[int(row.base_index), int(row.layer_index), :]
        target = x[int(row.moved_index), int(row.layer_index), :]
        deltas.append(target - source)
    frame["delta_vector"] = deltas
    return frame


def physical_displacements(model: LoadedModel, variant: str) -> pd.DataFrame:
    maps = build_sample_maps(model.metadata)
    x = model.x_norm if variant == "normalized" else model.x_raw
    rows = []
    for layer_index, layer_name in enumerate(model.layer_names):
        for (pair_id, appearance), info in sorted(maps["physical"].items()):
            possible = x[info["possible_index"], layer_index, :]
            impossible = x[info["impossible_index"], layer_index, :]
            delta = impossible - possible
            magnitude = float(np.linalg.norm(delta))
            floor = float(
                estimate_quantization_threshold(
                    possible[None, :], impossible[None, :]
                )[0]
            )
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": variant,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "scene_id": info["scene_id"],
                    "pair_id": pair_id,
                    "source_id": f"{pair_id}:{appearance}",
                    "transform": f"physics__{appearance}",
                    "dose": appearance,
                    "appearance_condition": appearance,
                    "possible_index": info["possible_index"],
                    "impossible_index": info["impossible_index"],
                    "magnitude": magnitude,
                    "noise_floor": floor,
                    "is_numerically_valid": bool(magnitude > floor),
                    "delta_vector": delta,
                }
            )
    return pd.DataFrame(rows)


def tdc_null_unrelated_clean(
    clean_vectors: np.ndarray,
    clean_scenes: np.ndarray,
    source_order: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_scene = {}
    for index, scene in enumerate(clean_scenes.tolist()):
        by_scene.setdefault(str(scene), []).append(index)
    scene_values = sorted(by_scene.keys())
    out = np.empty(draws, dtype=np.float32)
    for draw_index in range(draws):
        deltas = np.empty_like(clean_vectors[source_order])
        for sample_index, source_idx in enumerate(source_order.tolist()):
            source_scene = str(clean_scenes[source_idx])
            candidate_scenes = [scene for scene in scene_values if scene != source_scene]
            target_scene = candidate_scenes[int(rng.integers(0, len(candidate_scenes)))]
            target_candidates = by_scene[target_scene]
            target_idx = target_candidates[int(rng.integers(0, len(target_candidates)))]
            deltas[sample_index] = clean_vectors[target_idx] - clean_vectors[source_idx]
        magnitudes = np.linalg.norm(deltas, axis=1, keepdims=True)
        units = deltas / np.maximum(magnitudes, 1e-12)
        out[draw_index] = mean_pairwise_cosine_from_unit(units)
    return out


def tdc_null_shuffled_correspondence(
    base_vectors: np.ndarray,
    moved_vectors: np.ndarray,
    scenes: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scene_array = scenes.astype(str)
    scene_to_indices = {
        scene: np.flatnonzero(scene_array == scene).tolist()
        for scene in sorted(set(scene_array.tolist()))
    }
    all_indices = np.arange(len(scene_array))
    out = np.empty(draws, dtype=np.float32)
    for draw_index in range(draws):
        chosen = np.empty_like(all_indices)
        for idx in all_indices.tolist():
            candidates = [j for j in all_indices.tolist() if scene_array[j] != scene_array[idx]]
            chosen[idx] = candidates[int(rng.integers(0, len(candidates)))]
        deltas = moved_vectors[chosen] - base_vectors
        magnitudes = np.linalg.norm(deltas, axis=1, keepdims=True)
        units = deltas / np.maximum(magnitudes, 1e-12)
        out[draw_index] = mean_pairwise_cosine_from_unit(units)
    return out


def tdc_null_label_permutation(
    displacement_frame: pd.DataFrame,
    *,
    transform: str,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = displacement_frame[displacement_frame["transform"].isin(sorted(displacement_frame["transform"].unique()))]
    target = displacement_frame[displacement_frame["transform"] == transform]
    target_n = len(target)
    out = np.empty(draws, dtype=np.float32)
    pool_vectors = np.stack(pool["delta_vector"].to_numpy())
    for draw_index in range(draws):
        indices = rng.choice(len(pool_vectors), size=target_n, replace=False)
        sample = pool_vectors[indices]
        magnitudes = np.linalg.norm(sample, axis=1, keepdims=True)
        units = sample / np.maximum(magnitudes, 1e-12)
        out[draw_index] = mean_pairwise_cosine_from_unit(units)
    return out


def summarize_directional_coherence(
    model: LoadedModel,
    displacement_frame: pd.DataFrame,
    *,
    bootstrap_draws: int,
    null_draws: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    for layer_index, layer_name in enumerate(model.layer_names):
        clean_vectors = (
            model.x_norm[:, layer_index, :]
            if displacement_frame["embedding_variant"].iloc[0] == "normalized"
            else model.x_raw[:, layer_index, :]
        )
        clean_mask = model.metadata["appearance_condition"].astype(str).eq("clean").to_numpy()
        clean_vectors = clean_vectors[clean_mask]
        clean_scenes = model.metadata.loc[clean_mask, "scene_id"].astype(str).to_numpy()
        clean_names = model.metadata.loc[clean_mask, "name"].astype(str).to_numpy()
        clean_name_to_position = {
            name: index for index, name in enumerate(clean_names.tolist())
        }
        for transform, group in displacement_frame[
            displacement_frame["layer_index"].eq(layer_index)
        ].groupby("transform", sort=True):
            valid = group[group["is_numerically_valid"]].copy()
            if len(valid) < 2:
                continue
            vectors = np.stack(valid["delta_vector"].to_numpy())
            magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
            units = vectors / np.maximum(magnitudes, 1e-12)
            scenes = valid["scene_id"].astype(str).to_numpy()
            mean_cos, mean_draws, c_value, c_draws = tdc_scene_bootstrap(
                units, scenes, draws=bootstrap_draws, rng=rng
            )
            pairwise = pairwise_cosine_distribution(units)
            if "base_index" in valid and valid["source_id"].astype(str).isin(clean_name_to_position).all():
                source_order = valid["source_id"].astype(str).map(clean_name_to_position).to_numpy(dtype=np.int64)
            else:
                source_order = np.arange(len(valid), dtype=np.int64) % len(clean_vectors)
            unrelated_null = tdc_null_unrelated_clean(
                clean_vectors,
                clean_scenes,
                source_order,
                draws=null_draws,
                rng=rng,
            )
            if {"base_index", "moved_index"}.issubset(valid.columns):
                base_stack = np.stack(
                    [
                        (
                            model.x_norm[int(row.base_index), layer_index, :]
                            if displacement_frame["embedding_variant"].iloc[0] == "normalized"
                            else model.x_raw[int(row.base_index), layer_index, :]
                        )
                        for row in valid.itertuples()
                    ]
                )
                moved_stack = np.stack(
                    [
                        (
                            model.x_norm[int(row.moved_index), layer_index, :]
                            if displacement_frame["embedding_variant"].iloc[0] == "normalized"
                            else model.x_raw[int(row.moved_index), layer_index, :]
                        )
                        for row in valid.itertuples()
                    ]
                )
            else:
                base_stack = np.stack(
                    [
                        (
                            model.x_norm[int(row.possible_index), layer_index, :]
                            if displacement_frame["embedding_variant"].iloc[0] == "normalized"
                            else model.x_raw[int(row.possible_index), layer_index, :]
                        )
                        for row in valid.itertuples()
                    ]
                )
                moved_stack = np.stack(
                    [
                        (
                            model.x_norm[int(row.impossible_index), layer_index, :]
                            if displacement_frame["embedding_variant"].iloc[0] == "normalized"
                            else model.x_raw[int(row.impossible_index), layer_index, :]
                        )
                        for row in valid.itertuples()
                    ]
                )
            shuffled_null = tdc_null_shuffled_correspondence(
                base_stack,
                moved_stack,
                scenes,
                draws=null_draws,
                rng=rng,
            )
            if transform.startswith("physics__"):
                permuted_null = np.empty(0, dtype=np.float32)
            else:
                permuted_null = tdc_null_label_permutation(
                    displacement_frame[
                        displacement_frame["layer_index"].eq(layer_index)
                        & displacement_frame["is_numerically_valid"]
                    ],
                    transform=transform,
                    draws=null_draws,
                    rng=rng,
                )
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": displacement_frame["embedding_variant"].iloc[0],
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "transform": transform,
                    "n_total": int(len(group)),
                    "n_valid": int(len(valid)),
                    "n_below_noise_floor": int(len(group) - len(valid)),
                    "mean_pairwise_cosine": mean_cos,
                    "mean_pairwise_cosine_ci_low": ci_from_draws(mean_draws)[0],
                    "mean_pairwise_cosine_ci_high": ci_from_draws(mean_draws)[1],
                    "mean_resultant_length": c_value,
                    "mean_resultant_length_ci_low": ci_from_draws(c_draws)[0],
                    "mean_resultant_length_ci_high": ci_from_draws(c_draws)[1],
                    "pairwise_q10": float(np.quantile(pairwise, 0.1)),
                    "pairwise_q25": float(np.quantile(pairwise, 0.25)),
                    "pairwise_median": float(np.quantile(pairwise, 0.5)),
                    "pairwise_q75": float(np.quantile(pairwise, 0.75)),
                    "pairwise_q90": float(np.quantile(pairwise, 0.9)),
                    "tdc_resultant_identity_abs_error": abs(
                        mean_cos - implied_pairwise_from_resultant(c_value, len(valid))
                    ),
                    "null_unrelated_mean": float(np.mean(unrelated_null)),
                    "null_unrelated_ci_low": ci_from_draws(unrelated_null)[0],
                    "null_unrelated_ci_high": ci_from_draws(unrelated_null)[1],
                    "null_shuffled_mean": float(np.mean(shuffled_null)),
                    "null_shuffled_ci_low": ci_from_draws(shuffled_null)[0],
                    "null_shuffled_ci_high": ci_from_draws(shuffled_null)[1],
                    "null_permuted_mean": float(np.mean(permuted_null))
                    if permuted_null.size
                    else float("nan"),
                    "null_permuted_ci_low": ci_from_draws(permuted_null)[0]
                    if permuted_null.size
                    else float("nan"),
                    "null_permuted_ci_high": ci_from_draws(permuted_null)[1]
                    if permuted_null.size
                    else float("nan"),
                    "empirical_p_shuffled": float(
                        (1.0 + np.sum(shuffled_null >= mean_cos)) / (1.0 + len(shuffled_null))
                    ),
                    "z_dir_shuffled": float(
                        (mean_cos - np.mean(shuffled_null))
                        / max(np.std(np.concatenate([mean_draws, shuffled_null])), 1e-12)
                    ),
                }
            )
            for null_type, draws_array in [
                ("unrelated_clean", unrelated_null),
                ("shuffled_correspondence", shuffled_null),
                ("transform_label_permutation", permuted_null),
            ]:
                for draw_index, value in enumerate(draws_array.tolist()):
                    null_rows.append(
                        {
                            "model": model.label,
                            "embedding_variant": displacement_frame["embedding_variant"].iloc[0],
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "transform": transform,
                            "null_type": null_type,
                            "draw_index": draw_index,
                            "metric": "mean_pairwise_cosine",
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def summarize_alignment(
    model: LoadedModel,
    displacement_frame: pd.DataFrame,
    *,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    sample_maps = build_sample_maps(model.metadata)
    scene_lookup = {
        source_id: info["scene_id"]
        for source_id, info in sample_maps["by_name"].items()
    }
    for layer_index, layer_name in enumerate(model.layer_names):
        layer_group = displacement_frame[
            displacement_frame["layer_index"].eq(layer_index)
            & displacement_frame["is_numerically_valid"]
        ]
        if layer_group.empty:
            continue
        pivot = {}
        for transform, group in layer_group.groupby("transform", sort=True):
            pivot[transform] = {
                str(row.source_id): row.delta_vector for row in group.itertuples()
            }
        transform_names = sorted(pivot.keys())
        for left, right in combinations(transform_names, 2):
            shared = sorted(set(pivot[left]).intersection(pivot[right]))
            if len(shared) < 2:
                continue
            values = []
            scene_ids = []
            for source_id in shared:
                left_delta = pivot[left][source_id]
                right_delta = pivot[right][source_id]
                denom = max(
                    float(np.linalg.norm(left_delta) * np.linalg.norm(right_delta)),
                    1e-12,
                )
                values.append(float(np.dot(left_delta, right_delta) / denom))
                scene_ids.append(scene_lookup.get(source_id, source_id.split(":")[0].split("/")[-1]))
            frame = pd.DataFrame(
                {"scene_id": scene_ids, "value": values}
            )
            mean_value, mean_draws = scene_bootstrap_stat(
                frame,
                "value",
                draws=bootstrap_draws,
                rng=rng,
                stat="mean",
            )
            median_value, median_draws = scene_bootstrap_stat(
                frame,
                "value",
                draws=bootstrap_draws,
                rng=rng,
                stat="median",
            )
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": displacement_frame["embedding_variant"].iloc[0],
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "transform_left": left,
                    "transform_right": right,
                    "n_shared": len(shared),
                    "mean_alignment": mean_value,
                    "mean_alignment_ci_low": ci_from_draws(mean_draws)[0],
                    "mean_alignment_ci_high": ci_from_draws(mean_draws)[1],
                    "median_alignment": median_value,
                    "median_alignment_ci_low": ci_from_draws(median_draws)[0],
                    "median_alignment_ci_high": ci_from_draws(median_draws)[1],
                    "mean_abs_alignment": float(np.mean(np.abs(values))),
                }
            )
    return pd.DataFrame(rows)


def random_overlap_null(
    *,
    ambient_dim: int,
    k: int,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    scores = np.empty(draws, dtype=np.float32)
    for draw_index in range(draws):
        left = random_orthonormal_basis(rng, ambient_dim, k)
        right = random_orthonormal_basis(rng, ambient_dim, k)
        scores[draw_index] = principal_overlap_score(left, right, k)
    return scores


def summarize_translation(
    model: LoadedModel,
    displacement_frame: pd.DataFrame,
    *,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    scene_ids = sorted(displacement_frame["scene_id"].astype(str).unique().tolist())
    folds = {
        scene: index % 5 for index, scene in enumerate(scene_ids)
    }
    rows = []
    for layer_index, layer_name in enumerate(model.layer_names):
        for transform, group in displacement_frame[
            displacement_frame["layer_index"].eq(layer_index)
            & displacement_frame["is_numerically_valid"]
        ].groupby("transform", sort=True):
            fold_rows = []
            for fold in range(5):
                train = group[group["scene_id"].astype(str).map(folds) != fold]
                test = group[group["scene_id"].astype(str).map(folds) == fold]
                if train.empty or test.empty:
                    continue
                v_t = np.mean(np.stack(train["delta_vector"].to_numpy()), axis=0)
                for row in test.itertuples():
                    delta = row.delta_vector
                    denom = max(float(np.dot(delta, delta)), 1e-12)
                    relative_error = float(np.dot(delta - v_t, delta - v_t) / denom)
                    cosine = float(
                        np.dot(delta, v_t)
                        / max(np.linalg.norm(delta) * np.linalg.norm(v_t), 1e-12)
                    )
                    fold_rows.append(
                        {
                            "scene_id": str(row.scene_id),
                            "relative_translation_error": relative_error,
                            "translation_cosine": cosine,
                        }
                    )
            if not fold_rows:
                continue
            fold_frame = pd.DataFrame(fold_rows)
            mean_error, error_draws = scene_bootstrap_stat(
                fold_frame,
                "relative_translation_error",
                draws=bootstrap_draws,
                rng=rng,
                stat="mean",
            )
            mean_cosine, cosine_draws = scene_bootstrap_stat(
                fold_frame,
                "translation_cosine",
                draws=bootstrap_draws,
                rng=rng,
                stat="mean",
            )
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": displacement_frame["embedding_variant"].iloc[0],
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "transform": transform,
                    "mean_relative_translation_error": mean_error,
                    "mean_relative_translation_error_ci_low": ci_from_draws(error_draws)[0],
                    "mean_relative_translation_error_ci_high": ci_from_draws(error_draws)[1],
                    "mean_translation_cosine": mean_cosine,
                    "mean_translation_cosine_ci_low": ci_from_draws(cosine_draws)[0],
                    "mean_translation_cosine_ci_high": ci_from_draws(cosine_draws)[1],
                    "zero_baseline_relative_error": 1.0,
                    "improvement_over_zero": 1.0 - mean_error,
                }
            )
    return pd.DataFrame(rows)


def summarize_subspaces(
    model: LoadedModel,
    appearance_frame: pd.DataFrame,
    physics_frame: pd.DataFrame,
    *,
    bootstrap_draws: int,
    random_draws: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_rows = []
    overlap_rows = []
    pao_rows = []
    scene_ids = sorted(appearance_frame["scene_id"].astype(str).unique().tolist())
    folds = {scene: index % 5 for index, scene in enumerate(scene_ids)}
    for layer_index, layer_name in enumerate(model.layer_names):
        layer_app = appearance_frame[
            appearance_frame["layer_index"].eq(layer_index)
            & appearance_frame["embedding_variant"].eq("normalized")
            & appearance_frame["is_numerically_valid"]
        ]
        layer_phys = physics_frame[
            physics_frame["layer_index"].eq(layer_index)
            & physics_frame["embedding_variant"].eq("normalized")
            & physics_frame["appearance_condition"].eq("clean")
            & physics_frame["is_numerically_valid"]
        ]
        if layer_app.empty or layer_phys.empty:
            continue
        factor_bases = {}
        for transform, group in layer_app.groupby("transform", sort=True):
            matrix = np.stack(group["delta_vector"].to_numpy())
            basis, singular_values = orthonormal_basis_from_rows(
                matrix,
                max_rank=min(max(PRIMARY_K_VALUES), len(group)),
            )
            factor_bases[transform] = basis
            energy = explained_energy_curve(singular_values, PRIMARY_K_VALUES)
            factor_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "factor": transform,
                    "effective_rank_participation": participation_ratio(singular_values),
                    **{
                        f"energy_top_{k}": energy[k]
                        for k in PRIMARY_K_VALUES
                    },
                }
            )
        app_basis, app_singular = orthonormal_basis_from_rows(
            np.stack(layer_app["delta_vector"].to_numpy()),
            max_rank=min(max(PRIMARY_K_VALUES), len(layer_app)),
        )
        factor_rows.append(
            {
                "model": model.label,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "factor": "appearance_combined",
                "effective_rank_participation": participation_ratio(app_singular),
                **{
                    f"energy_top_{k}": explained_energy_curve(app_singular, PRIMARY_K_VALUES)[k]
                    for k in PRIMARY_K_VALUES
                },
            }
        )
        phys_basis, phys_singular = orthonormal_basis_from_rows(
            np.stack(layer_phys["delta_vector"].to_numpy()),
            max_rank=min(max(PRIMARY_K_VALUES), len(layer_phys)),
        )
        factor_rows.append(
            {
                "model": model.label,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "factor": "physics_clean",
                "effective_rank_participation": participation_ratio(phys_singular),
                **{
                    f"energy_top_{k}": explained_energy_curve(phys_singular, PRIMARY_K_VALUES)[k]
                    for k in PRIMARY_K_VALUES
                },
            }
        )
        combined_bases = dict(factor_bases)
        combined_bases["appearance_combined"] = app_basis
        combined_bases["physics_clean"] = phys_basis
        keys = sorted(combined_bases.keys())
        ambient_dim = app_basis.shape[0]
        random_overlap_by_k = {
            k: random_overlap_null(
                ambient_dim=ambient_dim, k=k, draws=random_draws, rng=rng
            )
            for k in PRIMARY_K_VALUES
            if min(
                [basis.shape[1] for basis in combined_bases.values()] + [ambient_dim]
            )
            >= k
        }
        for left, right in combinations(keys, 2):
            for k in PRIMARY_K_VALUES:
                if (
                    combined_bases[left].shape[1] < k
                    or combined_bases[right].shape[1] < k
                ):
                    continue
                observed = principal_overlap_score(
                    combined_bases[left], combined_bases[right], k
                )
                random_scores = random_overlap_by_k[k]
                overlap_rows.append(
                    {
                        "model": model.label,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "factor_left": left,
                        "factor_right": right,
                        "k": k,
                        "observed_overlap": observed,
                        "random_overlap_mean": float(np.mean(random_scores)),
                        "random_overlap_ci_low": ci_from_draws(random_scores)[0],
                        "random_overlap_ci_high": ci_from_draws(random_scores)[1],
                        "excess_over_random": observed - float(np.mean(random_scores)),
                    }
                )
        for k in PRIMARY_K_VALUES:
            if app_basis.shape[1] < k:
                continue
            scene_values = []
            for fold in range(5):
                train = layer_app[layer_app["scene_id"].astype(str).map(folds) != fold]
                test = layer_phys[layer_phys["scene_id"].astype(str).map(folds) == fold]
                if train.empty or test.empty:
                    continue
                train_basis, _ = orthonormal_basis_from_rows(
                    np.stack(train["delta_vector"].to_numpy()),
                    max_rank=k,
                )
                train_basis = train_basis[:, :k]
                for row in test.itertuples():
                    delta = row.delta_vector.astype(np.float64, copy=False)
                    denom = max(float(np.dot(delta, delta)), 1e-12)
                    proj = train_basis.T @ delta
                    pao = float(np.dot(proj, proj) / denom)
                    scene_values.append({"scene_id": str(row.scene_id), "value": pao})
            if not scene_values:
                continue
            pao_frame = pd.DataFrame(scene_values)
            observed, draws = scene_bootstrap_stat(
                pao_frame,
                "value",
                draws=bootstrap_draws,
                rng=rng,
                stat="mean",
            )
            random_scores = []
            for _ in range(random_draws):
                basis = random_orthonormal_basis(rng, ambient_dim, k)
                scene_means = []
                for row in layer_phys.itertuples():
                    delta = row.delta_vector.astype(np.float64, copy=False)
                    denom = max(float(np.dot(delta, delta)), 1e-12)
                    proj = basis.T @ delta
                    scene_means.append((str(row.scene_id), float(np.dot(proj, proj) / denom)))
                random_frame = pd.DataFrame(scene_means, columns=["scene_id", "value"])
                random_scores.append(
                    float(random_frame.groupby("scene_id", as_index=False)["value"].mean()["value"].mean())
                )
            random_scores = np.asarray(random_scores, dtype=np.float32)
            pao_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "k": k,
                    "pao_raw": observed,
                    "pao_ci_low": ci_from_draws(draws)[0],
                    "pao_ci_high": ci_from_draws(draws)[1],
                    "pao_random_mean": float(np.mean(random_scores)),
                    "pao_random_ci_low": ci_from_draws(random_scores)[0],
                    "pao_random_ci_high": ci_from_draws(random_scores)[1],
                    "pao_excess": observed - float(np.mean(random_scores)),
                    "pao_ratio": observed / max(float(np.mean(random_scores)), 1e-12),
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(overlap_rows), pd.DataFrame(pao_rows)


def run_synthetic_sanity_tests(rng: np.random.Generator) -> list[dict]:
    results = []
    # Case 1: identical directions -> TDC ~ 1.
    direction = np.zeros(8, dtype=np.float32)
    direction[0] = 1.0
    units = np.stack([direction for _ in range(12)])
    results.append(
        {
            "case": "same_direction",
            "metric": "mean_pairwise_cosine",
            "observed": mean_pairwise_cosine_from_unit(units),
            "expected": "near_1",
            "pass": mean_pairwise_cosine_from_unit(units) > 0.999,
        }
    )
    # Case 2: isotropic random directions -> near null.
    random = rng.standard_normal((256, 32), dtype=np.float32)
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    random_tdc = mean_pairwise_cosine_from_unit(random)
    results.append(
        {
            "case": "random_directions",
            "metric": "mean_pairwise_cosine",
            "observed": random_tdc,
            "expected": "near_0",
            "pass": abs(random_tdc) < 0.05,
        }
    )
    # Case 3/4: PAO orthogonal vs aligned.
    app = np.zeros((32, 2), dtype=np.float32)
    app[:2, :2] = np.eye(2, dtype=np.float32)
    phys_orth = np.zeros(32, dtype=np.float32)
    phys_orth[2] = 1.0
    phys_in = np.zeros(32, dtype=np.float32)
    phys_in[0] = 1.0
    results.append(
        {
            "case": "pao_orthogonal",
            "metric": "pao",
            "observed": float(np.sum((app.T @ phys_orth) ** 2) / np.sum(phys_orth**2)),
            "expected": "near_0",
            "pass": float(np.sum((app.T @ phys_orth) ** 2)) < 1e-6,
        }
    )
    results.append(
        {
            "case": "pao_inside",
            "metric": "pao",
            "observed": float(np.sum((app.T @ phys_in) ** 2) / np.sum(phys_in**2)),
            "expected": "near_1",
            "pass": abs(float(np.sum((app.T @ phys_in) ** 2) / np.sum(phys_in**2)) - 1.0)
            < 1e-6,
        }
    )
    # Case 5: global contraction with constant relative geometry.
    clean = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    moved = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    clean2 = clean * 0.5
    moved2 = moved * 0.5
    d1 = cosine_distance(clean, moved)
    d2 = cosine_distance(clean2, moved2)
    results.append(
        {
            "case": "cosine_scale_invariance",
            "metric": "distance",
            "observed": float(np.max(np.abs(d1 - d2))),
            "expected": "0",
            "pass": float(np.max(np.abs(d1 - d2))) < 1e-7,
        }
    )
    # Case 6: circular hue plane beats one-direction translation.
    angles = np.deg2rad(np.arange(0, 360, 45))
    trajectory = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    _, singular_values = orthonormal_basis_from_rows(trajectory)
    energy = explained_energy_curve(singular_values, [1, 2])
    results.append(
        {
            "case": "circular_hue_plane",
            "metric": "energy_top_2",
            "observed": energy[2],
            "expected": "near_1",
            "pass": energy[2] > 0.99 and energy[1] < 0.8,
        }
    )
    failing = [result for result in results if not result["pass"]]
    if failing:
        raise RuntimeError(f"Synthetic sanity tests failed: {failing}")
    return results


def draw_global_scale(global_scale: pd.DataFrame, output_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, group in global_scale.groupby("model", sort=True):
        ax.plot(
            group["layer_index"],
            group["global_unrelated_clean_scale"],
            marker="o",
            linewidth=2,
            label=model,
        )
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Median unrelated clean cosine distance")
    ax.set_title("Global contraction control")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=PLOT_DPI)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def draw_magnitude(appearance_distances: pd.DataFrame, physical_distances: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for model, group in appearance_distances.groupby("model", sort=True):
        summary = (
            group.groupby(["layer_index", "layer_name"], as_index=False)
            .agg(
                appearance_global_median=("global_normalized_distance", "median"),
                appearance_local_median=("local_normalized_distance", "median"),
            )
            .sort_values("layer_index")
        )
        axes[0].plot(summary["layer_index"], summary["appearance_global_median"], marker="o", label=f"{model} global")
        axes[0].plot(summary["layer_index"], summary["appearance_local_median"], marker="x", linestyle="--", label=f"{model} local")
    axes[0].set_title("Appearance magnitude")
    axes[0].set_ylabel("Normalized distance")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for model, group in physical_distances[physical_distances["appearance_condition"] == "clean"].groupby("model", sort=True):
        summary = (
            group.groupby(["layer_index", "layer_name"], as_index=False)
            .agg(
                physical_global_median=("global_normalized_distance", "median"),
                physical_local_median=("local_normalized_distance", "median"),
            )
            .sort_values("layer_index")
        )
        axes[1].plot(summary["layer_index"], summary["physical_global_median"], marker="o", label=f"{model} global")
        axes[1].plot(summary["layer_index"], summary["physical_local_median"], marker="x", linestyle="--", label=f"{model} local")
    axes[1].set_title("Physical magnitude")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Layer index")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=PLOT_DPI)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def draw_tdc(tdc_frame: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    normalized = tdc_frame[tdc_frame["embedding_variant"] == "normalized"]
    for model, model_group in normalized.groupby("model", sort=True):
        for transform, group in model_group[~model_group["transform"].str.startswith("physics__")].groupby("transform", sort=True):
            axes[0].plot(group["layer_index"], group["mean_pairwise_cosine"], marker="o", label=f"{model} {transform}")
            axes[0].fill_between(
                group["layer_index"],
                group["null_shuffled_mean"],
                group["null_shuffled_mean"],
                alpha=0.0,
            )
        physics = model_group[model_group["transform"] == "physics__clean"]
        if not physics.empty:
            axes[1].plot(physics["layer_index"], physics["mean_pairwise_cosine"], marker="o", label=f"{model} physics clean")
            axes[1].plot(physics["layer_index"], physics["null_shuffled_mean"], linestyle="--", label=f"{model} shuffled null")
    axes[0].set_ylabel("Appearance TDC")
    axes[1].set_ylabel("Physics TDC")
    axes[1].set_xlabel("Layer index")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=PLOT_DPI)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def draw_alignment_heatmaps(alignment: pd.DataFrame, output_dir: Path) -> None:
    normalized = alignment[alignment["embedding_variant"] == "normalized"]
    for (model, layer_name), group in normalized.groupby(["model", "layer_name"], sort=True):
        labels = sorted(set(group["transform_left"]).union(group["transform_right"]))
        matrix = np.eye(len(labels), dtype=np.float32)
        for row in group.itertuples():
            i = labels.index(row.transform_left)
            j = labels.index(row.transform_right)
            matrix[i, j] = row.mean_alignment
            matrix[j, i] = row.mean_alignment
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{model} {layer_name} mean alignment")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        prefix = output_dir / f"alignment_heatmap_{model.replace(' ', '_').lower()}_{layer_name}"
        fig.savefig(prefix.with_suffix(".png"), dpi=PLOT_DPI)
        fig.savefig(prefix.with_suffix(".pdf"))
        plt.close(fig)


def draw_overlap(overlap: pd.DataFrame, pao: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    target = overlap[
        (overlap["factor_left"] == "appearance_combined")
        & (overlap["factor_right"] == "physics_clean")
        & (overlap["k"] == 4)
    ]
    for model, group in target.groupby("model", sort=True):
        axes[0].plot(group["layer_index"], group["observed_overlap"], marker="o", label=f"{model} observed")
        axes[0].plot(group["layer_index"], group["random_overlap_mean"], linestyle="--", label=f"{model} random")
    axes[0].set_title("Subspace overlap k=4")
    axes[0].set_ylabel("Mean cos^2 principal angle")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    target_pao = pao[pao["k"] == 4]
    for model, group in target_pao.groupby("model", sort=True):
        axes[1].plot(group["layer_index"], group["pao_raw"], marker="o", label=f"{model} PAO")
        axes[1].plot(group["layer_index"], group["pao_random_mean"], linestyle="--", label=f"{model} random")
    axes[1].set_title("Physics-Appearance overlap k=4")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Layer index")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=PLOT_DPI)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def draw_spectra(factor_spectra: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, factor_name in zip(axes, ["appearance_combined", "physics_clean"]):
        subset = factor_spectra[factor_spectra["factor"] == factor_name]
        for model, group in subset.groupby("model", sort=True):
            axes_map = group.sort_values("layer_index")
            axes[0 if factor_name == "appearance_combined" else 1].plot(
                axes_map["layer_index"],
                axes_map["energy_top_1"],
                marker="o",
                label=f"{model} top1",
            )
            axes[0 if factor_name == "appearance_combined" else 1].plot(
                axes_map["layer_index"],
                axes_map["energy_top_4"],
                marker="x",
                linestyle="--",
                label=f"{model} top4",
            )
        axis.set_title(factor_name.replace("_", " "))
        axis.set_xlabel("Layer index")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Explained energy")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=PLOT_DPI)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def metric_recommendations() -> dict[str, str]:
    return {
        "global_normalized_distance": "KEEP AS SECONDARY",
        "local_normalized_distance": "KEEP AS SECONDARY",
        "Transformation Direction Coherence (TDC)": "KEEP AS PRIMARY",
        "mean_pairwise_directional_cosine": "KEEP AS PRIMARY",
        "cross_factor_directional_alignment": "KEEP AS SECONDARY",
        "principal_angle_factor_overlap": "KEEP AS SECONDARY",
        "Physics-Appearance Overlap (PAO)": "KEEP AS PRIMARY",
        "translation_generalization_error": "DIAGNOSTIC ONLY",
        "raw_cosine_distance_reanalysis": "REJECT",
    }


def write_metric_definitions(path: Path) -> None:
    recommendations = metric_recommendations()
    cards = [
        (
            "Global normalized distance",
            "D_global(i,T) = d(z_i,z_i^T) / G_l, with d(x,y)=1-cos(x,y) and G_l the median clean-clean unrelated-scene cosine distance at layer l.",
            "Relative displacement magnitude against a global unrelated-scene clean reference.",
            "Not directional coherence or subspace separation.",
            "Contracts with any global shrinkage; identical under raw vs normalized cosine distance.",
        ),
        (
            "Local normalized distance",
            "D_local(i,T) = d(z_i,z_i^T) / S_i, with S_i the median clean-clean unrelated-scene cosine distance from source i.",
            "Relative displacement magnitude against a source-conditioned unrelated reference.",
            "Not a direction metric; depends on the chosen local denominator for physical pairs.",
            "Can amplify noise for unusually isolated sources.",
        ),
        (
            "Transformation Direction Coherence (TDC)",
            "TDC(T) = mean_{i<j, scene_i != scene_j} u_i^T dot u_j^T where u_i^T = Delta_i^T / ||Delta_i^T||.",
            "Whether one factor induces a shared direction across scenes.",
            "Not invariance, not translation accuracy, not disentanglement.",
            "Undefined or unstable when displacement magnitudes are near the quantization floor.",
        ),
        (
            "Mean pairwise directional cosine",
            "Equivalent scalar summary of TDC for valid unit displacements.",
            "Average alignment of scene-distinct unit displacement vectors.",
            "Not robust without a null; raw values alone are hard to interpret in high dimension.",
            "Sensitive to a few coherent clusters that cancel globally.",
        ),
        (
            "Cross-factor directional alignment",
            "Alignment_i(A,B) = cos(Delta_i^A, Delta_i^B).",
            "Whether two factors move the same source in similar directions.",
            "Not evidence that two factors share the same low-dimensional subspace across scenes.",
            "Magnitude-zero directions make cosine unstable.",
        ),
        (
            "Principal-angle factor overlap",
            "FactorSubspaceOverlap(A,B;k) = mean_j cos^2(theta_j) for the top-k principal angles.",
            "Subspace orientation overlap for fixed k.",
            "Not causal factor disentanglement and not invariant to k or ambient dimension without a random-subspace baseline.",
            "Can look nonzero in high dimension even for unrelated subspaces.",
        ),
        (
            "Physics-Appearance Overlap (PAO)",
            "PAO_i = ||U_app^T Delta_phys||^2 / ||Delta_phys||^2 with U_app fit on train scenes only.",
            "How much held-out physical displacement energy lies in the learned appearance subspace.",
            "Not proof that appearance causes or explains physical readout differences.",
            "Depends on k and needs a random-subspace baseline; descriptive if fit/eval are not separated.",
        ),
        (
            "Translation generalization error",
            "||Delta_i^T - v_T||^2 / ||Delta_i^T||^2 with v_T fit on train scenes only.",
            "How well a global translation direction predicts held-out displacements.",
            "Not a full equivariance test and not enough to justify a linear-map claim.",
            "Can fail simply because a factor is multi-directional rather than unstructured.",
        ),
    ]
    lines = ["# Metric Definitions", ""]
    for title, equation, measures, not_measures, failure_modes in cards:
        recommendation = recommendations.get(title, recommendations.get(title.split(" (")[0], "DIAGNOSTIC ONLY"))
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Provisional name: {title}",
                f"- Exact equation: {equation}",
                f"- What it measures: {measures}",
                f"- What it does not measure: {not_measures}",
                f"- Expected null: problem-dependent; see report tables for bootstrap/permutation or random-subspace baselines.",
                f"- Failure modes: {failure_modes}",
                f"- Literature status: standard ingredients with project-specific adaptation and naming.",
                f"- Recommendation: {recommendation}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def best_five_findings(
    global_scale: pd.DataFrame,
    tdc: pd.DataFrame,
    overlap: pd.DataFrame,
    pao: pd.DataFrame,
    translation: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    for model in ["V-JEPA2-G", "VideoMAE2-G"]:
        scale = global_scale[global_scale["model"] == model].sort_values("layer_index")
        if not scale.empty:
            ratio = float(scale["global_unrelated_clean_scale"].iloc[-1] / scale["global_unrelated_clean_scale"].iloc[0])
            findings.append(
                f"{model}: global unrelated clean scale contracts by {ratio:.3f}x from {scale['layer_name'].iloc[0]} to {scale['layer_name'].iloc[-1]}."
            )
    target = tdc[
        (tdc["embedding_variant"] == "normalized")
        & (tdc["transform"] == "hue_090")
    ].sort_values(["mean_pairwise_cosine"], ascending=False)
    if not target.empty:
        row = target.iloc[0]
        findings.append(
            f"{row['model']} {row['transform']} peaks at {row['layer_name']} with TDC {row['mean_pairwise_cosine']:.3f} against shuffled-null mean {row['null_shuffled_mean']:.3f}."
        )
    target = pao[pao["k"] == 4].sort_values("pao_excess", ascending=False)
    if not target.empty:
        row = target.iloc[0]
        findings.append(
            f"{row['model']} PAO(k=4) is highest at {row['layer_name']}: raw {row['pao_raw']:.3f}, random {row['pao_random_mean']:.3f}, excess {row['pao_excess']:.3f}."
        )
    target = overlap[
        (overlap["factor_left"] == "appearance_combined")
        & (overlap["factor_right"] == "physics_clean")
        & (overlap["k"] == 4)
    ].sort_values("excess_over_random", ascending=False)
    if not target.empty:
        row = target.iloc[0]
        findings.append(
            f"{row['model']} appearance-vs-physics subspace overlap(k=4) peaks at {row['layer_name']} with excess-over-random {row['excess_over_random']:.3f}."
        )
    target = translation.sort_values("mean_relative_translation_error")
    if not target.empty:
        row = target.iloc[0]
        findings.append(
            f"{row['model']} translation test is best for {row['transform']} at {row['layer_name']}: relative error {row['mean_relative_translation_error']:.3f}, cosine {row['mean_translation_cosine']:.3f}."
        )
    return findings[:5]


def build_summary_json(
    reproduction_status: str,
    data_inventory: dict,
    findings: list[str],
    tdc: pd.DataFrame,
    pao: pd.DataFrame,
    overlap: pd.DataFrame,
    translation: pd.DataFrame,
) -> dict:
    recommendations = metric_recommendations()
    return {
        "reproduction": {"status": reproduction_status},
        "data_inventory": data_inventory,
        "main_findings": findings,
        "metric_cards": {
            key: {"recommendation": value}
            for key, value in recommendations.items()
        },
        "layerwise": {
            "tdc_peak_rows": tdc.sort_values("mean_pairwise_cosine", ascending=False)
            .head(10)
            .to_dict(orient="records"),
            "pao_peak_rows": pao.sort_values("pao_excess", ascending=False)
            .head(10)
            .to_dict(orient="records"),
            "overlap_peak_rows": overlap.sort_values("excess_over_random", ascending=False)
            .head(10)
            .to_dict(orient="records"),
        },
        "null_results": {
            "tdc": tdc[
                [
                    "model",
                    "embedding_variant",
                    "layer_name",
                    "transform",
                    "mean_pairwise_cosine",
                    "null_shuffled_mean",
                    "null_shuffled_ci_low",
                    "null_shuffled_ci_high",
                    "empirical_p_shuffled",
                ]
            ].to_dict(orient="records")
        },
        "robustness": {
            "translation": translation.head(20).to_dict(orient="records"),
            "normalized_vs_raw_tdc": tdc[
                [
                    "model",
                    "embedding_variant",
                    "layer_name",
                    "transform",
                    "mean_pairwise_cosine",
                ]
            ].to_dict(orient="records"),
        },
        "paper_claim_candidates": [
            "Global contraction with depth is measurable and must be shown alongside appearance-distance shrinkage.",
            "Any appearance/physics separation claim must be conditioned on random-subspace and directional-coherence nulls.",
        ],
        "claims_not_supported": [
            "Clean geometrical separation alone proves physical understanding.",
            "A global translation direction exists whenever appearance magnitude shrinks.",
            "Low appearance distance implies invariance.",
        ],
        "open_questions": [
            "Whether five extracted depths are enough to localize any mid-layer transition sharply.",
            "Whether full 41-layer cached features are needed for a stronger geometry-evolution story.",
        ],
    }


def write_reproduction_check(
    path: Path,
    summaries: pd.DataFrame,
    appearance_distances: pd.DataFrame,
    physical_distances: pd.DataFrame,
    global_scale: pd.DataFrame,
) -> str:
    status = "PASS" if summaries["pass"].all() else "FAIL"
    lines = [
        "# Reproduction Check",
        "",
        f"Status: {status}",
        "",
        "## Scope",
        "- Recomputed the current corrected-development layerwise geometry directly from the cached packed representation NPZ files.",
        "- Verified the exact appearance-condition set from code and manifests: `clean`, `hue_090`, `brightness_0p45`, `selective_recolor_180`.",
        "- Compared recomputed layerwise geometry metrics against the committed `geometry/layerwise_geometry_metrics.csv` outputs.",
        "",
        "## Result",
    ]
    for _, row in summaries.iterrows():
        row_pass = bool(row["pass"])
        lines.append(
            f"- {row['model']}: {'PASS' if row_pass else 'FAIL'}, rows={row['rows']}, checked_columns={row['checked_columns']}, max_abs_diff={row['max_abs_diff']:.3e}, failed_columns={row['failed_columns'] or 'none'}."
        )
    lines.extend(
        [
            "",
            "## Verified Current Geometry Inputs",
            f"- Appearance-distance rows: {len(appearance_distances)}",
            f"- Physical-distance rows: {len(physical_distances)}",
            f"- Global-scale rows: {len(global_scale)}",
            f"- Scene count in corrected development split: {appearance_distances['scene_id'].nunique()}",
            "",
            "## Extension Gate",
            "- Proceed with the extended geometry audit only because the legacy geometry metrics reproduced to numerical tolerance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return status


def write_run_commands(path: Path) -> None:
    lines = [
        "# Run Commands",
        "",
        "```powershell",
        "python scripts\\run_intphys2_geometry_audit.py --output-dir reports\\geometry_audit --overwrite",
        "pytest tests\\test_intphys2_geometry_audit_metrics.py",
        "```",
        "",
        "The audit uses only cached packed corrected-development representations under `results/24-00-17-intphys2-corrected-development-probes/`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_block(frame: pd.DataFrame, max_rows: int = 20) -> str:
    preview = frame.head(max_rows).copy()
    return "```text\n" + preview.to_string(index=False) + "\n```"


def write_main_report(
    path: Path,
    *,
    reproduction_status: str,
    findings: list[str],
    tdc: pd.DataFrame,
    pao: pd.DataFrame,
    overlap: pd.DataFrame,
    translation: pd.DataFrame,
    global_scale: pd.DataFrame,
    factor_spectra: pd.DataFrame,
) -> None:
    recommendations = metric_recommendations()
    strongest = findings[:10]
    negative = []
    raw_vs_norm = tdc.groupby(["model", "layer_name", "transform"]).filter(
        lambda frame: frame["embedding_variant"].nunique() == 2
    )
    if not raw_vs_norm.empty:
        pivot = raw_vs_norm.pivot_table(
            index=["model", "layer_name", "transform"],
            columns="embedding_variant",
            values="mean_pairwise_cosine",
        ).reset_index()
        diffs = pivot.assign(diff=(pivot["raw"] - pivot["normalized"]).abs()).sort_values(
            "diff", ascending=False
        )
        if not diffs.empty:
            row = diffs.iloc[0]
            negative.append(
                f"Raw-vs-normalized directional coherence diverges most for {row['model']} {row['transform']} at {row['layer_name']} by {row['diff']:.3f}; conclusions cannot silently collapse the two variants."
            )
    weak_tdc = tdc[tdc["mean_pairwise_cosine"] <= tdc["null_shuffled_ci_high"]]
    if not weak_tdc.empty:
        row = weak_tdc.sort_values("mean_pairwise_cosine").iloc[0]
        negative.append(
            f"{row['model']} {row['transform']} at {row['layer_name']} is at-or-below the shuffled TDC null ({row['mean_pairwise_cosine']:.3f} vs null high {row['null_shuffled_ci_high']:.3f})."
        )
    best_story = [
        "The data support a bounded geometry story: depth changes both the magnitude and direction structure of appearance and physical displacements, but global contraction and null baselines materially affect the interpretation.",
        "Any claim of appearance/physics separation must be phrased as descriptive overlap relative to matched nulls, not as proof of disentanglement or physical understanding.",
    ]
    dont_claim = [
        "appearance disappears",
        "low appearance distance means invariance",
        "subspace separation proves physical understanding",
        "a global transform translation exists whenever TDC is positive",
        "VideoMAE2 and V-JEPA2 differences isolate architecture or objective causally",
    ]
    next_steps = [
        "1. Recover or repack the full 41-layer corrected-development features if available; five layers are enough for trends but not for precise transition localization.",
        "2. Repeat the same audit on the sealed split only after freezing the exact metric set and reporting template.",
        "3. If the hue grid can be extracted cheaply, run the circular trajectory analysis as a second-stage experiment rather than inferring circular structure from a single 90-degree point.",
        "4. Only if more scenes are available, test a low-dimensional regularized affine map; the current sample size is too small for a high-capacity equivariant model.",
    ]
    lines = [
        "# Executive Result",
        "",
        "## Reproduction Status",
        reproduction_status,
        "",
        "## Strongest Findings",
    ]
    lines.extend([f"- {item}" for item in strongest])
    lines.extend(
        [
            "",
            "## Surprising / Negative Findings",
        ]
    )
    lines.extend([f"- {item}" for item in negative or ["No additional negative finding exceeded the report threshold beyond the main findings."]])
    lines.extend(["", "## Metric Recommendation"])
    for metric, recommendation in recommendations.items():
        lines.append(f"- {metric}: {recommendation}")
    lines.extend(["", "## Best Paper-Facing Story Supported by Data"])
    lines.extend([f"- {item}" for item in best_story])
    lines.extend(["", "## Claims We Must NOT Make"])
    lines.extend([f"- {item}" for item in dont_claim])
    lines.extend(["", "## Next Experiments Worth Running"])
    lines.extend(next_steps)
    lines.extend(
        [
            "",
            "# Detailed Sections",
            "",
            "## Global Contraction Control",
            dataframe_block(global_scale),
            "",
            "## Directional Coherence Highlights",
            dataframe_block(tdc.sort_values("mean_pairwise_cosine", ascending=False)),
            "",
            "## Physics-Appearance Overlap Highlights",
            dataframe_block(pao.sort_values("pao_excess", ascending=False)),
            "",
            "## Principal-Angle Overlap Highlights",
            dataframe_block(overlap.sort_values("excess_over_random", ascending=False)),
            "",
            "## Translation Diagnostic Highlights",
            dataframe_block(translation.sort_values("mean_relative_translation_error")),
            "",
            "## Factor Spectra Highlights",
            dataframe_block(factor_spectra),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(230902)

    base_dir = Path("results/24-00-17-intphys2-corrected-development-probes")
    models = [
        load_model("vjepa2", "V-JEPA2-G", base_dir),
        load_model("videomae2", "VideoMAE2-G", base_dir),
        load_model("pixel_summary", "PixelSummary", base_dir),
    ]
    sanity = run_synthetic_sanity_tests(rng)

    reproduction_summaries = []
    for model in models:
        summary, _, _ = reproduce_legacy_geometry(model)
        reproduction_summaries.append(summary)
    reproduction_summary = pd.concat(reproduction_summaries, ignore_index=True)

    appearance_frames = []
    physical_frames = []
    global_frames = []
    directional_frames = []
    null_frames = []
    alignment_frames = []
    translation_frames = []
    factor_frames = []
    overlap_frames = []
    pao_frames = []

    for model in models[:2]:
        appearance_distances, physical_distances, global_scale = prepare_distance_tables(model)
        appearance_frames.append(appearance_distances)
        physical_frames.append(physical_distances)
        global_frames.append(global_scale)
        for variant in ["normalized", "raw"]:
            appearance_disp = same_source_displacements(model, variant)
            physics_disp = physical_displacements(model, variant)
            tdc_appearance, null_appearance = summarize_directional_coherence(
                model, appearance_disp, bootstrap_draws=args.bootstrap_draws, null_draws=args.null_draws, rng=rng
            )
            tdc_physics, null_physics = summarize_directional_coherence(
                model, physics_disp, bootstrap_draws=args.bootstrap_draws, null_draws=args.null_draws, rng=rng
            )
            directional_frames.extend([tdc_appearance, tdc_physics])
            null_frames.extend([null_appearance, null_physics])
            alignment_frames.append(
                summarize_alignment(
                    model,
                    pd.concat([appearance_disp, physics_disp], ignore_index=True, sort=False),
                    bootstrap_draws=args.bootstrap_draws,
                    rng=rng,
                )
            )
            translation_frames.append(
                summarize_translation(
                    model, appearance_disp, bootstrap_draws=args.bootstrap_draws, rng=rng
                )
            )
            if variant == "normalized":
                factor_spectra, overlap, pao = summarize_subspaces(
                    model,
                    appearance_disp,
                    physics_disp,
                    bootstrap_draws=args.bootstrap_draws,
                    random_draws=args.random_subspace_draws,
                    rng=rng,
                )
                factor_frames.append(factor_spectra)
                overlap_frames.append(overlap)
                pao_frames.append(pao)

    appearance_distances = pd.concat(appearance_frames, ignore_index=True)
    physical_distances = pd.concat(physical_frames, ignore_index=True)
    global_scale = pd.concat(global_frames, ignore_index=True)
    tdc = pd.concat(directional_frames, ignore_index=True)
    nulls = pd.concat(null_frames, ignore_index=True)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    translation = pd.concat(translation_frames, ignore_index=True)
    factor_spectra = pd.concat(factor_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    pao = pd.concat(pao_frames, ignore_index=True)

    reproduction_status = write_reproduction_check(
        args.output_dir / "REPRODUCTION_CHECK.md",
        reproduction_summary,
        appearance_distances,
        physical_distances,
        global_scale,
    )
    if reproduction_status != "PASS":
        raise RuntimeError("Legacy geometry reproduction failed; stopping extension.")

    draw_global_scale(global_scale, plots_dir / "figure_a_global_scale")
    draw_magnitude(appearance_distances, physical_distances, plots_dir / "figure_b_magnitude")
    draw_tdc(tdc, plots_dir / "figure_c_tdc")
    draw_alignment_heatmaps(alignment, plots_dir)
    draw_overlap(overlap, pao, plots_dir / "figure_e_f_overlap")
    draw_spectra(factor_spectra, plots_dir / "figure_g_spectra")

    results_long_parts = [
        appearance_distances.assign(metric="appearance_distance"),
        physical_distances.assign(metric="physical_distance"),
        tdc.assign(metric="tdc_summary"),
        alignment.assign(metric="alignment_summary"),
        translation.assign(metric="translation_summary"),
        overlap.assign(metric="subspace_overlap"),
        pao.assign(metric="pao"),
    ]
    results_long = pd.concat(results_long_parts, ignore_index=True, sort=False)
    summary_statistics = pd.concat(
        [
            global_scale,
            tdc,
            alignment,
            translation,
            overlap,
            pao,
            factor_spectra,
        ],
        ignore_index=True,
        sort=False,
    )
    bootstrap_intervals = pd.concat(
        [
            tdc[
                [
                    "model",
                    "embedding_variant",
                    "layer_name",
                    "transform",
                    "mean_pairwise_cosine",
                    "mean_pairwise_cosine_ci_low",
                    "mean_pairwise_cosine_ci_high",
                    "mean_resultant_length",
                    "mean_resultant_length_ci_low",
                    "mean_resultant_length_ci_high",
                ]
            ],
            alignment[
                [
                    "model",
                    "embedding_variant",
                    "layer_name",
                    "transform_left",
                    "transform_right",
                    "mean_alignment",
                    "mean_alignment_ci_low",
                    "mean_alignment_ci_high",
                ]
            ],
            translation[
                [
                    "model",
                    "embedding_variant",
                    "layer_name",
                    "transform",
                    "mean_relative_translation_error",
                    "mean_relative_translation_error_ci_low",
                    "mean_relative_translation_error_ci_high",
                    "mean_translation_cosine",
                    "mean_translation_cosine_ci_low",
                    "mean_translation_cosine_ci_high",
                ]
            ],
            pao[
                [
                    "model",
                    "layer_name",
                    "k",
                    "pao_raw",
                    "pao_ci_low",
                    "pao_ci_high",
                    "pao_random_mean",
                    "pao_random_ci_low",
                    "pao_random_ci_high",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    findings = best_five_findings(global_scale, tdc, overlap, pao, translation)
    data_inventory = {
        "models": [model.label for model in models[:2]],
        "packed_feature_paths": {
            model.label: str(model.representations_path)
            for model in models
        },
        "scene_count": int(appearance_distances["scene_id"].nunique()),
        "pair_count": int(appearance_distances["pair_id"].nunique()),
        "source_count": int(appearance_distances["source_id"].nunique()),
        "appearance_conditions": sorted(
            appearance_distances["transform"].astype(str).unique().tolist()
        ),
        "layers": models[0].layer_names,
        "synthetic_sanity": sanity,
    }
    summary_json = build_summary_json(
        reproduction_status,
        data_inventory,
        findings,
        tdc,
        pao,
        overlap,
        translation,
    )

    write_metric_definitions(args.output_dir / "METRIC_DEFINITIONS.md")
    write_run_commands(args.output_dir / "RUN_COMMANDS.md")
    write_main_report(
        args.output_dir / "LATENT_GEOMETRY_AUDIT.md",
        reproduction_status=reproduction_status,
        findings=findings,
        tdc=tdc,
        pao=pao,
        overlap=overlap,
        translation=translation,
        global_scale=global_scale,
        factor_spectra=factor_spectra,
    )
    (args.output_dir / "RESULTS_SUMMARY.json").write_text(
        json.dumps(summary_json, indent=2) + "\n",
        encoding="utf-8",
    )
    results_long.to_csv(args.output_dir / "results_long.csv", index=False)
    summary_statistics.to_csv(args.output_dir / "summary_statistics.csv", index=False)
    bootstrap_intervals.to_csv(args.output_dir / "bootstrap_intervals.csv", index=False)
    nulls.to_csv(args.output_dir / "null_distributions.csv", index=False)

    print(str((args.output_dir / "LATENT_GEOMETRY_AUDIT.md").resolve()))
    print(str((args.output_dir / "RESULTS_SUMMARY.json").resolve()))
    print(reproduction_status)
    for finding in findings:
        print(finding)
    recommendations = metric_recommendations()
    keep_primary = sorted(
        [name for name, label in recommendations.items() if label == "KEEP AS PRIMARY"]
    )
    reject = sorted([name for name, label in recommendations.items() if label == "REJECT"])
    print("KEEP AS PRIMARY: " + ", ".join(keep_primary))
    print("REJECT: " + ", ".join(reject))
    print(
        "Blocker: only five cached corrected-development depths are available locally; full-depth geometry conclusions remain coarse unless the 41-layer corrected cache is recovered."
    )


if __name__ == "__main__":
    main()
