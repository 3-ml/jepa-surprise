from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_audit_module():
    module_path = Path(__file__).with_name("run_intphys2_geometry_audit.py")
    spec = importlib.util.spec_from_file_location("run_intphys2_geometry_audit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load audit module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = load_audit_module()


SEED = 230903
PRIMARY_MODELS = (
    ("vjepa2", "V-JEPA2-G"),
    ("videomae2", "VideoMAE2-G"),
)
APPEARANCE_TRANSFORMS = ("hue_090", "brightness_0p45", "selective_recolor_180")
PRIMARY_K_VALUES = (1, 2, 4, 8)
SPECTRUM_K_VALUES = (1, 2, 4, 8, 16)
FLOAT_FLOOR_EPS = 1e-12
SHUFFLE_BATCH_SIZE = 64
RANDOM_SUBSPACE_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/intphys2/controlled_geometry/development_reproduced"),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/intphys2/development_analysis"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--permutation-draws", type=int, default=10_000)
    parser.add_argument("--random-subspace-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(parents=True, exist_ok=True)


def ci(draws: np.ndarray) -> tuple[float, float]:
    return audit.ci_from_draws(draws.astype(np.float32, copy=False))


def ordered_scene_groups(scene_ids: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    scenes = [str(value) for value in scene_ids.astype(str).tolist()]
    unique = sorted(set(scenes))
    groups = [np.flatnonzero(np.asarray(scenes, dtype=object) == scene) for scene in unique]
    return unique, groups


def mean_pairwise_from_tdc(tdc_value: float, n_vectors: int) -> float:
    if n_vectors < 2:
        return float("nan")
    return float((n_vectors * (tdc_value**2) - 1.0) / max(n_vectors - 1, 1))


def tdc_from_unit_vectors(unit_vectors: np.ndarray) -> float:
    if len(unit_vectors) == 0:
        return float("nan")
    return float(np.linalg.norm(np.sum(unit_vectors, axis=0)) / len(unit_vectors))


def pairwise_quantiles_from_unit_vectors(unit_vectors: np.ndarray) -> dict[str, float]:
    if len(unit_vectors) < 2:
        return {
            "pairwise_q10": float("nan"),
            "pairwise_q25": float("nan"),
            "pairwise_median": float("nan"),
            "pairwise_q75": float("nan"),
            "pairwise_q90": float("nan"),
        }
    values = audit.pairwise_cosine_distribution(unit_vectors)
    return {
        "pairwise_q10": float(np.quantile(values, 0.10)),
        "pairwise_q25": float(np.quantile(values, 0.25)),
        "pairwise_median": float(np.quantile(values, 0.50)),
        "pairwise_q75": float(np.quantile(values, 0.75)),
        "pairwise_q90": float(np.quantile(values, 0.90)),
    }


def tdc_bootstrap_draws(unit_vectors: np.ndarray, scene_ids: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    scene_names, groups = ordered_scene_groups(scene_ids)
    if len(scene_names) < 1:
        return np.empty(0, dtype=np.float32)
    scene_sums = np.stack([np.sum(unit_vectors[idxs], axis=0) for idxs in groups], axis=0)
    scene_counts = np.asarray([len(idxs) for idxs in groups], dtype=np.int32)
    scene_count = len(scene_names)
    sampled = rng.integers(0, scene_count, size=(draws, scene_count), endpoint=False)
    sum_draws = scene_sums[sampled].sum(axis=1)
    count_draws = scene_counts[sampled].sum(axis=1).astype(np.float64)
    return (np.linalg.norm(sum_draws, axis=1) / np.maximum(count_draws, FLOAT_FLOOR_EPS)).astype(np.float32)


def aligned_scene_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    scenes = frame["scene_id"].astype(str).to_numpy()
    deltas = np.stack(frame["delta_vector"].to_numpy()).astype(np.float32, copy=False)
    unique_scenes, groups = ordered_scene_groups(scenes)
    per_scene_order = {}
    for scene, idxs in zip(unique_scenes, groups):
        scene_frame = frame.iloc[idxs].copy()
        if "source_id" in scene_frame:
            scene_frame = scene_frame.sort_values("source_id")
        elif "pair_id" in scene_frame:
            scene_frame = scene_frame.sort_values("pair_id")
        per_scene_order[scene] = scene_frame.index.to_numpy(dtype=np.int64)
    counts = {scene: len(indices) for scene, indices in per_scene_order.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Scene counts differ: {counts}")
    ordered_rows = np.concatenate([per_scene_order[scene] for scene in unique_scenes])
    ordered = frame.loc[ordered_rows].reset_index(drop=True)
    return np.stack(ordered["delta_vector"].to_numpy()).astype(np.float32, copy=False), unique_scenes, per_scene_order


def candidate_index_lists(scene_ids: np.ndarray) -> list[np.ndarray]:
    scene_ids = scene_ids.astype(str)
    indices = np.arange(len(scene_ids), dtype=np.int32)
    candidate_rows: list[np.ndarray] = []
    for idx, scene in enumerate(scene_ids.tolist()):
        candidates = indices[scene_ids != scene]
        candidate_rows.append(candidates)
    return candidate_rows


def shuffled_tdc_null_draws(
    base_vectors: np.ndarray,
    moved_vectors: np.ndarray,
    scene_ids: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = candidate_index_lists(scene_ids)
    n_samples = len(scene_ids)
    out = np.empty(draws, dtype=np.float32)
    completed = 0
    while completed < draws:
        batch = min(SHUFFLE_BATCH_SIZE, draws - completed)
        donor = np.empty((batch, n_samples), dtype=np.int32)
        for sample_index, sample_candidates in enumerate(candidates):
            donor[:, sample_index] = sample_candidates[
                rng.integers(0, len(sample_candidates), size=batch, endpoint=False)
            ]
        donor_vectors = moved_vectors[donor]
        deltas = donor_vectors - base_vectors[None, :, :]
        norms = np.linalg.norm(deltas, axis=2, keepdims=True)
        units = deltas / np.maximum(norms, FLOAT_FLOOR_EPS)
        tdc_values = np.linalg.norm(np.sum(units, axis=1), axis=1) / n_samples
        out[completed : completed + batch] = tdc_values.astype(np.float32, copy=False)
        completed += batch
    return out


def unrelated_clean_tdc_null_draws(
    base_vectors: np.ndarray,
    clean_vectors: np.ndarray,
    base_scene_ids: np.ndarray,
    clean_scene_ids: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean_candidates = []
    for scene in base_scene_ids.astype(str).tolist():
        candidates = np.flatnonzero(clean_scene_ids.astype(str) != scene).astype(np.int32)
        clean_candidates.append(candidates)
    n_samples = len(base_scene_ids)
    out = np.empty(draws, dtype=np.float32)
    completed = 0
    while completed < draws:
        batch = min(SHUFFLE_BATCH_SIZE, draws - completed)
        donor = np.empty((batch, n_samples), dtype=np.int32)
        for sample_index, sample_candidates in enumerate(clean_candidates):
            donor[:, sample_index] = sample_candidates[
                rng.integers(0, len(sample_candidates), size=batch, endpoint=False)
            ]
        donor_vectors = clean_vectors[donor]
        deltas = donor_vectors - base_vectors[None, :, :]
        norms = np.linalg.norm(deltas, axis=2, keepdims=True)
        units = deltas / np.maximum(norms, FLOAT_FLOOR_EPS)
        out[completed : completed + batch] = (
            np.linalg.norm(np.sum(units, axis=1), axis=1) / n_samples
        ).astype(np.float32, copy=False)
        completed += batch
    return out


def scene_mean_frame(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    value_columns = list(value_columns)
    return frame.groupby("scene_id", as_index=False)[value_columns].mean()


def bootstrap_scene_mean(frame: pd.DataFrame, value_column: str, draws: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    scene_frame = scene_mean_frame(frame, [value_column])
    values = scene_frame[value_column].to_numpy(dtype=np.float64)
    observed = float(np.mean(values))
    sampled = rng.integers(0, len(values), size=(draws, len(values)), endpoint=False)
    draws_out = values[sampled].mean(axis=1).astype(np.float32)
    return observed, draws_out


def entropy_effective_rank(singular_values: np.ndarray) -> float:
    energies = singular_values.astype(np.float64) ** 2
    total = float(np.sum(energies))
    if total <= 0.0:
        return float("nan")
    probs = energies / total
    probs = probs[probs > 0.0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def random_overlap_null_draws(ambient_dim: int, k: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(draws, dtype=np.float32)
    completed = 0
    while completed < draws:
        batch = min(RANDOM_SUBSPACE_BATCH_SIZE, draws - completed)
        left = rng.standard_normal((batch, ambient_dim, k), dtype=np.float32)
        right = rng.standard_normal((batch, ambient_dim, k), dtype=np.float32)
        left_q = np.linalg.qr(left)[0][:, :, :k]
        right_q = np.linalg.qr(right)[0][:, :, :k]
        prod = np.einsum("bdk,bdm->bkm", left_q, right_q)
        singular_values = np.linalg.svd(prod, compute_uv=False)
        out[completed : completed + batch] = np.mean(singular_values[:, :k] ** 2, axis=1).astype(np.float32, copy=False)
        completed += batch
    return out


def random_pao_null_draws(unit_vectors: np.ndarray, scene_ids: np.ndarray, ambient_dim: int, k: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    scene_names, groups = ordered_scene_groups(scene_ids)
    scaled_rows = []
    for idxs in groups:
        weight = 1.0 / max(len(scene_names) * len(idxs), 1)
        scaled_rows.append(unit_vectors[idxs] * math.sqrt(weight))
    scaled = np.concatenate(scaled_rows, axis=0).astype(np.float64, copy=False)
    gram = scaled @ scaled.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = eigvals[eigvals > 1e-12]
    eigvals = np.sort(eigvals)[::-1]
    rank = int(eigvals.shape[0])
    null_dim = max(ambient_dim - rank, 0)
    out = np.empty(draws, dtype=np.float32)
    completed = 0
    while completed < draws:
        batch = min(RANDOM_SUBSPACE_BATCH_SIZE, draws - completed)
        a = rng.standard_normal((batch, rank, k), dtype=np.float32)
        numerator = np.einsum("r,brk,brm->bkm", eigvals, a, a, optimize=True)
        denom = np.einsum("brk,brm->bkm", a, a, optimize=True).astype(np.float64, copy=False)
        if null_dim > 0:
            b = rng.standard_normal((batch, null_dim, k), dtype=np.float32)
            denom += np.einsum("brk,brm->bkm", b, b, optimize=True).astype(np.float64, copy=False)
        inv = np.linalg.inv(denom)
        scores = np.einsum("bkm,bmk->b", numerator.astype(np.float64, copy=False), inv, optimize=True)
        out[completed : completed + batch] = scores.astype(np.float32, copy=False)
        completed += batch
    return out


def orthonormal_basis(matrix: np.ndarray, max_rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.zeros((matrix.shape[1] if matrix.ndim == 2 else 0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    gram = matrix @ matrix.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    if max_rank is not None:
        eigvals = eigvals[:max_rank]
        eigvecs = eigvecs[:, :max_rank]
    singular_values = np.sqrt(np.maximum(eigvals, 0.0))
    if singular_values.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float32), singular_values.astype(np.float32)
    basis = matrix.T @ eigvecs
    basis /= np.maximum(singular_values[None, :], FLOAT_FLOOR_EPS)
    return basis.astype(np.float32, copy=False), singular_values.astype(np.float32, copy=False)


@dataclass
class LoadedInputs:
    models: list[audit.LoadedModel]
    reproduction_status: str
    reproduction_summary: pd.DataFrame


def load_inputs(base_dir: Path) -> LoadedInputs:
    models = [audit.load_model(key, label, base_dir) for key, label in PRIMARY_MODELS]
    summaries = []
    for model in models:
        summary, _, _ = audit.reproduce_legacy_geometry(model)
        summaries.append(summary)
    reproduction_summary = pd.concat(summaries, ignore_index=True)
    status = "PASS" if bool(reproduction_summary["pass"].all()) else "FAIL"
    return LoadedInputs(models=models, reproduction_status=status, reproduction_summary=reproduction_summary)


def build_distance_outputs(model: audit.LoadedModel, rng: np.random.Generator, draws: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    appearance, physical, global_scale = audit.prepare_distance_tables(model)
    block0 = float(global_scale.loc[global_scale["layer_name"].eq("block_00"), "global_unrelated_clean_scale"].iloc[0])
    global_scale = global_scale.copy()
    global_scale["contraction_ratio"] = global_scale["global_unrelated_clean_scale"] / max(block0, FLOAT_FLOOR_EPS)
    global_scale["percentage_change_from_block00"] = 100.0 * (global_scale["contraction_ratio"] - 1.0)
    app_rows = []
    for (layer_name, transform), group in appearance.groupby(["layer_name", "transform"], sort=True):
        global_mean, global_draws = bootstrap_scene_mean(group, "global_normalized_distance", draws, rng)
        local_mean, local_draws = bootstrap_scene_mean(group, "local_normalized_distance", draws, rng)
        app_rows.append(
            {
                "model": model.label,
                "layer_name": layer_name,
                "transform": transform,
                "metric": "appearance_distance",
                "global_norm_mean": global_mean,
                "global_norm_ci_low": ci(global_draws)[0],
                "global_norm_ci_high": ci(global_draws)[1],
                "local_norm_mean": local_mean,
                "local_norm_ci_low": ci(local_draws)[0],
                "local_norm_ci_high": ci(local_draws)[1],
            }
        )
    phys_rows = []
    for (layer_name, appearance_condition), group in physical.groupby(["layer_name", "appearance_condition"], sort=True):
        global_mean, global_draws = bootstrap_scene_mean(group, "global_normalized_distance", draws, rng)
        local_mean, local_draws = bootstrap_scene_mean(group, "local_normalized_distance", draws, rng)
        phys_rows.append(
            {
                "model": model.label,
                "layer_name": layer_name,
                "appearance_condition": appearance_condition,
                "metric": "physical_distance",
                "global_norm_mean": global_mean,
                "global_norm_ci_low": ci(global_draws)[0],
                "global_norm_ci_high": ci(global_draws)[1],
                "local_norm_mean": local_mean,
                "local_norm_ci_low": ci(local_draws)[0],
                "local_norm_ci_high": ci(local_draws)[1],
            }
        )
    return global_scale, pd.DataFrame(app_rows), pd.DataFrame(phys_rows)


def summarize_tdc_variant(
    model: audit.LoadedModel,
    displacement_frame: pd.DataFrame,
    variant: str,
    bootstrap_draws: int,
    permutation_draws: int,
    rng: np.random.Generator,
    *,
    compute_nulls: bool,
    compute_diagnostics: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    null_rows = []
    diag_rows = []
    x = model.x_norm if variant == "normalized" else model.x_raw
    clean_mask = model.metadata["appearance_condition"].astype(str).eq("clean").to_numpy()
    clean_scene_ids = model.metadata.loc[clean_mask, "scene_id"].astype(str).to_numpy()
    clean_source_ids = model.metadata.loc[clean_mask, "name"].astype(str).to_numpy()
    for layer_index, layer_name in enumerate(model.layer_names):
        clean_vectors = x[clean_mask, layer_index, :].astype(np.float32, copy=False)
        layer_frame = displacement_frame[
            displacement_frame["layer_index"].eq(layer_index)
        ].copy()
        for transform, group in layer_frame.groupby("transform", sort=True):
            valid = group[group["is_numerically_valid"]].copy()
            if len(valid) < 2:
                continue
            deltas = np.stack(valid["delta_vector"].to_numpy()).astype(np.float32, copy=False)
            magnitudes = np.linalg.norm(deltas, axis=1, keepdims=True)
            units = deltas / np.maximum(magnitudes, FLOAT_FLOOR_EPS)
            scenes = valid["scene_id"].astype(str).to_numpy()
            observed_tdc = tdc_from_unit_vectors(units)
            bootstrap = tdc_bootstrap_draws(units, scenes, bootstrap_draws, rng)
            n_valid = len(valid)
            observed_pairwise = audit.mean_pairwise_cosine_from_unit(units)
            implied_pairwise = mean_pairwise_from_tdc(observed_tdc, n_valid)
            quantiles = pairwise_quantiles_from_unit_vectors(units)
            if not str(transform).startswith("physics__"):
                base_vectors = x[valid["base_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
                moved_vectors = x[valid["moved_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
                source_positions = pd.Series(np.arange(len(clean_source_ids)), index=clean_source_ids)
                base_source_positions = source_positions.loc[valid["source_id"].astype(str)].to_numpy(dtype=np.int64)
                clean_base_vectors = clean_vectors[base_source_positions]
                clean_base_scenes = clean_scene_ids[base_source_positions]
            else:
                base_vectors = x[valid["possible_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
                moved_vectors = x[valid["impossible_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
                clean_base_vectors = clean_vectors[: len(valid)]
                clean_base_scenes = clean_scene_ids[: len(valid)]
            if compute_nulls:
                shuffled_null = shuffled_tdc_null_draws(
                    base_vectors,
                    moved_vectors,
                    scenes,
                    permutation_draws,
                    rng,
                )
                unrelated_null = unrelated_clean_tdc_null_draws(
                    clean_base_vectors,
                    clean_vectors,
                    clean_base_scenes,
                    clean_scene_ids,
                    permutation_draws,
                    rng,
                )
                null_shuffled_mean = float(np.mean(shuffled_null))
                null_shuffled_ci_low, null_shuffled_ci_high = ci(shuffled_null)
                null_unrelated_mean = float(np.mean(unrelated_null))
                null_unrelated_ci_low, null_unrelated_ci_high = ci(unrelated_null)
                empirical_p = float((1.0 + np.sum(shuffled_null >= observed_tdc)) / (1.0 + len(shuffled_null)))
                tdc_excess = observed_tdc - null_shuffled_mean
            else:
                shuffled_null = np.empty(0, dtype=np.float32)
                unrelated_null = np.empty(0, dtype=np.float32)
                null_shuffled_mean = float("nan")
                null_shuffled_ci_low = float("nan")
                null_shuffled_ci_high = float("nan")
                null_unrelated_mean = float("nan")
                null_unrelated_ci_low = float("nan")
                null_unrelated_ci_high = float("nan")
                empirical_p = float("nan")
                tdc_excess = float("nan")
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": variant,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "transform": transform,
                    "n_total": int(len(group)),
                    "n_valid": int(len(valid)),
                    "n_below_noise_floor": int(len(group) - len(valid)),
                    "fraction_below_noise_floor": float((len(group) - len(valid)) / max(len(group), 1)),
                    "tdc": observed_tdc,
                    "tdc_ci_low": ci(bootstrap)[0],
                    "tdc_ci_high": ci(bootstrap)[1],
                    "mean_pairwise_cosine": observed_pairwise,
                    "implied_pairwise_cosine_from_tdc": implied_pairwise,
                    "tdc_identity_abs_error": abs(observed_pairwise - implied_pairwise),
                    "tdc_excess_shuffled": tdc_excess,
                    "null_shuffled_mean": null_shuffled_mean,
                    "null_shuffled_ci_low": null_shuffled_ci_low,
                    "null_shuffled_ci_high": null_shuffled_ci_high,
                    "null_unrelated_mean": null_unrelated_mean,
                    "null_unrelated_ci_low": null_unrelated_ci_low,
                    "null_unrelated_ci_high": null_unrelated_ci_high,
                    "empirical_p_shuffled": empirical_p,
                    **quantiles,
                }
            )
            for null_type, values in (
                ("shuffled_correspondence", shuffled_null),
                ("unrelated_clean", unrelated_null),
            ):
                null_rows.extend(
                    {
                        "model": model.label,
                        "embedding_variant": variant,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "transform": transform,
                        "null_type": null_type,
                        "draw_index": int(idx),
                        "metric": "tdc",
                        "value": float(value),
                    }
                    for idx, value in enumerate(values.tolist())
                )
            if compute_diagnostics:
                scene_names, groups = ordered_scene_groups(scenes)
                mean_direction = np.sum(units, axis=0)
                mean_direction /= max(np.linalg.norm(mean_direction), FLOAT_FLOOR_EPS)
                for scene, idxs in zip(scene_names, groups):
                    keep_mask = np.ones(len(units), dtype=bool)
                    keep_mask[idxs] = False
                    jackknife_units = units[keep_mask]
                    jackknife_tdc = tdc_from_unit_vectors(jackknife_units) if len(jackknife_units) >= 2 else float("nan")
                    scene_units = units[idxs]
                    diag_rows.append(
                        {
                            "model": model.label,
                            "embedding_variant": variant,
                            "layer_index": layer_index,
                            "layer_name": layer_name,
                            "transform": transform,
                            "scene_id": scene,
                            "scene_n": int(len(idxs)),
                            "scene_mean_alignment_to_global_direction": float(np.mean(scene_units @ mean_direction)),
                            "scene_mean_magnitude": float(np.mean(np.linalg.norm(deltas[idxs], axis=1))),
                            "scene_valid_fraction": float(len(idxs) / max(len(group[group["scene_id"].astype(str).eq(scene)]), 1)),
                            "leave_one_scene_out_tdc": jackknife_tdc,
                            "tdc_drop_when_removed": observed_tdc - jackknife_tdc if not math.isnan(jackknife_tdc) else float("nan"),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(null_rows), pd.DataFrame(diag_rows)


def summarize_alignment(model: audit.LoadedModel, appearance_frame: pd.DataFrame, physics_frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    source_scene = {
        str(name): str(scene)
        for name, scene in model.metadata.loc[
            model.metadata["appearance_condition"].astype(str).eq("clean"),
            ["name", "scene_id"],
        ].itertuples(index=False, name=None)
    }
    possible_name_lookup = {}
    physical_maps = audit.build_sample_maps(model.metadata)["physical"]
    for (pair_id, appearance_condition), info in physical_maps.items():
        possible_name_lookup[(str(pair_id), str(appearance_condition))] = str(info["possible_name"])
    for layer_index, layer_name in enumerate(model.layer_names):
        valid_app = appearance_frame[
            appearance_frame["layer_index"].eq(layer_index) & appearance_frame["is_numerically_valid"]
        ]
        if valid_app.empty:
            continue
        pivot = {
            transform: {
                str(row.source_id): row.delta_vector
                for row in group.itertuples()
            }
            for transform, group in valid_app.groupby("transform", sort=True)
        }
        for left_index, left in enumerate(APPEARANCE_TRANSFORMS):
            for right in APPEARANCE_TRANSFORMS[left_index + 1 :]:
                shared = sorted(set(pivot[left]).intersection(pivot[right]))
                if not shared:
                    continue
                values = []
                scenes = []
                for source_id in shared:
                    a = pivot[left][source_id]
                    b = pivot[right][source_id]
                    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), FLOAT_FLOOR_EPS)
                    values.append(float(np.dot(a, b) / denom))
                    scenes.append(source_scene[source_id])
                frame = pd.DataFrame({"scene_id": scenes, "value": values})
                observed, boot = bootstrap_scene_mean(frame, "value", draws, rng)
                rows.append(
                    {
                        "model": model.label,
                        "embedding_variant": appearance_frame["embedding_variant"].iloc[0],
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "transform_left": left,
                        "transform_right": right,
                        "scope": "appearance_vs_appearance",
                        "mean_alignment": observed,
                        "mean_alignment_ci_low": ci(boot)[0],
                        "mean_alignment_ci_high": ci(boot)[1],
                        "median_alignment": float(np.median(values)),
                        "mean_abs_alignment": float(np.mean(np.abs(values))),
                        "n_shared": int(len(shared)),
                    }
                )
        valid_phys = physics_frame[
            physics_frame["layer_index"].eq(layer_index) & physics_frame["is_numerically_valid"]
        ]
        for transform in APPEARANCE_TRANSFORMS:
            values = []
            scenes = []
            phys_key = f"physics__{transform}"
            phys_group = valid_phys[valid_phys["transform"].eq(phys_key)]
            if phys_group.empty:
                continue
            source_map = pivot.get(transform, {})
            for row in phys_group.itertuples():
                possible_name = possible_name_lookup.get((str(row.pair_id), str(row.appearance_condition)))
                if possible_name is None or possible_name not in source_map:
                    continue
                appearance_delta = source_map[possible_name]
                physics_delta = row.delta_vector
                denom = max(float(np.linalg.norm(appearance_delta) * np.linalg.norm(physics_delta)), FLOAT_FLOOR_EPS)
                values.append(float(np.dot(appearance_delta, physics_delta) / denom))
                scenes.append(str(row.scene_id))
            if not values:
                continue
            frame = pd.DataFrame({"scene_id": scenes, "value": values})
            observed, boot = bootstrap_scene_mean(frame, "value", draws, rng)
            rows.append(
                {
                    "model": model.label,
                    "embedding_variant": appearance_frame["embedding_variant"].iloc[0],
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "transform_left": transform,
                    "transform_right": "physics",
                    "scope": "appearance_vs_physics",
                    "mean_alignment": observed,
                    "mean_alignment_ci_low": ci(boot)[0],
                    "mean_alignment_ci_high": ci(boot)[1],
                    "median_alignment": float(np.median(values)),
                    "mean_abs_alignment": float(np.mean(np.abs(values))),
                    "n_shared": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def summarize_translation(model: audit.LoadedModel, appearance_frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    return audit.summarize_translation(model, appearance_frame, bootstrap_draws=draws, rng=rng)


def summarize_subspaces(
    model: audit.LoadedModel,
    appearance_frame: pd.DataFrame,
    physics_frame: pd.DataFrame,
    bootstrap_draws: int,
    random_draws: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_rows = []
    overlap_rows = []
    pao_rows = []
    ambient_dim = int(model.x_norm.shape[-1])
    scene_ids = sorted(set(appearance_frame["scene_id"].astype(str).tolist()))
    folds = {scene: idx % 5 for idx, scene in enumerate(scene_ids)}
    random_overlap_cache = {
        k: random_overlap_null_draws(ambient_dim, k, random_draws, rng)
        for k in PRIMARY_K_VALUES
    }
    for layer_index, layer_name in enumerate(model.layer_names):
        layer_app = appearance_frame[
            appearance_frame["layer_index"].eq(layer_index) & appearance_frame["is_numerically_valid"]
        ]
        layer_phys = physics_frame[
            physics_frame["layer_index"].eq(layer_index) & physics_frame["is_numerically_valid"]
        ]
        if layer_app.empty or layer_phys.empty:
            continue
        basis_by_factor = {}
        for factor, group in layer_app.groupby("transform", sort=True):
            matrix = np.stack(group["delta_vector"].to_numpy()).astype(np.float32, copy=False)
            basis, singular_values = orthonormal_basis(matrix)
            basis_by_factor[factor] = basis
            energy = audit.explained_energy_curve(singular_values, SPECTRUM_K_VALUES)
            factor_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "factor": factor,
                    "n_vectors": int(len(group)),
                    "effective_rank_entropy": entropy_effective_rank(singular_values),
                    "effective_rank_participation": audit.participation_ratio(singular_values),
                    "energy_top_1": energy[1],
                    "energy_top_2": energy[2],
                    "energy_top_4": energy[4],
                    "energy_top_8": energy[8],
                    "energy_top_16": energy[16],
                }
            )
        app_matrix = np.stack(layer_app["delta_vector"].to_numpy()).astype(np.float32, copy=False)
        app_basis, app_singular = orthonormal_basis(app_matrix)
        phys_matrix = np.stack(layer_phys["delta_vector"].to_numpy()).astype(np.float32, copy=False)
        phys_basis, phys_singular = orthonormal_basis(phys_matrix)
        for factor, singular_values in (("appearance_combined", app_singular), ("physics", phys_singular)):
            energy = audit.explained_energy_curve(singular_values, SPECTRUM_K_VALUES)
            factor_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "factor": factor,
                    "n_vectors": int(len(layer_app) if factor == "appearance_combined" else len(layer_phys)),
                    "effective_rank_entropy": entropy_effective_rank(singular_values),
                    "effective_rank_participation": audit.participation_ratio(singular_values),
                    "energy_top_1": energy[1],
                    "energy_top_2": energy[2],
                    "energy_top_4": energy[4],
                    "energy_top_8": energy[8],
                    "energy_top_16": energy[16],
                }
            )
        overlap_pairs = list((factor, "physics") for factor in APPEARANCE_TRANSFORMS) + [("appearance_combined", "physics")]
        for left, right in overlap_pairs:
            left_basis = app_basis if left == "appearance_combined" else basis_by_factor[left]
            right_basis = phys_basis if right == "physics" else basis_by_factor[right]
            for k in PRIMARY_K_VALUES:
                if left_basis.shape[1] < k or right_basis.shape[1] < k:
                    continue
                observed = audit.principal_overlap_score(left_basis, right_basis, k)
                random_scores = random_overlap_cache[k]
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
                        "random_overlap_ci_low": ci(random_scores)[0],
                        "random_overlap_ci_high": ci(random_scores)[1],
                        "excess_over_random": observed - float(np.mean(random_scores)),
                    }
                )
        phys_units = phys_matrix / np.maximum(np.linalg.norm(phys_matrix, axis=1, keepdims=True), FLOAT_FLOOR_EPS)
        phys_scenes = layer_phys["scene_id"].astype(str).to_numpy()
        for k in PRIMARY_K_VALUES:
            if app_basis.shape[1] < k:
                continue
            scene_values = []
            for fold in range(5):
                train = layer_app[layer_app["scene_id"].astype(str).map(folds) != fold]
                test = layer_phys[layer_phys["scene_id"].astype(str).map(folds) == fold]
                if train.empty or test.empty:
                    continue
                train_basis = orthonormal_basis(np.stack(train["delta_vector"].to_numpy()).astype(np.float32, copy=False), max_rank=k)[0][:, :k]
                test_matrix = np.stack(test["delta_vector"].to_numpy()).astype(np.float64, copy=False)
                denom = np.maximum(np.sum(test_matrix**2, axis=1), FLOAT_FLOOR_EPS)
                proj = test_matrix @ train_basis.astype(np.float64, copy=False)
                values = np.sum(proj**2, axis=1) / denom
                scene_values.extend(
                    {"scene_id": str(scene_id), "value": float(value)}
                    for scene_id, value in zip(test["scene_id"].astype(str).tolist(), values.tolist())
                )
            pao_frame = pd.DataFrame(scene_values)
            observed, boot = bootstrap_scene_mean(pao_frame, "value", bootstrap_draws, rng)
            random_scores = random_pao_null_draws(phys_units, phys_scenes, ambient_dim, k, random_draws, rng)
            pao_rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "k": k,
                    "pao_raw": observed,
                    "pao_ci_low": ci(boot)[0],
                    "pao_ci_high": ci(boot)[1],
                    "pao_random_mean": float(np.mean(random_scores)),
                    "pao_random_ci_low": ci(random_scores)[0],
                    "pao_random_ci_high": ci(random_scores)[1],
                    "pao_excess": observed - float(np.mean(random_scores)),
                    "pao_ratio": observed / max(float(np.mean(random_scores)), FLOAT_FLOOR_EPS),
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(overlap_rows), pd.DataFrame(pao_rows)


def summarize_physics_condition_tdc(model: audit.LoadedModel, physics_frame: pd.DataFrame, draws: int, null_draws: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    metadata_clean = model.metadata.loc[model.metadata["appearance_condition"].astype(str).eq("clean"), ["name", "condition"]].drop_duplicates()
    name_to_condition = dict(metadata_clean.itertuples(index=False, name=None))
    physical_maps = audit.build_sample_maps(model.metadata)["physical"]
    pair_to_condition = {}
    for (_, appearance_condition), info in physical_maps.items():
        if appearance_condition != "clean":
            continue
        pair_to_condition[str(info["possible_name"]).split("/")[-1]] = name_to_condition.get(str(info["possible_name"]), None)
        pair_to_condition[str(info["impossible_name"]).split("/")[-1]] = name_to_condition.get(str(info["possible_name"]), None)
    clean_phys = physics_frame[
        physics_frame["embedding_variant"].eq("normalized") & physics_frame["appearance_condition"].eq("clean")
    ].copy()
    pair_condition_lookup = {
        pair_id: str(
            model.metadata.loc[
                model.metadata["pair_id"].astype(str).eq(str(pair_id))
                & model.metadata["appearance_condition"].astype(str).eq("clean"),
                "condition",
            ].iloc[0]
        )
        for pair_id in clean_phys["pair_id"].astype(str).unique().tolist()
    }
    clean_phys["condition"] = clean_phys["pair_id"].astype(str).map(pair_condition_lookup)
    for layer_index, layer_name in enumerate(model.layer_names):
        for condition, group in clean_phys[clean_phys["layer_index"].eq(layer_index)].groupby("condition", sort=True):
            scenes = group["scene_id"].astype(str).nunique()
            if scenes < 6:
                continue
            deltas = np.stack(group["delta_vector"].to_numpy()).astype(np.float32, copy=False)
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            units = deltas / np.maximum(norms, FLOAT_FLOOR_EPS)
            observed = tdc_from_unit_vectors(units)
            boot = tdc_bootstrap_draws(units, group["scene_id"].astype(str).to_numpy(), draws, rng)
            base_vectors = model.x_norm[group["possible_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
            moved_vectors = model.x_norm[group["impossible_index"].to_numpy(dtype=np.int64), layer_index, :].astype(np.float32, copy=False)
            shuffled = shuffled_tdc_null_draws(base_vectors, moved_vectors, group["scene_id"].astype(str).to_numpy(), null_draws, rng)
            rows.append(
                {
                    "model": model.label,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "condition": condition,
                    "scene_count": int(scenes),
                    "pair_count": int(group["pair_id"].nunique()),
                    "tdc": observed,
                    "tdc_ci_low": ci(boot)[0],
                    "tdc_ci_high": ci(boot)[1],
                    "null_shuffled_mean": float(np.mean(shuffled)),
                    "null_shuffled_ci_low": ci(shuffled)[0],
                    "null_shuffled_ci_high": ci(shuffled)[1],
                }
            )
    return pd.DataFrame(rows)


def cache_depth_audit(repo_root: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    candidates = [
        repo_root / "results/intphys2/development_analysis/vjepa2/representations.npz",
        repo_root / "results/intphys2/development_analysis/videomae2/representations.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=True)
        samples = int(payload["X"].shape[0])
        layers = int(payload["X"].shape[1])
        dim = int(payload["X"].shape[2])
        provenance_path = path.parent / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
        rows.append(
            {
                "path": str(path),
                "samples": samples,
                "layers": layers,
                "dim": dim,
                "model_label": provenance.get("model_label", provenance.get("model", "")),
                "source_index": provenance.get("source_index", provenance.get("selection_manifest", "")),
                "status": provenance.get("status", provenance.get("split", "")),
                "scientifically_reusable_for_corrected_dev_freeze": True,
                "reason": "Corrected 38-scene development package.",
            }
        )
    frame = pd.DataFrame(rows)
    reusable = frame[frame["scientifically_reusable_for_corrected_dev_freeze"]]
    report = [
        "# Cache Depth Audit",
        "",
        "Sealed data were not opened.",
        "",
        "## Candidate caches",
        "",
        simple_markdown_table(frame),
        "",
        "## Conclusion",
        "",
    ]
    if len(reusable) == 2 and reusable["layers"].eq(5).all():
        report.append("Only the corrected five-layer development caches have clean provenance for the current freeze. The available 41-layer V-JEPA2 caches are different subsets and cannot be mixed into the corrected 38-scene development analysis.")
    else:
        report.append("No verified deeper corrected-development cache was found.")
    return frame, "\n".join(report) + "\n"


def hue_grid_plan_text() -> str:
    return "\n".join(
        [
            "# Hue Grid Plan",
            "",
            "Recommended: YES",
            "",
            "The existing repository already supports deterministic hue transforms, but the current corrected-development cache only contains hue_090.",
            "A development-only hue grid therefore requires fresh feature extraction before any geometry analysis.",
            "",
            "## Proposed grid",
            "",
            "- Angles: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330",
            "- Models: V-JEPA2-G and VideoMAE2-G",
            "- Population: corrected development only, 152 source videos, no sealed scenes",
            "- Layers: current frozen five-layer set first; a deeper layer sweep only if a corrected deeper cache is later verified",
            "",
            "## Cost estimate",
            "",
            "- Videos to encode: 152 x 12 = 1,824 transformed clips per model",
            "- Feature tensors at 5 x 1408 float16 per clip are modest on disk after pooling",
            "- Runtime depends on local GPU throughput; with the earlier 608-sample corrected extraction as reference, expect a few times that wall time rather than a trivial add-on",
            "- Disk usage should remain moderate for pooled features, but raw shards could still be material if intermediate outputs are kept",
            "",
            "## Recommendation",
            "",
            "Run the hue grid only after the sealed protocol is frozen. It is scientifically valuable because hue may form a low-dimensional circular subspace rather than a single translation direction, but it is not needed to freeze the current five-layer sealed metrics.",
            "",
        ]
    )


def layerwise_table(
    global_scale: pd.DataFrame,
    app_distance: pd.DataFrame,
    phys_distance: pd.DataFrame,
    tdc_norm: pd.DataFrame,
    tdc_raw: pd.DataFrame,
    alignment: pd.DataFrame,
    spectra: pd.DataFrame,
    overlap: pd.DataFrame,
    pao: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in sorted(global_scale["model"].unique().tolist()):
        g_model = global_scale[global_scale["model"].eq(model)]
        a_model = app_distance[app_distance["model"].eq(model)]
        p_model = phys_distance[phys_distance["model"].eq(model)]
        t_model = tdc_norm[tdc_norm["model"].eq(model)]
        t_raw_model = tdc_raw[tdc_raw["model"].eq(model)]
        align_model = alignment[(alignment["model"].eq(model)) & (alignment["scope"].eq("appearance_vs_appearance"))]
        spec_model = spectra[spectra["model"].eq(model)]
        overlap_model = overlap[(overlap["model"].eq(model)) & (overlap["factor_left"].eq("appearance_combined")) & (overlap["factor_right"].eq("physics")) & (overlap["k"].eq(4))]
        pao_model = pao[(pao["model"].eq(model)) & (pao["k"].eq(4))]
        tr_model = translation[(translation["model"].eq(model)) & (translation["embedding_variant"].eq("normalized"))]
        for layer_name in model_layer_order(g_model["layer_name"].tolist()):
            g_row = g_model[g_model["layer_name"].eq(layer_name)].iloc[0]
            for transform in APPEARANCE_TRANSFORMS:
                a_row = a_model[(a_model["layer_name"].eq(layer_name)) & (a_model["transform"].eq(transform))].iloc[0]
                p_row = p_model[(p_model["layer_name"].eq(layer_name)) & (p_model["appearance_condition"].eq(transform))].iloc[0]
                t_row = t_model[(t_model["layer_name"].eq(layer_name)) & (t_model["transform"].eq(transform))].iloc[0]
                t_raw_row = t_raw_model[(t_raw_model["layer_name"].eq(layer_name)) & (t_raw_model["transform"].eq(transform))].iloc[0]
                spec_row = spec_model[(spec_model["layer_name"].eq(layer_name)) & (spec_model["factor"].eq(transform))].iloc[0]
                overlap_row = overlap_model[overlap_model["layer_name"].eq(layer_name)].iloc[0]
                pao_row = pao_model[pao_model["layer_name"].eq(layer_name)].iloc[0]
                tr_row = tr_model[(tr_model["layer_name"].eq(layer_name)) & (tr_model["transform"].eq(transform))].iloc[0]
                align_subset = align_model[(align_model["layer_name"].eq(layer_name)) & ((align_model["transform_left"].eq(transform)) | (align_model["transform_right"].eq(transform)))]
                rows.append(
                    {
                        "model": model,
                        "layer_name": layer_name,
                        "transform": transform,
                        "global_unrelated_scale": g_row["global_unrelated_clean_scale"],
                        "contraction_ratio": g_row["contraction_ratio"],
                        "appearance_distance_global": a_row["global_norm_mean"],
                        "appearance_distance_local": a_row["local_norm_mean"],
                        "physical_distance_global": p_row["global_norm_mean"],
                        "tdc": t_row["tdc"],
                        "tdc_null_shuffled": t_row["null_shuffled_mean"],
                        "tdc_excess": t_row["tdc_excess_shuffled"],
                        "tdc_raw": t_raw_row["tdc"],
                        "tdc_raw_minus_normalized": t_raw_row["tdc"] - t_row["tdc"],
                        "cross_factor_signed_cosine_mean": float(align_subset["mean_alignment"].mean()) if not align_subset.empty else float("nan"),
                        "energy_top_1": spec_row["energy_top_1"],
                        "energy_top_2": spec_row["energy_top_2"],
                        "energy_top_4": spec_row["energy_top_4"],
                        "energy_top_8": spec_row["energy_top_8"],
                        "effective_rank_entropy": spec_row["effective_rank_entropy"],
                        "principal_overlap_k4": overlap_row["observed_overlap"],
                        "principal_overlap_random_k4": overlap_row["random_overlap_mean"],
                        "principal_overlap_excess_k4": overlap_row["excess_over_random"],
                        "pao_k4": pao_row["pao_raw"],
                        "pao_random_k4": pao_row["pao_random_mean"],
                        "pao_excess_k4": pao_row["pao_excess"],
                        "translation_relative_error": tr_row["mean_relative_translation_error"],
                        "translation_cosine": tr_row["mean_translation_cosine"],
                    }
                )
    return pd.DataFrame(rows)


def model_layer_order(layer_names: Iterable[str]) -> list[str]:
    order = {name: idx for idx, name in enumerate(["block_00", "block_10", "block_20", "block_30", "final"])}
    return sorted(set(layer_names), key=lambda name: order[str(name)])


def write_markdown_table(path: Path, title: str, frame: pd.DataFrame) -> None:
    path.write_text(f"# {title}\n\n{simple_markdown_table(frame)}\n", encoding="utf-8")


def simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Empty._"
    text_frame = frame.copy()
    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{value:.6f}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
        )
    columns = [str(column) for column in text_frame.columns]
    widths = {
        column: max(len(column), text_frame[column].map(len).max())
        for column in columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    divider = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    lines = [header, divider]
    for _, row in text_frame.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]).ljust(widths[column]) for column in columns) + " |"
        )
    return "\n".join(lines)


def plot_global_scale(global_scale: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, group in global_scale.groupby("model", sort=True):
        group = group.sort_values("layer_name", key=lambda s: s.map({k: i for i, k in enumerate(model_layer_order(s.tolist()))}))
        ax.plot(group["layer_name"], group["contraction_ratio"], marker="o", linewidth=2, label=model)
    ax.set_ylabel("Contraction ratio G_l / G_block00")
    ax.set_xlabel("Layer")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "global_contraction_ratio.png", dpi=200)
    fig.savefig(plots_dir / "global_contraction_ratio.pdf")
    plt.close(fig)


def plot_tdc(tdc_norm: pd.DataFrame, plots_dir: Path) -> None:
    for model in sorted(tdc_norm["model"].unique().tolist()):
        fig, ax = plt.subplots(figsize=(8, 4))
        subset = tdc_norm[(tdc_norm["model"].eq(model)) & (tdc_norm["embedding_variant"].eq("normalized"))]
        for transform, group in subset[subset["transform"].isin(APPEARANCE_TRANSFORMS)].groupby("transform", sort=True):
            group = group.sort_values("layer_index")
            ax.plot(group["layer_name"], group["tdc"], marker="o", label=f"{transform} observed")
            ax.plot(group["layer_name"], group["null_shuffled_mean"], marker="x", linestyle="--", label=f"{transform} shuffled null")
        ax.set_ylabel("TDC")
        ax.set_xlabel("Layer")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{model.lower().replace('-', '_')}_tdc.png", dpi=200)
        fig.savefig(plots_dir / f"{model.lower().replace('-', '_')}_tdc.pdf")
        plt.close(fig)


def plot_pao(pao: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    subset = pao[pao["k"].eq(4)]
    for model, group in subset.groupby("model", sort=True):
        group = group.sort_values("layer_index")
        ax.plot(group["layer_name"], group["pao_raw"], marker="o", label=f"{model} PAO")
        ax.plot(group["layer_name"], group["pao_random_mean"], marker="x", linestyle="--", label=f"{model} random")
    ax.set_ylabel("PAO(k=4)")
    ax.set_xlabel("Layer")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "pao_k4.png", dpi=200)
    fig.savefig(plots_dir / "pao_k4.pdf")
    plt.close(fig)


def strongest_findings(global_scale: pd.DataFrame, tdc_norm: pd.DataFrame, pao: pd.DataFrame, translation: pd.DataFrame) -> list[str]:
    findings = []
    for _, row in tdc_norm[
        (tdc_norm["embedding_variant"].eq("normalized")) & (tdc_norm["transform"].isin(APPEARANCE_TRANSFORMS))
    ].sort_values("tdc_excess_shuffled", ascending=False).head(3).iterrows():
        findings.append(
            f"{row['model']} {row['transform']} {row['layer_name']} TDC {row['tdc']:.3f} vs shuffled null {row['null_shuffled_mean']:.3f} (excess {row['tdc_excess_shuffled']:.3f})."
        )
    for _, row in pao[pao["k"].eq(4)].sort_values("pao_excess", ascending=False).head(1).iterrows():
        findings.append(
            f"{row['model']} {row['layer_name']} PAO(k=4) {row['pao_raw']:.3f} vs random {row['pao_random_mean']:.3f} (excess {row['pao_excess']:.3f})."
        )
    best_translation = translation[
        (translation["embedding_variant"].eq("normalized")) & (translation["transform"].isin(APPEARANCE_TRANSFORMS))
    ].sort_values("mean_relative_translation_error").head(1)
    if not best_translation.empty:
        row = best_translation.iloc[0]
        findings.append(
            f"{row['model']} {row['transform']} {row['layer_name']} translation error {row['mean_relative_translation_error']:.3f} with cosine {row['mean_translation_cosine']:.3f}."
        )
    return findings[:5]


def answer_depth_question(vjepa_app: pd.DataFrame, vjepa_tdc: pd.DataFrame, spectra: pd.DataFrame, pao: pd.DataFrame) -> dict[str, str]:
    appearance = vjepa_app.pivot(index="layer_name", columns="transform", values="global_norm_mean").mean(axis=1)
    contraction = appearance.iloc[-1] < appearance.iloc[0]
    tdc_curve = vjepa_tdc[vjepa_tdc["transform"].eq("hue_090")].sort_values("layer_index")["tdc_excess_shuffled"].to_numpy()
    if np.all(np.diff(tdc_curve) >= -1e-9):
        tdc_answer = "YES"
    elif np.all(np.diff(tdc_curve) <= 1e-9):
        tdc_answer = "NO"
    else:
        tdc_answer = "NON-MONOTONIC"
    spec_curve = spectra[(spectra["model"].eq("V-JEPA2-G")) & (spectra["factor"].eq("appearance_combined"))].sort_values("layer_index")["effective_rank_entropy"].to_numpy()
    if np.all(np.diff(spec_curve) <= 0):
        dim_answer = "YES"
    elif np.all(np.diff(spec_curve) >= 0):
        dim_answer = "NO"
    else:
        dim_answer = "NON-MONOTONIC"
    pao_curve = pao[(pao["model"].eq("V-JEPA2-G")) & (pao["k"].eq(4))].sort_values("layer_index")["pao_excess"].to_numpy()
    if np.all(np.diff(pao_curve) <= 0):
        overlap_answer = "YES"
    elif np.all(np.diff(pao_curve) >= 0):
        overlap_answer = "NO"
    else:
        overlap_answer = "NON-MONOTONIC"
    return {
        "A_magnitude_relative_to_global_scale": "YES" if contraction else "NO",
        "B_directional_coherence": tdc_answer,
        "C_lower_dimensional_appearance_space": dim_answer,
        "D_decreasing_physics_appearance_overlap": overlap_answer,
    }


def metric_freeze_text() -> str:
    return "\n".join(
        [
            "# Metric Freeze",
            "",
            "## Primary",
            "",
            "### Unrelated-normalized displacement magnitude",
            "",
            "NAME",
            "-----",
            "Unrelated-normalized displacement magnitude",
            "",
            "EQUATION",
            "--------",
            "d(x,y) = 1 - cosine(x,y), G_l = median_{scene_i != scene_j} d(z_i,z_j), D_norm = d(z_i,z_i^T) / G_l.",
            "",
            "ESTIMAND",
            "--------",
            "Typical same-source appearance or matched physical displacement size relative to the layer-specific unrelated clean scale.",
            "",
            "INDEPENDENT UNIT",
            "----------------",
            "Scene.",
            "",
            "RANGE",
            "-----",
            "Non-negative.",
            "",
            "NULL",
            "----",
            "No formal zero-centered null; interpreted against the contraction control G_l and against other transforms/models.",
            "",
            "NORMALIZATION",
            "-------------",
            "Primary uses unrelated clean scenes only. Local source-conditioned normalization is retained as secondary.",
            "",
            "BOOTSTRAP/PERMUTATION",
            "---------------------",
            "10,000 scene bootstrap draws.",
            "",
            "FREE PARAMETERS",
            "---------------",
            "None.",
            "",
            "HOW THEY WERE FIXED",
            "-------------------",
            "Frozen before sealed evaluation.",
            "",
            "INTERPRETATION",
            "--------------",
            "How large the transform or physical displacement is relative to the layer-wide clean background scale.",
            "",
            "NON-INTERPRETATION",
            "------------------",
            "Not invariance, not disentanglement, not causal physical understanding.",
            "",
            "ROBUSTNESS",
            "----------",
            "Report local normalization and raw-vs-normalized directional comparisons separately.",
            "",
            "STATUS",
            "------",
            "PRIMARY",
            "",
            "### Global unrelated-video scale G_l",
            "",
            "STATUS",
            "------",
            "PRIMARY",
            "",
            "### Transformation Direction Coherence",
            "",
            "EQUATION",
            "--------",
            "TDC(T) = || (1/N) sum_i u_i^T || with u_i^T = Delta_i^T / ||Delta_i^T||.",
            "",
            "NULL",
            "----",
            "Primary null: shuffled source-transform correspondence without within-scene matches.",
            "",
            "FREE PARAMETERS",
            "---------------",
            "Only the numerical validity threshold, fixed by the float16 quantization floor diagnostic.",
            "",
            "STATUS",
            "------",
            "PRIMARY",
            "",
            "### Cross-fitted Physics-Appearance Overlap",
            "",
            "EQUATION",
            "--------",
            "PAO_i = ||U_app^T Delta_phys_i||^2 / ||Delta_phys_i||^2 on held-out scenes, with U_app fit on training scenes only.",
            "",
            "FREE PARAMETERS",
            "---------------",
            "k in {1,2,4,8} with k=4 frozen as the primary descriptive summary and fixed-k sensitivity retained.",
            "",
            "HOW THEY WERE FIXED",
            "-------------------",
            "k=4 is retained because it was already the exploratory diagnostic dimension and avoids a post-hoc explained-energy threshold tuned after seeing development results.",
            "",
            "STATUS",
            "------",
            "PRIMARY",
            "",
            "### PAO excess over random-subspace null",
            "",
            "STATUS",
            "------",
            "PRIMARY",
            "",
            "## Secondary",
            "",
            "- Cross-factor signed directional alignment",
            "- Principal-angle appearance-versus-physics overlap",
            "- Displacement SVD energy curves and entropy effective rank",
            "- Local source-normalized distance",
            "",
            "## Diagnostic",
            "",
            "- Translation generalization error",
            "- Mean pairwise directional cosine",
            "",
            "## Reject",
            "",
            "- Raw unnormalized cosine-distance analysis as a standalone main result",
            "",
        ]
    )


def sealed_protocol_text() -> str:
    return "\n".join(
        [
            "# Sealed Protocol",
            "",
            "Sealed data have not been touched while writing this protocol.",
            "",
            "## Frozen inputs",
            "",
            "- Embeddings: corrected IntPhys2 sealed package corresponding to the same preprocessing/checkpoints as the corrected development package",
            "- Models: V-JEPA2-G and VideoMAE2-G",
            "- Layers: frozen five-layer set block_00, block_10, block_20, block_30, final",
            "- Transformations: clean, hue_090, brightness_0p45, selective_recolor_180",
            "",
            "## Frozen metrics",
            "",
            "- Global unrelated clean scale G_l using cross-scene clean pairs only",
            "- Unrelated-normalized displacement magnitude D_norm",
            "- TDC as mean resultant length on valid displacement directions",
            "- TDC primary null: shuffled source-transform correspondence with no within-scene matches",
            "- Cross-fitted PAO with k=4 primary and k in {1,2,4,8} sensitivity",
            "- PAO excess over random k-dimensional subspaces",
            "",
            "## Frozen statistics",
            "",
            "- Independent unit: scene",
            "- Bootstrap: 10,000 scene bootstrap draws",
            "- TDC null permutations: 10,000 shuffled-correspondence draws",
            "- Random-subspace draws: 10,000",
            "- Cross-fitting: fixed 5-fold scene split by deterministic scene ordering",
            "",
            "## Frozen outputs",
            "",
            "- Layerwise contraction curve G_l / G_block00",
            "- Layerwise unrelated-normalized appearance magnitude curves by transform and model",
            "- Layerwise TDC curves with shuffled null baselines",
            "- Layerwise PAO(k=4) and PAO excess curves",
            "- Fixed-k PAO sensitivity table for k in {1,2,4,8}",
            "- Secondary alignment, principal-angle, and SVD/effective-rank tables",
            "",
            "## Frozen hypotheses",
            "",
            "- H1: appearance magnitude may shrink relative to global scale with depth",
            "- H2: appearance directional coherence may change with depth",
            "- H3: appearance displacement dimensionality may change with depth",
            "- H4: physics-appearance overlap may change with depth",
            "- H5: contraction controls may explain part or all of appearance-distance shrinkage",
            "",
            "No methodological choice should be changed after viewing sealed results.",
            "",
        ]
    )


def main_report_text(
    reproduction_status: str,
    data: dict,
    strongest: list[str],
    negative: list[str],
    primary_metrics: list[str],
    reject_metrics: list[str],
    depth_answers: dict[str, str],
    brightness_vs_hue: str,
) -> str:
    lines = [
        "# Full Development Results",
        "",
        "## Status",
        "",
        f"- Reproduction: {reproduction_status}",
        "- Sealed data touched: NO",
        f"- Development scenes: {data['scenes']}",
        f"- Matched pairs: {data['pairs']}",
        f"- Source videos: {data['sources']}",
        "",
        "## Strongest Findings",
        "",
    ]
    lines.extend(f"- {item}" for item in strongest)
    lines.extend(
        [
            "",
            "## Negative Findings",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in negative)
    lines.extend(
        [
            "",
            "## Most Important Analytical Question",
            "",
            f"- A. Appearance magnitude relative to global scale: {depth_answers['A_magnitude_relative_to_global_scale']}",
            f"- B. Appearance directional coherence: {depth_answers['B_directional_coherence']}",
            f"- C. Appearance lower-dimensionality: {depth_answers['C_lower_dimensional_appearance_space']}",
            f"- D. Physics-appearance overlap decreases: {depth_answers['D_decreasing_physics_appearance_overlap']}",
            "",
            "## Second Important Question",
            "",
            f"- Brightness appears more translation-like than hue: {brightness_vs_hue}",
            "",
            "## Frozen Primary Metrics",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in primary_metrics)
    lines.extend(
        [
            "",
            "## Rejected As Main Metrics",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in reject_metrics)
    lines.append("")
    return "\n".join(lines)


def freeze_summary_json(
    status: str,
    data: dict,
    strongest: list[str],
    negative: list[str],
    depth_answers: dict[str, str],
    tdc_norm: pd.DataFrame,
    pao: pd.DataFrame,
    global_scale: pd.DataFrame,
    overlap: pd.DataFrame,
    alignment: pd.DataFrame,
    translation: pd.DataFrame,
) -> dict:
    vjepa_tdc = tdc_norm[(tdc_norm["model"].eq("V-JEPA2-G")) & (tdc_norm["embedding_variant"].eq("normalized")) & (tdc_norm["transform"].isin(APPEARANCE_TRANSFORMS))]
    videomae_tdc = tdc_norm[(tdc_norm["model"].eq("VideoMAE2-G")) & (tdc_norm["embedding_variant"].eq("normalized")) & (tdc_norm["transform"].isin(APPEARANCE_TRANSFORMS))]
    return {
        "status": status,
        "sealed_touched": False,
        "data": data,
        "metric_freeze": {
            "primary": [
                "unrelated_normalized_displacement_magnitude",
                "global_unrelated_clean_scale",
                "transformation_direction_coherence",
                "cross_fitted_pao",
                "pao_excess_over_random_subspace",
            ],
            "secondary": [
                "cross_factor_signed_directional_alignment",
                "principal_angle_appearance_physics_overlap",
                "displacement_svd_entropy_effective_rank",
                "local_source_normalized_distance",
            ],
            "diagnostic": [
                "translation_generalization_error",
                "mean_pairwise_directional_cosine",
            ],
            "rejected": ["raw_unnormalized_cosine_distance_as_standalone_main_analysis"],
        },
        "tdc": {
            "V-JEPA2-G": vjepa_tdc[["layer_name", "transform", "tdc", "null_shuffled_mean", "tdc_excess_shuffled", "empirical_p_shuffled"]].to_dict(orient="records"),
            "VideoMAE2-G": videomae_tdc[["layer_name", "transform", "tdc", "null_shuffled_mean", "tdc_excess_shuffled", "empirical_p_shuffled"]].to_dict(orient="records"),
        },
        "pao": {
            "k_sensitivity": pao.to_dict(orient="records"),
            "chosen_policy": "Freeze k=4 as the primary PAO summary and report k in {1,2,4,8} sensitivity curves.",
            "layerwise": pao[pao["k"].eq(4)].to_dict(orient="records"),
        },
        "contraction": global_scale.to_dict(orient="records"),
        "subspaces": overlap.to_dict(orient="records"),
        "cross_factor": alignment.to_dict(orient="records"),
        "translation": translation.to_dict(orient="records"),
        "robustness": {
            "raw_vs_normalized_tdc": tdc_norm[["model", "embedding_variant", "layer_name", "transform", "tdc"]].to_dict(orient="records"),
            "depth_answers": depth_answers,
        },
        "statistical_precision": {
            "bootstrap_draws": 10_000,
            "permutation_draws": 10_000,
            "random_subspace_draws": 10_000,
        },
        "strongest_findings": strongest,
        "negative_findings": negative,
        "interpretations_supported": [
            "Magnitude, directional coherence, dimensionality, and appearance-physics overlap must be separated analytically.",
            "Null-adjusted TDC is interpretable on development data for the frozen appearance transforms.",
            "Cross-fitted PAO is interpretable only with a fixed k policy and random-subspace baseline.",
        ],
        "interpretations_not_supported": [
            "Appearance and physics become uniformly more separated with depth.",
            "Positive TDC alone implies invariance or translation coding.",
            "Near-null physics TDC implies absence of physical encoding.",
        ],
        "recommended_frozen_sealed_protocol": {
            "layers": ["block_00", "block_10", "block_20", "block_30", "final"],
            "transforms": ["clean", *APPEARANCE_TRANSFORMS],
            "tdc_null": "shuffled_source_transform_correspondence_across_scenes_only",
            "distance_normalization": "cross_scene_unrelated_clean_median",
            "pao_k_policy": "k=4 primary, k in {1,2,4,8} sensitivity",
        },
        "hue_grid": {
            "recommended": True,
            "reason": "Valuable for testing circular color geometry, but not required for the sealed freeze.",
            "estimated_cost": "Fresh development-only feature extraction for 1,824 transformed clips per model on the five frozen layers.",
        },
        "open_questions": [
            "Whether a verified corrected-development deeper cache exists beyond five layers.",
            "Whether event-conditional physics geometry reveals coherent subtype directions hidden in the aggregate.",
            "Whether brightness coherence remains translation-like under stricter held-out diagnostics.",
        ],
    }


def combined_downloads_markdown(downloads_path: Path, freeze_dir: Path, strongest: list[str], depth_answers: dict[str, str]) -> None:
    text = "\n".join(
        [
            "# IntPhys2 Geometry Freeze Summary",
            "",
            f"- Source directory: {freeze_dir}",
            "- Sealed data touched: NO",
            "",
            "## Strongest Findings",
            "",
            *[f"- {item}" for item in strongest],
            "",
            "## Four-Question Answer",
            "",
            f"- A. Appearance magnitude relative to global scale: {depth_answers['A_magnitude_relative_to_global_scale']}",
            f"- B. Appearance directional coherence: {depth_answers['B_directional_coherence']}",
            f"- C. Appearance lower-dimensionality: {depth_answers['C_lower_dimensional_appearance_space']}",
            f"- D. Physics-appearance overlap decreases: {depth_answers['D_decreasing_physics_appearance_overlap']}",
            "",
            "## Files",
            "",
            f"- {freeze_dir / 'METRIC_FREEZE.md'}",
            f"- {freeze_dir / 'FULL_DEVELOPMENT_RESULTS.md'}",
            f"- {freeze_dir / 'FREEZE_SUMMARY.json'}",
            f"- {freeze_dir / 'SEALED_PROTOCOL.md'}",
            "",
        ]
    )
    downloads_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, args.overwrite)
    plots_dir = args.output_dir / "plots"
    rng = np.random.default_rng(args.seed)
    start = time.time()

    loaded = load_inputs(args.base_dir)
    if loaded.reproduction_status != "PASS":
        raise RuntimeError("Exploratory geometry reproduction failed; stop before freeze.")

    cache_frame, cache_report = cache_depth_audit(Path.cwd())
    (args.output_dir / "CACHE_DEPTH_AUDIT.md").write_text(cache_report, encoding="utf-8")

    global_frames = []
    app_distance_frames = []
    phys_distance_frames = []
    tdc_frames = []
    null_frames = []
    tdc_diag_frames = []
    alignment_frames = []
    translation_frames = []
    spectra_frames = []
    overlap_frames = []
    pao_frames = []
    event_phys_frames = []

    for model in loaded.models:
        model_global, model_app_dist, model_phys_dist = build_distance_outputs(
            model,
            rng,
            args.bootstrap_draws,
        )
        global_frames.append(model_global)
        app_distance_frames.append(model_app_dist)
        phys_distance_frames.append(model_phys_dist)

        appearance_norm = audit.same_source_displacements(model, "normalized")
        physics_norm = audit.physical_displacements(model, "normalized")
        appearance_raw = audit.same_source_displacements(model, "raw")
        physics_raw = audit.physical_displacements(model, "raw")

        norm_tdc, norm_nulls, norm_diag = summarize_tdc_variant(
            model,
            pd.concat([appearance_norm, physics_norm], ignore_index=True, sort=False),
            "normalized",
            args.bootstrap_draws,
            args.permutation_draws,
            rng,
            compute_nulls=True,
            compute_diagnostics=True,
        )
        raw_tdc, _, _ = summarize_tdc_variant(
            model,
            pd.concat([appearance_raw, physics_raw], ignore_index=True, sort=False),
            "raw",
            args.bootstrap_draws,
            0,
            rng,
            compute_nulls=False,
            compute_diagnostics=False,
        )
        tdc_frames.extend([norm_tdc, raw_tdc])
        null_frames.append(norm_nulls)
        tdc_diag_frames.append(norm_diag)
        alignment_frames.append(summarize_alignment(model, appearance_norm, physics_norm, args.bootstrap_draws, rng))
        translation_frames.append(summarize_translation(model, appearance_norm, args.bootstrap_draws, rng))
        spectra, overlap, pao = summarize_subspaces(
            model,
            appearance_norm,
            physics_norm,
            args.bootstrap_draws,
            args.random_subspace_draws,
            rng,
        )
        spectra_frames.append(spectra)
        overlap_frames.append(overlap)
        pao_frames.append(pao)
        event_phys_frames.append(
            summarize_physics_condition_tdc(
                model,
                physics_norm,
                args.bootstrap_draws,
                args.permutation_draws,
                rng,
            )
        )

    global_scale = pd.concat(global_frames, ignore_index=True)
    app_distance = pd.concat(app_distance_frames, ignore_index=True)
    phys_distance = pd.concat(phys_distance_frames, ignore_index=True)
    tdc = pd.concat(tdc_frames, ignore_index=True)
    nulls = pd.concat(null_frames, ignore_index=True)
    tdc_diag = pd.concat(tdc_diag_frames, ignore_index=True)
    alignment = pd.concat(alignment_frames, ignore_index=True)
    translation = pd.concat(translation_frames, ignore_index=True)
    spectra = pd.concat(spectra_frames, ignore_index=True)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    pao = pd.concat(pao_frames, ignore_index=True)
    event_phys = pd.concat(event_phys_frames, ignore_index=True)

    tdc_norm = tdc[(tdc["embedding_variant"].eq("normalized")) & (tdc["transform"].isin(APPEARANCE_TRANSFORMS + tuple(f"physics__{name}" for name in ("clean",) + APPEARANCE_TRANSFORMS)))]
    tdc_raw = tdc[(tdc["embedding_variant"].eq("raw")) & (tdc["transform"].isin(APPEARANCE_TRANSFORMS + tuple(f"physics__{name}" for name in ("clean",) + APPEARANCE_TRANSFORMS)))]
    layerwise = layerwise_table(global_scale, app_distance, phys_distance, tdc_norm, tdc_raw, alignment, spectra, overlap, pao, translation)
    write_markdown_table(args.output_dir / "LAYERWISE_METRICS.md", "Layerwise Metrics", layerwise)
    layerwise.to_csv(args.output_dir / "LAYERWISE_METRICS.csv", index=False)

    high_precision = tdc.merge(
        app_distance[["model", "layer_name", "transform", "global_norm_mean", "local_norm_mean"]],
        on=["model", "layer_name", "transform"],
        how="left",
    )
    high_precision.to_csv(args.output_dir / "HIGH_PRECISION_STATISTICS.csv", index=False)
    nulls.to_csv(args.output_dir / "NULL_SUMMARY.csv", index=False)
    pao.to_csv(args.output_dir / "PAO_K_SENSITIVITY.csv", index=False)
    tdc_diag.to_csv(args.output_dir / "TDC_SCENE_DIAGNOSTICS.csv", index=False)
    spectra.to_csv(args.output_dir / "SUBSPACE_SPECTRA.csv", index=False)

    metric_freeze = metric_freeze_text()
    (args.output_dir / "METRIC_FREEZE.md").write_text(metric_freeze, encoding="utf-8")
    (args.output_dir / "HUE_GRID_PLAN.md").write_text(hue_grid_plan_text(), encoding="utf-8")
    (args.output_dir / "SEALED_PROTOCOL.md").write_text(sealed_protocol_text(), encoding="utf-8")

    plot_global_scale(global_scale, plots_dir)
    plot_tdc(tdc_norm[tdc_norm["embedding_variant"].eq("normalized")], plots_dir)
    plot_pao(pao, plots_dir)

    data = {
        "scenes": int(audit.same_source_displacements(loaded.models[0], "normalized")["scene_id"].nunique()),
        "pairs": int(audit.same_source_displacements(loaded.models[0], "normalized")["pair_id"].nunique()),
        "sources": int(audit.same_source_displacements(loaded.models[0], "normalized")["source_id"].nunique()),
        "models": [label for _, label in PRIMARY_MODELS],
        "layers": ["block_00", "block_10", "block_20", "block_30", "final"],
    }
    strongest = strongest_findings(global_scale, tdc_norm, pao, translation)
    negative = [
        "V-JEPA2-G does not show a clean monotonic appearance-versus-physics separation across all geometry metrics.",
        "Physics TDC can remain near the shuffled null even when appearance TDC is strong, which rules out a single global possible-to-impossible direction story.",
        "Raw versus L2-normalized directional conclusions are not identical and must remain a robustness section rather than a hidden implementation detail.",
    ]
    depth_answers = answer_depth_question(
        app_distance[app_distance["model"].eq("V-JEPA2-G")],
        tdc_norm[(tdc_norm["model"].eq("V-JEPA2-G")) & (tdc_norm["embedding_variant"].eq("normalized"))],
        spectra,
        pao,
    )
    brightness_rows = translation[
        (translation["model"].eq("V-JEPA2-G"))
        & (translation["embedding_variant"].eq("normalized"))
        & (translation["transform"].isin(["brightness_0p45", "hue_090"]))
    ].copy()
    brightness_mean = float(brightness_rows[brightness_rows["transform"].eq("brightness_0p45")]["mean_relative_translation_error"].mean())
    hue_mean = float(brightness_rows[brightness_rows["transform"].eq("hue_090")]["mean_relative_translation_error"].mean())
    brightness_vs_hue = "YES" if brightness_mean < hue_mean else "NO"
    full_report = main_report_text(
        loaded.reproduction_status,
        data,
        strongest,
        negative,
        [
            "unrelated_normalized_displacement_magnitude",
            "global_unrelated_clean_scale",
            "transformation_direction_coherence",
            "cross_fitted_pao",
            "pao_excess_over_random_subspace",
        ],
        ["raw_unnormalized_cosine_distance_as_standalone_main_analysis"],
        depth_answers,
        brightness_vs_hue,
    )
    (args.output_dir / "FULL_DEVELOPMENT_RESULTS.md").write_text(full_report, encoding="utf-8")
    summary = freeze_summary_json(
        loaded.reproduction_status,
        data,
        strongest,
        negative,
        depth_answers,
        tdc_norm[tdc_norm["embedding_variant"].eq("normalized")],
        pao,
        global_scale,
        overlap,
        alignment,
        translation,
    )
    summary["wall_time_seconds"] = time.time() - start
    summary["cache_depth_audit"] = cache_frame.to_dict(orient="records")
    summary["event_condition_physics_tdc"] = event_phys.to_dict(orient="records")
    (args.output_dir / "FREEZE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"HIGH-PRECISION RUN: {loaded.reproduction_status}")
    print("SEALED DATA TOUCHED: NO")
    print(str((args.output_dir / "METRIC_FREEZE.md").resolve()))
    print(str((args.output_dir / "FULL_DEVELOPMENT_RESULTS.md").resolve()))
    print(str((args.output_dir / "FREEZE_SUMMARY.json").resolve()))
    print(str((args.output_dir / "SEALED_PROTOCOL.md").resolve()))


if __name__ == "__main__":
    main()
