#!/usr/bin/env python3
"""Run the post-sealed exploratory transformation-dose manifold audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from transformation_manifold_metrics import (
    cosine_rows,
    curvature_metrics,
    direction_kernel,
    l2_normalize,
    mean_resultant_length,
    orthonormal_basis,
    projection_fraction,
    svd_energy_rank,
    tangent_vectors,
    trajectory_straightness,
)


STATUS = "EXPLORATORY POST-SEALED ANALYSIS"
LAYERS = ["block_00", "block_06", "block_12", "block_18", "final"]
KS = (1, 2, 3, 4, 8)
PROJECTION_KS = (1, 2, 4, 8, 16)
FACTOR_SPECS = {
    "hue": {
        "conditions": ["hue_015", "hue_045", "hue_090"],
        "nominal": [15.0, 45.0, 90.0],
        "path": [15.0, 45.0, 90.0],
        "units": "degrees",
    },
    "brightness": {
        "conditions": ["brightness_0p85", "brightness_0p65", "brightness_0p45"],
        "nominal": [0.85, 0.65, 0.45],
        "path": [0.15, 0.35, 0.55],
        "units": "multiplicative factor",
    },
    "selective_recolor": {
        "conditions": [
            "selective_recolor_045",
            "selective_recolor_090",
            "selective_recolor_180",
        ],
        "nominal": [45.0, 90.0, 180.0],
        "path": [45.0, 90.0, 180.0],
        "units": "degrees within pair-specific hue band",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--appearance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--null-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=260903)
    parser.add_argument("--model-label", default="V-JEPA2-L")
    parser.add_argument("--layer-names", default=",".join(LAYERS))
    parser.add_argument("--expected-scenes", type=int, default=10)
    parser.add_argument("--expected-pairs", type=int, default=20)
    parser.add_argument("--expected-sources", type=int, default=40)
    parser.add_argument("--image-dose-from-samples", action="store_true")
    parser.add_argument("--skip-grid-inventory", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_ci(values: np.ndarray, scenes: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    frame = pd.DataFrame({"value": values, "scene": scenes}).dropna()
    scene_means = frame.groupby("scene", sort=True)["value"].mean().to_numpy()
    if len(scene_means) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(scene_means.mean())
    indices = rng.integers(0, len(scene_means), size=(draws, len(scene_means)))
    boot = scene_means[indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, float(low), float(high)


def mrl_ci(vectors: np.ndarray, scenes: np.ndarray, draws: int, rng: np.random.Generator, floor: float) -> tuple[float, float, float, int]:
    units, valid = l2_normalize(vectors, floor)
    units = units[valid]
    valid_scenes = scenes[valid]
    if len(units) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    unique = np.unique(valid_scenes)
    sums = np.stack([units[valid_scenes == scene].sum(axis=0) for scene in unique])
    counts = np.array([(valid_scenes == scene).sum() for scene in unique], dtype=float)
    observed = float(np.linalg.norm(sums.sum(axis=0) / counts.sum()))
    samples = np.empty(draws, dtype=float)
    batch = 256
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        index = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        total = sums[index].sum(axis=1)
        denominator = counts[index].sum(axis=1)
        samples[start:stop] = np.linalg.norm(total / denominator[:, None], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return observed, float(low), float(high), int((~valid).sum())


def scene_derangement_indices(scenes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(scenes)
    groups = {scene: np.flatnonzero(scenes == scene) for scene in unique}
    sizes = {len(indices) for indices in groups.values()}
    if len(sizes) != 1:
        raise ValueError("balanced per-scene source counts are required for strict bijective nulls")
    order = rng.permutation(unique)
    shift = int(rng.integers(1, len(order)))
    mapping = {order[index]: order[(index + shift) % len(order)] for index in range(len(order))}
    result = np.empty(len(scenes), dtype=int)
    for scene in unique:
        target = groups[scene]
        donor = rng.permutation(groups[mapping[scene]])
        result[target] = donor
    if np.any(scenes == scenes[result]) or len(np.unique(result)) != len(result):
        raise AssertionError("invalid cross-scene bijection")
    return result


def shuffled_mrl_null(clean: np.ndarray, transformed: np.ndarray, scenes: np.ndarray, draws: int, rng: np.random.Generator, floor: float) -> np.ndarray:
    output = np.empty(draws, dtype=float)
    batch = 64
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        permutations = np.stack(
            [scene_derangement_indices(scenes, rng) for _ in range(stop - start)]
        )
        differences = transformed[None, :, :] - clean[permutations]
        norms = np.linalg.norm(differences, axis=2)
        valid = norms > floor
        units = np.divide(
            differences,
            norms[:, :, None],
            out=np.zeros_like(differences),
            where=valid[:, :, None],
        )
        counts = valid.sum(axis=1)
        means = units.sum(axis=1) / np.maximum(counts[:, None], 1)
        output[start:stop] = np.linalg.norm(means, axis=1)
    return output


def shuffled_tangent_mrl_nulls(points: np.ndarray, doses: np.ndarray, scenes: np.ndarray, draws: int, rng: np.random.Generator, floors: np.ndarray) -> np.ndarray:
    """Break dose-to-source trajectories while preserving each dose population."""
    count = points.shape[1]
    identity = np.eye(count, dtype=np.float64)
    weights = np.gradient(identity, doses, axis=0, edge_order=2)
    output = np.empty((draws, count), dtype=float)
    batch = 64
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        current = stop - start
        shuffled = [np.broadcast_to(points[None, :, 0, :], (current, points.shape[0], points.shape[2]))]
        for dose_index in range(1, count):
            permutations = np.stack(
                [scene_derangement_indices(scenes, rng) for _ in range(current)]
            )
            shuffled.append(points[permutations, dose_index, :])
        for tangent_index in range(count):
            tangent = sum(
                weights[tangent_index, dose_index] * shuffled[dose_index]
                for dose_index in range(count)
            )
            norms = np.linalg.norm(tangent, axis=2)
            valid = norms > floors[tangent_index]
            units = np.divide(
                tangent,
                norms[:, :, None],
                out=np.zeros_like(tangent),
                where=valid[:, :, None],
            )
            means = units.sum(axis=1) / np.maximum(valid.sum(axis=1)[:, None], 1)
            output[start:stop, tangent_index] = np.linalg.norm(means, axis=1)
    return output


def read_frames(video_dir: Path, indices: np.ndarray) -> np.ndarray:
    paths = sorted((video_dir / "scene").glob("scene_*.png"))
    if not paths:
        paths = sorted((video_dir / "scene").glob("*.png"))
    arrays = []
    for index in indices:
        with Image.open(paths[int(index)]) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(arrays)


def manifest_l1_map(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for pair in payload["pairs"]:
        for condition in pair["conditions"]:
            for video in condition["videos"]:
                result[str(Path(video["output"]).resolve())] = float(
                    video["mean_absolute_pixel_delta"]
                ) / 255.0
    return result


def image_dose_table(samples: pd.DataFrame, sampled_indices: np.ndarray, manifest_path: Path) -> pd.DataFrame:
    full_l1 = manifest_l1_map(manifest_path)
    clean_rows = samples[samples.condition == "clean"].set_index("base_video_id")
    clean_cache: dict[str, np.ndarray] = {}
    rows = []
    for row_index, row in samples[samples.condition != "clean"].iterrows():
        base_id = row.base_video_id
        if base_id not in clean_cache:
            clean_row = clean_rows.loc[base_id]
            clean_cache[base_id] = read_frames(
                Path(clean_row.video_dir), sampled_indices[int(clean_row._row_index)]
            )
        clean = clean_cache[base_id]
        transformed = read_frames(Path(row.video_dir), sampled_indices[row_index])
        diff = transformed.astype(np.int16) - clean.astype(np.int16)
        changed = np.any(diff != 0, axis=-1)
        transformed_boundary = np.any((transformed == 0) | (transformed == 255), axis=-1)
        clean_boundary = np.any((clean == 0) | (clean == 255), axis=-1)
        factor, nominal, path_dose = condition_metadata(row.condition)
        rows.append(
            {
                "sample_id": row.sample_id,
                "base_video_id": base_id,
                "scene": row.scene,
                "split": row.split,
                "factor": factor,
                "condition": row.condition,
                "nominal_dose": nominal,
                "path_dose": path_dose,
                "pixel_l1": float(np.mean(np.abs(diff)) / 255.0),
                "pixel_l2": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)) / 255.0),
                "fraction_changed_pixels": float(changed.mean()),
                "clipping_fraction": float(transformed_boundary.mean()),
                "new_clipping_fraction": float((transformed_boundary & ~clean_boundary).mean()),
                "full_video_pixel_l1": full_l1.get(str(Path(row.video_dir).resolve()), float("nan")),
            }
        )
    return pd.DataFrame(rows)


def condition_metadata(condition: str) -> tuple[str, float, float]:
    for factor, spec in FACTOR_SPECS.items():
        if condition in spec["conditions"]:
            index = spec["conditions"].index(condition)
            return factor, float(spec["nominal"][index]), float(spec["path"][index])
    raise KeyError(condition)


def trajectory_for_factor(samples: pd.DataFrame, features: np.ndarray, factor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = FACTOR_SPECS[factor]
    clean = samples[samples.condition == "clean"].sort_values("base_video_id")
    source_ids = clean.base_video_id.to_numpy()
    scenes = clean.scene.to_numpy()
    splits = clean.split.to_numpy()
    indices = [clean.index.to_numpy()]
    for condition in spec["conditions"]:
        current = samples[samples.condition == condition].set_index("base_video_id").loc[source_ids]
        indices.append(current._row_index.to_numpy(dtype=int))
    points = np.stack([features[index] for index in indices], axis=1)
    return points, source_ids, scenes, splits


def response_fits(x_train: np.ndarray, y_train: np.ndarray, scene_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, scene_test: np.ndarray) -> list[dict]:
    models: dict[str, np.ndarray] = {}
    models["constant"] = np.full_like(y_test, np.mean(y_train))
    for degree, name in ((1, "linear"), (2, "quadratic")):
        coefficients = np.polyfit(x_train, y_train, degree)
        models[name] = np.polyval(coefficients, x_test)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x_train, y_train)
    models["monotonic_isotonic"] = iso.predict(x_test)

    def saturation(x, amplitude, rate):
        return amplitude * (1.0 - np.exp(-rate * np.maximum(x, 0)))

    try:
        popt, _ = curve_fit(
            saturation,
            x_train,
            y_train,
            p0=(max(float(np.max(y_train)), 1e-6), 1.0 / max(float(np.max(x_train)), 1e-6)),
            bounds=(0, np.inf),
            maxfev=20_000,
        )
        models["saturating"] = saturation(x_test, *popt)
    except (RuntimeError, ValueError):
        models["saturating"] = np.full_like(y_test, np.nan)
    rows = []
    for name, prediction in models.items():
        per_scene = []
        for scene in np.unique(scene_test):
            keep = scene_test == scene
            per_scene.append(np.sqrt(np.mean((prediction[keep] - y_test[keep]) ** 2)))
        rows.append({"fit_model": name, "heldout_scene_rmse": float(np.mean(per_scene))})
    return rows


def hue_models(points: np.ndarray, scenes: np.ndarray, splits: np.ndarray, angles: np.ndarray, rng: np.random.Generator | None = None, null_draws: int = 10_000) -> dict:
    train = splits == "development"
    test = splits == "test"
    delta = points - points[:, :1]
    train_rows = delta[train].reshape(-1, delta.shape[-1])
    _, _, vh = np.linalg.svd(train_rows, full_matrices=False)
    direction = vh[0]
    train_projection = np.einsum("sdf,f->sd", delta[train], direction)
    alpha = train_projection.mean(axis=0)
    ray_prediction = alpha[None, :, None] * direction[None, None, :]

    radians = np.deg2rad(angles)
    design = np.stack([np.cos(radians) - 1.0, np.sin(radians)], axis=1)
    repeated = np.tile(design, (train.sum(), 1))
    coefficients, *_ = np.linalg.lstsq(repeated, train_rows, rcond=None)
    harmonic_prediction = design @ coefficients

    actual = delta[test, 1:]
    denominator = np.sum(actual**2, axis=-1)
    ray_num = np.sum((actual - ray_prediction[:, 1:]) ** 2, axis=-1)
    harmonic_num = np.sum((actual - harmonic_prediction[None, 1:]) ** 2, axis=-1)
    valid = denominator > 1e-16
    ray_error = ray_num[valid] / denominator[valid]
    harmonic_error = harmonic_num[valid] / denominator[valid]

    rank_values = []
    for source in points[test]:
        centered = source - source.mean(axis=0)
        rank_values.append(svd_energy_rank(centered, (1, 2, 3)))

    pair_angle, pair_distance = [], []
    for source in points[test]:
        for a in range(len(angles)):
            for b in range(a + 1, len(angles)):
                pair_angle.append(min(abs(angles[a] - angles[b]), 360 - abs(angles[a] - angles[b])))
                left, _ = l2_normalize(source[[a]])
                right, _ = l2_normalize(source[[b]])
                pair_distance.append(1.0 - float(left[0] @ right[0]))
    ordering = float(spearmanr(pair_angle, pair_distance).statistic)
    angle_null = np.full(null_draws, np.nan)
    if rng is not None:
        angle_array = np.asarray(angles)
        for draw in range(null_draws):
            shuffled = rng.permutation(angle_array)
            shuffled_separation = []
            for _source in points[test]:
                for a in range(len(angles)):
                    for b in range(a + 1, len(angles)):
                        difference = abs(shuffled[a] - shuffled[b])
                        shuffled_separation.append(min(difference, 360 - difference))
            angle_null[draw] = spearmanr(shuffled_separation, pair_distance).statistic
    angle_low, angle_high = np.nanquantile(angle_null, [0.025, 0.975])

    scene_bases = {}
    for scene in np.unique(scenes):
        current = delta[scenes == scene, 1:].reshape(-1, delta.shape[-1])
        scene_bases[scene] = orthonormal_basis(current, 2)
    overlaps = []
    ordered_scenes = sorted(scene_bases)
    for left_index, left in enumerate(ordered_scenes):
        for right in ordered_scenes[left_index + 1:]:
            singular = np.linalg.svd(
                scene_bases[left].T @ scene_bases[right], compute_uv=False
            )
            overlaps.append(float(np.mean(singular**2)))
    return {
        "ray_relative_error": float(np.mean(ray_error)),
        "harmonic_relative_error": float(np.mean(harmonic_error)),
        "harmonic_improvement": float(np.mean(ray_error) - np.mean(harmonic_error)),
        "per_source_energy_1_mean": float(np.mean([row["energy"][1] for row in rank_values])),
        "per_source_energy_2_mean": float(np.mean([row["energy"][2] for row in rank_values])),
        "angular_order_spearman": ordering,
        "angular_order_null_mean": float(np.nanmean(angle_null)),
        "angular_order_null_low": float(angle_low),
        "angular_order_null_high": float(angle_high),
        "angular_order_empirical_p": float(
            (1 + np.sum(np.abs(angle_null) >= abs(ordering))) / (null_draws + 1)
        ),
        "shared_plane_overlap_mean": float(np.mean(overlaps)),
        "shared_plane_overlap_median": float(np.median(overlaps)),
        "shared_plane_random_expectation": 2.0 / points.shape[-1],
        "closure_status": "not_estimable_no_360_degree_endpoint",
    }


def physics_displacements(samples: pd.DataFrame, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = samples[samples.condition == "clean"]
    vectors, scenes, pairs = [], [], []
    for pair_id, frame in clean.groupby("base_pair_id", sort=True):
        if set(frame.possible.astype(bool)) != {False, True}:
            raise ValueError(f"pair orientation invalid: {pair_id}")
        possible = int(frame[frame.possible.astype(bool)].index[0])
        impossible = int(frame[~frame.possible.astype(bool)].index[0])
        vectors.append(features[impossible] - features[possible])
        scenes.append(frame.scene.iloc[0])
        pairs.append(pair_id)
    return np.stack(vectors), np.asarray(scenes), np.asarray(pairs)


def beta_null(k: int, dimension: int, observation_count: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(k / 2.0, (dimension - k) / 2.0, size=(draws, observation_count)).mean(axis=1)


def write_inventory(output: Path) -> None:
    rows = []
    for factor, spec in FACTOR_SPECS.items():
        for dose in spec["nominal"]:
            rows.append({
                "factor": factor,
                "dose": dose,
                "units": spec["units"],
                "implementation": "scripts/appearance_interventions.py",
                "split": "Original IntPhys O1 local pilot",
                "scenes": 10,
                "videos": 40,
                "models": "V-JEPA2-L feasibility",
                "layers": "0,6,12,18,24",
                "transformed_video_available": True,
                "embedding_available": True,
                "joint_grid": False,
                "notes": "Measured locally; not a substitute for either requested giant model.",
            })
    for block in ("O1", "O2", "O3"):
        for factor, spec in FACTOR_SPECS.items():
            for dose in spec["nominal"]:
                rows.append({
                    "factor": factor,
                    "dose": dose,
                    "units": spec["units"],
                    "implementation": "scripts/appearance_interventions.py",
                    "split": f"Original IntPhys {block}",
                    "scenes": 10,
                    "videos": 120 if block != "O1-local-pilot" else 40,
                    "models": "none locally for full grid",
                    "layers": "none locally for full grid",
                    "transformed_video_available": "remote archive only",
                    "embedding_available": False,
                    "joint_grid": False,
                    "notes": "Independent transform; full generation provenance in results/20-*.",
                })
    for family, doses, units in (
        ("brightening", [1.15, 1.35, 1.60], "multiplicative factor"),
        ("foreground_recolor", [45, 90, 180], "degrees"),
        ("background_recolor", [45, 90, 180], "degrees"),
    ):
        for dose in doses:
            rows.append({
                "factor": family,
                "dose": dose,
                "units": units,
                "implementation": "scripts/harder_appearance_interventions.py",
                "split": "Original IntPhys O2 diagnostic",
                "scenes": 10,
                "videos": 120,
                "models": "none",
                "layers": "none",
                "transformed_video_available": "remote archive only",
                "embedding_available": False,
                "joint_grid": False,
                "notes": "Diagnostic O2 grid; foreground uses non-border instance-mask labels.",
            })
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "GRID_INVENTORY.csv", index=False)
    local_rows = []
    for factor, spec in FACTOR_SPECS.items():
        for dose in spec["nominal"]:
            local_rows.append(
                f"| {factor} | {dose:g} | {spec['units']} | O1 scenes 01-10, 20 pairs, 40 videos | yes | V-JEPA2-L newly extracted; giant models unavailable |"
            )
    text = f"""# Grid Inventory

**{STATUS}.** This inventory was rebuilt from pre-2026-09-03 09:36:06 +03:00 code, manifests, experiment records, and external caches. Files created or modified in the preceding five hours were excluded as inputs.

## Locally Usable Grid

| Factor | Dose | Units | Population | Videos present | Compatible embeddings |
|---|---:|---|---|---|---|
{chr(10).join(local_rows)}

The local O1 pilot contains 360 transformed videos plus 40 clean sources. It has three independently applied factors and three nonzero doses per factor. All source videos have 100 PNG frames. The appearance manifest reports all integrity checks passed.

## Verified Implementations

- Hue: global HSV hue shift, circular parameter, +15/+45/+90 degrees.
- Brightness: RGB multiplication and uint8 clipping, scale parameter, factors 0.85/0.65/0.45.
- Selective recolor: pair-specific saturated hue-band shift, +45/+90/+180 degrees; the selected band is not a semantic object mask.
- O2 diagnostic only: brightening 1.15/1.35/1.60 and foreground/background recoloring +45/+90/+180.

No saturation scale, RGB-channel grid, joint factor grid, or ordered-composition grid was found. Therefore saturation, RGB, interaction, and commutativity conclusions are unavailable rather than negative.

## Cache Findings

- The documented 70 MB partial V-JEPA2-G appearance matrix stores predictor-error maps and scalar losses, not pooled layer embeddings. It is invalid for this geometry audit.
- The corrected development and sealed giant-model caches contain only clean and the three strongest appearance conditions. They do not provide a dose grid and were not recomputed or modified.
- No local V-JEPA2-G or VideoMAE2-G checkpoint/cache supports the multi-dose O1 grid.
- A verified V-JEPA2-L checkpoint is local and supports a bounded feasibility extraction. Its results are not evidence about either giant model.

`GRID_INVENTORY.csv` records the full O1/O2/O3 and O2-diagnostic provenance-level inventory. Full transformed trees are documented in durable remote archives but were not downloaded for this local run.
"""
    (output / "GRID_INVENTORY.md").write_text(text, encoding="utf-8")


def plot_outputs(output: Path, dose: pd.DataFrame, direction: pd.DataFrame, kernels: pd.DataFrame, spectra: pd.DataFrame, straight: pd.DataFrame, curvature: pd.DataFrame, hue: pd.DataFrame, physics: pd.DataFrame) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, factor in zip(axes, FACTOR_SPECS):
        subset = dose[dose.factor == factor]
        for layer in LAYERS:
            current = subset[subset.layer == layer].sort_values("pixel_l1_mean")
            axis.errorbar(
                current.pixel_l1_mean,
                current.rdm_mean,
                yerr=np.vstack([
                    current.rdm_mean - current.rdm_ci_low,
                    current.rdm_ci_high - current.rdm_mean,
                ]),
                marker="o",
                capsize=2,
                label=layer,
            )
        axis.set_title(factor.replace("_", " "))
        axis.set_xlabel("realized RGB mean absolute difference")
    axes[0].set_ylabel("unrelated-normalized latent magnitude")
    axes[-1].legend(frameon=False, fontsize=7)
    model_label = dose.model.iloc[0] if not dose.empty else "latent model"
    fig.suptitle(f"Post-sealed exploratory dose response: {model_label}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_1_dose_response.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, factor in zip(axes, FACTOR_SPECS):
        subset = direction[direction.factor == factor]
        for layer in LAYERS:
            current = subset[subset.layer == layer].sort_values("path_dose")
            axis.errorbar(
                current.path_dose,
                current.mrl_excess,
                yerr=np.vstack([
                    current.mrl_observed - current.mrl_ci_low,
                    current.mrl_ci_high - current.mrl_observed,
                ]),
                marker="o",
                capsize=2,
                label=layer,
            )
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set_title(factor.replace("_", " "))
        axis.set_xlabel("factor-native path dose")
    axes[0].set_ylabel("MRL observed - shuffled mean")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Null-adjusted directional coherence by dose")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_2_direction_by_dose.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 5, figsize=(13, 7.5), sharex=False, sharey=False)
    for row, factor in enumerate(FACTOR_SPECS):
        labels = ["mild", "medium", "strong"]
        for column, layer in enumerate(LAYERS):
            matrix = kernels[(kernels.factor == factor) & (kernels.layer == layer)].pivot(index="dose_a", columns="dose_b", values="cosine_mean").to_numpy()
            image = axes[row, column].imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            axes[row, column].set_xticks(range(3), labels, rotation=45, ha="right", fontsize=7)
            axes[row, column].set_yticks(range(3), labels, fontsize=7)
            if row == 0:
                axes[row, column].set_title(layer)
            if column == 0:
                axes[row, column].set_ylabel(factor.replace("_", " "))
    fig.subplots_adjust(top=0.9, right=0.88, wspace=0.35, hspace=0.45)
    color_axis = fig.add_axes([0.91, 0.16, 0.015, 0.68])
    fig.colorbar(image, cax=color_axis, label="matched-source signed cosine")
    fig.suptitle("Dose-by-dose displacement-direction kernels")
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_3_dose_direction_kernels.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for factor in FACTOR_SPECS:
        current = spectra[(spectra.factor == factor) & (spectra.k == 1)]
        axes[0].plot(current.layer, current.explained_energy, marker="o", label=factor)
        rank = spectra[(spectra.factor == factor) & (spectra.k == 1)]
        axes[1].plot(rank.layer, rank.effective_rank, marker="o", label=factor)
    axes[0].set_ylabel("top-1 displacement energy")
    axes[1].set_ylabel("entropy effective rank")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.set_xlabel("layer")
    axes[1].legend(frameon=False)
    fig.suptitle("Trajectory dimensionality")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_4_trajectory_dimensionality.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for factor in FACTOR_SPECS:
        current = straight[straight.factor == factor]
        axes[0].plot(current.layer, current.straightness_mean, marker="o", label=factor)
        turn = curvature[curvature.factor == factor].groupby("layer", sort=False).turn_cosine_mean.mean().reindex(LAYERS)
        axes[1].plot(LAYERS, turn, marker="o", label=factor)
    axes[0].set_ylabel("endpoint / path length")
    axes[1].set_ylabel("adjacent-segment turn cosine")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.set_xlabel("layer")
    axes[1].legend(frameon=False)
    fig.suptitle("Trajectory straightness and turning")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_5_straightness_curvature.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 3.6))
    axis.plot(hue.layer, hue.ray_relative_error, marker="o", label="shared ray")
    axis.plot(hue.layer, hue.harmonic_relative_error, marker="o", label="2D harmonic")
    axis.axhline(1, color="#777777", lw=0.8, label="zero-displacement baseline")
    axis.set_ylabel("held-out relative reconstruction error")
    axis.set_xlabel("layer")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.set_title("Hue model comparison; only 0-90 degrees observed")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_6_hue_geometry.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    for space in ("endpoint_strong", "all_dose", "local_tangent"):
        current = physics[(physics.space == space) & (physics.k == 4)]
        axis.errorbar(
            current.layer,
            current.projection_mean,
            yerr=np.vstack([
                current.projection_mean - current.ci_low,
                current.ci_high - current.projection_mean,
            ]),
            marker="o",
            capsize=3,
            label=space,
        )
    random = physics[(physics.space == "all_dose") & (physics.k == 4)]
    axis.plot(random.layer, random.random_mean, linestyle="--", color="#555555", label="isotropic random k=4")
    axis.set_ylabel("held-out physical projection fraction")
    axis.set_xlabel("layer")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    axis.set_title("Exploratory appearance spaces vs physics")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"figure_7_appearance_tangent_vs_physics.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global LAYERS
    args = parse_args()
    LAYERS = args.layer_names.split(",")
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_grid_inventory:
        write_inventory(output)
    bundle = np.load(args.features)
    raw = bundle["pooled"].astype(np.float64)
    native_layers = bundle["layers"].tolist()
    sampled_indices = bundle["sampled_source_indices"]
    samples = pd.DataFrame(json.loads(args.samples.read_text(encoding="utf-8")))
    samples["_row_index"] = np.arange(len(samples), dtype=int)
    if len(samples) != len(raw) or raw.shape[1] != 5:
        raise ValueError("feature/sample shape mismatch")
    if (samples.scene.nunique(), samples.base_pair_id.nunique(), samples.base_video_id.nunique()) != (args.expected_scenes, args.expected_pairs, args.expected_sources):
        raise ValueError("unexpected O1 pilot population")
    expected_conditions = {"clean"} | {condition for spec in FACTOR_SPECS.values() for condition in spec["conditions"]}
    if set(samples.condition) != expected_conditions:
        raise ValueError("condition set mismatch")

    if args.image_dose_from_samples:
        image_dose = samples[samples.condition != "clean"][[
            "sample_id", "base_video_id", "scene", "split", "factor", "condition",
            "nominal_dose", "path_dose", "rgb_mean_absolute_difference",
            "rgb_root_mean_square_difference", "changed_pixel_fraction", "clipping_fraction",
        ]].rename(columns={
            "rgb_mean_absolute_difference": "pixel_l1",
            "rgb_root_mean_square_difference": "pixel_l2",
            "changed_pixel_fraction": "fraction_changed_pixels",
        })
        image_dose["new_clipping_fraction"] = image_dose["clipping_fraction"]
        image_dose["full_video_pixel_l1"] = np.nan
    else:
        image_dose = image_dose_table(samples, sampled_indices, args.appearance_manifest)
    image_dose.to_csv(output / "IMAGE_SPACE_DOSE.csv", index=False)
    pixel_by_sample = image_dose.set_index("sample_id")
    dose_rows, response_fit_rows, direction_rows, kernel_rows = [], [], [], []
    tangent_rows, curvature_rows, straight_rows = [], [], []
    spectra_rows, rank_rows, ray_rows, cross_rows, hue_rows = [], [], [], [], []
    physics_rows, statistics_rows = [], []
    rng = np.random.default_rng(args.seed)

    normalized, valid_embeddings = l2_normalize(raw.reshape(-1, raw.shape[-1]))
    if not np.all(valid_embeddings):
        raise ValueError("zero or invalid pooled embedding")
    normalized = normalized.reshape(raw.shape)

    for layer_index, layer in enumerate(LAYERS):
        layer_values = normalized[:, layer_index]
        layer_raw = raw[:, layer_index]
        clean_frame = samples[samples.condition == "clean"].sort_values("base_video_id")
        clean_indices = clean_frame.index.to_numpy()
        clean_values = layer_values[clean_indices]
        clean_scenes = clean_frame.scene.to_numpy()
        clean_cos = clean_values @ clean_values.T
        cross_scene = clean_scenes[:, None] != clean_scenes[None, :]
        upper = np.triu(np.ones_like(cross_scene, dtype=bool), 1)
        global_scale = float(np.median((1.0 - clean_cos)[cross_scene & upper]))
        statistics_rows.append({"model": args.model_label, "layer": layer, "factor": "clean", "dose": 0, "metric": "global_unrelated_scale", "estimate": global_scale, "ci_low": np.nan, "ci_high": np.nan, "null_mean": np.nan, "null_low": np.nan, "null_high": np.nan, "status": STATUS})

        factor_cache = {}
        for factor, spec in FACTOR_SPECS.items():
            points, source_ids, scenes, splits = trajectory_for_factor(samples, layer_values, factor)
            raw_points, _, _, _ = trajectory_for_factor(samples, layer_raw, factor)
            delta = points[:, 1:] - points[:, :1]
            raw_delta = raw_points[:, 1:] - raw_points[:, :1]
            floor = max(1e-10, float(np.quantile(np.linalg.norm(delta.reshape(-1, delta.shape[-1]), axis=1), 0.01)) * 1e-6)
            tangents = tangent_vectors(points, np.asarray([0.0, *spec["path"]]))
            factor_cache[factor] = (points, delta, tangents, source_ids, scenes, splits, floor)

            spectrum = svd_energy_rank(delta.reshape(-1, delta.shape[-1]), KS)
            for k in KS:
                spectra_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "k": k, "explained_energy": spectrum["energy"][k], "effective_rank": spectrum["effective_rank"], "sample_count": len(delta) * delta.shape[1], "status": STATUS})
            rank_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "effective_rank": spectrum["effective_rank"], "top1_energy": spectrum["energy"][1], "top2_energy": spectrum["energy"][2], "status": STATUS})

            straightness = trajectory_straightness(points)
            s_mean, s_low, s_high = scene_ci(straightness, scenes, args.bootstrap_draws, rng)
            straight_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "straightness_mean": s_mean, "ci_low": s_low, "ci_high": s_high, "scene_count": len(np.unique(scenes)), "status": STATUS})

            turns, normalized_curvature = curvature_metrics(points)
            for turn_index in range(turns.shape[1]):
                center = float(spec["path"][turn_index])
                mean, low, high = scene_ci(turns[:, turn_index], scenes, args.bootstrap_draws, rng)
                curv_mean, curv_low, curv_high = scene_ci(normalized_curvature[:, turn_index], scenes, args.bootstrap_draws, rng)
                curvature_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "turn_index": turn_index, "center_path_dose": center, "turn_cosine_mean": mean, "turn_cosine_ci_low": low, "turn_cosine_ci_high": high, "normalized_direction_change_mean": curv_mean, "normalized_direction_change_ci_low": curv_low, "normalized_direction_change_ci_high": curv_high, "status": STATUS})

            tangent_norms = np.linalg.norm(tangents, axis=2)
            tangent_floors = np.maximum(
                1e-12, np.quantile(tangent_norms, 0.01, axis=0) * 1e-6
            )
            tangent_nulls = shuffled_tangent_mrl_nulls(
                points,
                np.asarray([0.0, *spec["path"]]),
                scenes,
                args.null_draws,
                rng,
                tangent_floors,
            )
            for tangent_index, path_dose in enumerate([0.0, *spec["path"]]):
                tangent = tangents[:, tangent_index]
                tangent_floor = float(tangent_floors[tangent_index])
                observed, ci_low, ci_high, tiny = mrl_ci(tangent, scenes, args.bootstrap_draws, rng, tangent_floor)
                tangent_spectrum = svd_energy_rank(tangent, KS)
                tangent_null = tangent_nulls[:, tangent_index]
                tangent_null_low, tangent_null_high = np.quantile(
                    tangent_null, [0.025, 0.975]
                )
                tangent_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "path_dose": path_dose, "difference_scheme": "second-order one-sided boundary" if tangent_index in (0, len(spec["path"])) else "central irregular-grid finite difference", "tangent_norm_mean": float(np.mean(np.linalg.norm(tangent, axis=1))), "tangent_mrl": observed, "tangent_mrl_ci_low": ci_low, "tangent_mrl_ci_high": ci_high, "tangent_null_mean": float(tangent_null.mean()), "tangent_null_low": float(tangent_null_low), "tangent_null_high": float(tangent_null_high), "tangent_mrl_excess": observed - float(tangent_null.mean()), "tangent_empirical_p": float((1 + np.sum(tangent_null >= observed)) / (len(tangent_null) + 1)), "tangent_effective_rank": tangent_spectrum["effective_rank"], "tiny_count": tiny, "floor": tangent_floor, "null_status": "dose-wise cross-scene bijective source shuffle", "status": STATUS})

            kernel = direction_kernel(delta, floor)
            for a in range(3):
                for b in range(3):
                    values = cosine_rows(delta[:, a], delta[:, b], floor)
                    mean, low, high = scene_ci(values, scenes, args.bootstrap_draws, rng)
                    kernel_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "dose_a": float(spec["nominal"][a]), "dose_b": float(spec["nominal"][b]), "cosine_mean": mean, "ci_low": low, "ci_high": high, "matrix_check": kernel[a, b], "status": STATUS})

            factor_rdm_values = []
            factor_pixel_values = []
            for dose_index, (condition, nominal, path_dose) in enumerate(zip(spec["conditions"], spec["nominal"], spec["path"])):
                transformed_frame = samples[samples.condition == condition].set_index("base_video_id").loc[source_ids]
                transformed_indices = transformed_frame.index.to_numpy()
                raw_cosine_distance = 1.0 - np.einsum("ij,ij->i", points[:, 0], points[:, dose_index + 1])
                rdm = raw_cosine_distance / global_scale
                r_mean, r_low, r_high = scene_ci(rdm, scenes, args.bootstrap_draws, rng)
                image_current = pixel_by_sample.loc[transformed_frame.sample_id]
                factor_rdm_values.append(rdm)
                factor_pixel_values.append(image_current.pixel_l1.to_numpy())
                dose_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "condition": condition, "nominal_dose": nominal, "path_dose": path_dose, "units": spec["units"], "global_scale": global_scale, "raw_cosine_distance_mean": float(np.mean(raw_cosine_distance)), "raw_displacement_norm_mean": float(np.mean(np.linalg.norm(raw_delta[:, dose_index], axis=1))), "rdm_mean": r_mean, "rdm_ci_low": r_low, "rdm_ci_high": r_high, "pixel_l1_mean": float(image_current.pixel_l1.mean()), "pixel_l2_mean": float(image_current.pixel_l2.mean()), "fraction_changed_pixels_mean": float(image_current.fraction_changed_pixels.mean()), "clipping_fraction_mean": float(image_current.clipping_fraction.mean()), "new_clipping_fraction_mean": float(image_current.new_clipping_fraction.mean()), "status": STATUS})

                observed, ci_low, ci_high, tiny = mrl_ci(delta[:, dose_index], scenes, args.bootstrap_draws, rng, floor)
                null = shuffled_mrl_null(points[:, 0], points[:, dose_index + 1], scenes, args.null_draws, rng, floor)
                null_low, null_high = np.quantile(null, [0.025, 0.975])
                raw_mrl = mean_resultant_length(raw_delta[:, dose_index], floor)
                direction_rows.append({"model": args.model_label, "layer": layer, "factor": factor, "condition": condition, "nominal_dose": nominal, "path_dose": path_dose, "mrl_observed": observed, "mrl_ci_low": ci_low, "mrl_ci_high": ci_high, "shuffled_mean": float(null.mean()), "shuffled_low": float(null_low), "shuffled_high": float(null_high), "mrl_excess": observed - float(null.mean()), "empirical_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)), "raw_embedding_mrl": raw_mrl, "raw_minus_normalized": raw_mrl - observed, "tiny_count": tiny, "tiny_fraction": tiny / len(delta), "floor": floor, "status": STATUS})

            train = splits == "development"
            test = splits == "test"
            train_vectors = delta[train].reshape(-1, delta.shape[-1])
            train_units, train_valid = l2_normalize(train_vectors, floor)
            ray_direction = train_units[train_valid].mean(axis=0)
            ray_norm = np.linalg.norm(ray_direction)
            if not np.isfinite(ray_norm) or ray_norm <= floor:
                raise ValueError(f"undefined train-scene ray for {layer}/{factor}")
            ray_direction /= ray_norm
            for dose_index, (condition, nominal, path_dose) in enumerate(
                zip(spec["conditions"], spec["nominal"], spec["path"])
            ):
                heldout_delta = delta[test, dose_index]
                heldout_scenes = scenes[test]
                directional_cosine = cosine_rows(
                    heldout_delta,
                    np.broadcast_to(ray_direction, heldout_delta.shape),
                    floor,
                )
                projection = directional_cosine**2
                orthogonal = 1.0 - projection
                cosine_mean, cosine_low, cosine_high = scene_ci(
                    directional_cosine,
                    heldout_scenes,
                    args.bootstrap_draws,
                    rng,
                )
                projection_mean, projection_low, projection_high = scene_ci(
                    projection,
                    heldout_scenes,
                    args.bootstrap_draws,
                    rng,
                )
                orthogonal_mean, orthogonal_low, orthogonal_high = scene_ci(
                    orthogonal,
                    heldout_scenes,
                    args.bootstrap_draws,
                    rng,
                )
                ray_rows.append({
                    "model": args.model_label,
                    "layer": layer,
                    "factor": factor,
                    "condition": condition,
                    "nominal_dose": nominal,
                    "path_dose": path_dose,
                    "direction_fit": "normalized mean of train-scene unit displacements across all nonzero doses",
                    "directional_cosine_mean": cosine_mean,
                    "directional_cosine_ci_low": cosine_low,
                    "directional_cosine_ci_high": cosine_high,
                    "projection_fraction_mean": projection_mean,
                    "projection_fraction_ci_low": projection_low,
                    "projection_fraction_ci_high": projection_high,
                    "orthogonal_residual_fraction_mean": orthogonal_mean,
                    "orthogonal_residual_fraction_ci_low": orthogonal_low,
                    "orthogonal_residual_fraction_ci_high": orthogonal_high,
                    "train_scene_count": len(np.unique(scenes[train])),
                    "heldout_scene_count": len(np.unique(heldout_scenes)),
                    "status": STATUS,
                })
            y_matrix = np.column_stack([np.zeros(len(scenes)), *factor_rdm_values])
            path_matrix = np.broadcast_to(
                np.asarray([0.0, *spec["path"]]), y_matrix.shape
            )
            pixel_matrix = np.column_stack(
                [np.zeros(len(scenes)), *factor_pixel_values]
            )
            for scale_name, x_matrix in (
                ("factor_native_path_dose", path_matrix),
                ("realized_rgb_l1", pixel_matrix),
            ):
                fits = response_fits(
                    x_matrix[train].ravel(),
                    y_matrix[train].ravel(),
                    np.repeat(scenes[train], y_matrix.shape[1]),
                    x_matrix[test].ravel(),
                    y_matrix[test].ravel(),
                    np.repeat(scenes[test], y_matrix.shape[1]),
                )
                for fit in fits:
                    response_fit_rows.append({
                        "model": args.model_label,
                        "layer": layer,
                        "factor": factor,
                        "dose_scale": scale_name,
                        **fit,
                        "train_scenes": int(len(np.unique(scenes[train]))),
                        "heldout_scenes": int(len(np.unique(scenes[test]))),
                        "status": STATUS,
                    })

        for left_index, left in enumerate(FACTOR_SPECS):
            for right in list(FACTOR_SPECS)[left_index + 1:]:
                left_tangent = factor_cache[left][2][:, 0]
                right_tangent = factor_cache[right][2][:, 0]
                scenes = factor_cache[left][4]
                values = cosine_rows(left_tangent, right_tangent)
                mean, low, high = scene_ci(values, scenes, args.bootstrap_draws, rng)
                cross_rows.append({"model": args.model_label, "layer": layer, "factor_a": left, "factor_b": right, "location": "clean-side tangent", "signed_cosine_mean": mean, "ci_low": low, "ci_high": high, "status": STATUS})

        hue_points, _, _, _, hue_scenes, hue_splits, _ = factor_cache["hue"]
        hue_result = hue_models(
            hue_points,
            hue_scenes,
            hue_splits,
            np.asarray([0.0, 15.0, 45.0, 90.0]),
            rng,
            args.null_draws,
        )
        hue_rows.append({"model": args.model_label, "layer": layer, **hue_result, "status": STATUS})

        physical, physical_scenes, _ = physics_displacements(samples, layer_values)
        heldout_scene_ids = samples.loc[samples.split == "test", "scene"].unique()
        heldout = np.isin(physical_scenes, heldout_scene_ids)
        physical_test = physical[heldout]
        physical_test_scenes = physical_scenes[heldout]
        appearance_train = []
        endpoint_train = []
        tangent_train = []
        for factor in FACTOR_SPECS:
            points, delta, tangents, _, _, splits, _ = factor_cache[factor]
            train = splits == "development"
            appearance_train.append(delta[train].reshape(-1, delta.shape[-1]))
            endpoint_train.append(delta[train, -1])
            tangent_train.append(tangents[train, 0])
        spaces = {
            "endpoint_strong": np.concatenate(endpoint_train),
            "all_dose": np.concatenate(appearance_train),
            "local_tangent": np.concatenate(tangent_train),
        }
        for space_name, matrix in spaces.items():
            max_rank = min(matrix.shape)
            for k in PROJECTION_KS:
                if k > max_rank:
                    continue
                basis = orthonormal_basis(matrix, k)
                values = projection_fraction(physical_test, basis)
                mean, low, high = scene_ci(values, physical_test_scenes, args.bootstrap_draws, rng)
                null = beta_null(k, physical_test.shape[1], len(values), args.null_draws, rng)
                null_low, null_high = np.quantile(null, [0.025, 0.975])
                physics_rows.append({"model": args.model_label, "layer": layer, "space": space_name, "k": k, "projection_mean": mean, "ci_low": low, "ci_high": high, "random_mean": float(null.mean()), "random_low": float(null_low), "random_high": float(null_high), "excess": mean - float(null.mean()), "theoretical_k_over_d": k / physical_test.shape[1], "train_scene_count": int(samples.loc[samples.split == "development", "scene"].nunique()), "heldout_scene_count": int(len(np.unique(physical_test_scenes))), "heldout_pair_count": len(values), "random_null": "Haar-isotropic marginal Beta Monte Carlo; dependence approximation", "status": STATUS})

    dose = pd.DataFrame(dose_rows)
    response_fits_frame = pd.DataFrame(response_fit_rows)
    direction = pd.DataFrame(direction_rows)
    kernels = pd.DataFrame(kernel_rows)
    tangents = pd.DataFrame(tangent_rows)
    curvature = pd.DataFrame(curvature_rows)
    straight = pd.DataFrame(straight_rows)
    spectra = pd.DataFrame(spectra_rows)
    ranks = pd.DataFrame(rank_rows)
    rays = pd.DataFrame(ray_rows)
    hue = pd.DataFrame(hue_rows)
    cross = pd.DataFrame(cross_rows)
    physics = pd.DataFrame(physics_rows)
    statistics = pd.DataFrame(statistics_rows)
    outputs = {
        "DOSE_RESPONSE.csv": dose,
        "DOSE_RESPONSE_FITS.csv": response_fits_frame,
        "DIRECTION_BY_DOSE.csv": direction,
        "DOSE_DIRECTION_KERNELS.csv": kernels,
        "TANGENT_METRICS.csv": tangents,
        "CURVATURE_METRICS.csv": curvature,
        "STRAIGHTNESS_METRICS.csv": straight,
        "SUBSPACE_SPECTRA.csv": spectra,
        "EFFECTIVE_RANK.csv": ranks,
        "RAY_METRICS.csv": rays,
        "HUE_CIRCULARITY.csv": hue,
        "CROSS_FACTOR_ALIGNMENT.csv": cross,
        "PHYSICS_APPEARANCE_MANIFOLD.csv": physics,
        "STATISTICAL_SUMMARY.csv": statistics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    pd.DataFrame([{
        "model": "not_applicable",
        "layer": "not_applicable",
        "factor_a": "not_available",
        "factor_b": "not_available",
        "dose_a": np.nan,
        "dose_b": np.nan,
        "interaction_fraction": np.nan,
        "status": STATUS,
        "reason": "No joint transformation grid exists",
    }]).to_csv(output / "INTERACTIONS.csv", index=False)
    plot_outputs(output, dose, direction, kernels, spectra, straight, curvature, hue, physics)

    final = direction[direction.layer == "final"]
    final_rank = ranks[ranks.layer == "final"].set_index("factor")
    final_spectra = spectra[spectra.layer == "final"].set_index(["factor", "k"])
    final_straight = straight[straight.layer == "final"].set_index("factor")
    classification = {}
    for factor in FACTOR_SPECS:
        rank = final_rank.loc[factor]
        top4_energy = float(final_spectra.loc[(factor, 4), "explained_energy"])
        line = final_straight.loc[factor]
        if rank.top1_energy >= 0.85 and line.straightness_mean >= 0.90:
            label = "APPROXIMATELY 1D / RAY-LIKE"
        elif rank.top2_energy >= 0.78 and top4_energy >= 0.88 and line.straightness_mean < 0.95:
            label = "CURVED LOW-DIMENSIONAL"
        elif top4_energy < 0.85 or rank.effective_rank >= 5:
            label = "HIGHER-DIMENSIONAL"
        elif line.straightness_mean < 0.80:
            label = "SCENE-DEPENDENT"
        else:
            label = "INCONCLUSIVE"
        classification[factor] = {"label": label, "top1_energy": float(rank.top1_energy), "top2_energy": float(rank.top2_energy), "effective_rank": float(rank.effective_rank), "straightness": float(line.straightness_mean)}

    summary = {
        "status": "PASS_WITH_MODEL_COVERAGE_LIMITATION",
        "analysis_status": "post-sealed exploratory",
        "fresh_start_cutoff": "2026-09-03T09:36:06+03:00",
        "sealed_confirmatory_analysis_modified": False,
        "grid_inventory": {"local_population": "Original IntPhys O1 pilot", "scenes": 10, "pairs": 20, "source_videos": 40, "conditions": 10, "factors": {key: value["nominal"] for key, value in FACTOR_SPECS.items()}, "joint_grids": False},
        "models": {
            "V-JEPA2-G": {"status": "not_evaluated", "reason": "no compatible multi-dose embedding cache or locally runnable checkpoint"},
            "VideoMAE2-G": {"status": "not_evaluated", "reason": "no compatible multi-dose embedding cache or locally runnable checkpoint"},
            "V-JEPA2-L": {"status": "measured_feasibility_pilot", "layers": LAYERS, "native_depths": native_layers, "classification": classification},
        },
        "factors": classification,
        "cross_factor": {"clean_side_tangent_final": cross[cross.layer == "final"].to_dict("records")},
        "interactions": {"status": "unavailable", "reason": "no joint grids"},
        "physics_projection": {"final_k4": physics[(physics.layer == "final") & (physics.k == 4)].to_dict("records")},
        "hue": hue.to_dict("records"),
        "claims_supported": ["On the V-JEPA2-L O1 feasibility population, dose geometry differs across the three implemented appearance factors if the reported curves and intervals differ.", "The local grid is insufficient to test hue periodicity or closure."],
        "claims_not_supported": ["Any claim about V-JEPA2-G or VideoMAE2-G dose manifolds", "A latent color wheel", "Disentanglement, physical understanding, or causal factorization", "Additivity or commutativity of factors"],
        "paper_recommendation": {"main_text_worthy": False, "reason": "measured model and population differ from the paper-facing giant-model corrected IntPhys2 evaluation"},
        "resampling": {"scene_bootstrap_draws": args.bootstrap_draws, "direction_null_draws": args.null_draws, "seed": args.seed},
        "provenance": {"features_sha256": sha256(args.features), "samples_sha256": sha256(args.samples), "appearance_manifest_sha256": sha256(args.appearance_manifest), "script_sha256": sha256(Path(__file__)), "python": platform.python_version(), "numpy": np.__version__},
    }
    (output / "TRANSFORMATION_MANIFOLD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "HUE_CIRCULARITY.md").write_text(f"""# Hue Circularity

**{STATUS}.** Only 0, 15, 45, and 90 degree hue states exist locally. This covers one quarter-turn and contains neither negative rotations nor a 360-degree endpoint. Consequently periodicity, closure, and a latent color-wheel claim are not identifiable.

The held-out shared-ray and 2D harmonic errors are reported in `HUE_CIRCULARITY.csv`. A lower harmonic error is descriptive evidence that two fixed trigonometric basis functions fit this observed arc better; it is not evidence of a closed circle. Angular-order Spearman values use within-source latent distances over the four observed angles.
""", encoding="utf-8")
    elapsed = time.perf_counter() - started
    (output / "RUN_METADATA.json").write_text(json.dumps({"elapsed_seconds": elapsed, "status": STATUS, "outputs": sorted(outputs)}, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "elapsed_seconds": elapsed, "classification": classification}, indent=2))


if __name__ == "__main__":
    main()
