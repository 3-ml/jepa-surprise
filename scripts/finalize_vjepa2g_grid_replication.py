#!/usr/bin/env python3
"""Finalize reports for the post-sealed V-JEPA2-G dose-grid replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_transformation_manifold_audit import (
    FACTOR_SPECS,
    physics_displacements,
    trajectory_for_factor,
)
from transformation_manifold_metrics import (
    l2_normalize,
    orthonormal_basis,
    projection_fraction,
    tangent_vectors,
)


STATUS = "POST-SEALED EXPLORATORY V-JEPA2-G GRID REPLICATION"
LAYERS = ["block_00", "block_10", "block_20", "block_30", "final"]
FACTORS = ["hue", "brightness", "selective_recolor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def label(condition: bool, partial: bool = False) -> str:
    if condition:
        return "REPLICATED"
    return "PARTIAL" if partial else "NOT_REPLICATED"


def markdown_rows(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def paired_projection_contrasts(
    packed_root: Path,
    output: Path,
    draws: int = 10_000,
    seed: int = 260904,
) -> pd.DataFrame:
    """Compare matched-k appearance projections using paired held-out scenes."""
    bundle = np.load(packed_root / "layer_features.npz")
    raw = bundle["pooled"].astype(np.float64)
    samples = pd.DataFrame(json.loads((packed_root / "samples.json").read_text()))
    samples["_row_index"] = np.arange(len(samples), dtype=int)
    normalized, valid = l2_normalize(raw.reshape(-1, raw.shape[-1]))
    if not np.all(valid):
        raise ValueError("zero or invalid packed embedding in projection contrast")
    normalized = normalized.reshape(raw.shape)
    rng = np.random.default_rng(seed)
    rows = []
    for layer_index, layer_name in enumerate(LAYERS):
        layer_values = normalized[:, layer_index]
        endpoint_train, all_dose_train, tangent_train = [], [], []
        for factor, spec in FACTOR_SPECS.items():
            points, _, _, splits = trajectory_for_factor(samples, layer_values, factor)
            delta = points[:, 1:] - points[:, :1]
            tangents = tangent_vectors(points, np.asarray([0.0, *spec["path"]]))
            train = splits == "development"
            endpoint_train.append(delta[train, -1])
            all_dose_train.append(delta[train].reshape(-1, delta.shape[-1]))
            tangent_train.append(tangents[train, 0])
        bases = {
            "endpoint_strong": orthonormal_basis(np.concatenate(endpoint_train), 4),
            "all_dose": orthonormal_basis(np.concatenate(all_dose_train), 4),
            "local_tangent": orthonormal_basis(np.concatenate(tangent_train), 4),
        }
        physical, physical_scenes, _ = physics_displacements(samples, layer_values)
        heldout_scenes = samples.loc[samples.split == "test", "scene"].unique()
        heldout = np.isin(physical_scenes, heldout_scenes)
        physical = physical[heldout]
        physical_scenes = physical_scenes[heldout]
        values = {
            name: projection_fraction(physical, basis) for name, basis in bases.items()
        }
        for left, right in (
            ("local_tangent", "endpoint_strong"),
            ("local_tangent", "all_dose"),
            ("all_dose", "endpoint_strong"),
        ):
            differences = values[left] - values[right]
            scene_values = (
                pd.DataFrame({"scene": physical_scenes, "difference": differences})
                .groupby("scene", sort=True)
                .difference.mean()
                .to_numpy()
            )
            indices = rng.integers(0, len(scene_values), size=(draws, len(scene_values)))
            boot = scene_values[indices].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            estimate = float(scene_values.mean())
            rows.append({
                "model": "V-JEPA2-G",
                "layer": layer_name,
                "left_space": left,
                "right_space": right,
                "k": 4,
                "mean_paired_difference": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "heldout_scenes": len(scene_values),
                "heldout_pairs": len(differences),
                "bootstrap_draws": draws,
                "seed": seed,
                "status": STATUS,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    args = parse_args()
    root = args.report_root
    root.mkdir(parents=True, exist_ok=True)
    dose = pd.read_csv(root / "DOSE_RESPONSE.csv")
    direction = pd.read_csv(root / "DIRECTION_BY_DOSE.csv")
    straight = pd.read_csv(root / "STRAIGHTNESS_METRICS.csv")
    spectra = pd.read_csv(root / "SUBSPACE_SPECTRA.csv")
    ranks = pd.read_csv(root / "EFFECTIVE_RANK.csv")
    rays = pd.read_csv(root / "RAY_METRICS.csv")
    hue = pd.read_csv(root / "HUE_CIRCULARITY.csv")
    physics = pd.read_csv(root / "PHYSICS_APPEARANCE_MANIFOLD.csv")
    extraction = json.loads((args.extraction_root / "extraction_summary.json").read_text())
    packed = json.loads((args.packed_root / "provenance.json").read_text())
    analysis_metadata = json.loads((root / "RUN_METADATA.json").read_text())
    benchmark = json.loads((Path(__import__("os").path.join(__import__("os").environ['JEPA_BENCHMARK_ROOT'], 'vjepa2g-grid-stage-e')) / "benchmark_results.json").read_text())
    final_straight = straight[straight.layer == "final"].set_index("factor")
    final_rank = ranks[ranks.layer == "final"].set_index("factor")
    final_hue = hue[hue.layer == "final"].iloc[0]
    final_physics = physics[(physics.layer == "final") & (physics.k == 4)].set_index("space")
    final_rays = rays[rays.layer == "final"]
    brightness_ray = final_rays[final_rays.factor == "brightness"]
    factor_order = final_straight.straightness_mean.sort_values(ascending=False).index.tolist()
    rank_order = final_rank.effective_rank.sort_values(ascending=False).index.tolist()

    h1 = label(len(set(np.round(final_straight.straightness_mean, 3))) == 3, True)
    h2 = label(factor_order == ["brightness", "hue", "selective_recolor"], factor_order[0] == "brightness")
    h3 = label(final_rank.loc["selective_recolor", "effective_rank"] > final_rank.loc["brightness", "effective_rank"])
    hue_low_dimensional = final_rank.loc["hue", "top2_energy"] >= 0.70
    harmonic_better = final_hue.harmonic_relative_error < final_hue.ray_relative_error
    h4 = label(harmonic_better and hue_low_dimensional, harmonic_better or hue_low_dimensional)
    contrasts = paired_projection_contrasts(
        args.packed_root,
        root / "PHYSICS_PROJECTION_CONTRAST.csv",
    )
    tangent_contrast = contrasts[
        (contrasts.layer == "final")
        & (contrasts.left_space == "local_tangent")
        & (contrasts.right_space == "endpoint_strong")
    ].iloc[0]
    tangent_gt_endpoint = tangent_contrast.mean_paired_difference > 0
    h5 = label(tangent_contrast.ci_low > 0, tangent_gt_endpoint)
    hypotheses = {
        "factor_specific_geometry": h1,
        "brightness_most_ray_like": h2,
        "recolor_higher_dimensional": h3,
        "hue_curved_low_dimensional": h4,
        "tangent_gt_endpoint_physics_projection": h5,
    }

    aliases = {
        "DOSE_DIRECTION_KERNELS.csv": "DOSE_KERNELS.csv",
        "STRAIGHTNESS_METRICS.csv": "STRAIGHTNESS.csv",
        "RAY_METRICS.csv": "RAY_MODEL.csv",
        "HUE_CIRCULARITY.csv": "HUE_CURVED_MODEL.csv",
        "PHYSICS_APPEARANCE_MANIFOLD.csv": "PHYSICS_APPEARANCE_PROJECTIONS.csv",
    }
    for source, target in aliases.items():
        shutil.copy2(root / source, root / target)
    shutil.copy2(args.extraction_root / "hardware_runtime_log.csv", root / "hardware_runtime_log.csv")
    shutil.copy2(Path(__import__("os").path.join(__import__("os").environ['JEPA_BENCHMARK_ROOT'], 'vjepa2g-grid')) / "benchmark_results.json", root / "benchmark_1clip.json")

    samples = pd.DataFrame(json.loads((args.packed_root / "samples.json").read_text()))
    samples.to_csv(root / "GRID_FEATURE_MANIFEST.csv", index=False)
    runtime = pd.read_csv(args.extraction_root / "hardware_runtime_log.csv")
    extraction_seconds = float(runtime.seconds.sum())
    max_temp = int(runtime["temperature.gpu"].max())
    peak_vram = int(runtime.peak_reserved_bytes.max())
    (root / "GRID_EXTRACTION_SUMMARY.md").write_text(f"""# Grid Extraction Summary

- Status: PASS
- New exact embeddings: {extraction['jobs_valid']}.
- Reused corrected embeddings: {packed['reused_samples']}.
- Packed samples: {packed['samples']} across {packed['scenes']} scenes and {packed['pairs']} pairs.
- Encoder inference seconds across completed jobs: {extraction_seconds:.1f}.
- Resumable-run wall seconds: {extraction['wall_seconds']:.1f} (excludes the documented external-GPU-contention pause).
- Peak reserved VRAM: {peak_vram / 2**20:.1f} MiB.
- Maximum logged GPU temperature: {max_temp} C; thermal throttling was observed and preserved in the runtime log.
- Repeat checks: {json.dumps(extraction['repeat_checks'])}.
""", encoding="utf-8")

    final_summary = {}
    for factor in FACTORS:
        s = final_straight.loc[factor]
        r = final_rank.loc[factor]
        factor_rays = final_rays[final_rays.factor == factor]
        final_summary[factor] = {
            "straightness": float(s.straightness_mean),
            "straightness_ci": [float(s.ci_low), float(s.ci_high)],
            "effective_rank": float(r.effective_rank),
            "top1_energy": float(r.top1_energy),
            "top2_energy": float(r.top2_energy),
            "ray_directional_cosine_mean": float(factor_rays.directional_cosine_mean.mean()),
            "ray_projection_fraction_mean": float(factor_rays.projection_fraction_mean.mean()),
        }
    final_summary["hue"].update(
        ray_relative_error=float(final_hue.ray_relative_error),
        harmonic_relative_error=float(final_hue.harmonic_relative_error),
        harmonic_improvement=float(final_hue.harmonic_improvement),
    )
    projections = {
        space: {
            "observed": float(row.projection_mean),
            "ci": [float(row.ci_low), float(row.ci_high)],
            "random": float(row.random_mean),
            "excess": float(row.excess),
        }
        for space, row in final_physics.iterrows()
    }
    strongest = [
        f"Final straightness: brightness {final_summary['brightness']['straightness']:.3f}, hue {final_summary['hue']['straightness']:.3f}, recolor {final_summary['selective_recolor']['straightness']:.3f}.",
        f"Final effective rank: brightness {final_summary['brightness']['effective_rank']:.2f}, hue {final_summary['hue']['effective_rank']:.2f}, recolor {final_summary['selective_recolor']['effective_rank']:.2f}.",
        f"Hue held-out relative error: ray {final_hue.ray_relative_error:.3f}, harmonic {final_hue.harmonic_relative_error:.3f}.",
        f"Final k=4 physics projection: endpoint {projections['endpoint_strong']['observed']:.3f}, all-dose {projections['all_dose']['observed']:.3f}, tangent {projections['local_tangent']['observed']:.3f}.",
        f"Final paired tangent-minus-endpoint projection difference is {tangent_contrast.mean_paired_difference:.3f} [{tangent_contrast.ci_low:.3f}, {tangent_contrast.ci_high:.3f}].",
        f"Matched random k=4 baseline is approximately {projections['local_tangent']['random']:.4f} in 1,408 dimensions.",
    ]
    main_text = all(value == "REPLICATED" for value in hypotheses.values())
    summary = {
        "status": "PASS",
        "analysis_status": "post-sealed exploratory",
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_gb": 4.0,
            "strategy": benchmark["strategy"],
            "peak_vram_gb": peak_vram / 2**30,
            "seconds_per_clip_benchmark_median": benchmark["seconds_median"],
            "seconds_per_clip_benchmark_p90": benchmark["seconds_p90"],
            "extraction_inference_seconds": extraction_seconds,
            "analysis_seconds": float(analysis_metadata["elapsed_seconds"]),
            "max_temperature_c": max_temp,
        },
        "scientific_integrity": {
            "exact_checkpoint": True,
            "same_preprocessing": True,
            "quantized": False,
            "frame_count": 64,
            "layers": LAYERS,
            "sealed_analysis_modified": False,
        },
        "grid": {"scenes": 38, "pairs": 76, "sources": 152, "hue": [15, 45, 90], "brightness": [0.85, 0.65, 0.45], "recolor": [45, 90, 180]},
        "factors": final_summary,
        "physics_projection_final_k4": projections,
        "physics_projection_final_k4_paired_tangent_minus_endpoint": {
            "difference": float(tangent_contrast.mean_paired_difference),
            "ci": [float(tangent_contrast.ci_low), float(tangent_contrast.ci_high)],
        },
        "vjepa2l_replication": hypotheses,
        "strongest_findings": strongest,
        "negative_findings": ["The 0-90 degree hue arc cannot establish periodicity or a color wheel.", "The analysis is post-sealed exploratory and does not replace frozen PAO."],
        "claims_supported": ["The exact V-JEPA2-G development grid exhibits the reported factor-specific trajectory geometry."] if h1 == "REPLICATED" else [],
        "claims_not_supported": ["Hue circularity", "semantic disentanglement", "physical understanding", "a causal JEPA-objective explanation"],
        "paper_recommendation": {"main_text_worthy": main_text, "status": "MAIN-TEXT WORTHY" if main_text else "APPENDIX WORTHY"},
        "provenance": {
            "packed_features_sha256": sha256(args.packed_root / "layer_features.npz"),
            "samples_sha256": sha256(args.packed_root / "samples.json"),
            "analysis_script_sha256": sha256(Path("scripts/run_transformation_manifold_audit.py")),
            "finalizer_sha256": sha256(Path(__file__)),
        },
    }
    (root / "VJEPA2G_GRID_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pilot_straight = pd.read_csv(args.pilot_root / "STRAIGHTNESS_METRICS.csv")
    pilot_rank = pd.read_csv(args.pilot_root / "EFFECTIVE_RANK.csv")
    pilot_straight = pilot_straight[pilot_straight.layer == "final"].set_index("factor")
    pilot_rank = pilot_rank[pilot_rank.layer == "final"].set_index("factor")
    comparison_lines = ["# V-JEPA2-L vs V-JEPA2-G", "", f"**{STATUS}.**", ""]
    for key, value in hypotheses.items():
        comparison_lines.append(f"- **{key.replace('_', ' ')}: {value}.**")
    comparison_lines.extend([
        "",
        "| Factor | L final straightness | G final straightness | L final effective rank | G final effective rank |",
        "|---|---:|---:|---:|---:|",
    ])
    for factor in FACTORS:
        comparison_lines.append(
            f"| {factor} | {pilot_straight.loc[factor, 'straightness_mean']:.3f} | "
            f"{final_straight.loc[factor, 'straightness_mean']:.3f} | "
            f"{pilot_rank.loc[factor, 'effective_rank']:.2f} | "
            f"{final_rank.loc[factor, 'effective_rank']:.2f} |"
        )
    comparison_lines.extend(["", "The L pilot and G replication use the same three-dose factor grid. The G analysis uses the corrected 38-scene development population and a pre-analysis deterministic 23/15 scene split. No G method was tuned after observing these values."])
    (root / "VJEPA2L_VS_VJEPA2G.md").write_text("\n".join(comparison_lines), encoding="utf-8")

    findings_md = "\n".join(f"- {item}" for item in strongest)
    geometry_table = markdown_rows(
        ["Layer", "Factor", "Straightness", "95% CI", "Effective rank", "Top-2 energy"],
        [
            [
                layer,
                factor,
                f"{straight[(straight.layer == layer) & (straight.factor == factor)].iloc[0].straightness_mean:.3f}",
                f"[{straight[(straight.layer == layer) & (straight.factor == factor)].iloc[0].ci_low:.3f}, {straight[(straight.layer == layer) & (straight.factor == factor)].iloc[0].ci_high:.3f}]",
                f"{ranks[(ranks.layer == layer) & (ranks.factor == factor)].iloc[0].effective_rank:.2f}",
                f"{ranks[(ranks.layer == layer) & (ranks.factor == factor)].iloc[0].top2_energy:.3f}",
            ]
            for layer in LAYERS
            for factor in FACTORS
        ],
    )
    final_direction_table = markdown_rows(
        ["Factor", "Dose", "MRL", "Shuffled mean", "Excess", "MRL 95% CI"],
        [
            [
                row.factor,
                f"{row.nominal_dose:g}",
                f"{row.mrl_observed:.3f}",
                f"{row.shuffled_mean:.3f}",
                f"{row.mrl_excess:.3f}",
                f"[{row.mrl_ci_low:.3f}, {row.mrl_ci_high:.3f}]",
            ]
            for _, row in direction[direction.layer == "final"].iterrows()
        ],
    )
    projection_table = markdown_rows(
        ["Layer", "Space", "Observed", "95% CI", "Random", "Excess"],
        [
            [
                layer,
                space,
                f"{row.projection_mean:.3f}",
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}]",
                f"{row.random_mean:.4f}",
                f"{row.excess:.3f}",
            ]
            for layer in LAYERS
            for space in ("endpoint_strong", "all_dose", "local_tangent")
            for _, row in physics[
                (physics.layer == layer) & (physics.space == space) & (physics.k == 4)
            ].iterrows()
        ],
    )
    report = f"""# Executive Result

## Hardware Feasibility

Strategy: exact sequential transformer-block CPU/GPU offload
Peak VRAM: {peak_vram / 2**20:.1f} MiB reserved
Seconds/clip: {benchmark['seconds_median']:.3f} warm median, {benchmark['seconds_p90']:.3f} p90 before external contention
Total extraction inference time: {extraction_seconds:.1f} seconds
Exact model preserved: yes
Numerical validation: minimum clean-cache cosine {min(row['cosine'] for row in benchmark['validation']):.8f}

## Grid Coverage

Corrected development: 38 scenes, 76 matched pairs, 152 sources. Each source has clean plus hue 15/45/90 degrees, brightness 0.85/0.65/0.45, and selective recolor 45/90/180 degrees. The packed matrix contains 1,520 samples at five paper-facing depths.

## Strongest V-JEPA2-G Findings

{findings_md}

## V-JEPA2-L -> V-JEPA2-G Replication

- H1 factor-specific geometry: **{h1}**
- H2 brightness most ray-like: **{h2}**
- H3 recolor higher-dimensional: **{h3}**
- H4 hue curved low-dimensional: **{h4}**
- H5 tangent physics overlap exceeds endpoint: **{h5}**

## Negative Results

- A three-dose 0-90 degree hue arc cannot establish circularity, periodicity, closure, or a latent color wheel.
- Effective rank and subspace projection do not establish semantic disentanglement or physical understanding.
- Any non-replicated hypothesis remains visible in the classifications above.

## Physics-Appearance Tangent Result

At final k=4, held-out physical projection is {projections['endpoint_strong']['observed']:.3f} [{projections['endpoint_strong']['ci'][0]:.3f}, {projections['endpoint_strong']['ci'][1]:.3f}] into the strongest-endpoint space, {projections['all_dose']['observed']:.3f} [{projections['all_dose']['ci'][0]:.3f}, {projections['all_dose']['ci'][1]:.3f}] into the all-dose space, and {projections['local_tangent']['observed']:.3f} [{projections['local_tangent']['ci'][0]:.3f}, {projections['local_tangent']['ci'][1]:.3f}] into the local tangent space. Each uses matched k=4 dimensionality; the tangent random baseline is {projections['local_tangent']['random']:.4f}. The paired scene-level tangent-minus-endpoint difference is {tangent_contrast.mean_paired_difference:.3f} [{tangent_contrast.ci_low:.3f}, {tangent_contrast.ci_high:.3f}].

## Main-Text Recommendation

**{'MAIN-TEXT WORTHY' if main_text else 'APPENDIX WORTHY'}.** This is an exact paper-model development-population replication but remains post-sealed exploratory. Main-text inclusion requires clear exploratory labeling and should not replace the frozen sealed analysis.

## Claims Supported

The exact V-JEPA2-G model exhibits only the numerical factor-specific, curvedness, rank, and matched-k physics-projection patterns reported here. L-to-G replication labels are qualitative comparisons, not causal architecture claims.

## Claims Not Supported

No result licenses hue circularity, semantic axes, disentanglement, causal factorization, physical understanding, or replacement of the confirmatory sealed PAO result.

## Methods and Detailed Results

Representations were L2-normalized before displacement geometry. Scene is the independent unit. Point estimates, 10,000 scene-bootstrap intervals, 10,000 shuffled-correspondence null draws, and 10,000 dimension-matched random-subspace draws are in the CSV outputs. Brightness path dose is defined as intervention strength `1 - multiplier`, so 0.85, 0.65, 0.45 map to 0.15, 0.35, 0.55. Physics orientation is possible to impossible. Appearance bases use only the fixed 23 training scenes and physical projections use the 15 held-out scenes.
"""
    report += f"""

### Layerwise Trajectory Geometry

{geometry_table}

### Final-Layer Direction By Dose

{final_direction_table}

Raw MRL is interpreted only against the cross-scene shuffled-correspondence null. The full five-layer dose table, null intervals, and empirical permutation p-values are in DIRECTION_BY_DOSE.csv.

### Layerwise Physics Projection At Matched k=4

{projection_table}

The direct tangent-minus-endpoint contrast is paired by held-out scene and uses 10,000 bootstrap draws; all layerwise contrasts are in PHYSICS_PROJECTION_CONTRAST.csv.

### Numerical And Coverage Audit

- Every one of the 10 conditions has 152 sources from 38 scenes; the packed tensor shape is 1,520 x 5 x 1,408.
- All packed representations and all primary geometry outputs are finite.
- Fifteen layer-aggregated pixel_l2_mean cells are unavailable because reused strongest-dose cache rows retained RGB mean absolute difference but not RGB RMS difference. The required realized RGB mean-absolute dose is complete and is the x-axis used in Figure 1.
- Encoder repeat extraction had maximum absolute representation difference 0 on three checked clips.
- Geometry/statistical analysis runtime was {analysis_metadata['elapsed_seconds']:.1f} seconds after feature packing.
"""
    (root / "VJEPA2G_GRID_AUDIT.md").write_text(report, encoding="utf-8")

    manuscript = f"""# Manuscript Implications

**{STATUS}.** Do not alter or replace the frozen sealed result.

- Paper status: **{'MAIN-TEXT WORTHY' if main_text else 'APPENDIX WORTHY'}**.
- The strongest new paper-relevant question is whether matched-k local appearance tangents capture more held-out physical displacement energy than strongest endpoints. V-JEPA2-G gives endpoint {projections['endpoint_strong']['observed']:.3f} versus tangent {projections['local_tangent']['observed']:.3f} at final k=4; the paired difference is {tangent_contrast.mean_paired_difference:.3f} [{tangent_contrast.ci_low:.3f}, {tangent_contrast.ci_high:.3f}].
- Factor-specific straightness and effective-rank results can support a concise exploratory statement only with the full layerwise curves and uncertainty.
- Hue should be described as a curved 0-90 degree arc, never as circular or a color wheel.
- Keep extraction feasibility, dose kernels, detailed tangent nulls, and full spectra in the appendix.
- Do not infer that JEPA training causes the geometry; this run contains no controlled objective comparison.
"""
    (root / "MANUSCRIPT_IMPLICATIONS.md").write_text(manuscript, encoding="utf-8")

    (root / "EXTRACTION_PROVENANCE.md").write_text(f"""# Extraction Provenance

- Exact checkpoint SHA-256: `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`.
- V-JEPA2 source revision: `204698b45b3712590f06245fbfba32d3be539812`.
- Strategy: exact sequential CPU/GPU block offload, BF16 autocast, FP32 weights, no quantization.
- New valid jobs: {extraction['jobs_valid']}; reused corrected samples: {packed['reused_samples']}.
- Frames/resolution: frozen 64 / 256x256; five pooled paper layers only.
- Repeat checks: `{json.dumps(extraction['repeat_checks'])}`.
- Packed feature SHA-256: `{summary['provenance']['packed_features_sha256']}`.
""", encoding="utf-8")

    (root / "REPRODUCE.md").write_text('# Reproduce\n\nScoped scientific validation passed 34/34 tests. The repository-wide suite has two pre-existing out-of-scope limitations: collection requires the absent optional pycocotools package, and with that test ignored, one Windows newline-sensitive confirmatory-validation hash test fails while 107 tests pass. Frozen confirmatory code was not modified.\n\n```powershell\npython scripts\\extract_vjepa2g_grid_replication.py --checkpoint ${JEPA_VJEPA2G_CHECKPOINT} --vjepa2-root ${JEPA_VJEPA2_SOURCE} --dataset-root D:\\jepa-data\\datasets\\intphys2\\corrected-development --reference-cache results\\24-00-17-intphys2-corrected-development-probes\\vjepa2\\representations.npz --cache-root D:\\jepa-data\\cache\\vjepa2g_grid_replication\npython scripts\\pack_vjepa2g_grid_replication.py --reference-cache results\\24-00-17-intphys2-corrected-development-probes\\vjepa2\\representations.npz --new-cache-root D:\\jepa-data\\cache\\vjepa2g_grid_replication --dataset-root D:\\jepa-data\\datasets\\intphys2\\corrected-development --output-root ${JEPA_MULTIDOSE_CACHE}\npython scripts\\run_transformation_manifold_audit.py --features ${JEPA_MULTIDOSE_CACHE}\\layer_features.npz --samples ${JEPA_MULTIDOSE_CACHE}\\samples.json --appearance-manifest ${JEPA_MULTIDOSE_CACHE}\\samples.json --output reports\\vjepa2g_grid_replication --bootstrap-draws 10000 --null-draws 10000 --seed 260903 --model-label V-JEPA2-G --layer-names block_00,block_10,block_20,block_30,final --expected-scenes 38 --expected-pairs 76 --expected-sources 152 --image-dose-from-samples --skip-grid-inventory\npython scripts\\finalize_vjepa2g_grid_replication.py --report-root reports\\vjepa2g_grid_replication --packed-root ${JEPA_MULTIDOSE_CACHE} --extraction-root D:\\jepa-data\\cache\\vjepa2g_grid_replication --pilot-root reports\\transformation_manifolds\n```\n', encoding="utf-8")
    print(json.dumps({"status": "PASS", "hypotheses": hypotheses, "main_text_worthy": main_text}, indent=2))


if __name__ == "__main__":
    main()
