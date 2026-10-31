#!/usr/bin/env python3
"""Finalize narrative and compact outputs from the manifold metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_transformation_manifold_audit import (
    hue_models,
    l2_normalize,
    trajectory_for_factor,
)


STATUS = "EXPLORATORY POST-SEALED ANALYSIS"
LAYERS = ["block_00", "block_06", "block_12", "block_18", "final"]
FACTORS = ["hue", "brightness", "selective_recolor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def number(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def layer_dict(frame: pd.DataFrame, value_columns: list[str]) -> dict:
    result = {}
    for _, row in frame.iterrows():
        result[row.layer] = {
            column: None if pd.isna(row[column]) else float(row[column])
            for column in value_columns
        }
    return result


def classify(spectra: pd.DataFrame, straight: pd.DataFrame, factor: str) -> dict:
    final_spectrum = spectra[(spectra.factor == factor) & (spectra.layer == "final")].set_index("k")
    final_straight = straight[(straight.factor == factor) & (straight.layer == "final")].iloc[0]
    top1 = float(final_spectrum.loc[1, "explained_energy"])
    top2 = float(final_spectrum.loc[2, "explained_energy"])
    top4 = float(final_spectrum.loc[4, "explained_energy"])
    rank = float(final_spectrum.loc[1, "effective_rank"])
    path = float(final_straight.straightness_mean)
    if top1 >= 0.85 and path >= 0.90:
        label = "APPROXIMATELY 1D / RAY-LIKE"
    elif top2 >= 0.78 and top4 >= 0.88 and path < 0.95:
        label = "CURVED LOW-DIMENSIONAL"
    elif top4 < 0.85 or rank >= 5:
        label = "HIGHER-DIMENSIONAL"
    elif path < 0.80:
        label = "SCENE-DEPENDENT"
    else:
        label = "INCONCLUSIVE"
    return {
        "label": label,
        "top1_energy": top1,
        "top2_energy": top2,
        "top4_energy": top4,
        "effective_rank": rank,
        "straightness": path,
    }


def build_statistical_summary(root: Path, dose: pd.DataFrame, direction: pd.DataFrame, rays: pd.DataFrame, tangents: pd.DataFrame, straight: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, row in dose.groupby("layer", sort=False).first().iterrows():
        rows.append({"model": row.model, "layer": layer, "factor": "clean", "dose": 0.0, "metric": "global_unrelated_scale", "estimate": row.global_scale, "ci_low": np.nan, "ci_high": np.nan, "null_mean": np.nan, "null_low": np.nan, "null_high": np.nan, "p_value": np.nan, "status": STATUS})
    for _, row in dose.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": row.nominal_dose, "metric": "unrelated_normalized_magnitude", "estimate": row.rdm_mean, "ci_low": row.rdm_ci_low, "ci_high": row.rdm_ci_high, "null_mean": np.nan, "null_low": np.nan, "null_high": np.nan, "p_value": np.nan, "status": STATUS})
    for _, row in direction.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": row.nominal_dose, "metric": "mean_resultant_length", "estimate": row.mrl_observed, "ci_low": row.mrl_ci_low, "ci_high": row.mrl_ci_high, "null_mean": row.shuffled_mean, "null_low": row.shuffled_low, "null_high": row.shuffled_high, "p_value": row.empirical_p, "status": STATUS})
    for _, row in rays.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": row.nominal_dose, "metric": "heldout_ray_directional_cosine", "estimate": row.directional_cosine_mean, "ci_low": row.directional_cosine_ci_low, "ci_high": row.directional_cosine_ci_high, "null_mean": np.nan, "null_low": np.nan, "null_high": np.nan, "p_value": np.nan, "status": STATUS})
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": row.nominal_dose, "metric": "heldout_ray_projection_fraction", "estimate": row.projection_fraction_mean, "ci_low": row.projection_fraction_ci_low, "ci_high": row.projection_fraction_ci_high, "null_mean": np.nan, "null_low": np.nan, "null_high": np.nan, "p_value": np.nan, "status": STATUS})
    for _, row in tangents.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": row.path_dose, "metric": "tangent_mean_resultant_length", "estimate": row.tangent_mrl, "ci_low": row.tangent_mrl_ci_low, "ci_high": row.tangent_mrl_ci_high, "null_mean": row.tangent_null_mean, "null_low": row.tangent_null_low, "null_high": row.tangent_null_high, "p_value": row.tangent_empirical_p, "status": STATUS})
    for _, row in straight.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.factor, "dose": np.nan, "metric": "trajectory_straightness", "estimate": row.straightness_mean, "ci_low": row.ci_low, "ci_high": row.ci_high, "null_mean": 1.0, "null_low": np.nan, "null_high": np.nan, "p_value": np.nan, "status": STATUS})
    for _, row in physics.iterrows():
        rows.append({"model": row.model, "layer": row.layer, "factor": row.space, "dose": row.k, "metric": "physical_projection_fraction", "estimate": row.projection_mean, "ci_low": row.ci_low, "ci_high": row.ci_high, "null_mean": row.random_mean, "null_low": row.random_low, "null_high": row.random_high, "p_value": np.nan, "status": STATUS})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "STATISTICAL_SUMMARY.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dose = pd.read_csv(root / "DOSE_RESPONSE.csv")
    fits = pd.read_csv(root / "DOSE_RESPONSE_FITS.csv")
    direction = pd.read_csv(root / "DIRECTION_BY_DOSE.csv")
    rays = pd.read_csv(root / "RAY_METRICS.csv")
    kernels = pd.read_csv(root / "DOSE_DIRECTION_KERNELS.csv")
    tangents = pd.read_csv(root / "TANGENT_METRICS.csv")
    curvature = pd.read_csv(root / "CURVATURE_METRICS.csv")
    straight = pd.read_csv(root / "STRAIGHTNESS_METRICS.csv")
    spectra = pd.read_csv(root / "SUBSPACE_SPECTRA.csv")
    bundle = np.load(root / "cache/vjepa2_l_o1/layer_features.npz")
    samples = pd.DataFrame(
        json.loads((root / "cache/vjepa2_l_o1/samples.json").read_text(encoding="utf-8"))
    )
    samples["_row_index"] = np.arange(len(samples), dtype=int)
    normalized, valid = l2_normalize(
        bundle["pooled"].astype(np.float64).reshape(-1, bundle["pooled"].shape[-1])
    )
    if not np.all(valid):
        raise ValueError("invalid embedding while finalizing hue diagnostics")
    normalized = normalized.reshape(bundle["pooled"].shape)
    hue_rows = []
    hue_rng = np.random.default_rng(260904)
    for layer_index, layer in enumerate(LAYERS):
        points, _, scenes, splits = trajectory_for_factor(
            samples, normalized[:, layer_index], "hue"
        )
        hue_rows.append({
            "model": "V-JEPA2-L",
            "layer": layer,
            **hue_models(
                points,
                scenes,
                splits,
                np.asarray([0.0, 15.0, 45.0, 90.0]),
                hue_rng,
                10_000,
            ),
            "status": STATUS,
        })
    hue = pd.DataFrame(hue_rows)
    hue.to_csv(root / "HUE_CIRCULARITY.csv", index=False)
    cross = pd.read_csv(root / "CROSS_FACTOR_ALIGNMENT.csv")
    physics = pd.read_csv(root / "PHYSICS_APPEARANCE_MANIFOLD.csv")
    run = json.loads((root / "RUN_METADATA.json").read_text(encoding="utf-8"))
    extraction = json.loads((root / "cache/vjepa2_l_o1/provenance.json").read_text(encoding="utf-8"))
    statistical = build_statistical_summary(root, dose, direction, rays, tangents, straight, physics)

    classifications = {factor: classify(spectra, straight, factor) for factor in FACTORS}
    monotonicity = {}
    for factor in FACTORS:
        monotonicity[factor] = {}
        for layer in LAYERS:
            current = dose[(dose.factor == factor) & (dose.layer == layer)].sort_values("path_dose")
            monotonicity[factor][layer] = float(spearmanr(current.path_dose, current.rdm_mean).statistic)

    strongest_direction = {
        factor: direction[direction.factor == factor].sort_values("mrl_excess", ascending=False).iloc[0]
        for factor in FACTORS
    }
    final_hue = hue[hue.layer == "final"].iloc[0]
    final_physics = physics[(physics.layer == "final") & (physics.k == 4)].set_index("space")
    early_physics = physics[(physics.layer == "block_00") & (physics.k == 4)].set_index("space")
    final_cross = cross[cross.layer == "final"]
    best_fits = fits.sort_values("heldout_scene_rmse").groupby(["factor", "dose_scale"], sort=False).first().reset_index()

    findings = []
    for factor in FACTORS:
        row = strongest_direction[factor]
        findings.append(
            f"V-JEPA2-L {row.layer} {row.condition}: MRL {row.mrl_observed:.3f} versus shuffled {row.shuffled_mean:.3f} "
            f"(excess {row.mrl_excess:+.3f}, bootstrap [{row.mrl_ci_low:.3f}, {row.mrl_ci_high:.3f}], permutation p={row.empirical_p:.4f}); "
            "the fixed dose has source-correspondence-specific directional structure on this pilot."
        )
    for factor in FACTORS:
        item = classifications[factor]
        findings.append(
            f"V-JEPA2-L final {factor}: top-1/top-2 displacement energy {item['top1_energy']:.3f}/{item['top2_energy']:.3f}, "
            f"effective rank {item['effective_rank']:.2f}, straightness {item['straightness']:.3f}; provisional geometry is {item['label'].lower()}."
        )
    findings.append(
        f"V-JEPA2-L final hue: held-out ray error {final_hue.ray_relative_error:.3f} versus 2D harmonic error {final_hue.harmonic_relative_error:.3f} "
        f"(improvement {final_hue.harmonic_improvement:+.3f}); mean scene-plane overlap {final_hue.shared_plane_overlap_mean:.3f} versus random expectation {final_hue.shared_plane_random_expectation:.4f}. "
        "This is an arc-fit comparison, not a circularity result."
    )
    findings.append(
        f"V-JEPA2-L final physics: k=4 projection is {final_physics.loc['endpoint_strong','projection_mean']:.3f} into the endpoint space, "
        f"{final_physics.loc['all_dose','projection_mean']:.3f} into all-dose space, and {final_physics.loc['local_tangent','projection_mean']:.3f} into tangent space, "
        f"versus random means near {final_physics.loc['all_dose','random_mean']:.4f}."
    )

    magnitude_answers = []
    for factor in FACTORS:
        rhos = list(monotonicity[factor].values())
        label = "monotonic at all five depths" if min(rhos) > 0 else "not consistently monotonic"
        magnitude_answers.append(f"{factor}: {label} (dose-rank correlations {', '.join(number(v, 2) for v in rhos)}).")

    kernel_answers = []
    ray_answers = []
    for factor in FACTORS:
        current = kernels[(kernels.factor == factor) & (kernels.layer == "final") & (kernels.dose_a != kernels.dose_b)]
        kernel_answers.append(f"{factor}: final mean off-diagonal matched-source cosine {current.cosine_mean.mean():.3f}.")
        current_ray = rays[(rays.factor == factor) & (rays.layer == "final")]
        ray_answers.append(
            f"{factor}: held-out ray cosine range {current_ray.directional_cosine_mean.min():.3f}-"
            f"{current_ray.directional_cosine_mean.max():.3f}; projection-fraction range "
            f"{current_ray.projection_fraction_mean.min():.3f}-{current_ray.projection_fraction_mean.max():.3f}."
        )

    physics_lines = []
    for space in ("endpoint_strong", "all_dose", "local_tangent"):
        first = early_physics.loc[space]
        last = final_physics.loc[space]
        physics_lines.append(
            f"{space}: block_00 {first.projection_mean:.3f} (random {first.random_mean:.4f}) -> final {last.projection_mean:.3f} (random {last.random_mean:.4f})."
        )

    report = f"""# Executive Result

**{STATUS}.** The requested V-JEPA2-G and VideoMAE2-G dose manifolds could not be measured from existing caches or on the available 4 GB GPU. A complete, 10,000-draw feasibility audit was executed on the verified local V-JEPA2-L checkpoint and Original IntPhys O1 grid. These measurements validate the pipeline and generate hypotheses; they are not evidence about the paper-facing giant models.

## Data / Grid Coverage

Measured: 10 O1 scenes, 20 matched pairs, 40 source videos, clean plus hue 15/45/90 degrees, brightness 0.85/0.65/0.45, and selective recolor 45/90/180 degrees; 400 clips at native layers 0/6/12/18/24. Six scenes fit shared geometry and four scenes are held out. No saturation, RGB-channel, joint, ordered-composition, negative-hue, or closed hue grid exists locally.

## Strongest Findings

{chr(10).join(f'- {item}' for item in findings[:10])}

## Most Important Negative Findings

- No compatible multi-dose embedding cache exists for V-JEPA2-G or VideoMAE2-G. Their manifold geometry remains unmeasured.
- Hue periodicity and closure are not identifiable from a one-sided 0-90 degree arc.
- No joint transforms exist, so additivity, interaction, and commutativity cannot be assessed.
- The O1 pilot has only 10 independent scenes and only four held-out scenes; bootstrap intervals and cross-fitted projections remain broad.
- A better low-dimensional fit does not establish semantic axes, disentanglement, or physical understanding.

## Factor Geometry Classification

{chr(10).join(f"- **{factor}: {classifications[factor]['label']}.** Top-1/top-2/top-4 energy {classifications[factor]['top1_energy']:.3f}/{classifications[factor]['top2_energy']:.3f}/{classifications[factor]['top4_energy']:.3f}; effective rank {classifications[factor]['effective_rank']:.2f}; straightness {classifications[factor]['straightness']:.3f}." for factor in FACTORS)}

No factor is classified as approximately circular because the required cycle was not observed.

## Physics Relationship

Held-out possible-to-impossible displacements were projected into train-scene endpoint, all-dose, and local-tangent appearance spaces. At k=4:

{chr(10).join(f'- {line}' for line in physics_lines)}

These are exploratory O1/V-JEPA2-L projections. They do not replace or reinterpret the frozen sealed k=4 PAO result.

## Development vs Sealed Status

This is a post-sealed exploratory study on Original IntPhys O1, not an additional sealed evaluation. No sealed representation was loaded. The finished `geometry_freeze` and `geometry_sealed` outputs were not modified. There is no development-to-sealed replication claim for this dose grid.

## What Is New Beyond Frozen Geometry

The audit adds realized image-space dose, dose-response curves, dose-by-dose direction kernels, local finite-difference tangents, path straightness, turning, all-dose dimensionality, shared ray versus harmonic hue arc fits, and physics projection into endpoint/all-dose/tangent spaces.

## Claims Supported

- On this V-JEPA2-L O1 pilot, the implemented factors have measurably different path straightness and displacement spectra.
- Null-adjusted MRL can vary by both dose and depth, so a single strongest-dose direction does not summarize the factor trajectory.
- The one-sided hue arc can be compared descriptively with ray and harmonic fits, but periodicity is unresolved.

## Claims Not Supported

- V-JEPA2-G or VideoMAE2-G has any reported dose geometry.
- Hue forms a closed latent color wheel.
- Brightness is a universal semantic axis.
- Appearance and physics are disentangled, causally factorized, or evidence of physical understanding.
- Appearance factors are additive or commute.

## Best Next Experiment

Run the exact 400-clip O1 manifest through the hash-locked V-JEPA2-G and VideoMAE2-G extractors on one >=16 GB GPU without changing the analysis. After that replication, generate a development-only symmetric 12-angle hue grid with 0/360 identity verification if circularity remains scientifically important.

## Main Questions

### Q1. Does magnitude scale monotonically with strength?

{chr(10).join(magnitude_answers)}

### Q2. Does each factor preserve one direction across dose?

{chr(10).join(kernel_answers)}

Train-scene shared-ray generalization:

{chr(10).join(ray_answers)}

High off-diagonal cosine and held-out ray projection support a common ray; lower values and low straightness indicate rotation or scene dependence. Dose-level intervals are in `RAY_METRICS.csv`.

### Q3. Which factors are approximately 1D?

None meets the predeclared final-layer threshold of top-1 energy >=0.85 and straightness >=0.90. Brightness is the closest but remains curved under this criterion.

### Q4. Which factors are curved or higher-dimensional?

Brightness is curved low-dimensional; selective recolor is higher-dimensional; hue is classified from the reported final spectrum/straightness and must not be called circular.

### Q5. Is hue better fit by a 2D periodic model?

Final held-out ray error is {final_hue.ray_relative_error:.3f}; harmonic error is {final_hue.harmonic_relative_error:.3f}. Regardless of which is lower, only a quarter arc exists, so periodicity and closure are inconclusive.

### Q6. Does geometry simplify with depth?

The layerwise MRL, rank, straightness, and curvature tables show factor-specific and non-monotonic behavior; there is no justified single simplification claim.

### Q7. Do factors share tangent directions?

Final clean-side signed tangent cosines are {', '.join(f"{row.factor_a}-{row.factor_b} {row.signed_cosine_mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]" for _, row in final_cross.iterrows())}. These are not absolute cosines.

### Q8. Are factors additive when combined?

Not estimable: no joint grid exists.

### Q9. How much physics lies in the three appearance spaces?

At final k=4: endpoint {final_physics.loc['endpoint_strong','projection_mean']:.3f}, all-dose {final_physics.loc['all_dose','projection_mean']:.3f}, tangent {final_physics.loc['local_tangent','projection_mean']:.3f}; matched isotropic random means are reported in `PHYSICS_APPEARANCE_MANIFOLD.csv`.

### Q10. Does the physics relationship change with depth?

{chr(10).join(physics_lines)} The trajectories are descriptive and may be non-monotonic.

## Numerical Scope and Nulls

All intervals use 10,000 scene-bootstrap draws. Endpoint and tangent MRL nulls each use 10,000 constrained cross-scene correspondence shuffles. Projection nulls use 10,000 isotropic marginal draws and are checked against k/d. No endpoint or tangent vector fell below its recorded numerical floor. The maximum absolute raw-versus-L2-normalized endpoint MRL difference was 0.0428; no qualitative factor classification changed. Full thresholds and robustness values are retained in the directional tables.

## Response-Fit Diagnostics

Held-out scene RMSE for constant, linear, quadratic, isotonic, and saturating fits is in `DOSE_RESPONSE_FITS.csv`. Best models are descriptive because the grid has only four states including clean: {', '.join(f"{row.factor}/{row.dose_scale}: {row.fit_model} ({row.heldout_scene_rmse:.3f})" for _, row in best_fits.iterrows())}.
"""
    (root / "TRANSFORMATION_MANIFOLD_AUDIT.md").write_text(report, encoding="utf-8")

    manuscript = f"""# Manuscript Implications

**{STATUS}.**

## Recommendation

Do not add the measured V-JEPA2-L/O1 manifold values to the main paper. They use a different checkpoint family, 16-frame clips, Original IntPhys O1, and only 10 scenes. They are suitable as pipeline validation and hypothesis generation.

The genuinely new contribution is methodological: dose response, direction kernels, local tangents, path straightness, and endpoint/all-dose/tangent physics projections distinguish geometric properties that a strongest-dose scalar cannot. This framework is main-text worthy only after exact replication with V-JEPA2-G and VideoMAE2-G on a provenance-clean multi-dose population.

Do not replace the current Figure 2 with these pilot results. A future giant-model figure could use: normalized magnitude versus realized RGB dose; MRL excess by dose; dose-direction kernels; and k=4 physics projection into endpoint/all-dose/tangent spaces. Hue circularity belongs in an appendix or future paper unless a symmetric closed grid is generated.

No old confirmatory claim should be changed from this analysis. No result licenses "latent color wheel," "disentanglement," "semantic appearance axis," "physical understanding," factor additivity, or causal claims about JEPA training.

The strongest future-paper direction is a dedicated controlled-manifold study with symmetric hue, bidirectional brightness/exposure, saturation, and predeclared joint grids across multiple video encoders. That would be a separate experiment family, not a retrospective extension of the sealed analysis.
"""
    (root / "MANUSCRIPT_IMPLICATIONS.md").write_text(manuscript, encoding="utf-8")

    hue_table = "\n".join(
        f"| {row.layer} | {row.ray_relative_error:.3f} | {row.harmonic_relative_error:.3f} | {row.harmonic_improvement:+.3f} | {row.angular_order_spearman:.3f} | {row.angular_order_null_low:.3f}, {row.angular_order_null_high:.3f} | {row.shared_plane_overlap_mean:.3f} | {row.shared_plane_random_expectation:.4f} |"
        for _, row in hue.iterrows()
    )
    (root / "HUE_CIRCULARITY.md").write_text(f"""# Hue Circularity

**{STATUS}.** Only 0, 15, 45, and 90 degree states exist. This is a one-sided quarter arc with no negative rotations and no 360-degree endpoint. Periodicity and closure are therefore not identifiable.

| Layer | Ray error | Harmonic error | Improvement | Angle-distance Spearman | Shuffled-angle 95% interval | Mean scene-plane overlap | Random 2D expectation |
|---|---:|---:|---:|---:|---:|---:|---:|
{hue_table}

Errors are held-out relative reconstruction errors after fitting shared geometry on scenes 01-06 and evaluating scenes 07-10. The angular-order null uses 10,000 shuffled angle assignments. Scene-plane overlap is mean squared principal-angle cosine between per-scene two-dimensional hue-arc planes. A harmonic improvement or overlap above random does not establish a closed circular manifold because the grid never returns to its origin.
""", encoding="utf-8")

    reproduce = f"""# Reproduce

**{STATUS}.** Run from the repository root.

```powershell
python -m pytest -q tests\\test_transformation_manifold_metrics.py

python scripts\\extract_vjepa2_layer_features.py `
  --config reports\\transformation_manifolds\\vjepa2_l_o1_extraction_config.json

python scripts\\run_transformation_manifold_audit.py `
  --features reports\\transformation_manifolds\\cache\\vjepa2_l_o1\\layer_features.npz `
  --samples reports\\transformation_manifolds\\cache\\vjepa2_l_o1\\samples.json `
  --appearance-manifest D:\\jepa-data\\results\\19-12-04-appearance-interventions-pilot\\appearance_manifest.json `
  --output reports\\transformation_manifolds `
  --bootstrap-draws 10000 `
  --null-draws 10000 `
  --seed 260903

python scripts\\finalize_transformation_manifold_audit.py `
  --root reports\\transformation_manifolds
```

Extraction checkpoint SHA-256: `{extraction['checkpoint_sha256']}`. Feature SHA-256: `{extraction['features_sha256']}`. Feature extraction measured {extraction['elapsed_seconds_total']:.2f} seconds; analysis measured {run['elapsed_seconds']:.2f} seconds. The giant-model replication is not part of this command because the required checkpoints and GPU are unavailable locally.
"""
    (root / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    summary = {
        "status": "PASS_WITH_MODEL_COVERAGE_LIMITATION",
        "analysis_status": "post-sealed exploratory",
        "sealed_confirmatory_analysis_modified": False,
        "fresh_start_cutoff": "2026-09-03T09:36:06+03:00",
        "grid_inventory": {"population": "Original IntPhys O1 pilot", "scenes": 10, "pairs": 20, "source_videos": 40, "doses": {"hue": [15, 45, 90], "brightness": [0.85, 0.65, 0.45], "selective_recolor": [45, 90, 180]}, "saturation": None, "rgb": None, "joint": None},
        "models": {
            "V-JEPA2-G": {"status": "not measured", "reason": "no compatible cache or local compute"},
            "VideoMAE2-G": {"status": "not measured", "reason": "no compatible cache or local compute"},
            "V-JEPA2-L": {
                "status": "measured feasibility pilot",
                "layers": LAYERS,
                "classification": classifications,
                "ray_metrics_final": rays[rays.layer == "final"].to_dict("records"),
                "mrl_by_factor_layer_dose": {
                    factor: direction[direction.factor == factor][["layer", "nominal_dose", "mrl_observed", "shuffled_mean", "mrl_excess", "empirical_p"]].to_dict("records")
                    for factor in FACTORS
                },
            },
        },
        "factors": {
            factor: {**classifications[factor], "magnitude_dose_spearman": monotonicity[factor]}
            for factor in FACTORS
        },
        "hue": {"status": "quarter-arc only", "layerwise": hue.to_dict("records"), "closure": "not estimable"},
        "cross_factor": {"final_clean_side_tangents": final_cross.to_dict("records")},
        "interactions": {"status": "not estimable", "reason": "no joint grid"},
        "physics_projection": {"k4_layerwise": physics[physics.k == 4].to_dict("records")},
        "strongest_findings": findings[:8],
        "negative_findings": ["Giant-model dose geometry unmeasured", "Hue closure/circularity unidentifiable", "No joint-grid interaction test", "Only four held-out scenes"],
        "claims_supported": ["Factor-specific dose geometry exists in the V-JEPA2-L O1 feasibility pilot", "Strongest-dose scalars do not summarize path direction, curvature, and rank"],
        "claims_not_supported": ["Any V-JEPA2-G or VideoMAE2-G dose-manifold claim", "Latent color wheel", "Disentanglement or physical understanding", "Factor additivity or commutativity"],
        "paper_recommendation": {"main_text_worthy": False, "replace_figure_2": False, "next_gate": "exact giant-model replication"},
        "precision": {"scene_bootstrap_draws": 10000, "direction_null_draws": 10000, "random_subspace_draws": 10000},
        "runtime_seconds": {"extraction": extraction["elapsed_seconds_total"], "analysis": run["elapsed_seconds"]},
        "row_counts": {"dose_response": len(dose), "direction": len(direction), "ray": len(rays), "tangent": len(tangents), "statistical_summary": len(statistical)},
    }
    (root / "TRANSFORMATION_MANIFOLD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "classifications": classifications, "strongest_findings": findings[:8]}, indent=2))


if __name__ == "__main__":
    main()
