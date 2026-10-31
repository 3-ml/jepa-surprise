# Corrected IntPhys2 Sealed-Test Analysis Specification

Date frozen: 2026-08-24

Status: draft to commit before any corrected sealed-test extraction or analysis

## Scope

This specification freezes the analysis choices for the corrected IntPhys2 Main
sealed test split produced by the union-of-inspected-scenes split lock. The
development evidence used to choose these settings is recorded in
`results/24-00-17-intphys2-corrected-development-probes/`.

No corrected sealed-test model features, pixel features, probes, metrics, or
plots may be inspected before this file is committed.

## Fixed Inputs

- Dataset: IntPhys2 Main sealed test from `data/manifests/intphys2_confirmatory_samples.csv`.
- Split contract: exclude the union of scenes touched by the 169-pair endpoint
  selection and the 24-pair workshop selection.
- Test size: 114 pairs / 57 scenes as locked in
  `results/23-22-48-intphys2-corrected-union-split-lock/`.
- Frame sampling: exactly 64 shared decoded source frames per sample, using
  `round(linspace(0, decoded_frame_count - 1, 64))`.
- Appearance conditions: `clean`, `hue_090`, `brightness_0p45`,
  `selective_recolor_180`.
- Models and comparators:
  - V-JEPA2-G, checkpoint SHA-256
    `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`
  - VideoMAE2-G, checkpoint SHA-256
    `a231169cd13307aeb29bc56259a494ef273bfd9b7232f881ac4e8d7e39713add`
  - PixelSummary low-level baseline.
- Extracted depths: `block_00`, `block_10`, `block_20`, `block_30`, `final`
  for V-JEPA2-G and VideoMAE2-G; `pixel_summary` for PixelSummary.

## Fixed Probe Protocol

- Probe implementation: `scripts/train_vjepa2_layerwise_probes.py`.
- Classifier: `StandardScaler` plus balanced logistic regression.
- Seeds: `0,1,2,3,4`.
- Pair-held-out split: 60% train, 20% validation, 20% held-out test by
  deterministic pair hashing.
- Regularization: `C=1.0`.
- Maximum iterations: `2000`.
- Tasks:
  - `appearance`: four-way appearance-condition readout over all samples.
  - `physics_all`: possible/impossible readout over all appearances.
  - `physics_clean_only`: possible/impossible readout over clean samples only.

## Fixed Primary Layers

The following layers are frozen from the corrected development split and must be
used for confirmatory CIs and model comparisons on the sealed split:

| Model | appearance | physics_all | physics_clean_only |
|---|---|---|---|
| V-JEPA2-G | `block_20` | `block_30` | `block_30` |
| VideoMAE2-G | `block_30` | `block_00` | `block_30` |
| PixelSummary | `pixel_summary` | `pixel_summary` | `pixel_summary` |

Layerwise curves for all available layers may be plotted descriptively, but the
primary statistical claims must use the frozen layers above.

## Fixed Statistics

Run paired bootstrap over held-out pairs with 10,000 draws for the final sealed
test report.

Primary report:

- V-JEPA2-G `physics_all` pair accuracy and 95% CI.
- VideoMAE2-G `physics_all` pair accuracy and 95% CI.
- PixelSummary `physics_all` pair accuracy and 95% CI.
- V-JEPA2-G minus VideoMAE2-G `physics_all` paired delta and 95% CI.
- V-JEPA2-G minus PixelSummary `physics_all` paired delta and 95% CI.
- V-JEPA2-G `physics_clean_only` accuracy and paired deltas versus VideoMAE2-G
  and PixelSummary.
- Appearance accuracy for all three families as a nuisance/static-sensitivity
  check, not as a physics claim.
- Physical-distance-vs-depth trend for V-JEPA2-G and VideoMAE2-G, and their
  paired bootstrap difference.

Secondary robustness:

- Per-appearance `physics_all` breakdown: `clean`, `hue_090`,
  `brightness_0p45`, `selective_recolor_180`.
- Per-condition, per-difficulty, and per-camera breakdowns.
- Layerwise descriptive plots for probe accuracy, appearance displacement,
  physical displacement, and cross-appearance retrieval.

## Figure Plan

1. Layerwise probe curves: appearance vs `physics_all` vs `physics_clean_only`
   for V-JEPA2-G and VideoMAE2-G.
2. Geometry curves: same-video appearance displacement, matched physical
   displacement, and cross-appearance retrieval by depth.
3. Final sealed-test primary bars with paired-bootstrap CIs: V-JEPA2-G,
   VideoMAE2-G, PixelSummary, and V-JEPA2-G deltas.
4. Stratified robustness panel by appearance condition, with condition,
   difficulty, and camera in supplemental tables if space is tight.

## Claim Boundary

If the sealed test repeats the corrected development pattern, the workshop claim
is:

Frozen V-JEPA2-G representations contain a modest but testable physical-state
readout on corrected IntPhys2 that is stronger than a VideoMAE2-G comparator and
not explained by simple low-level pixel summaries, while appearance information
remains highly accessible. This supports a construct-validity story about
separating appearance accessibility from physical readout, not a claim that raw
latent surprise is a calibrated physical-surprise scalar.

If the sealed test does not repeat, report the development/test mismatch and do
not claim confirmatory physical readout.
