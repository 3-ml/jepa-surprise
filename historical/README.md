# Anonymous artifact: latent prediction error in video world models

This archive accompanies the anonymous paper *When Does Latent Prediction Error Support Physical-Surprise Claims in Video World Models?* It contains paper-facing source snapshots, locked manifests, compact features, predictions, statistics, validation records, figures, and environment specifications. Third-party datasets, checkpoints, and the large Kubric render tree are not redistributed.

## Quick start

Use CPython 3.12.13 from the archive root:

```powershell
python3.12 -m venv <workspace>/artifact-smoke
<workspace>/artifact-smoke/bin/python -m pip install -r environments/smoke/requirements.txt
<workspace>/artifact-smoke/bin/python -m scripts.run_archive_smoke --output-dir <workspace>/recreated-figures
```

Windows uses `<workspace>\artifact-smoke\Scripts\python.exe` in the final two commands. The smoke verifies the complete byte-exact archive manifest, current IntPhys2 readout and geometry values, the DINO-WM and JEPA-WM PushT records, the focused tests, and all four figures rebuilt from bundled tables.

## Headline results

- On the corrected IntPhys2 sealed population, five-fold scene cross-fitting and a global fixed-prediction scene bootstrap give V-JEPA2-G physics accuracy 0.579 [0.542, 0.618], VideoMAE2-G 0.492 [0.473, 0.509], and PixelSummary 0.500 [0.489, 0.510].
- V-JEPA2-G exceeds VideoMAE2-G and PixelSummary by 0.087 [0.044, 0.133] and 0.079 [0.041, 0.118] on the all-appearance physics task.
- Appearance accuracy under the same audit is 0.977 for V-JEPA2-G, 0.953 for VideoMAE2-G, and 0.791 for PixelSummary.
- Controlled geometry gives V-JEPA2-G physical projection into the cross-fitted four-dimensional appearance space of 0.401 [0.320, 0.483] at block 00 and 0.060 [0.050, 0.071] at final, against matched random means near 0.00284.
- The post-sealed V-JEPA2-G dose grid gives final straightness 0.907 for brightness, 0.692 for hue, and 0.553 for selective recoloring; its corresponding effective ranks are 5.43, 9.83, and 13.72.
- LeWorldModel residuals increase with action inconsistency in TwoRoom and PushT, while simulator-state-preserving hue changes can match or exceed the action-mismatch effect.
- DINO-WM assigns lower visual latent error to hue than to matched opposite actions in PushT; a post-hoc JEPA-WM extension over the same target bytes shows the same ordering under its native predictor and preprocessing.
- The eight-scene Kubric pilot does not detect scalar separation between matched appearance and direct state-displacement interventions.
- The Physion++ diagnostic finds high-error near-static windows whose frozen-frame residuals are nonpositive.

The IntPhys2 development and sealed populations are scene-disjoint; every sealed scene is held out exactly once and contributes one aggregated scene value. The current paper-facing result is the post-hoc corrective analysis in `results/intphys2/scene_crossfit_probe_audit/`; intervals omit probe-refitting and fold-assignment variability. Pair-held-out and repeated scene-holdout outputs remain as superseded audit trails.

## Archive map

- `scripts/` contains source snapshots for the paper-facing experiments and analyses.
- `tests/` contains focused tests for those scripts.
- `configs/` contains the IntPhys2 extraction, readout, controlled-geometry, and multi-dose contracts. `configs/original_sealed_analysis_spec/README.md` is the byte-exact original frozen specification.
- `manifests/` contains the locked IntPhys2 inventory, pairs, samples, transform metrics, and inspected-scene exclusions.
- `environments/` contains dependency specifications and the Kubric renderer environment.
- `results/intphys2/` contains development and sealed compact representations, superseded probe outputs, the current scene-cross-fit audit, controlled geometry, and multi-dose trajectory records.
- `results/leworldmodel/` contains TwoRoom and PushT action-consistency results plus the PushT role-color decomposition.
- `results/dinowm/` contains the paper-facing DINO-WM PushT controlled-displacement result.
- `results/jepa_wm/` contains the post-hoc JEPA-WM PushT robustness check.
- `results/kubric/` contains the renderer determinism gate and exact-rerender pilot.
- `results/static_content_diagnostic/` contains the V-JEPA2-L Physion++ window, pixel-motion, and frozen-frame audit.
- `figures/` contains metadata-stripped active-paper assets; `scripts/make_paper_figures.py` rebuilds all four from bundled tables.
- `archive_manifest.json` records final archive-relative SHA-256 values after anonymization.
- `archive_audit.json` records the final comment, identity, secret, ignored-file, and structure checks.

## External assets

`environments/external_assets/README.md` gives public retrieval locations, revisions, expected hashes, and the raw-versus-compact reproduction boundary. Paths represented as `<artifact-root>`, `<dataset-cache>`, `<checkpoint-cache>`, `<archive-root>`, or `<workspace>` must be supplied locally. The bundled compact artifacts reproduce paper-facing statistics and figures without those downloads; raw inference requires the separately licensed datasets and checkpoints. The generated 1.188 GB Kubric tree is omitted, but its generator, fixed seeds, intervention pipeline, validation records, inventory, and checksum-manifest hash are retained.

## Third-party terms

This archive does not redistribute third-party datasets or checkpoints. V-JEPA2-G's model card lists Apache-2.0; V-JEPA2-L lists MIT; V-JEPA2 and VideoMAE V2 source repositories are MIT; the evaluated VideoMAE2-G checkpoint and IntPhys2 are CC BY-NC 4.0; LeWorldModel and DINO-WM source are MIT; JEPA-WM source and model releases are CC BY-NC 4.0; DINOv2 is Apache-2.0; Kubric is Apache-2.0. No separate Physion++ dataset license was found in the downloaded archive, so only derived numeric diagnostics and source identifiers are retained here.

## Anonymization boundary

Personal names, hostnames, home directories, private storage addresses, cloud instance identifiers, and internal source-history identifiers were removed or replaced. Image metadata was stripped by lossless pixel-preserving re-encoding. Allowed `README.md` files provide navigation and protocol summaries. Hashes embedded in execution provenance refer to the pre-anonymization execution artifacts; `archive_manifest.json` gives hashes of the final anonymous files. The paper's original-specification SHA-256 `f7e0ca1651c9586c180ba89c8be347d02d99de9045a8b1b9b75ed6d9fac7b465` is directly verifiable against `configs/original_sealed_analysis_spec/README.md`; the later anonymized corrective cross-fit specification is `configs/intphys2_sealed_analysis_spec.json`.
