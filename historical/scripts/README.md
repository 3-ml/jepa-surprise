# Source map

## IntPhys2

- `prepare_intphys2_confirmatory_manifest.py`, `build_intphys2_confirmatory_manifest.py`, and `validate_intphys2_confirmatory_manifest.py` construct and validate the locked split and transforms.
- `extract_vjepa2_intphys2_layers.py` and `extract_videomae2_intphys2_layers.py` extract five matched relative depths.
- `intphys2_confirmatory_common.py` and `intphys2_confirmatory_extraction.py` implement shared decoding, transformation, shard, locking, and provenance logic.
- `validate_intphys2_feature_parity.py` checks exact cross-model sample alignment.
- `pack_intphys2_feature_shards.py` creates the compact representation arrays in `results/intphys2/`.
- `extract_intphys2_pixel_summary.py` computes the 100-dimensional low-level comparator.
- `train_vjepa2_layerwise_probes.py` is retained for the superseded pair-held-out audit trail. `train_intphys2_grouped_probes.py` supports deterministic five-fold scene cross-fitting; `analyze_intphys2_grouped_probe_statistics.py` validates complete cross-model support and computes the global fixed-prediction scene bootstrap.
- `run_intphys2_geometry_audit.py`, `run_intphys2_geometry_freeze.py`, and `run_intphys2_geometry_sealed.py` reproduce the development-frozen controlled geometry analysis from bundled compact representations.
- `benchmark_vjepa2g_grid_replication.py`, `extract_vjepa2g_grid_replication.py`, and `pack_vjepa2g_grid_replication.py` reproduce the exact multi-dose feature package from IntPhys2 and the released V-JEPA2-G checkpoint. `transformation_manifold_metrics.py`, `run_transformation_manifold_audit.py`, and `finalize_vjepa2g_grid_replication.py` compute and validate the trajectory analysis.
- `make_paper_figures.py` rebuilds all four active paper figures from bundled result tables.

## LeWorldModel

- `run_lewm_prediction_error_audit.py` provides model, dataset, preprocessing, intervention, and metric helpers.
- `run_lewm_tworoom_controlled_displacement.py` and `run_lewm_pusht_controlled_displacement.py` implement identical-target lawful, zero-action, opposite-action, and hue controls.
- `run_lewm_pusht_role_color_audit.py` produces the matched role-color experiment and latent decomposition fields.
- `analyze_lewm_role_color_decomposition.py` aggregates encoder and predictor transport metrics.

## Frozen-DINO PushT controls

- `run_dinowm_pusht_native_error_audit.py` provides the released DINO-WM loading and scoring helpers.
- `run_dinowm_pusht_controlled_displacement.py` applies the 256-episode controlled PushT construction to DINO-WM.
- `export_pusht_controlled_audit_bundle.py` exports the selected four-frame clips without redistributing the source dataset.
- `run_jepa_wm_pusht_controlled_displacement.py` applies the same controlled construction to the released JEPA-WM PushT checkpoint while retaining its native preprocessing and predictor.
- `validate_jepa_wm_pusht_controlled_displacement.py` verifies complete support, target identity with DINO-WM, summaries, and paper-facing ordering.

## Kubric

- `kubric_movia_base_worker.py` creates the MOVi-A base scenes and retains its upstream attribution through `KUBRIC_NOTICE` and `KUBRIC_LICENSE`.
- `kubric_render_intervention.py`, `prepare_kubric_intervention_manifest.py`, and `canonicalize_kubric_intervention_context.py` create controlled variants with shared contexts.
- `render_kubric_intervention_grid.py`, `validate_kubric_intervention_blends.py`, and `analyze_kubric_intervention_rendering.py` render and validate the grid.
- `vjepa2_prediction_core.py` and `run_vjepa2_kubric_construct_validity.py` score the frozen intervention matrix.
- `analyze_vjepa2_kubric_construct_validity.py` computes scene-bootstrap contrasts.

## Static-content diagnostic

- `run_vjepa2_section_surprise.py` computes native V-JEPA2-L prediction residuals over fixed windows.
- `batch_static_surprise_audit.py` adds pixel-motion and repeated-frame controls across the selected Physion++ videos.

Run scripts from the archive root as modules, for example `python -m scripts.run_archive_smoke --output-dir _recreated_figures` or `python -m scripts.train_intphys2_grouped_probes --help`. Dataset- and checkpoint-dependent commands require the external assets described in the root README.
