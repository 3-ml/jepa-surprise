# IntPhys2 sealed analysis

This directory contains 912 appearance-conditioned samples from 114 pairs in 57 scenes. The frozen primary layers and statistics in `configs/intphys2_sealed_analysis_spec.json` were applied without sealed-population layer selection.

- `vjepa2/`, `videomae2/`, and `pixel_summary/` contain compact representations, aligned sample metadata, superseded pair-held-out probe outputs, and descriptive geometry.
- `statistics/` contains the superseded fixed-prediction pair-bootstrap intervals, model deltas, stratified summaries, and held-out predictions.
- `feature-parity.json` records exact cross-model sample alignment.

The pair-held-out probe outputs in this folder are retained only as a superseded audit trail. Current paper-facing IntPhys2 inference uses `../scene_crossfit_probe_audit/sealed/`, where every scene is held out exactly once, pair predictions are aggregated to one scene value, and the fixed-prediction bootstrap resamples all 57 unique scenes globally.
