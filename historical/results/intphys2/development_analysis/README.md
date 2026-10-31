# IntPhys2 development analysis

This directory contains 608 appearance-conditioned samples from 76 pairs. V-JEPA2-G and VideoMAE2-G each provide five 1408-dimensional pooled depths; PixelSummary provides one 100-dimensional low-level view.

The development analysis selected V-JEPA2-G `block_30` and VideoMAE2-G `block_00` for all-appearance physics, `block_30` for both clean-only physics readouts, and `block_20`/`block_30` for V-JEPA2-G/VideoMAE2-G appearance. These choices are recorded in `configs/intphys2_sealed_analysis_spec.json`.

Development pair-held-out probe outputs are retained as a superseded layer-selection audit trail. The current out-of-fold check is in `../scene_crossfit_probe_audit/development/`; it preserves V-JEPA2-G `block_30` for `physics_all`, while its comparator deltas include zero and clean-only development results remain secondary.
