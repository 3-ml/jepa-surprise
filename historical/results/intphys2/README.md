# Corrected IntPhys2 result map

- `development_extraction/` records cross-model parity for the corrected 76-pair development extraction.
- `development_analysis/` contains compact V-JEPA2-G, VideoMAE2-G, and PixelSummary representations used to choose layers and freeze the analysis.
- `sealed_analysis/` contains the corresponding 114-pair sealed-population compact representations, descriptive geometry, and superseded pair-held-out probe outputs.
- `scene_crossfit_probe_audit/` contains the current paper-facing five-fold out-of-fold predictions, exact coverage records, and global scene-bootstrap intervals for development and sealed populations.
- `scene_grouped_probe_audit/` contains the superseded repeated scene-holdout correction.
- `controlled_geometry/` contains the development-frozen magnitude, shuffled-direction, cross-fitted projection, sensitivity, and stability records for the sealed population.
- `multidose_geometry/` contains the post-sealed exploratory V-JEPA2-G trajectory analysis and its exact extraction, feature, runtime, and statistical provenance.

The current sealed `physics_all` results are V-JEPA2-G 0.579 [0.542, 0.618], VideoMAE2-G 0.492 [0.473, 0.509], and PixelSummary 0.500 [0.489, 0.510], with V-JEPA2-G deltas of 0.087 [0.044, 0.133] and 0.079 [0.041, 0.118]. The cross-fit protocol is a post-hoc correction after sealed-population inspection; intervals condition on fitted predictions.
