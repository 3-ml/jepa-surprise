# PushT role-color decomposition

This mechanistic rerun uses the same 256 sources as the role-color confirmation and retains baseline residuals, predictor shifts, target-encoder shifts, residual shifts, and the exact linear term for each intervention.

`scores.csv` contains per-episode decomposition fields. `transport_summary.csv` aggregates shift magnitudes and cosines; `decomposition_summary.json` records reconstruction accuracy; the remaining tables and manifests align the rerun with the confirmed contrasts.

The maximum reconstruction error is 2.38e-7. Target-only block-blue versus orange and agent-gray versus purple score gaps nearly equal their target-encoder shift gaps, locating the abrupt effect primarily in encoder color-direction sensitivity. Persistent predictor and target shifts are directionally aligned but transport is incomplete.
