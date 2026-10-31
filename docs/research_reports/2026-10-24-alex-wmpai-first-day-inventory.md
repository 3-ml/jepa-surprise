# Alex WMPAI first-day inventory (read-only)

**Date:** 2026-10-24. **Repo revision:** `23da932` (main). **Status:** inventory only. No experiment was run, no model was scored, and no existing result directory was modified. Per-file SHA-256 hashes for 180 files are in [`2026-10-24-alex-wmpai-inventory-hashes.csv`](2026-10-24-alex-wmpai-inventory-hashes.csv).

## Inventory

| Record | Files | Size | Role in A1-A4 |
|---|---|---|---|
| `results/21-08-30-intphys2-main-one-third-endpoints-vjepa2-g/` | 10 | 0.4 MB | A1 source: `pair_surprises.csv` (1,183 rows = 169 pairs x 7 conditions), `selection-manifest.json`, `summary.json` |
| `results/22-19-03-intphys2-endpoint-analysis-gate/` | 9 | 0.1 MB | A1 reference: stopped gate (STOP, no Holm-adjusted pass), `summary.json`, window diagnostics |
| `results/23-05-00-vjepa2g-kubric-construct-validity-pilot8/` | 57 | 7.7 MB | Regression fixture only (8 scenes): `frozen_manifest.json`, `analysis/per_scene_deltas.csv` |
| `results/23-22-48-intphys2-corrected-union-split-lock/` | 1 | <0.1 MB | Scene-disjoint split lock (76 pairs / 38 scenes development, 114 pairs / 57 scenes sealed) |
| `results/24-01-50-intphys2-corrected-sealed-test/` | 52 | 29.2 MB | Superseded pair-held-out probe record (scene leakage found) |
| `reports/geometry_freeze/` | 21 | 136.1 MB | A3 reference: frozen geometry metrics |
| `reports/geometry_sealed/` | 30 | 136.6 MB | A3 reference: sealed geometry evaluation |

External data: `${JEPA_INTPHYS2_ENDPOINT}\extracted\intphys2-appearance-completion\main` is present (conditions: clean, hue/brightness mild-medium-strong, ...). The archive's recorded SHA-256 is `6a72dd77...efca6` (from the gate README) and has **not** been re-verified yet.

Reference scripts for A1/A3: `analyze_intphys2_endpoint_gate.py`, `analyze_intphys2_layerwise_geometry.py`, `run_intphys2_geometry_freeze.py`, `run_intphys2_geometry_sealed.py`, `extract_vjepa2_layerwise_representations.py`, `analyze_vjepa2_kubric_construct_validity.py`.

## Findings that affect the plan

1. **`results/30-19-46-intphys2-scene-grouped-probe-audit/` is not a folder in this checkout, but its outputs are tracked in `jepa-anonymous-archive/results/intphys2/scene_grouped_probe_audit/`** (added by commit `e710ed9`; logged in `Science_log.md` as 30-19-46). Checked against the archive's `sealed/statistics/` CSVs: V-JEPA2-G `physics_all` `block_30` 0.5604 [0.526, 0.597], VideoMAE2-G 0.5137, PixelSummary 0.4979 and the two deltas (0.0470, 0.0623) match the logged numbers. Recomputing the point estimates from `selected_predictions.csv` gives the same values (0.5604 / 0.5135 / 0.4979); a separately reconstructed scene-bootstrap interval for V-JEPA2-G was [0.523, 0.608] against the archived [0.526, 0.597], so the archive uses a different resampling scheme and the intervals were not reproduced exactly.
2. **A1 scene IDs exist but are not in the score table.** `pair_surprises.csv` has only `pair` (e.g. `1/1`). The selection manifest maps each pair to `SceneIndex` and `member`. The 169 pairs cover **143 distinct SceneIndex values** (117 scenes with 1 pair, 26 with 2). The project's own `scripts/prepare_intphys2_confirmatory_manifest.py` groups and splits by `SceneIndex`, so it is the established scene unit; it has since been checked against the dataset's own `Main/metadata.csv` inside `IntPhys2.zip`: all 253 Main scenes have exactly 4 videos (two matched pairs), each scene has a single game, environment, condition, camera and difficulty, and all 169 manifest pairs agree with the metadata. So `SceneIndex` is the dataset's scene field. Whether different `SceneIndex` values can share underlying content (same game and environment) is not checked, so scene-level intervals remain exploratory.
3. **`pair_surprises.csv` already holds `possible_surprise` and `impossible_surprise` per pair per condition**, so the A1 margin is computable without any new inference. Raw per-window scores are in the extracted archive.
4. **No violation timestamps exist for IntPhys2 Main** (gate README). Event-localization claims are out of scope for A1.
5. **Overlap risk:** the 169-pair set and the workshop selection (`results/22-20-13-...`) are both inspected, so none of this is fresh confirmatory data.

## Proposed scientific metadata schema (not an execution assignment)

- Per clip: `scene_id`, `base_trajectory_id`, `split`, `physical_condition`, `appearance_condition`, dose, state-trace SHA-256, context/target input SHA-256, render path and frame manifest, ground-truth event frame, checkpoint SHA-256, frozen frame selection, score-protocol ID.
- Scores table: raw native scalar, per-target-tubelet score with source-frame mapping, precision, input hashes. Missing or failed cells stay missing.
- Masks, visibility and affected-pixel fraction per clip (needed for the A4 shortcut checks), and RGB L1 plus optical-flow summaries.
- Confirmation that persistent colour variants share the clean clip's simulator state, and which condition is target-only abrupt.
- Total development-scene count and per-scene clip count, so the power study can be sized.

