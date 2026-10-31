# WMPAI statistics protocol (historical unsigned draft)

**Status at the 2026-10-25 draft:** proposal by Alex, not frozen; no confirmatory scene had yet been generated under this proposed design. The proposal was not jointly ratified; section 9 records unresolved scientific choices. It is not the executed final256 analysis protocol. This revision (2026-10-25) adapts the draft to Vasi's crossed 2x2 development design (`protocol/vasi/development24-protocol.md` on branch `wmpai/vasi-experiments`), which was not yet merged into `main` at that revision.

## 1. Question and unit

For one frozen V-JEPA2-G checkpoint and one frozen native score, does the score rank a physically inconsistent continuation above its matched lawful continuation across appearances, and does one development-calibrated threshold transfer, on scenes that were not used to design anything?

- **Development unit (development24):** the configuration (`dev24-001` to `dev24-024`). Its two worlds, four trajectory cells and three appearances stay together in every resample. The 24 configurations cross three factors (sphere radius 0.60, 0.65, 0.70; nominal fall time 1.18 s and 1.24 s; camera x of -8, -4, 4, 8) inside one shared procedural template, so they are **not 24 independent scenes**. Every development-stage interval is conditional on that template.
- **Confirmatory unit:** an independent scene of a final generator that differs from the development template in ways frozen before any confirmatory scoring (requirements to be agreed, section 9). Claims about new scenes rest only on this unit.
- **Splits:** `development` and `confirmatory`. No scene or base trajectory is in both. The old eight Kubric scenes, the archived 16-scene study and every inspected IntPhys2 Main scene are regression and context only.

## 2. Design and estimands

### 2.1 Design (development24)

Each configuration has two worlds, a **solid** bridge and a visible **gap**, and two trajectories, **stop** and **fall**, giving four physical cells:

| World | Trajectory | Label |
|---|---|---|
| solid | stop | lawful |
| solid | fall | inconsistent |
| gap | fall | lawful |
| gap | stop | inconsistent |

The same motion is lawful in one world and inconsistent in the other, so a pure motion or pixel-change shortcut cannot rank the labels. Each cell is rendered under three appearances: `clean`, `persistent_hue` (applied from frame 0) and `abrupt_event` (applied from frame 12). Abrupt is secondary and is never pooled with persistent appearance.

### 2.2 Margins

With `S` the raw native scalar (no fitted correction), for configuration `c` and appearance `a`:

- solid-world margin `M_solid(c,a) = S(solid, fall) - S(solid, stop)`;
- gap-world margin `M_gap(c,a) = S(gap, stop) - S(gap, fall)`;
- **configuration margin** `M(c,a) = (M_solid + M_gap) / 2`.

`M` equals half the world-by-trajectory interaction contrast, `[S(solid,fall) - S(solid,stop) - S(gap,fall) + S(gap,stop)] / 2`. Additive effects of world and of trajectory cancel in `M`; a positive `M` means the score is higher when the motion contradicts the visible support, beyond those additive effects. It does not by itself separate physics from visual factors that act only in specific world-trajectory combinations (occlusion, contact appearance, post-event position); section 8 addresses these.

### 2.3 Estimands (one primary violation type, frozen before confirmatory scoring)

| ID | Estimand | Role |
|---|---|---|
| E1 | Fraction of units with `M > 0` (ties reported separately, counted as not positive) | Primary |
| E2 | Mean and median `M`, unit-clustered 95% bootstrap interval | Primary |
| E3 | Single-clip AUROC (inconsistent versus lawful clips) with unit-clustered interval and, at one threshold `t*` frozen from development, miss rate (inconsistent below `t*`) and false-alarm rate (lawful at or above `t*`) on confirmatory units and appearances | Primary for the "when" claim |
| E4 | Same as E1-E3 for the abrupt-event appearance | Secondary, never pooled with persistent |
| E5 | `M_solid` and `M_gap` reported separately (both signs and sizes) | Secondary; a positive margin in only one world is reported as such |
| E6 | Exploratory severities, temporal windows and per-tubelet traces | Exploratory, labelled, with stated multiplicity handling |

The primary appearance, violation type and score aggregation are fixed in the manifest before any confirmatory score is opened. E1 and E2 use the primary appearance; E1-E3 are then reported per appearance as secondary. A paired ranking may survive a common-mode appearance offset while a universal threshold fails, so E1/E2 and E3 are judged separately.

## 3. Outcome definitions (set before data)

- **Supported (E1/E2):** the lower bound of the clustered 95% interval for mean `M` is above 0 and the lower bound of E1 is above 0.5, both at the smallest practically meaningful effect declared in section 6.
- **Supported (E3):** AUROC lower bound above 0.5 by the declared margin and false-alarm and miss rates at the frozen `t*` each below the pre-declared ceiling (to be agreed from development data; no value is chosen here).
- **Inconclusive:** an interval overlaps the null but is wider than the declared precision target. An interval that contains 0 is never read as "equal". Equivalence is claimed only with a pre-registered margin and a two-sided test with adequate power.
- **Failure:** the interval is narrower than the precision target and excludes the supported region, or `M` is negative.
- Verdicts are reported per estimand and per gate as supported / inconclusive / untested, never as a single headline.
- **Development-stage reading:** development24 results are labelled DEVELOPMENT and conditional on the template. They tune nothing confirmatory except where section 9 lists them as inputs, and they are never reported as confirmation.

## 4. Handling of data problems

- Missing or failed cells are `missing`, never zero. A configuration with any of its four physical cells missing for an appearance is omitted from that appearance's estimands, listed by ID with the reason, and counted in a denominator table. Failed candidates are preserved and never replaced.
- `a2_scene_stats` treats a cell as missing when its `output_status` is not `complete`, whatever its score field holds, and rejects a table in which a failed cell carries a score. Vasi's manifest validator runs first (complete grid, hash format, failed-cell blank); `a2_scene_stats.validate` runs second.
- Exclusion rules depend only on render, state and visibility audits made before scores are opened. Exclusions are not changed after scores are visible.
- **Visibility rule (to settle jointly):** the target mask is non-zero in every scored frame 12 to 47 in both branches of both worlds. Proposed floor: one 16x16 patch, 256 pixels (0.39% of the 256x256 frame); Vasi's current configuration uses 164 pixels (0.25%). Per-frame target areas are saved for every clip. Before any clip is scored, the minimum area over frames 12 to 47 is computed for all 288 clips and the number of clips and configurations below 164 and below 256 pixels is reported; the floor is applied from that table, not from scores. Clips are reported, not silently dropped.
- **Independent audit before scores (proposed):** The audit uses `scripts/wmpai2026/alex/a4_audit_development24.py` on a hash-verified input bundle (shared context, state invariance, cross-world stop and fall matching, first-divergence frame inside 12 to 24, rail and support contact from the states files, area traces, canonicalization bounds, hashes) without opening any score; each configuration receives a pass/fail scientific audit before batch analysis.
- Refuse a table whose context hashes differ between lawful and inconsistent branches of the same world and appearance, or whose state hashes differ across appearances of the same branch, unless the protocol defines that difference.

## 5. Inference

- Bootstrap: 10,000 draws, fixed seed recorded (development24 uses 20261025), resampling units (configurations); the cells of a unit move together. If anything is fitted (threshold, probes, subspaces), it is fitted on development units only, or inside explicit nested folds.
- Development24 sensitivity: leave-one-configuration-out range of mean `M`, and **leave-one-factor-level-out** for each factor (drop each radius, each fall time, each camera x). `scripts/wmpai2026/alex/a2_dev24_stats.py` implements the configuration margins, cluster statistics and these sensitivities, plus a naive per-world version kept only to show it is anti-conservative.
- Clip-level AUROC pooled across units without clustering is descriptive only and carries no interval.
- E3's `t*` maximizes balanced accuracy on development clips and is saved with the analysis manifest before confirmatory scores are revealed.
- Multiplicity: E1-E3 for the single primary appearance are the confirmatory family and are not adjusted against each other; everything else (other appearances, E4-E6) is secondary with Holm adjustment within its labelled family, or descriptive only.
- Plot per-unit paired points so influential units are visible; report leave-one-unit-out sensitivity of E2.

## 6. Sample size (to be filled from development data, not guessed)

1. From development24, estimate the between-configuration SD of `M` (balanced margin) and the SDs of `M_solid` and `M_gap`.
2. Declare the smallest margin that would change the paper's claim (joint decision; not a number chosen here).
3. Simulate clustered precision for candidate confirmatory sizes and pick the smallest size whose expected 95% half-width is below the declared margin with the chosen assurance. The parent plan's 128-256 range is a resource envelope, not a power guarantee.
4. **Template caveat:** development24 varies three factors inside one template. Between-scene SD in an independent generator is expected to be at least as large, so the simulation also reports SD multipliers of 1.5 and 2, and the confirmatory pilot of the final generator supplies its own SD before N is frozen.
5. Record assumptions, observed variance and cost; the size is frozen before any confirmatory score is examined.

Reference only, not development data: the archived 16-scene Kubric study gives a scene-level SD of 0.0033 to 0.0102 (median 0.0072) for an RGB-L1-matched physical-minus-appearance contrast, so the expected 95% half-width of a mean over N scenes at the median SD is 0.0017 for N = 64 and 0.0012 for N = 128 (`results/24-18-55-wmpai-alex-a2-multifamily16-variance-proxy/`). That contrast is not the lawful-versus-inconsistent margin; these numbers must not be used to choose N.

## 7. Parity and numerical tolerances (development only)

- Native scalar: absolute difference at most 0.002 between devices (Vasi's development tolerance). Observed on one scene across three sources: at most 1.7e-04 between his RTX 3080 and an RTX 3050, and 3.1e-04 from the archived A40 (`results/25-13-48-wmpai-alex-crossdevice-native-score-parity/`).
- Per-tubelet scores: proposed 5e-03 until more scenes are compared (observed at most 2.1e-03 on one scene).
- A bitwise gate such as 1e-06 holds only on the same device; predictor-output hashes differ across devices.
- Encoder features (A3 extraction): relative L2 difference at most 1e-02 per layer against a stored A40 shard (observed at most 4.0e-03 on one clip).

## 8. Geometry and shortcut analyses (A3/A4) rules

- Extraction runs through `scripts/wmpai2026/alex/a3_extract_fresh_scenes.py` (frozen weights, bfloat16 autocast, mean over tokens, batch 4) on the model-input frames listed in each world's manifest, with the table built by `a3_make_dev24_table.py`. The layer list is passed explicitly and frozen on development units. The exploratory runs used `block_09`, `block_19`, `block_29` and `final` only because they follow the project's relative-depth mapping; for the confirmatory test the list is a development decision.
- Geometry measure used in earlier exploratory runs: relative displacement from the same scene's clean clip, with a scene-held-out association with the native score change (`results/24-19-16-wmpai-alex-a3-pilot8-geometry-vs-error/`, `results/24-19-26-wmpai-alex-a3-multifamily16-geometry-vs-error/`). Because the native score compares predictor output with final-layer target tokens of the same encoder, such a link may be largely mechanical; the confirmatory geometry test must target the lawful-versus-inconsistent margin `M`, not the displacement itself. Held-out units are configurations (development) or independent scenes (confirmatory).
- Probes, appearance directions, ranks and layers are fixed on development units. Test on confirmatory units: association between appearance geometry and native-error shift, and between physical geometry and `M`, against equal-rank random directions, shuffled labels, unrelated scenes and a no-adjustment baseline.
- Association is reported as association. A causal wording requires a predictor-consistent intervention with equal-rank and sham controls (parent plan, Stage 2).
- Shortcut baselines (static frame, temporal shuffle, frame-difference motion, plus gap and fall indicators) use the same unit partitions and readout budget. In the 2x2 design the physical flag equals gap XOR fall, so the nested test asks whether the world-by-trajectory interaction adds held-out prediction beyond additive world, trajectory and frame statistics (`a3_a4_lib.nested_gain`); it is association only.
- Target-hold and target-shuffle controls are reported for every cell once the primary cells are agreed, not only for one configuration.

## 9. Unresolved scientific freeze checklist

- [ ] Manifest schema, Vasi's validator and Alex's guard accepted by both; fixture tests pass
- [ ] Primary violation type, appearance(s), dose, score aggregation, tie rule
- [ ] Margin definitions (section 2.2) and unit (configuration for development, independent scene for confirmation)
- [ ] Visibility floor (164 or 256 pixels) and the pre-scoring area table
- [ ] Independent audit note per configuration, written before scores were shared
- [ ] Smallest meaningful margin, precision target, FA/miss ceilings
- [ ] **Independent final generator specified** (what must differ from the development template, what stays fixed) and its development pilot
- [ ] Confirmatory scene count and its simulation record
- [ ] `t*`, scaler, ranks, layers saved from development
- [ ] Parity tolerances (section 7) agreed
- [ ] Analysis script revision and its SHA-256; test suite hash
- [ ] Zero scene overlap between splits (checked by `validate`)
- [ ] Record that the development24 design was fixed after one n=1 score had been seen

Scientific review criteria: Is the claim tested by native error on lawful/inconsistent pairs rather than feature classification? Can someone reproduce the unit groups, aggregation and threshold? Are limitations and non-significant results visible? Is the analysis independent of the renderer's label source? Is every claim mapped to data marked development, exploratory or confirmatory?
