# WMPAI Alex A3/A4 on the eight-scene Kubric pilot (v2, exploratory)

**Status: OBSERVED, exploratory code regression. 8 scenes; not held-out evidence and not a claim about geometry or shortcuts.** Stored tables only; no model inference, no frames, no GPU. Python 3.10.11, CPU, code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_a4_lib.py` and `a3_a4_pilot8.py`.

Command (permutation count 5,000, `numpy` seed 0 in the library):

```
python scripts/wmpai2026/alex/a3_a4_pilot8.py --out results/24-12-28-wmpai-alex-a3a4-pilot8-exploratory-v2
```

Inputs (hashes in `summary.json`): `model/scores.csv`, `render_validation/render_metrics.csv`, `render_validation/blend-validation.json` from `results/23-05-00-vjepa2g-kubric-construct-validity-pilot8/`.

**Relation to the sibling folder `results/24-12-27-wmpai-alex-a3a4-pilot8-exploratory/`:** its first run reported `pass: false` because my check assumed the stored physical-condition `target_transform_max_abs` would be positive; that key is absent from the physical rows (the blend validation file stores the displacement under other keys, see the correction below). v2 changes only that check. The A3/A4 numbers are identical in both runs. The first folder is kept unchanged.

## Independent validity check (from stored tables)

- 64 state-validation rows over 8 scenes, all passed. Non-target and pre-event transforms are exactly 0. Appearance conditions have zero target transform and equal material fields. Context bytes are exact in all 72 renders.
- **Correction (added after the first version of this record):** the original text here said the physical conditions' state change could not be confirmed because `target_transform_max_abs` is absent from those rows. That was incomplete. The same `blend-validation.json` records `z_displacement_world` (0.35, 0.70, 1.40 and 2.10 for `physical-z-0p25`, `-0p5`, `-1p0`, `-1p5`) and `target_after_declared_delta_max_abs` (at most 1.2e-07 for all 32 physical rows), so the declared displacement was applied after the event to within float precision, and all 32 rows passed. This confirms a keyed displacement only; it is not evidence of a lawful-versus-inconsistent contrast. The analysis script `a3_a4_pilot8.py` did not read these two fields.
- **Observed:** in 3 of 8 scenes (`seed-230201`, `seed-230209`, `seed-230211`) the target has zero segmentation pixels in at least one of the 36 target frames of the clean render (min over target frames; definition in `scripts/analyze_kubric_intervention_rendering.py`). Event-frame area is at least 796 px in all scenes. Interventions cannot change pixels of a frame where the target is invisible, so scores in those scenes mix visible and invisible frames. This was not a gate in the pilot protocol, which checked the event frame.

## A3: does encoder-side displacement predict the score change on held-out scenes?

Predictor: `target_shift_l1_layernorm` (the only geometry-like quantity stored for the pilot). Outcome: score change from the clean condition. 64 rows (8 scenes x 8 non-clean conditions); leave-one-scene-out linear fit; null = shift shuffled across scenes within condition.

| Quantity | Value |
|---|---|
| Leave-one-scene-out R^2 | -0.011 |
| Permutation-null mean R^2 | -0.039 |
| Permutation p | 0.31 |
| Pooled Spearman | 0.56 |

The pooled rank correlation is mostly a dose/condition effect; the held-out-scene prediction is not distinguishable from its null. This neither supports nor refutes a geometry-to-error link.

## A4: does the physical flag explain score change beyond rendered change?

Leave-one-scene-out R^2 of score change, base predictor alone vs. base plus a physical-family flag:

| Base predictor | Base R^2 | With physical flag | Gain |
|---|---|---|---|
| Target RGB L1 | -0.105 | -0.110 | -0.005 |
| Changed-pixel fraction | -0.047 | -0.051 | -0.004 |
| Encoder shift | -0.011 | -0.026 | -0.015 |

All R^2 are at or below zero: none of these simple predictors predicts held-out-scene score change, so there is nothing for the flag to add. This does not show appearance and physics are equivalent.

## Not done and why

The static-frame, temporal-shuffle and frame-difference baselines (`frame_features` in the library, unit-tested on synthetic clips) were **not run on the pilot**: its 3,456 rendered PNGs (1.19 GB) were only in the remote archive `s3://jepa/results/23-05-00-vjepa2g-kubric-construct-validity-pilot8/` when this record was written. They were downloaded afterwards, checked against the pilot's SHA-256 manifest, and the baselines were run: see `results/24-13-57-wmpai-alex-a4-pilot8-frame-baselines/`.
