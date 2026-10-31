# WMPAI Alex A3: feature displacement versus native score change, eight-scene Kubric pilot (exploratory)

**Status: OBSERVED, exploratory, 8 scenes (64 scene x condition rows). Not held-out evidence and not a claim about causal mediation.** CPU analysis of the stored features in `results/24-19-09-wmpai-alex-a3-pilot8-real-features/` and the pilot's stored native scores. Python 3.10.11. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_geometry_vs_error.py` and `a3_a4_lib.py`.

```
cd scripts/wmpai2026/alex
python a3_geometry_vs_error.py --features ../../../results/24-19-09-wmpai-alex-a3-pilot8-real-features --source pilot8 --out ../../../results/24-19-16-wmpai-alex-a3-pilot8-geometry-vs-error
```

**Question.** Does how far a clip's features move from its own scene's clean clip predict how much the native score changes, when predicting a scene that was left out of the fit?

**Definitions.** Relative displacement = ||f(clip) - f(clean clip of the same scene)|| / ||f(clean clip)||, at one layer. Outcome = native score (`scalar_l1`) minus the same scene's clean score. Rows: 4 hue and 4 physical-displacement conditions per scene. Fit: leave-one-scene-out linear regression; null: displacement shuffled across scenes within each condition (5,000 permutations, minimum p = 0.0002); so the null keeps each condition's displacement level and tests the scene-by-scene link. The physical flag marks the four displacement conditions.

| Layer | Leave-one-scene-out R^2 | Null mean R^2 | Permutation p | Pooled Spearman | R^2 with physical flag added |
|---|---|---|---|---|---|
| block_09 | -0.074 | -0.047 | 0.76 | 0.34 | -0.076 |
| block_19 | -0.045 | -0.051 | 0.56 | 0.45 | -0.039 |
| block_29 | 0.387 | -0.122 | 0.0002 | 0.57 | 0.390 |
| final | 0.442 | -0.149 | 0.0002 | 0.57 | 0.430 |

The physical flag alone gives R^2 = -0.011. Mean relative displacement for physical edits versus appearance edits: 0.0043 vs 0.0034 (block_09), 0.0098 vs 0.0074 (block_19), 0.0124 vs 0.0059 (block_29), 0.0839 vs 0.0326 (final).

**Reading.** At the two deepest layers the feature displacement predicts the native score change on left-out scenes beyond what the condition alone provides; at the two earlier layers it does not. Physical edits move the deepest features further than appearance edits. Adding the physical flag does not improve on displacement alone.

**What this does not show.** It does not show that geometry explains physical surprise or that the link is causal. The native score compares predictor output with final-layer target tokens of the same encoder, so a link between final-layer displacement and score change may be largely mechanical. It uses direct edits (hue and a keyed displacement), not lawful-versus-inconsistent pairs. Eight scenes give a very uncertain estimate, the R^2 values have no interval here, and layers and the displacement definition were not varied. This is an analysis-path regression; interpretation requires a separately documented evaluation on fresh development scenes.
