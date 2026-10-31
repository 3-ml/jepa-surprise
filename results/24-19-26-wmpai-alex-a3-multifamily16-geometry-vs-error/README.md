# WMPAI Alex A3: feature displacement versus native score change, archived 16-scene study (exploratory)

**Status: OBSERVED, exploratory, 16 scenes (144 scene x condition rows). Not held-out evidence and not a claim about causal mediation.** CPU analysis of the stored features in `results/24-19-16-wmpai-alex-a3-multifamily16-real-features/` and the archive's stored native responses. Python 3.10.11. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_geometry_vs_error.py` and `a3_a4_lib.py`.

```
cd scripts/wmpai2026/alex
python a3_geometry_vs_error.py --features ../../../results/24-19-16-wmpai-alex-a3-multifamily16-real-features --source multifamily16 --out ../../../results/24-19-26-wmpai-alex-a3-multifamily16-geometry-vs-error
```

Definitions and method are the same as in `results/24-19-16-wmpai-alex-a3-pilot8-geometry-vs-error/`. Rows: nine non-clean rendered conditions per scene (3 hue, 3 turn, 3 z displacement). Outcome: the archive's `response_from_clean`. Physical flag: turn and z displacement. The archive's full-frame RGB L1 (stored for all 144 rows used) is the rendered-change baseline.

| Layer | Leave-one-scene-out R^2 | Null mean | Permutation p | Pooled Spearman | R^2 with physical flag added | Gain from displacement over RGB L1 |
|---|---|---|---|---|---|---|
| block_09 | 0.247 | 0.045 | 0.0006 | 0.34 | 0.259 | 0.169 |
| block_19 | 0.445 | 0.025 | 0.0002 | 0.39 | 0.471 | 0.425 |
| block_29 | 0.760 | 0.057 | 0.0002 | 0.48 | 0.755 | 0.780 |
| final | 0.666 | 0.055 | 0.0002 | 0.44 | 0.660 | 0.650 |

The physical flag alone gives R^2 = 0.079 and RGB L1 alone gives -0.0002. Mean relative displacement, physical versus appearance edits: 0.0082 vs 0.0050 (block_09), 0.0134 vs 0.0107 (block_19), 0.0200 vs 0.0087 (block_29), 0.1231 vs 0.0519 (final).

**Reading.** The feature displacement from the clean clip predicts the native score change on left-out scenes at all four layers, most strongly at block_29 and the final layer, and far better than the rendered RGB change or the physical flag. At the deep layers the flag adds nothing once displacement is included. Physical edits move deep features about twice as far as appearance edits. The pilot (8 scenes) shows the same pattern at the deep layers.

**What this does not show.** The native score compares the predictor's output with final-layer target tokens of the same encoder, so a strong link between deep-layer displacement and score change may be largely mechanical and says little about physics. The outcome mixes conditions with very different response sizes (turn 180 degrees is much larger than hue), so the pooled R^2 is dominated by a few conditions; the within-condition null addresses the scene-by-scene link only. Direct edits, not lawful-versus-inconsistent pairs; 16 scenes; one displacement definition; no intervals on R^2; layers fixed in advance but not varied. Per-condition breakdowns and intervals were not executed here; interpretation requires a separately documented evaluation on fresh development scenes.
