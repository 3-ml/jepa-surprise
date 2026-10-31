# WMPAI Alex A4: frame-statistic baselines on the archived 16-scene Kubric study (exploratory)

**Status: OBSERVED, exploratory, 16 scenes (144 scene x condition rows). Not held-out evidence.** CPU only, no model inference. Python 3.10.11, opencv 4.12.0, Windows 11. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a4_multifamily16_frame_baselines.py` and `a3_a4_lib.py`.

```
cd scripts/wmpai2026/alex
python a4_multifamily16_frame_baselines.py --out ../../../results/24-19-18-wmpai-alex-a4-multifamily16-frame-baselines
```

**Data.** 7,680 rendered frames (16 scenes x 10 rendered conditions x 48 frames) from `s3://jepa/results/23-16-09-vjepa2g-kubric-multifamily-signal16/full-artifacts/objects/run/rendered/`, downloaded to `${JEPA_MULTIFAMILY16_FRAMES}`. Every frame matched the archive's `full-artifacts-sha256sums.txt` (0 mismatches, 0 missing; one stray partial download file from an interrupted attempt was removed). Per-clip hashes of the concatenated frames are in `summary.json`. Outcome: the archive's stored native response from the clean clip (`response_from_clean`); rows are the nine non-clean rendered conditions (3 hue, 3 turn, 3 z displacement). The archive's in-memory appearance conditions (brightness, grayscale, inversion) have no frames and are not covered by the frame statistics. Their full-frame RGB L1 and native responses are stored for all 18 non-control conditions (288 scene-condition rows); a comparison that uses only the stored RGB L1 on all 288 rows is possible and was not done here.

**Features** (64x64 colour, 12 context frames, as in the pilot version), as the change from the same scene's clean clip: `static`, `motion`, `shuffled` (see `a3_a4_lib.frame_features`), plus the archive's stored full-frame RGB L1. Leave-one-scene-out linear fit; null = feature shuffled across scenes within each condition (5,000 permutations). "Flag gain" = leave-one-scene-out R^2 gained by adding a physical-edit flag (turn and z displacement).

| Feature | Leave-one-scene-out R^2 | Permutation p | Pooled Spearman | R^2 with flag | Flag gain |
|---|---|---|---|---|---|
| static | 0.039 | 0.006 | 0.34 | 0.115 | +0.077 |
| motion | -0.445 | 1.00 | 0.26 | -0.307 | +0.138 |
| shuffled | -0.916 | 1.00 | 0.32 | -0.750 | +0.166 |
| archived RGB L1 | -0.000 | 0.45 | 0.39 | 0.045 | +0.045 |

**Reading.** Trivial frame statistics predict little of the native response on left-out scenes: the best (static change) gives R^2 = 0.04, and motion and shuffled motion are worse than predicting the mean. The physical-edit flag adds R^2 beyond every one of them, most clearly beyond static change (+0.077). So on these 16 scenes the physical edits change the native score by more than simple frame change accounts for. This agrees in direction with the archive's own matched-RGB analysis; it is not a separate confirmation.

**Does not show.** It does not show the response is physical surprise. Physical edits (turn, z displacement) also differ from hue edits in motion and in how long the change persists, and the flag collects all of that. Direct edits, not lawful-versus-inconsistent pairs; 16 scenes; one outcome definition; very negative R^2 for motion and shuffled motion come from a few rows with large frame differences. Compare with `results/24-13-57-wmpai-alex-a4-pilot8-frame-baselines/` (8 scenes, no predictive signal from these statistics).
