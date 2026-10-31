# WMPAI Alex A4: trivial-video-statistic baselines on the 169 IntPhys2 pairs (exploratory)

**Status: OBSERVED, exploratory, inspected data. Not confirmatory.** No model inference, CPU only (Python 3.10.11, opencv 4.12.0, Windows 11). Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a4_intphys2_shortcut_baselines.py`.

```
python scripts/wmpai2026/alex/a4_intphys2_shortcut_baselines.py --out results/24-12-36-wmpai-alex-a4-intphys2-shortcut-baselines --tmp <scratch>
```

Videos come from `${JEPA_INTPHYS2_ZIP}` (the 338 Main videos of the stored 169-pair selection; per-video SHA-256 in `per_pair_features.csv`). Which member is impossible comes from each video's stored metadata (`type`). Bootstrap: 10,000 draws over 143 `SceneIndex` units, seed 20261024. The direction (higher feature = impossible) was fixed before running; there is no fitting, so no train/test leakage.

## Result: pair-ordering accuracy, P(feature(impossible) > feature(possible))

| Feature | Accuracy [scene-bootstrap 95%] | Ties |
|---|---|---|
| Static (mean abs difference from the first frame) | 0.627 [0.553, 0.701] | 0 |
| Motion (mean abs frame-to-frame difference) | 0.479 [0.408, 0.552] | 0 |
| Temporal-shuffled motion | 0.473 [0.395, 0.549] | 0 |
| Frame count | 0.047 [0.017, 0.086] | 152 |

Reference, same pairs, from A1: V-JEPA2-G native score, clean condition, 0.633 [0.563, 0.703].

**What it shows.** A trivial statistic, how far the video drifts from its first frame, orders the pairs about as well as V-JEPA2-G's score (0.627 vs 0.633, heavily overlapping intervals). So the stored clean pair-ordering accuracy does not by itself demonstrate physics-specific signal. Motion magnitude and temporal order are not what drives it.

**What it does not show.** That V-JEPA2-G uses this statistic, or that its score is a pixel-change detector. Impossible events such as objects vanishing legitimately change frames, so "drift from the first frame" may partly be a property of the event itself. Frame count is uninformative: 152 of 169 pairs have equal length, so its accuracy of 0.047 is a tie artefact (ties count as not greater), not a reversed signal.

**Caveats.** Four features were tried with no multiplicity correction; only one clears 0.5. These are the same inspected scenes as the stopped endpoint gate, so this is context for a fresh test, not a claim. The static feature uses all frames of the full video, while V-JEPA2-G scored 48-frame windows, so the two are comparable in accuracy only, not in construction.

**Unexecuted follow-up proposal.** Compare the same static baseline (`a3_a4_lib.frame_features`) with the native score on fresh development scenes, and evaluate the physical flag's incremental association using `nested_gain`.
