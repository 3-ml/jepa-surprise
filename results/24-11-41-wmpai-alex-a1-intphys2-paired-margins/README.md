# WMPAI Alex A1: IntPhys2 paired-margin reanalysis (secondary, exploratory)

**Status: OBSERVED, exploratory, secondary.** Stored scores only; no inference was run and no GPU was used. The 169-pair set was already inspected, so this is not fresh confirmatory evidence and it does not resume the stopped Main matrix.

## Command and provenance

```powershell
python scripts/wmpai2026/alex/a1_intphys2_paired_margins.py --out results\24-11-41-wmpai-alex-a1-intphys2-paired-margins
```

- Code revision: `23da93255f30920d5205bd6a3e9f6e0cfc45e66f` plus the uncommitted A1 script. Python 3.10.11, Windows 11, CPU only.
- Seed `20261024`, 10,000 bootstrap draws, resampling unit = `SceneIndex` (143 units), all conditions of a scene kept together.
- Source: `results/21-08-30-intphys2-main-one-third-endpoints-vjepa2-g/` (hashes in `a1_summary.json`).
- Archive `archive.tar.zst` SHA-256 `6a72dd779a39488e1b6252ea24f639071602bdf6b05c14043857711d427efca6`: re-verified, matches the gate README.
- Checkpoint `vitg.pt` SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`: re-verified, matches the gate README.

## Definitions

- Margin = impossible surprise minus possible surprise, from the stored native score. Pair-ordering accuracy = fraction of pairs with margin > 0; a tie counts as incorrect.
- "Shared sign" = fraction of pairs where the possible and impossible videos both shift in the same direction relative to clean.
- `SceneIndex` comes from the selection manifest. It matches the dataset's own `Main/metadata.csv` (checked afterwards: 253 scenes, 4 videos each, all 169 pairs consistent). At byte level, the 338 videos of the 169 pairs have 338 distinct SHA-256 values and no hash occurs in more than one `SceneIndex` (from `results/24-12-36-wmpai-alex-a4-intphys2-shortcut-baselines/per_pair_features.csv`). Whether distinct `SceneIndex` values can share underlying content that differs at byte level is unchecked, so intervals stay exploratory.

## Results (143 scenes, 169 pairs)

| Condition | Mean margin | Pair-ordering accuracy [scene-bootstrap 95%] | Shared sign |
|---|---|---|---|
| clean | +0.00014 | 0.633 [0.563, 0.703] | n/a |
| brightness-mild | +0.00013 | 0.639 [0.571, 0.703] | 0.704 |
| brightness-strong | +0.00008 | 0.609 [0.539, 0.679] | 0.840 |
| hue-mild | +0.00008 | 0.609 [0.541, 0.678] | 0.734 |
| hue-strong | -0.00004 | 0.556 [0.484, 0.627] | 0.882 |
| selective-recolor-mild | +0.00003 | 0.586 [0.518, 0.653] | 0.793 |
| selective-recolor-strong | +0.00015 | 0.615 [0.551, 0.680] | 0.805 |

Mean-margin intervals are in `a1_summary.json`; every one spans zero.

Consistency check: clean to strong-hue accuracy changes by -7.7 percentage points, matching the -7.69 reported by the stopped gate.

## What this shows and does not show

- Shows: margins are tiny relative to their uncertainty, and when an appearance shift occurs the possible and impossible members usually move together (70-88% of pairs), which fits a common-mode appearance offset.
- Does not show: any appearance effect on the margin that is distinguishable from zero, equivalence, or any event-localized effect (IntPhys2 Main has no violation timestamps). No condition was selected post hoc.
- Shared-sign here is per pair; the gate's 76.8% figure is per aligned window, so the two are not directly comparable.

## Limitations

`SceneIndex` was checked against the dataset's `Main/metadata.csv` (see Definitions) but content-level sharing between scenes is unchecked; ties are treated as incorrect; the severity grid was fixed before this analysis but the outcome-driven reading above is exploratory. It has a `Science_log.md` entry (24-11-41) as an exploratory analysis of stored scores. Raw per-pair scores: `per_pair_scores.csv`.
