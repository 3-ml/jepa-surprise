# WMPAI Alex A4: frame-statistic baselines on the eight-scene Kubric pilot (exploratory)

**Status: OBSERVED, exploratory, 8 scenes (64 scene x condition rows). Not held-out evidence.** CPU only, no model inference. Python 3.10.11, opencv 4.12.0. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_a4_lib.py` and `a4_pilot8_frame_baselines.py`.

```
python scripts/wmpai2026/alex/a4_pilot8_frame_baselines.py --out results/24-13-57-wmpai-alex-a4-pilot8-frame-baselines
```

**Data.** The 3,456 rendered PNGs (8 scenes x 9 conditions x 48 frames) were downloaded from `s3://jepa/results/23-05-00-vjepa2g-kubric-construct-validity-pilot8/full-artifacts/objects/pilot/rendered/` (Cloudflare R2) to `${JEPA_PILOT8_FRAMES}` with `scripts/wmpai2026/alex/r2_fetch.py`. Each file's size was checked against the bucket listing; the count matches the pilot's validation summary (3,456 frames). Per-clip SHA-256 of the concatenated frames is in `summary.json`. All 3,456 files were then checked against the pilot's `full-artifacts-sha256sums.txt`: no mismatch, none extra, none missing (`frame_hash_verification.json`).

**Features** (64x64 colour, 12 context frames; `a3_a4_lib.frame_features`), taken as the change from the same scene's clean clip: `static` (mean abs difference of target frames from the last context frame), `motion` (mean frame-to-frame difference of target frames), `shuffled` (motion after a fixed random permutation of the target frames).

**Outcome:** native score change from the clean condition (`scalar_l1`). Leave-one-scene-out linear fit; null = feature shuffled across scenes within condition (5,000 permutations). "Flag gain" = leave-one-scene-out R^2 gained by adding a physical-family flag.

| Feature | LOSO R^2 | Permutation p | Pooled Spearman | Gain from physical flag |
|---|---|---|---|---|
| static | -0.030 | 0.59 | -0.006 | -0.002 |
| motion | -0.042 | 0.64 | 0.398 | -0.003 |
| shuffled | -1.558 | 1.00 | 0.232 | -0.023 |

**Reading.** None of the three frame statistics predicts the native score change on held-out scenes (all R^2 at or below zero, p well above 0.05). The physical flag adds nothing on top of them, but with nothing predictive there is nothing for it to add to. This neither supports nor refutes a shortcut explanation in this pilot, and it does not show appearance and displacement are equivalent. Motion has a moderate pooled rank correlation (0.40) that does not survive scene-held-out prediction. The very negative `shuffled` R^2 comes from a few influential rows, not a signal. The old pixel-level matching (RGB L1) result for the same data is in `results/24-12-28-wmpai-alex-a3a4-pilot8-exploratory-v2/`.

**Not covered.** The `target_hold` and `target_shuffle` conditions have no rendered folders (they were made at inference time) and are excluded. The pilot has no lawful-versus-inconsistent pairs, so this does not test discrimination; that requires a separately documented fresh-scene evaluation.
