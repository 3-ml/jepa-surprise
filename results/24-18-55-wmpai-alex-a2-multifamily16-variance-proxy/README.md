# WMPAI Alex A2: scene-level variance proxy from the archived 16-scene Kubric study (exploratory)

**Status: OBSERVED, exploratory planning input. Not development data for the primary test and not confirmatory evidence.** Stored tables only; no model inference and no GPU. Python 3.10.11, pandas, numpy, Windows 11. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a2_multifamily16_variance.py` and `a2_power_study.py`.

```
cd scripts/wmpai2026/alex
python a2_multifamily16_variance.py --record ${JEPA_MULTIFAMILY16_RECORD}\run --out ../../../results/24-18-55-wmpai-alex-a2-multifamily16-variance-proxy
```

**Data.** The archived record `s3://jepa/results/23-16-09-vjepa2g-kubric-multifamily-signal16/full-artifacts/` (16 fresh MOVi-A scenes, seeds 231601 to 231618, V-JEPA2-G checkpoint SHA-256 `67129f01...5cb8e`, run on one A40). The 94 small files used (3.6 MB: analysis tables, manifests, logs) were downloaded with `scripts/wmpai2026/alex/r2_fetch.py` and each matched the archive's own `full-artifacts-sha256sums.txt` (0 mismatches). The record is not in this repository's `Science_log.md`; this record only reuses its stored scores.

**What was computed.** For each of the six RGB-L1-matched (physical edit, appearance edit) pairs the archive defines, the per-scene contrast `response(physical) - response(appearance)`, where response is the change of the native scalar from the clean clip. The scene-bootstrap mean of that contrast is then simulated for candidate scene counts at the observed scene-level SD.

**Reproduction check.** Recomputed per-pair means equal the archive's reported `mean_physical_minus_appearance_response` to within 4.4e-17 for all six pairs.

| Scene-level SD of the contrast (score units) | Value |
|---|---|
| smallest pair | 0.00334 |
| median pair | 0.00723 |
| mean over pairs, then SD across scenes | 0.00490 |
| largest pair | 0.01022 |

Expected scene-bootstrap 95% half-width of the mean contrast at the median-pair SD:

| Scenes | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| Half-width | 0.00324 | 0.00239 | 0.00198 | 0.00173 | 0.00142 | 0.00124 | 0.00100 | 0.00087 |

The archived pair means range from 0.0014 to 0.0104, so these half-widths are the same order as the smaller effects. Half-widths for the other SDs are in `variance_and_precision.json`.

**Limitations.** The contrast compares direct edits (z displacement, turn) with appearance edits matched on rendered RGB change. It is not the lawful-versus-inconsistent margin of the primary test, so the SD is a rough proxy. Sixteen scenes give a noisy SD (relative standard error about 18% for a normal sample). The simulation assumes normal scene-level contrasts. Neither a smallest meaningful effect nor a target half-width is chosen here; neither was jointly frozen in this note. Direct development-scene estimates are needed instead of this proxy.
