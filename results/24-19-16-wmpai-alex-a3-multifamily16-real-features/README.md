# WMPAI Alex A3: real V-JEPA2-G features for the archived 16-scene Kubric study (exploratory)

**Status: OBSERVED, exploratory feature extraction. 16 scenes; not held-out and not confirmatory.** Same pipeline, environment and checkpoint as `results/24-19-09-wmpai-alex-a3-pilot8-real-features/` (read its README for the checkpoint, weight-shard hashes and processing details). Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_extract_fresh_scenes.py`, `gpu4gb_offload.py`, `a3_make_pilot8_table.py`.

```
python scripts/wmpai2026/alex/a3_make_pilot8_table.py --frames-root ${JEPA_MULTIFAMILY16_FRAMES} --out ${JEPA_MULTIFAMILY16_RECORD}_extract_table.csv
python scripts/wmpai2026/alex/a3_extract_fresh_scenes.py --scores ${JEPA_MULTIFAMILY16_RECORD}_extract_table.csv --frames-root ${JEPA_MULTIFAMILY16_FRAMES} --layers block_09 block_19 block_29 final --split development --batch 4 --out results/24-19-16-wmpai-alex-a3-multifamily16-real-features
```

**Data.** 7,680 rendered frames (16 scenes x 10 rendered conditions x 48 frames) of `s3://jepa/results/23-16-09-vjepa2g-kubric-multifamily-signal16/`, every frame checked against that archive's SHA-256 manifest (0 mismatches). The `development` split label is only needed by the extractor; these scenes were used for the archive's own analysis and are not fresh.

**Layers.** `block_09`, `block_19`, `block_29` and `final`, the same project mapping as the pilot, fixed before looking at any result.

**Result.** `features.npy` has shape (160, 4, 1408), all finite, 0 missing rows, aligned 1:1 with the input table by clip id and input hash (`feature_index.csv`). Extraction took about 9 minutes on the 4 GB GPU (batch 4, 160 clips, roughly 3.5 seconds per clip including loading; from file timestamps, with other applications open).

**Limitations.** One run, no determinism repeat. In-memory appearance conditions of the archive (brightness, grayscale, inversion) have no frames and no features. Checkpoint hash not re-computed in this run.
