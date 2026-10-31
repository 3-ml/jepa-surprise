# WMPAI Alex A3: real V-JEPA2-G features for the eight-scene Kubric pilot (exploratory regression)

**Status: OBSERVED, exploratory regression of the A3 extraction path. 8 scenes; not held-out and not confirmatory.** First end-to-end run of `a3_extract_fresh_scenes.py` with the real frozen encoder, on the 4 GB GPU (see `results/24-18-53-wmpai-alex-gpu4gb-offload-parity/`). Python 3.10.11, PyTorch 2.5.1+cu121, RTX 3050 Laptop GPU (4 GiB), Windows 11. Code revision `23da932` plus uncommitted `scripts/wmpai2026/alex/a3_extract_fresh_scenes.py`, `gpu4gb_offload.py`, `a3_make_pilot8_table.py`.

```
python scripts/wmpai2026/alex/a3_make_pilot8_table.py --frames-root ${JEPA_PILOT8_FRAMES} --out external-record/a2f271d920c7/pilot8_extract_table.csv
python scripts/wmpai2026/alex/a3_extract_fresh_scenes.py --scores external-record/a2f271d920c7/pilot8_extract_table.csv --frames-root ${JEPA_PILOT8_FRAMES} --layers block_09 block_19 block_29 final --split development --batch 4 --out results/24-19-09-wmpai-alex-a3-pilot8-real-features
```

**Checkpoint.** `vitg.pt`, SHA-256 `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e` (locked model config; not re-hashed in this run, recorded in `feature_meta.json` as not verified in this run). Per-block weight files and their hashes: `results/24-18-53-wmpai-alex-gpu4gb-offload-parity/shard_index.json`.

**Data.** 3,456 rendered frames of the pilot (8 scenes x 9 conditions x 48 frames), downloaded from R2 and checked against the pilot's SHA-256 manifest with no mismatch (`results/24-13-57-wmpai-alex-a4-pilot8-frame-baselines/frame_hash_verification.json`). Each clip's input hash in the table is computed from the frame files; the `development` split label is only needed by the extractor, these scenes are not fresh.

**Layers.** `block_09`, `block_19`, `block_29` (outputs of transformer blocks 10, 20 and 30, the project's relative-depth mapping) and `final` (encoder output). They were fixed by that mapping, not chosen from any result.

**Processing.** 48-frame clips at 256 pixels, batch 4, bfloat16 autocast with fp32 weights, mean over all tokens per layer, float32 output.

**Result.** `features.npy` has shape (72, 4, 1408), all values finite, 0 missing rows. `feature_index.csv` aligns rows 1:1 with the input table by clip id and input hash. Values range from -48.8 to 701.5; a few channels are very large, as is common in vision transformers.

**Limitations.** One run; no repeat for determinism. The 4 GB path was compared with an A40 shard on one clip only. The 8 scenes, 9 conditions and their native scores come from the earlier pilot, so any analysis on these features is exploratory. Conditions `target_hold` and `target_shuffle` have no frames and are absent.
