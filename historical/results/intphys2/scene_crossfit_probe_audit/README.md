# IntPhys2 five-fold scene cross-fit audit

This is the current paper-facing corrective analysis. It replaces repeated random scene holdouts by holding every scene out exactly once and aggregating its pair predictions to one scene value. It reuses the existing compact corrected representations; no video decoding, checkpoint loading, or GPU extraction is performed.

## Protocol

The 38 development scenes and 57 sealed scenes are independently hash-ordered into five balanced folds. For fold `k`, fold `k` is test, fold `(k+1) mod 5` is validation, and the remaining three folds are training. Every scene is tested once, validated once, and used for training three times. All pair members and appearance variants stay with their scene, and split assignments are identical across feature families.

The fixed probe is `StandardScaler` followed by class-balanced logistic regression with `C=1`, at most 2000 iterations, and fold index as random state. Task layers remain those already selected for the paper. Samples are averaged within pair and both pairs are averaged within scene. Ten thousand bootstrap draws resample the 57 unique sealed scene values globally; model deltas are paired by fold, scene, and pair.

This protocol was designed and executed after inspection of the sealed population and is therefore a post-hoc corrective analysis. Intervals condition on the five fitted probes and omit probe-refitting and fold-assignment variability.

## Sealed results

| Feature family | Appearance | Physics | Clean-only physics |
|---|---:|---:|---:|
| V-JEPA2-G | 0.977 [0.957, 0.993] | 0.579 [0.542, 0.618] | 0.548 [0.513, 0.588] |
| VideoMAE2-G | 0.953 [0.928, 0.976] | 0.492 [0.473, 0.509] | 0.496 [0.465, 0.522] |
| PixelSummary | 0.791 [0.737, 0.845] | 0.500 [0.489, 0.510] | 0.518 [0.487, 0.548] |

Primary all-appearance physics deltas are V-JEPA2-G minus VideoMAE2-G `0.087 [0.044, 0.133]` and V-JEPA2-G minus PixelSummary `0.079 [0.041, 0.118]`. The clean-only PixelSummary delta includes zero and is not interpreted as superiority. The development cross-fit retains V-JEPA2-G `block_30` as its best tested all-appearance physics layer; sealed layers were not reselected.

## Files

Each population contains `vjepa2/`, `videomae2/`, and `pixel_summary/` probe outputs plus `statistics/`. The statistics directory contains globally scene-clustered accuracy and delta intervals, per-fold estimates, exact selected predictions, split audits, and coverage audits. `run_record.json` preserves commands, input and code hashes, output hashes, software, measured runtime, and validation status.
