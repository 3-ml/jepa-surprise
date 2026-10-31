# Configurations

- `intphys2_confirmatory_shared.json` fixes the corrected IntPhys2 split, 64-frame sampling, four appearance conditions, five relative depths, shared feature views, determinism rules, and model-independent extraction contract.
- `vjepa2_intphys2_confirmatory.json` identifies the V-JEPA2-G checkpoint, architecture, preprocessing, and layer mapping.
- `videomae2_intphys2_confirmatory.json` identifies the VideoMAE2-G comparator, four-chunk temporal packing, preprocessing, and layer mapping.
- `intphys2_sealed_analysis_spec.json` records the preserved probe tasks and layers, deterministic five-fold scene cross-fit, complete out-of-fold coverage, and 10,000-draw global fixed-prediction scene bootstrap. It marks this protocol as a post-hoc correction after sealed-population inspection.
- `original_sealed_analysis_spec/README.md` is the byte-exact original specification frozen before sealed extraction; its SHA-256 is the value cited by the paper.
- `intphys2_controlled_geometry.json` records the development-frozen sealed geometry protocol, execution hashes, and integrity history.
- `intphys2_multidose_geometry.json` records the post-sealed exploratory V-JEPA2-G dose grid, extraction environment, feature hashes, and resampling budget.

Private storage locations and internal source-history identifiers were replaced without changing scientific parameters.
