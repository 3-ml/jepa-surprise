# IntPhys2 manifest map

- `intphys2_release_inventory.csv` lists released files and source hashes.
- `intphys2_confirmatory_pairs.csv` records pair, scene, label, metadata, and partition assignments.
- `intphys2_confirmatory_samples.csv` expands pairs across the four appearance conditions and fixed source-frame indices.
- `intphys2_confirmatory_transform_metrics.csv` contains per-source rendered-effect measurements.
- `intphys2_confirmatory_transform_full_audit.json` records the full transform-integrity run.
- `intphys2_confirmatory_validation.json` records the corrected split and package validation.
- `previously_inspected/` contains only the two selection artifacts needed to reconstruct the union-of-inspected-scenes exclusion; no outputs from those earlier experiments are included.

The corrected active population contains 76 development pairs from 38 scenes and 114 sealed pairs from 57 scenes after excluding 193 inspected pairs across 158 scenes.
