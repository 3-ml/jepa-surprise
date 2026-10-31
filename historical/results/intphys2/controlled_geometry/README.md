# Development-frozen controlled geometry

This directory contains the current paper-facing controlled geometry analysis for the corrected IntPhys2 representation package.

- `development_reference/` preserves the development quantities used to freeze the protocol.
- `sealed/` preserves the completed 57-scene sealed analysis, including relative displacement magnitude, mean resultant length above shuffled correspondence, cross-fitted appearance-subspace projection, random-subspace references, uncertainty intervals, sensitivity analyses, and stability records.
- `configs/intphys2_controlled_geometry.json` records the archive-facing protocol and the hashes of the scripts and protocol files used during the original execution.

The bundled compact representations under `results/intphys2/development_analysis/` and `results/intphys2/sealed_analysis/` are numerically identical to the executed arrays; only private source-revision metadata was replaced by `anonymous-source-snapshot`. Run the archive-compatible analysis with:

```powershell
python -m scripts.run_intphys2_geometry_sealed --output-dir <output-dir> --bootstrap-draws 10000 --permutation-draws 10000 --random-subspace-draws 10000 --overwrite
```

The original run used Python 3.10.11, NumPy 2.1.2, SciPy 1.15.3, and scikit-learn 1.7.2. Regeneration is deterministic for the fixed seed and draw budgets.
