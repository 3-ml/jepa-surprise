# Post-sealed exploratory multi-dose geometry

This directory contains the current appendix analysis for V-JEPA2-G on corrected IntPhys2 development: 38 scenes, 76 matched pairs, 152 source videos, 1,520 clean or transformed samples, and five encoder depths.

The grid contains three nonzero doses for hue, brightness, and selective recoloring. `VJEPA2G_GRID_SUMMARY.json` records the paper-facing results. `GRID_FEATURE_MANIFEST.csv` preserves sample identities and feature provenance; the CSV result tables preserve dose response, direction, straightness, effective rank, curvature, tangent geometry, and held-out physics projections. The compact 20 MiB representation package already under `results/intphys2/development_analysis/vjepa2/` supplies the 608 clean and strongest-dose samples reused during extraction.

## Reproduce from external assets

Obtain IntPhys2 from the direct URL in `configs/intphys2_confirmatory_shared.json`, the checkpoint from `configs/intphys2_multidose_geometry.json`, and V-JEPA2 source at its pinned revision. Verify the recorded SHA-256 values, then run:

```powershell
python -m scripts.extract_vjepa2g_grid_replication --checkpoint <checkpoint-cache>/vitg.pt --vjepa2-root <workspace>/vjepa2 --dataset-root <dataset-cache>/intphys2 --reference-cache results/intphys2/development_analysis/vjepa2/representations.npz --cache-root <workspace>/multidose-shards
python -m scripts.pack_vjepa2g_grid_replication --reference-cache results/intphys2/development_analysis/vjepa2/representations.npz --new-cache-root <workspace>/multidose-shards --dataset-root <dataset-cache>/intphys2 --output-root <workspace>/multidose-packed
python -m scripts.run_transformation_manifold_audit --features <workspace>/multidose-packed/layer_features.npz --samples <workspace>/multidose-packed/samples.json --appearance-manifest <workspace>/multidose-packed/samples.json --output <workspace>/multidose-analysis --bootstrap-draws 10000 --null-draws 10000 --seed 260903 --model-label V-JEPA2-G --layer-names block_00,block_10,block_20,block_30,final --expected-scenes 38 --expected-pairs 76 --expected-sources 152 --image-dose-from-samples --skip-grid-inventory
python -m scripts.finalize_vjepa2g_grid_replication --report-root <workspace>/multidose-analysis --packed-root <workspace>/multidose-packed --extraction-root <workspace>/multidose-shards --pilot-root results/intphys2/multidose_geometry/vjepa2l_reference --benchmark-result results/intphys2/multidose_geometry/benchmark_1clip.json
```

The extractor computes the six intermediate-dose conditions and the packer reuses the bundled clean and strongest-dose representations. Extraction used the exact float32 checkpoint with bfloat16 autocast and sequential transformer-block CPU/GPU offload, without quantization. `configs/intphys2_multidose_geometry.json` gives the complete grid, environment, budgets, and packed-input hashes. The generated packed tensor is omitted; its expected SHA-256 is recorded, while all paper-facing processed tables and figures are included.
