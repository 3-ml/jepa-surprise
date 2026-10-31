# Kubric environment

`Dockerfile` and `requirements.lock` define the Python 3.11, Blender Python 4.3, Kubric, PyBullet, and rendering dependencies used for MOVi-A generation. `hypertune.py` is the minimal runtime compatibility helper copied into the image. Build the image from the archive root and mount this archive, the pinned Kubric checkout, the pinned V-JEPA2 checkout, checkpoint, and a writable workspace.

For each seed in `230200 230201 230203 230207 230208 230209 230210 230211`, generate a base scene with:

```powershell
python scripts/kubric_movia_base_worker.py --camera=clevr --background=clevr --objects_set=clevr --min_num_objects=3 --max_num_objects=3 --frame_end=48 --resolution=256 --seed=<seed> --samples_per_pixel=64 --cycles_device=optix --denoiser=none --job-dir=<workspace>/base-<seed>/output --scratch_dir=<workspace>/scratch-<seed> --save_state
```

Then run the intervention and analysis chain:

```powershell
python -m scripts.prepare_kubric_intervention_manifest --base-output <workspace>/base-230200/output --base-output <workspace>/base-230201/output --base-output <workspace>/base-230203/output --base-output <workspace>/base-230207/output --base-output <workspace>/base-230208/output --base-output <workspace>/base-230209/output --base-output <workspace>/base-230210/output --base-output <workspace>/base-230211/output --output <workspace>/manifest.json
python -m scripts.render_kubric_intervention_grid --manifest <workspace>/manifest.json --output-root <workspace>/rendered --blender-python <workspace>/kubric-python --renderer-script scripts/kubric_render_intervention.py
python -m scripts.canonicalize_kubric_intervention_context --manifest <workspace>/manifest.json --rendered-root <workspace>/rendered
python -m scripts.validate_kubric_intervention_blends --manifest <workspace>/manifest.json --rendered-root <workspace>/rendered --output <workspace>/blend-validation.json
python -m scripts.analyze_kubric_intervention_rendering --manifest <workspace>/manifest.json --rendered-root <workspace>/rendered --blend-validation <workspace>/blend-validation.json --output-dir <workspace>/render-analysis
python -m scripts.run_vjepa2_kubric_construct_validity --checkpoint <checkpoint-cache>/vitg.pt --vjepa2-root <workspace>/vjepa2 --manifest <workspace>/manifest.json --rendered-root <workspace>/rendered --output-dir <workspace>/model
python -m scripts.analyze_vjepa2_kubric_construct_validity --scores <workspace>/model/scores.csv --render-metrics <workspace>/render-analysis/render_metrics.csv --output-dir <workspace>/analysis --bootstrap-seed 2308 --bootstrap-replicates 10000
```

Validation requires 48 outputs per render, byte-identical 12-frame contexts across conditions, zero maximum pre-event uint8 difference, finite values, visible event targets, hue-state invariance within `1e-6`, and physical translation error below `1e-6`. The executed tree had 72 condition outputs and 3,456 rendered PNG frames. The generated 1,187,641,118-byte tree is omitted; `results/kubric/construct_validity_pilot/` retains the frozen manifest, output inventory, checksum-manifest hash, successful readback record, and processed results.

The KuBasic asset-manifest hash was not captured. The public manifest URL and fixed scene parameters regenerate the same scientific construction, but future mutable asset releases may prevent byte-identical rerendering.
