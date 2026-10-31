# Environments

- `smoke/` pins the tested CPython 3.12.13 reviewer environment without GPU-only xFormers.
- `intphys2/` records the Python 3.11 GPU extraction requirements.
- `static_diagnostic/requirements.txt` covers V-JEPA2-L window scoring and pixel-motion analysis.
- `leworldmodel/requirements.txt` covers TwoRoom and PushT inference and rendering.
- `jepa_wm/requirements.txt` is the complete freeze for the JEPA-WM PushT run.
- `kubric/` contains the executed renderer Dockerfile, package lock, and runtime helper.
- `external_assets/README.md` lists public dataset, source, and checkpoint locations with revisions and hashes.
- `compute_resources.json` maps paper-facing stages to measured wall times, workloads, hardware, and retained evidence, including the Apple M4 five-fold IntPhys2 probe/bootstrap run.

GPU-specific libraries and public upstream source trees must match the revisions recorded in `configs/` and result provenance.
