# Tested archive smoke environment

The reviewer smoke is tested with CPython 3.12.13. Create a fresh environment with that interpreter and install the complete pinned lock:

```powershell
python3.12 -m venv <workspace>/artifact-smoke
<workspace>/artifact-smoke/bin/python -m pip install -r environments/smoke/requirements.txt
<workspace>/artifact-smoke/bin/python -m scripts.run_archive_smoke --output-dir <workspace>/recreated-figures
```

On Windows, use `<workspace>\artifact-smoke\Scripts\python.exe` for the final two commands. `xformers` is intentionally absent because no smoke test or compact-artifact analysis imports it and it has no macOS ARM64 wheel. GPU extraction environments remain separate and are recorded with each experiment.
