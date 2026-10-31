# Tests

These focused tests cover the retained paper-facing code:

- IntPhys2 manifest integrity, transformations, extraction contracts, feature parity, deterministic five-fold scene cross-fitting, exactly-once scene holdout coverage, cross-model support equality, globally clustered bootstrap behavior, controlled-geometry metrics, multi-dose trajectory metrics, and rejection of duplicate or incomplete support.
- LeWorldModel intervention construction, identical-target checks, role-color rendering, and paired analyses.
- V-JEPA2-L section scoring and static controls.
- Kubric intervention validation and construct-validity analysis.

Run from the archive root with `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p no:cacheprovider tests`. Tests requiring unavailable model or simulator dependencies skip or must be run in the corresponding environment under `environments/`.
