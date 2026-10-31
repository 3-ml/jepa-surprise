# Superseded IntPhys2 repeated scene-holdout audit

This directory preserves the first correction from pair-held-out to scene-held-out probe splits. It is not paper-facing. Across five repeated holdouts, only 42 of 57 sealed scenes were tested, recurring scenes received unequal weight, and the bootstrap did not preserve dependence when the same scene appeared in multiple test splits.

The replacement is `../scene_crossfit_probe_audit/`: deterministic five-fold scene cross-fitting holds every scene out exactly once, aggregates its pair predictions to one scene value, and resamples all unique scenes globally. Numeric outputs and embedded historical commands in this directory remain unchanged as provenance for the superseded analysis.
