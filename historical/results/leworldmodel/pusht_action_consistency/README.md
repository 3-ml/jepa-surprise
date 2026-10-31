# PushT action consistency

The 256 accepted episodes use simulator-lawful targets at five scales, identical-target action counterfactuals, and fixed-state hue controls. Two candidates violating the reset-state tolerance are retained in `rejected_candidates.csv`.

`scores.csv` contains raw temporal and final MSE values. `same_target_contrasts.csv`, `construct_comparisons.csv`, `condition_summary.csv`, and `block_motion_interactions.csv` contain processed results. Selection, environment, provenance, and status files preserve execution details.

At full scale, zero-minus-lawful is 0.1668 and opposite-minus-lawful is 0.3834. Closely rendered-effect-matched 30-degree and 60-degree hue interventions exceed their opposite-action matches by 0.146 and 0.642.
