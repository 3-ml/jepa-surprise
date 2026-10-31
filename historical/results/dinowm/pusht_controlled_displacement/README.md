# DINO-WM PushT controlled-displacement audit

This paper-facing confirmatory run applies the PushT controlled construction to 256 episodes. Each lawful rendered target is reused under lawful, zero, and opposite final action blocks. Hue controls rotate the static target while preserving simulator state.

The score is final-step DINO-WM visual latent MSE against the encoded target. Zero-minus-lawful is 0.00052 [0.00046, 0.00059] at scale 0.25 and 0.00183 [0.00152, 0.00215] at scale 1.0. Opposite-minus-lawful is 0.00104 [0.00091, 0.00118] and 0.00364 [0.00304, 0.00428]. Hue-minus-opposite is -0.03082 [-0.04028, -0.02273] at 30 degrees/0.25 and -0.10987 [-0.12416, -0.09614] at 60 degrees/0.75.

`scores.csv` contains the raw episode-condition matrix. `condition_summary.csv`, `same_target_contrasts.csv`, and `construct_comparisons.csv` contain processed statistics. `selection_manifest.csv`, `provenance.json`, and `status.json` preserve the validated support and execution record. The checkpoint and source dataset are external assets and are not redistributed.
