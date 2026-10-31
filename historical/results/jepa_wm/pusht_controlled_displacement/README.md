# JEPA-WM PushT controlled-displacement audit

This post-hoc exploratory run uses the same 256 episode identities, target bytes, action interventions, rendered-effect matches, and episode bootstrap as the paper-facing DINO-WM PushT control. JEPA-WM retains its released 224-pixel preprocessing, action and proprioception statistics, three-frame context, AdaLN plus RoPE predictor, and frozen DINOv2 ViT-S/14 patch tokens.

The score is final one-step visual embedding MSE and excludes proprioceptive prediction loss. Zero-minus-lawful is 0.00049 [0.00042, 0.00056] at scale 0.25 and 0.00179 [0.00151, 0.00209] at scale 1.0. Opposite-minus-lawful is 0.00098 [0.00084, 0.00112] and 0.00351 [0.00293, 0.00411]. Hue-minus-opposite is -0.03670 [-0.04434, -0.03008] at 30 degrees/0.25 and -0.12481 [-0.13905, -0.11114] at 60 degrees/0.75.

All checkpoint, target-identity, simulator, embedding-invariance, determinism, support, finite-score, and independent bootstrap-recomputation gates pass. The qualitative ordering agrees with DINO-WM under a revised predictor recipe over the same encoder family. Architecture, conditioning, context, and rollout training vary jointly, so this result does not isolate a cause.

`scores.csv` contains the raw 4,608-row episode-condition matrix. `condition_summary.csv`, `same_target_contrasts.csv`, and `construct_comparisons.csv` contain processed statistics. `selection_manifest.csv`, `provenance.json`, `smoke_validation.json`, `validation.json`, and `status.json` preserve the validated support and execution record. The source dataset, compact image bundle, checkpoints, and upstream source trees are external assets and are not redistributed.
