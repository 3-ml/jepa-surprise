# Kubric rendering gate

Four configurations compare 16 versus 64 samples per pixel and denoising choices on one 48-frame MOVi-A scene. Each subdirectory contains the renderer manifest and wall time. `determinism.json` compares repeated decoded frames; `provenance.json`, `environment.txt`, `nvidia-smi-q.txt`, and `key_checksums.sha256` record runtime and artifact integrity.

Raw 64-sample OptiX without denoising took 39.65 and 57.97 seconds. Repeated RGB L1 was 1.12e-8 with a maximum one-level uint8 difference, selecting the eight-scene pilot branch.
