# Cross-device native-score parity: laptop RTX 3050 vs A40 and RTX 3080

Status: OBSERVED, engineering parity check on one scene, not a scientific result.

Scene seed-230200, four conditions (clean, hue-030, target_hold, target_shuffle), Vasi's disk-block-offload regression script run unchanged except for the device-specific edits listed in provenance.json.

## Scalar and per-tubelet comparison

| condition | A40 scalar | Vasi RTX 3080 scalar | laptop scalar | abs diff vs A40 | abs diff vs Vasi | max tubelet diff vs A40 | max tubelet diff vs Vasi |
|---|---|---|---|---|---|---|---|
| clean | 0.507334 | 0.507037 | 0.507027 | 3.065e-04 | 9.716e-06 | 2.052e-03 | 5.879e-04 |
| hue-030 | 0.507470 | 0.507724 | 0.507724 | 2.541e-04 | 2.384e-07 | 1.656e-03 | 5.801e-04 |
| target_hold | 0.517055 | 0.516921 | 0.517088 | 3.278e-05 | 1.668e-04 | 1.063e-03 | 6.678e-04 |
| target_shuffle | 0.516226 | 0.515987 | 0.516048 | 1.785e-04 | 6.068e-05 | 1.391e-03 | 5.035e-04 |

Maximum absolute scalar difference: 3.065e-04 against the A40 archive and 1.668e-04 against Vasi's RTX 3080 run.
Maximum absolute per-tubelet difference: 2.052e-03 against the A40 archive and 6.678e-04 against Vasi's RTX 3080 run.
The model-input SHA-256 digests match Vasi's RTX 3080 digests for all four conditions; the predictor-output digests differ for all four (the cause was not isolated; the scalar differences above are the measured effect).

Run: GPU NVIDIA GeForce RTX 3050 Laptop GPU, torch 2.5.1+cu121 (CUDA 12.1), Python 3.10.11. Peak torch-allocated GPU memory per forward pass 493827584 bytes (471.0 MiB); peak reserved 681574400 bytes (650.0 MiB). Wall time of the reported run including model load: 118 s. Model load time: 15.6 s.

## Limitations

- One scene and one run. Nothing here generalises beyond seed-230200 or beyond these four conditions.
- The A40 and RTX 3080 numbers are reference values, not re-measured here. Different GPUs, drivers and kernels can change bf16 results.
- The 16 GB checkpoint's SHA-256 was not recomputed in this run; the script checked its byte size only. The hash recorded in provenance is the value supplied for this run.
- Device-specific edits were made: paths, the GPU free-memory guard (5 GiB lowered to 3 GiB on this 4 GiB card), the output folder, and a Python 3.10 hashing fix. None changes the model computation, but they are part of the record.
- The 0.002 scalar tolerance is Vasi's development tolerance fixed before the run. Passing it is not a device-equivalence test.
- The frames were read from the alex pilot8 render folder rather than Vasi's v1-fixture folder; the script's archive-hash check against the A40 frame sums passed.
- The torch.cuda peak counters reset per condition and exclude the load phase; total nvidia-smi usage was not sampled.

Files: scores.json (raw run output), comparison.csv, provenance.json (commands, versions, hashes, edits, memory, time), script_provenance.json (the script's own record), scripts_used/ (the edited copies).
